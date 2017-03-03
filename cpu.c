/*
 * Copyright 2012 INRIA Paris-Rocquencourt
 * Copyright 2012 Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Tobias Grosser, INRIA Paris-Rocquencourt,
 * Domaine de Voluceau, Rocquenqourt, B.P. 105,
 * 78153 Le Chesnay Cedex France
 * and Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <limits.h>
#include <stdio.h>
#include <string.h>

#include <isl/aff.h>
#include <isl/ctx.h>
#include <isl/flow.h>
#include <isl/map.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>
#include <isl/constraint.h>
#include <pet.h>

#include "ppcg.h"
#include "ppcg_options.h"
#include "cpu.h"
#include "print.h"
#include "schedule.h"
#include "util.h"

/* Representation of a statement inside a generated AST.
 *
 * "stmt" refers to the original statement.
 * "ref2expr" maps the reference identifier of each access in
 * the statement to an AST expression that should be printed
 * at the place of the access.
 */
struct ppcg_stmt {
	struct pet_stmt *stmt;

	isl_id_to_ast_expr *ref2expr;
};

static void ppcg_stmt_free(void *user)
{
	struct ppcg_stmt *stmt = user;

	if (!stmt)
		return;

	isl_id_to_ast_expr_free(stmt->ref2expr);

	free(stmt);
}

/* Derive the output file name from the input file name.
 * 'input' is the entire path of the input file. The output
 * is the file name plus the additional extension.
 *
 * We will basically replace everything after the last point
 * with '.ppcg.c'. This means file.c becomes file.ppcg.c
 */
static FILE *get_output_file(const char *input, const char *output)
{
	char name[PATH_MAX];
	const char *ext;
	const char ppcg_marker[] = ".ppcg";
	int len;
	FILE *file;

	len = ppcg_extract_base_name(name, input);

	strcpy(name + len, ppcg_marker);
	ext = strrchr(input, '.');
	strcpy(name + len + sizeof(ppcg_marker) - 1, ext ? ext : ".c");

	if (!output)
		output = name;

	file = fopen(output, "w");
	if (!file) {
		fprintf(stderr, "Unable to open '%s' for writing\n", output);
		return NULL;
	}

	return file;
}

/* Data used to annotate for nodes in the ast.
 */
struct ast_node_userinfo {
	/* The for node is an openmp parallel for node. */
	int is_openmp;
};

/* Information used while building the ast.
 */
struct ast_build_userinfo {
	/* The current ppcg scop. */
	struct ppcg_scop *scop;

	/* Are we currently in a parallel for loop? */
	int in_parallel_for;
};

/* Check if the scheduling "dimension" in the "schedule" map is parallel.
 *
 * We check for parallelism by verifying that the loop does not carry any
 * dependences.
 * If the live_range_reordering option is set, then this currently
 * includes the order dependences.  In principle, non-zero order dependences
 * could be allowed, but this would require privatization and/or expansion.
 *
 * Parallelism test: if the distance is zero in all outer dimensions, then it
 * has to be zero in the current dimension as well.
 * Implementation: first, translate dependences into time space, then force
 * outer dimensions to be equal.  If the distance is zero in the current
 * dimension, then the loop is parallel.
 * The distance is zero in the current dimension if it is a subset of a map
 * with equal values for the current dimension.
 */
static int schedule_dim_is_parallel(__isl_take isl_union_map *schedule,
	unsigned dimension, struct ppcg_scop *scop)
{
	isl_union_map *deps;
	isl_map *test, *schedule_deps;
	unsigned i, is_parallel;

	deps = isl_union_map_copy(scop->dep_flow);
	deps = isl_union_map_union(deps, isl_union_map_copy(scop->dep_false));
	if (scop->options->live_range_reordering) {
		isl_union_map *order = isl_union_map_copy(scop->dep_order);
		deps = isl_union_map_union(deps, order);
	}
	deps = isl_union_map_apply_range(deps, isl_union_map_copy(schedule));
	deps = isl_union_map_apply_domain(deps, schedule);

	if (isl_union_map_is_empty(deps)) {
		isl_union_map_free(deps);
		return 1;
	}

	schedule_deps = isl_map_from_union_map(deps);

	for (i = 0; i < dimension; i++)
		schedule_deps = isl_map_equate(schedule_deps, isl_dim_out, i,
					       isl_dim_in, i);

	test = isl_map_universe(isl_map_get_space(schedule_deps));
	test = isl_map_equate(test, isl_dim_out, dimension, isl_dim_in,
			      dimension);
	is_parallel = isl_map_is_subset(schedule_deps, test);

	isl_map_free(test);
	isl_map_free(schedule_deps);

	return is_parallel;
}

/* Check if the current scheduling dimension is parallel.
 */
static int ast_schedule_dim_is_parallel(__isl_keep isl_ast_build *build,
	struct ppcg_scop *scop)
{
	isl_union_map *schedule, *deps;
	isl_map *schedule_deps, *test;
	isl_space *schedule_space;
	unsigned dimension;

	schedule = isl_ast_build_get_schedule(build);
	schedule_space = isl_ast_build_get_schedule_space(build);

	dimension = isl_space_dim(schedule_space, isl_dim_out) - 1;
	isl_space_free(schedule_space);
	return schedule_dim_is_parallel(schedule, dimension, scop);
}

/* Mark a for node openmp parallel, if it is the outermost parallel for node.
 */
static void mark_openmp_parallel(__isl_keep isl_ast_build *build,
	struct ast_build_userinfo *build_info,
	struct ast_node_userinfo *node_info)
{
	if (build_info->in_parallel_for)
		return;

	if (ast_schedule_dim_is_parallel(build, build_info->scop)) {
		build_info->in_parallel_for = 1;
		node_info->is_openmp = 1;
	}
}

/* Allocate an ast_node_info structure and initialize it with default values.
 */
static struct ast_node_userinfo *allocate_ast_node_userinfo()
{
	struct ast_node_userinfo *node_info;
	node_info = (struct ast_node_userinfo *)
		malloc(sizeof(struct ast_node_userinfo));
	node_info->is_openmp = 0;
	return node_info;
}

/* Free an ast_node_info structure.
 */
static void free_ast_node_userinfo(void *ptr)
{
	struct ast_node_userinfo *info;
	info = (struct ast_node_userinfo *) ptr;
	free(info);
}

/* This method is executed before the construction of a for node. It creates
 * an isl_id that is used to annotate the subsequently generated ast for nodes.
 *
 * In this function we also run the following analyses:
 *
 * 	- Detection of openmp parallel loops
 */
static __isl_give isl_id *ast_build_before_for(
	__isl_keep isl_ast_build *build, void *user)
{
	isl_id *id;
	struct ast_build_userinfo *build_info;
	struct ast_node_userinfo *node_info;

	build_info = (struct ast_build_userinfo *) user;
	node_info = allocate_ast_node_userinfo();
	id = isl_id_alloc(isl_ast_build_get_ctx(build), "", node_info);
	id = isl_id_set_free_user(id, free_ast_node_userinfo);

	mark_openmp_parallel(build, build_info, node_info);

	return id;
}

/* This method is executed after the construction of a for node.
 *
 * It performs the following actions:
 *
 * 	- Reset the 'in_parallel_for' flag, as soon as we leave a for node,
 * 	  that is marked as openmp parallel.
 *
 */
static __isl_give isl_ast_node *ast_build_after_for(
	__isl_take isl_ast_node *node, __isl_keep isl_ast_build *build,
	void *user)
{
	isl_id *id;
	struct ast_build_userinfo *build_info;
	struct ast_node_userinfo *info;

	id = isl_ast_node_get_annotation(node);
	info = isl_id_get_user(id);

	if (info && info->is_openmp) {
		build_info = (struct ast_build_userinfo *) user;
		build_info->in_parallel_for = 0;
	}

	isl_id_free(id);

	return node;
}

/* Find the element in scop->stmts that has the given "id".
 */
static struct pet_stmt *find_stmt(struct ppcg_scop *scop, __isl_keep isl_id *id)
{
	int i;

	for (i = 0; i < scop->pet->n_stmt; ++i) {
		struct pet_stmt *stmt = scop->pet->stmts[i];
		isl_id *id_i;

		id_i = isl_set_get_tuple_id(stmt->domain);
		isl_id_free(id_i);

		if (id_i == id)
			return stmt;
	}

	isl_die(isl_id_get_ctx(id), isl_error_internal,
		"statement not found", return NULL);
}

/* Print a user statement in the generated AST.
 * The ppcg_stmt has been attached to the node in at_each_domain.
 */
static __isl_give isl_printer *print_user(__isl_take isl_printer *p,
	__isl_take isl_ast_print_options *print_options,
	__isl_keep isl_ast_node *node, void *user)
{
	struct ppcg_stmt *stmt;
	isl_id *id;

	id = isl_ast_node_get_annotation(node);
	stmt = isl_id_get_user(id);
	isl_id_free(id);

	p = pet_stmt_print_body(stmt->stmt, p, stmt->ref2expr);

	isl_ast_print_options_free(print_options);

	return p;
}


/* Print a for loop node as an openmp parallel loop.
 *
 * To print an openmp parallel loop we print a normal for loop, but add
 * "#pragma openmp parallel for" in front.
 *
 * Variables that are declared within the body of this for loop are
 * automatically openmp 'private'. Iterators declared outside of the
 * for loop are automatically openmp 'shared'. As ppcg declares all iterators
 * at the position where they are assigned, there is no need to explicitly mark
 * variables. Their automatically assigned type is already correct.
 *
 * This function only generates valid OpenMP code, if the ast was generated
 * with the 'atomic-bounds' option enabled.
 *
 */
static __isl_give isl_printer *print_for_with_openmp(
	__isl_keep isl_ast_node *node, __isl_take isl_printer *p,
	__isl_take isl_ast_print_options *print_options)
{
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "#pragma omp parallel for");
	p = isl_printer_end_line(p);

	p = isl_ast_node_for_print(node, p, print_options);

	return p;
}

/* Print a for node.
 *
 * Depending on how the node is annotated, we either print a normal
 * for node or an openmp parallel for node.
 */
static __isl_give isl_printer *print_for(__isl_take isl_printer *p,
	__isl_take isl_ast_print_options *print_options,
	__isl_keep isl_ast_node *node, void *user)
{
	isl_id *id;
	int openmp;

	openmp = 0;
	id = isl_ast_node_get_annotation(node);

	if (id) {
		struct ast_node_userinfo *info;

		info = (struct ast_node_userinfo *) isl_id_get_user(id);
		if (info && info->is_openmp)
			openmp = 1;
	}

	if (openmp)
		p = print_for_with_openmp(node, p, print_options);
	else
		p = isl_ast_node_for_print(node, p, print_options);

	isl_id_free(id);

	return p;
}

/* Index transformation callback for pet_stmt_build_ast_exprs.
 *
 * "index" expresses the array indices in terms of statement iterators
 * "iterator_map" expresses the statement iterators in terms of
 * AST loop iterators.
 *
 * The result expresses the array indices in terms of
 * AST loop iterators.
 */
static __isl_give isl_multi_pw_aff *pullback_index(
	__isl_take isl_multi_pw_aff *index, __isl_keep isl_id *id, void *user)
{
	isl_pw_multi_aff *iterator_map = user;

	iterator_map = isl_pw_multi_aff_copy(iterator_map);
	return isl_multi_pw_aff_pullback_pw_multi_aff(index, iterator_map);
}

/* Transform the accesses in the statement associated to the domain
 * called by "node" to refer to the AST loop iterators, construct
 * corresponding AST expressions using "build",
 * collect them in a ppcg_stmt and annotate the node with the ppcg_stmt.
 */
static __isl_give isl_ast_node *at_each_domain(__isl_take isl_ast_node *node,
	__isl_keep isl_ast_build *build, void *user)
{
	struct ppcg_scop *scop = user;
	isl_ast_expr *expr, *arg;
	isl_ctx *ctx;
	isl_id *id;
	isl_map *map;
	isl_pw_multi_aff *iterator_map;
	struct ppcg_stmt *stmt;

	ctx = isl_ast_node_get_ctx(node);
	stmt = isl_calloc_type(ctx, struct ppcg_stmt);
	if (!stmt)
		goto error;

	expr = isl_ast_node_user_get_expr(node);
	arg = isl_ast_expr_get_op_arg(expr, 0);
	isl_ast_expr_free(expr);
	id = isl_ast_expr_get_id(arg);
	isl_ast_expr_free(arg);
	stmt->stmt = find_stmt(scop, id);
	isl_id_free(id);
	if (!stmt->stmt)
		goto error;

	map = isl_map_from_union_map(isl_ast_build_get_schedule(build));
	map = isl_map_reverse(map);
	iterator_map = isl_pw_multi_aff_from_map(map);
	stmt->ref2expr = pet_stmt_build_ast_exprs(stmt->stmt, build,
				    &pullback_index, iterator_map, NULL, NULL);
	isl_pw_multi_aff_free(iterator_map);

	id = isl_id_alloc(isl_ast_node_get_ctx(node), NULL, stmt);
	id = isl_id_set_free_user(id, &ppcg_stmt_free);
	return isl_ast_node_set_annotation(node, id);
error:
	ppcg_stmt_free(stmt);
	return isl_ast_node_free(node);
}

/* Set *depth (initialized to 0 by the caller) to the maximum
 * of the schedule depths of the leaf nodes for which this function is called.
 */
static isl_bool update_depth(__isl_keep isl_schedule_node *node, void *user)
{
	int *depth = user;
	int node_depth;

	if (isl_schedule_node_get_type(node) != isl_schedule_node_leaf)
		return isl_bool_true;
	node_depth = isl_schedule_node_get_schedule_depth(node);
	if (node_depth > *depth)
		*depth = node_depth;

	return isl_bool_false;
}

/* This function is called for each node in a CPU AST.
 * In case of a user node, print the macro definitions required
 * for printing the AST expressions in the annotation, if any.
 * For other nodes, return true such that descendants are also
 * visited.
 *
 * In particular, print the macro definitions needed for the substitutions
 * of the original user statements.
 */
static isl_bool at_node(__isl_keep isl_ast_node *node, void *user)
{
	struct ppcg_stmt *stmt;
	isl_id *id;
	isl_printer **p = user;

	if (isl_ast_node_get_type(node) != isl_ast_node_user)
		return isl_bool_true;

	id = isl_ast_node_get_annotation(node);
	stmt = isl_id_get_user(id);
	isl_id_free(id);

	if (!stmt)
		return isl_bool_error;

	*p = ppcg_print_body_macros(*p, stmt->ref2expr);
	if (!*p)
		return isl_bool_error;

	return isl_bool_false;
}

/* Print the required macros for the CPU AST "node" to "p",
 * including those needed for the user statements inside the AST.
 */
static __isl_give isl_printer *cpu_print_macros(__isl_take isl_printer *p,
	__isl_keep isl_ast_node *node)
{
	if (isl_ast_node_foreach_descendant_top_down(node, &at_node, &p) < 0)
		return isl_printer_free(p);
	p = ppcg_print_macros(p, node);
	return p;
}

/* Code generate the scop 'scop' using "schedule"
 * and print the corresponding C code to 'p'.
 */
static __isl_give isl_printer *print_scop(struct ppcg_scop *scop,
	__isl_take isl_schedule *schedule, __isl_take isl_printer *p,
	struct ppcg_options *options)
{
	isl_ctx *ctx = isl_printer_get_ctx(p);
	isl_ast_build *build;
	isl_ast_print_options *print_options;
	isl_ast_node *tree;
	isl_id_list *iterators;
	struct ast_build_userinfo build_info;
	int depth;

	depth = 0;
	if (isl_schedule_foreach_schedule_node_top_down(schedule, &update_depth,
						&depth) < 0)
		goto error;

	build = isl_ast_build_alloc(ctx);
	iterators = ppcg_scop_generate_names(scop, depth, "c");
	build = isl_ast_build_set_iterators(build, iterators);
	build = isl_ast_build_set_at_each_domain(build, &at_each_domain, scop);

	if (options->openmp) {
		build_info.scop = scop;
		build_info.in_parallel_for = 0;

		build = isl_ast_build_set_before_each_for(build,
							&ast_build_before_for,
							&build_info);
		build = isl_ast_build_set_after_each_for(build,
							&ast_build_after_for,
							&build_info);
	}

	tree = isl_ast_build_node_from_schedule(build, schedule);
	isl_ast_build_free(build);

	print_options = isl_ast_print_options_alloc(ctx);
	print_options = isl_ast_print_options_set_print_user(print_options,
							&print_user, NULL);

	print_options = isl_ast_print_options_set_print_for(print_options,
							&print_for, NULL);

	p = cpu_print_macros(p, tree);
	p = isl_ast_node_print(tree, p, print_options);

	isl_ast_node_free(tree);

	return p;
error:
	isl_schedule_free(schedule);
	isl_printer_free(p);
	return NULL;
}

/* Tile the band node "node" with tile sizes "sizes" and
 * mark all members of the resulting tile node as "atomic".
 */
static __isl_give isl_schedule_node *tile(__isl_take isl_schedule_node *node,
	__isl_take isl_multi_val *sizes)
{
	node = isl_schedule_node_band_tile(node, sizes);
	node = ppcg_set_schedule_node_type(node, isl_ast_loop_atomic);

	return node;
}

struct filter_carried_dependences_data {
	isl_union_map *schedule;
	isl_union_map *result;
};

/* Carrying check: if all points in the scheduled dependence map domain are
 * lexicograhically strictly less than their range counterparts, the
 * dependence is carried (or strictly satisfied) by the given schedule.
 * In practice, we check if an intersection of the scheduled dependence map
 * with a less-than map is equal to the schedule depdendence map.
 */
static isl_stat filter_out_carried_dependences_one(__isl_take isl_map *dependence,
	void *user)
{
	struct filter_carried_dependences_data *data = user;
	isl_map *dep = isl_map_copy(dependence);
	isl_map *comparison;
	isl_union_map *udep = isl_union_map_from_map(dep);
	isl_space *space;
	isl_bool carried;

	udep = isl_union_map_apply_domain(udep,
		isl_union_map_copy(data->schedule));
	udep = isl_union_map_apply_range(udep,
		isl_union_map_copy(data->schedule));
	if (isl_union_map_is_empty(udep)) {
		isl_union_map_free(udep);
		isl_map_free(dependence);
		return isl_stat_ok;
	}
	dep = isl_map_from_union_map(udep);

	space = isl_map_get_space(dep);
	space = isl_space_domain(space);
	comparison = isl_map_lex_lt(space);
	comparison = isl_map_intersect(isl_map_copy(dep), comparison);
	carried = isl_map_is_equal(dep, comparison);
	isl_map_free(dep);
	isl_map_free(comparison);

	if (carried < 0) {
		isl_map_free(dependence);
		return isl_stat_error;
	}

	if (carried == isl_bool_false)
		data->result = isl_union_map_add_map(data->result, dependence);
	else
		isl_map_free(dependence);

	return isl_stat_ok;
}

/* Keep only those dependence maps from the union "dependences" that are
 * not carried (strictly satisfied) by the given "schedule".
 */
static __isl_give isl_union_map *filter_out_carried_dependences(
	__isl_take isl_union_map *dependences, __isl_keep isl_union_map *schedule)
{
	isl_space *space = isl_union_map_get_space(dependences);
	isl_union_map *result = isl_union_map_empty(space);
	struct filter_carried_dependences_data data = { schedule, result };
	isl_stat r = isl_union_map_foreach_map(dependences,
		&filter_out_carried_dependences_one, &data);
	isl_union_map_free(dependences);
	if (r == isl_stat_error)
		return isl_union_map_free(data.result);
	return data.result;
}

static isl_stat replicate_tag_one(__isl_take isl_map *map, void *user)
{
	isl_union_map **result = user;
	isl_id *id = isl_map_get_tuple_id(map, isl_dim_in);
	map = isl_map_set_tuple_id(map, isl_dim_out, id);
	*result = isl_union_map_add_map(*result, map);
	return isl_stat_ok;
}

static __isl_give isl_union_map *replicate_tag(__isl_take isl_union_map *umap)
{
	isl_space *space = isl_union_map_get_space(umap);
	isl_union_map *result = isl_union_map_empty(space);
	isl_stat r = isl_union_map_foreach_map(umap, &replicate_tag_one, &result);
	isl_union_map_free(umap);
	if (r == isl_stat_error)
			result = isl_union_map_free(result);
	return result;
}

static __isl_give isl_union_map *bandwise_dependences(
	__isl_take isl_schedule_node *node,
	__isl_keep isl_union_set *domain, struct ppcg_scop *scop)
{
	isl_schedule_node **band_nodes;
	int n_band_nodes = 1;
	isl_ctx *ctx;
	int i;
	isl_union_map *validity;
	isl_union_map *schedule, *partial_schedule;
	isl_space *space;

	fprintf(stderr, "tis is me\n");
	isl_schedule_node_dump(node);

	if (!node || isl_schedule_node_get_type(node) != isl_schedule_node_band)
		return NULL;

		fprintf(stderr, "tis is me\n");

	// FIXME: account for live-range reodering here
	validity = isl_union_map_copy(scop->dep_flow);
	validity = isl_union_map_union(validity,
		isl_union_map_copy(scop->dep_false));
	isl_union_map_dump(validity);

	ctx = isl_schedule_node_get_ctx(node);

	band_nodes = isl_alloc_array(ctx, isl_schedule_node *, 1);
	band_nodes[0] = isl_schedule_node_copy(node);

	// Find all bands from root to "node", in inverse order.
	while (isl_schedule_node_has_parent(node)) {
		node = isl_schedule_node_parent(node);
		if (isl_schedule_node_get_type(node) != isl_schedule_node_band)
			continue;
		++n_band_nodes;
		band_nodes = isl_realloc_array(ctx, band_nodes, isl_schedule_node *,
			n_band_nodes);
		band_nodes[n_band_nodes - 1] = isl_schedule_node_copy(node);
	}
	isl_schedule_node_free(node);

	// Filter out validity dependences that are satisfied by outer bands.
	// Construct schedule from partials in progress.
	// Do not take into account the current band (the dependences it carries
	// must be included in the result).
	space = isl_union_set_get_space(scop->domain);
	space = isl_space_set_from_params(space);
	schedule = isl_union_map_from_domain_and_range(
		isl_union_set_copy(scop->domain),
		isl_union_set_from_set(isl_set_universe(space)));

	space = isl_union_set_get_space(scop->domain);

	// Only keep those dependences that are relevant for the tile domain
	// validity = isl_union_map_intersect_domain(validity,
	// 	isl_union_set_copy(domain));
	// validity = isl_union_map_intersect_domain(validity,
	// 	isl_union_set_copy(domain));

	for (i = n_band_nodes - 1; i >= 0; --i) {
		node = band_nodes[i];
		partial_schedule =
			isl_schedule_node_band_get_partial_schedule_union_map(node);
		schedule =
			isl_union_map_flat_range_product(schedule, partial_schedule);

		// Dependences that are strongly satisfied by current schedule.
		if (i != 0)
			validity = filter_out_carried_dependences(validity, schedule);
		isl_schedule_node_free(node);
	}
	schedule = replicate_tag(schedule);
	// validity = isl_union_map_apply_domain(validity,
	// 	isl_union_map_copy(schedule));
	// validity = isl_union_map_apply_range(validity,
	// 	isl_union_map_copy(schedule));

	isl_space_free(space);
	isl_union_map_free(schedule);

	return validity;
}

static __isl_give isl_schedule_node *continue_find_next_child_node_band(
	__isl_take isl_schedule_node *node);

/* Finds first band child of "node" in depth-first traversal order,
 * including the given node.
 * Intended to be called on tree root.
 */
static __isl_give isl_schedule_node *find_next_child_node_band(
	__isl_take isl_schedule_node *node)
{
	int i, n;
	isl_schedule_node *child;

	if (!node)
		return NULL;

	if (isl_schedule_node_get_type(node) == isl_schedule_node_band)
		return node;
	if (isl_schedule_node_has_children(node) != isl_bool_true)
		return NULL;
	node = isl_schedule_node_first_child(node);
	child = find_next_child_node_band(isl_schedule_node_copy(node));
	if (child) {
		isl_schedule_node_free(node);
		return child;
	}
	return continue_find_next_child_node_band(node);
}

/* Continue looking for the first band child of "node" in the depth-first order
 * starting from the next sibling of "node" inclusive.  Assumes that the
 * subtree of the current node was already traversed.
 */
static __isl_give isl_schedule_node *continue_find_next_child_node_band(
	__isl_take isl_schedule_node *node)
{
	if (isl_schedule_node_has_next_sibling(node) != isl_bool_true)
		return NULL;

	node = isl_schedule_node_next_sibling(node);
	return find_next_child_node_band(node);
}

static __isl_give isl_schedule_node *schedule_get_single_node_band(
	__isl_take isl_schedule *schedule)
{
	isl_schedule_node *ancestor = NULL, *other;
	isl_schedule_node *node = isl_schedule_get_root(schedule);
	isl_ctx *ctx = isl_schedule_node_get_ctx(node);

	node = find_next_child_node_band(node);
	if (!node) // not found
		goto error;

	ancestor = isl_schedule_node_copy(node);

	while (1) {
		isl_bool r;
		other = continue_find_next_child_node_band(ancestor);
		if (other) {// found another one
			node = isl_schedule_node_free(node);
			break;
		}
		r = isl_schedule_node_has_parent(ancestor);
		if (r == isl_bool_error)
			goto error;
		if (r != isl_bool_true)
			break;
		ancestor = isl_schedule_node_parent(ancestor);
	}

	isl_schedule_node_free(ancestor);
	isl_schedule_free(schedule);
	return node;

error:
	isl_schedule_node_free(ancestor);
	isl_schedule_free(schedule);
	return NULL;
}

static inline __isl_give isl_set *fix_set_dims_as_params(
	__isl_take isl_set *bset, int n_out)
{
	int n_param, i;
	isl_local_space *local_space;
	isl_constraint *constraint;
	isl_space *space;

	bset = isl_set_add_dims(bset, isl_dim_param, n_out);
	n_param = isl_set_n_param(bset);
	for (i = 0; i < n_out; i++) {
		space = isl_set_get_space(bset);
		local_space = isl_local_space_from_space(space);
		constraint = isl_constraint_alloc_equality(local_space);
		constraint = isl_constraint_set_coefficient_si(constraint,
			isl_dim_set, i, -1);
		constraint = isl_constraint_set_coefficient_si(constraint,
			isl_dim_param, n_param - n_out + i, 1);
		bset = isl_set_add_constraint(bset, constraint);

		char name[10];
		snprintf(name, 10, "ppcgp%d", i);
		bset = isl_set_set_dim_name(bset, isl_dim_param,
			n_param - n_out + i, name);
	}
	return bset;
}

static __isl_give isl_map *fix_map_dims_as_params(
	__isl_take isl_map *map, int n_out)
{
	int n_param, i;
	isl_local_space *local_space;
	isl_constraint *constraint;
	isl_space *space;

	map = isl_map_add_dims(map, isl_dim_param, n_out);
	n_param = isl_map_n_param(map);
	for (i = 0; i < n_out; i++) {
		space = isl_map_get_space(map);
		local_space = isl_local_space_from_space(space);
		constraint = isl_constraint_alloc_equality(local_space);
		constraint = isl_constraint_set_coefficient_si(constraint,
			isl_dim_out, i, -1);
		constraint = isl_constraint_set_coefficient_si(constraint,
			isl_dim_param, n_param - n_out + i, 1);
		map = isl_map_add_constraint(map, constraint);

		char name[10];
		snprintf(name, 10, "ppcgp%d", i);
		map = isl_map_set_dim_name(map, isl_dim_param,
			n_param - n_out + i, name);
	}
	return map;
}

struct fix_umap_dims_data {
	isl_union_map *result;
	int n_out;
	int keep;
	int equal_id;
};

static isl_stat fix_umap_dims_one(__isl_take isl_map *map, void *user)
{
	struct fix_umap_dims_data *data = user;
	if (data->n_out != 0) {
		map = fix_map_dims_as_params(map, data->n_out);
		if (!data->keep)
			map = isl_map_project_out(map, isl_dim_out, 0, data->n_out);
	}
	if (data->equal_id)
		map = isl_map_set_tuple_id(map, isl_dim_out,
			isl_map_get_tuple_id(map, isl_dim_in));

	data->result = isl_union_map_add_map(data->result, map);
	return data->result ? isl_stat_ok : isl_stat_error;
}

static __isl_give isl_union_map *fix_union_map_dims_as_params(
	__isl_take isl_union_map *umap, int n_out, int keep, int equal_id)
{
	isl_space *space = isl_union_map_get_space(umap);
	struct fix_umap_dims_data data = { isl_union_map_empty(space), n_out,
		keep, equal_id };
	if (isl_union_map_foreach_map(umap, &fix_umap_dims_one, &data) < 0)
		isl_union_map_free(data.result);
	return data.result;
}

struct umap_project_out_data {
	isl_union_map *result;
	int n_dim;
};

static isl_stat map_project_out_one(__isl_take isl_map *map, void *user)
{
	struct umap_project_out_data *data = user;
	map = isl_map_project_out(map, isl_dim_out, 0, data->n_dim);
	data->result = isl_union_map_add_map(data->result, map);
	if (!data->result)
		return isl_stat_error;
	return isl_stat_ok;
}

static __isl_give isl_union_map *union_map_project_out_out_dims(
	__isl_take isl_union_map *umap, int n_dim)
{
	isl_space *space = isl_union_map_get_space(umap);
	struct umap_project_out_data data = { isl_union_map_empty(space), n_dim };
	if (isl_union_map_foreach_map(umap, &map_project_out_one, &data) < 0)
		data.result = isl_union_map_free(data.result);
	isl_union_map_free(umap);
	return data.result;
}

static unsigned schedule_node_prefix_n_member(
	__isl_keep isl_schedule_node *node)
{
	unsigned n_member = 0;
	isl_schedule_node *nd = isl_schedule_node_copy(node);
	if (!nd)
		return 0;

	while (isl_schedule_node_has_parent(nd) == isl_bool_true) {
		nd = isl_schedule_node_parent(nd);
		if (isl_schedule_node_get_type(nd) != isl_schedule_node_band)
			continue;
		n_member += isl_schedule_node_band_n_member(nd);
	}
	return n_member;
}

inline static unsigned schedule_node_total_n_member(
	__isl_keep isl_schedule_node *node)
{
	return schedule_node_prefix_n_member(node) +
		isl_schedule_node_band_n_member(node);
}

static isl_union_map *schedule_node_band_get_ascendant_schedule_step(
	__isl_take isl_schedule_node *node, __isl_take isl_union_map *sched)
{
	isl_union_map *partial_schedule;
	isl_bool has_parent;

	if (isl_schedule_node_get_type(node) == isl_schedule_node_band) {
		partial_schedule =
			isl_schedule_node_band_get_partial_schedule_union_map(node);
		sched = isl_union_map_flat_range_product(partial_schedule, sched);
	}

	has_parent = isl_schedule_node_has_parent(node);
	if (has_parent < 0) {
		isl_schedule_node_free(node);
		return isl_union_map_free(sched);
	}
	if (has_parent)
		return schedule_node_band_get_ascendant_schedule_step(
			isl_schedule_node_parent(node), sched);
	else
		return sched;
}

static isl_union_map *schedule_node_band_get_ascendant_schedule(
	__isl_keep isl_schedule_node *node)
{
	isl_union_set *domain;
	isl_space *space;
	isl_union_map *sched;

	if (!node)
		return NULL;

	domain = isl_schedule_node_get_universe_domain(node);
	space = isl_union_set_get_space(domain);
	space = isl_space_set_from_params(space);
	sched = isl_union_map_from_domain_and_range(domain,
		isl_union_set_from_set(isl_set_universe(space)));

	return schedule_node_band_get_ascendant_schedule_step(
		isl_schedule_node_copy(node), sched);
}

static isl_bool remove_extra_parameters(__isl_keep isl_schedule_node *node, void *user)
{
	// band -> update schedule to remove parameters
	// filter -> update filter
}

// point_node = point-loop from rescheduled
// tile_node = tile-loop from original
static __isl_give isl_schedule_node *remove_tile_parameters(
	__isl_take isl_schedule_node *point_node,
	__isl_keep isl_schedule_node *tile_node)
{
	isl_union_map *tile_schedule =
		isl_schedule_node_band_get_partial_schedule_union_map(tile_node);
	isl_union_map *point_schedule =
		isl_schedule_node_band_get_partial_schedule_union_map(point_node);
	unsigned n_member = schedule_node_total_n_member(tile_node);
	// isl_union_set *domain =
	// 	isl_schedule_node_get_domain(tile_node);

	// tile_schedule = isl_union_map_intersect_domain(tile_schedule, domain);
	tile_schedule = fix_union_map_dims_as_params(tile_schedule,
		n_member, 1, 1);

	// isl_schedule_node_foreach_descendant_top_down / remove_extra_parameters

	point_schedule = isl_union_map_flat_range_product(tile_schedule,
		point_schedule);
	point_schedule = isl_union_map_project_out(point_schedule, isl_dim_param,
		isl_union_map_dim(point_schedule, isl_dim_param) - n_member, n_member);
	point_schedule = union_map_project_out_out_dims(point_schedule, n_member);

	point_node = isl_schedule_node_band_reset_schedule(point_node,
		isl_multi_union_pw_aff_from_union_map(point_schedule));
	return point_node;
}


static __isl_give isl_union_set *bandwise_domain(
	__isl_keep isl_schedule_node *node, struct ppcg_scop *scop)
{
	isl_union_set *band_domain;

	if (isl_schedule_node_get_type(node) != isl_schedule_node_band)
		return NULL;

	// this should give only the dimension up to the end of current node (i.e. not including e.g. the point loop dimensions that were created during tiling)
	// band_domain = isl_schedule_node_get_universe_domain(node);
	// band_domain = isl_union_set_intersect(band_domain,
	// 	isl_union_set_copy(scop->domain));

// #if 0
		isl_schedule_node *point_loop = isl_schedule_node_copy(node);
		isl_union_set *dom = isl_schedule_node_get_domain(point_loop);
		isl_union_map *umap1 =
			isl_schedule_node_band_get_partial_schedule_union_map(point_loop);
		// point_loop = isl_schedule_node_first_child(point_loop);
		// isl_union_map *umap2 =
		// 	isl_schedule_node_band_get_partial_schedule_union_map(point_loop);
		// isl_union_map *sched = isl_union_map_flat_range_product(umap1, umap2);
		isl_union_map *sched = umap1;
		unsigned n_member = schedule_node_total_n_member(point_loop);


		dom = isl_union_set_apply(dom, isl_union_map_copy(sched));
		isl_set *dom_set = isl_set_from_union_set(dom);
		dom_set = fix_set_dims_as_params(dom_set, n_member);
		dom = isl_union_set_from_set(dom_set);
		// dom = isl_union_set_apply(dom, isl_union_map_reverse(sched));
		dom = isl_union_set_preimage_union_pw_multi_aff(dom,
			isl_union_pw_multi_aff_from_union_map(sched));

#if 0
		sched = fix_union_map_dims_as_params(sched, 3, 0, 1);

		dom = isl_union_set_apply(dom, isl_union_map_copy(sched));
#endif

#if 0
		isl_union_pw_multi_aff *upma =
			isl_union_pw_multi_aff_from_union_map(sched);
		dom = isl_union_set_preimage_union_pw_multi_aff(dom, upma);
#endif

#if 0
//v2
		sched = replicate_tag(sched);
		dom = isl_union_set_apply(dom, isl_union_map_copy(sched));
		isl_set *dom_set = isl_set_from_union_set(dom);

		dom_set = fix_set_dims_as_params(dom_set, 3);
		isl_id *id = isl_set_get_tuple_id(dom_set);
		dom_set = isl_set_project_out(dom_set, isl_dim_out, 0, 3);
		dom_set = isl_set_set_tuple_id(dom_set, id);
		dom = isl_union_set_from_set(dom_set);
#endif

		// isl_union_pw_multi_aff *upma = isl_union_pw_multi_aff_from_union_map(sched);
		// dom = isl_union_set_preimage_union_pw_multi_aff(dom, upma);

		fprintf(stderr, "[ppcg] domain: ");
		isl_union_set_dump(dom);

		band_domain = dom;
// #endif

	return band_domain;
}

static __isl_give isl_schedule_constraints *proximity_validity_constraints(
	__isl_take isl_schedule_node *node,
	__isl_take isl_schedule_node *tiled_node, struct ppcg_scop *scop)
{
	isl_union_set *domain;
	isl_union_map *validity, *proximity;
	isl_schedule_constraints *constraints;

	domain = bandwise_domain(tiled_node, scop);
	validity = bandwise_dependences(node, domain, scop);

	proximity = isl_union_map_copy(scop->dep_flow);
	proximity = isl_union_map_intersect_domain(proximity,
		isl_union_set_copy(domain));
	proximity = isl_union_map_intersect_range(proximity,
		isl_union_set_copy(domain));

	constraints = isl_schedule_constraints_on_domain(domain);
	constraints = isl_schedule_constraints_set_validity(constraints,
		validity);
	constraints = isl_schedule_constraints_set_proximity(constraints,
		proximity);// FIXME: account for live-range reodering

	return constraints;
}

static __isl_give isl_schedule_node *reschedule_whole_component(
	__isl_take isl_schedule_constraints *constraints)
{
	int orig_schedule_whole_component, orig_schedule_outer_coincidence;
	isl_schedule *schedule;
	isl_schedule_node *rescheduled_node;
	isl_ctx *ctx = isl_schedule_constraints_get_ctx(constraints);

	orig_schedule_whole_component =
		isl_options_get_schedule_whole_component(ctx);
	isl_options_set_schedule_whole_component(ctx, 1);
	orig_schedule_outer_coincidence =
		isl_options_get_schedule_outer_coincidence(ctx);
	isl_options_set_schedule_outer_coincidence(ctx, 0);

	// This schedule MUST have only one band, traverse the tree until we find
	// first band.  Continue traversal to ensure it is the only one.
	schedule = isl_schedule_constraints_compute_schedule(constraints);
	rescheduled_node = schedule_get_single_node_band(schedule);
	isl_options_set_schedule_whole_component(ctx,
		orig_schedule_whole_component);
	isl_options_set_schedule_outer_coincidence(ctx,
		orig_schedule_outer_coincidence);
	return rescheduled_node;
}

static __isl_give isl_schedule_node *reschedule_tile_loops(
	__isl_take isl_schedule_node *node, struct ppcg_scop *scop)
{
	isl_union_map *coincidence;
	isl_union_set *band_domain;
	isl_schedule_constraints *constraints;

	constraints = proximity_validity_constraints(
		isl_schedule_node_copy(node), isl_schedule_node_copy(node), scop);
	band_domain = isl_schedule_constraints_get_domain(constraints);

	coincidence = isl_union_map_union(isl_union_map_copy(scop->dep_flow),
		isl_union_map_copy(scop->dep_false)); // FIXME: account for live-range reodering
	coincidence = isl_union_map_intersect_domain(coincidence,
		isl_union_set_copy(band_domain));
	coincidence = isl_union_map_intersect_range(coincidence, band_domain);
	constraints = isl_schedule_constraints_set_coincidence(constraints,
		coincidence);

	return isl_schedule_constraints_recompute_schedule(constraints, node);

	// return reschedule_whole_component(constraints);
}

static __isl_give isl_schedule_node *reschedule_point_loops(
	__isl_take isl_schedule_node *node, __isl_take isl_schedule_node *tiled_node, struct ppcg_scop *scop)
{
	isl_schedule_constraints *constraints;
	isl_union_map *spatial_proximity;
	isl_union_set *band_domain, *access_set;
	isl_union_map *access_map, *counted_accesses;

	constraints = proximity_validity_constraints(
		isl_schedule_node_copy(node), tiled_node, scop);
	band_domain = isl_schedule_constraints_get_domain(constraints);
	// TODO: spatial proximity is computed always, but may be done on per-tile level
	// not sure whether it is faster or not (many small computations or one large)
	spatial_proximity = isl_union_map_copy(scop->retagged_dep);

	access_map = isl_union_map_copy(scop->counted_accesses);
	access_map = isl_union_set_unwrap(isl_union_map_domain(access_map));
	access_map = isl_union_map_universe(access_map);
	access_map = isl_union_map_intersect_domain(access_map,
		isl_union_set_copy(band_domain));
	access_set = isl_union_map_wrap(access_map);

	spatial_proximity = isl_union_map_intersect_domain(spatial_proximity,
		isl_union_set_copy(access_set));
	spatial_proximity = isl_union_map_intersect_range(spatial_proximity,
		isl_union_set_copy(access_set));

	counted_accesses = isl_union_map_copy(scop->counted_accesses);
	counted_accesses = isl_union_map_intersect_domain(counted_accesses,
		access_set);

	constraints = isl_schedule_constraints_set_spatial_proximity(constraints,
		spatial_proximity);
	constraints = isl_schedule_constraints_set_counted_accesses(constraints,
		counted_accesses);

	return isl_schedule_constraints_recompute_schedule(constraints, node);

	// return reschedule_whole_component(constraints);
}

isl_stat max_out_dim_map(__isl_take isl_map *map, void *user)
{
	int *max_dim = user;
	int dim = isl_map_dim(map, isl_dim_out);
	isl_map_free(map);
	if (dim > *max_dim)
		*max_dim = dim;
	return isl_stat_ok;
}

int max_out_dim(__isl_keep isl_union_map *umap)
{
	int max_dim = 0;
	if (isl_union_map_foreach_map(umap, &max_out_dim_map, &max_dim) < 0)
		return -1;
	return max_dim;
}

struct extend_map_data {
	isl_union_map *result;
	int max_dim;
};

isl_stat extend_map(__isl_take isl_map *map, void *user)
{
	struct extend_map_data *data = user;
	int dim = isl_map_dim(map, isl_dim_out);
	if (dim < data->max_dim)
		map = isl_map_insert_dims(map, isl_dim_out, dim, data->max_dim - dim);
	data->result = isl_union_map_add_map(data->result, map);
	return isl_stat_ok;
}

__isl_give isl_union_map *extend_to_max_dim(__isl_take isl_union_map *umap)
{
	isl_space *space = isl_union_map_get_space(umap);
	int max_dim = max_out_dim(umap);
	struct extend_map_data data = { isl_union_map_empty(space), max_dim };
	if (isl_union_map_foreach_map(umap, &extend_map, &data) < 0)
		data.result = isl_union_map_free(data.result);
	isl_union_map_free(umap);
	return data.result;
}

int check_validity(isl_schedule_node *node, struct ppcg_scop *scop)
{
	isl_schedule *sch = isl_schedule_node_get_schedule(node);
	isl_band_list *blist = isl_schedule_get_band_forest(isl_schedule_cow(sch));

	isl_union_map *validity = isl_union_map_copy(scop->dep_flow);
	isl_union_map *usch = isl_band_list_get_suffix_schedule(blist);
	usch = extend_to_max_dim(usch);

	isl_map *scheduled_validity;
	isl_space *schedule_space;
	isl_bool empty;

	validity = isl_union_map_apply_domain(validity, isl_union_map_copy(usch));
	validity = isl_union_map_apply_range(validity, usch);
	scheduled_validity = isl_map_from_union_map(validity);
	schedule_space = isl_map_get_space(scheduled_validity);
	schedule_space = isl_space_domain(schedule_space);
	scheduled_validity = isl_map_intersect(scheduled_validity,
		isl_map_lex_gt(schedule_space));
	empty = isl_map_is_empty(scheduled_validity);
	isl_map_free(scheduled_validity);
	fprintf(stderr, "validity check %d\n", empty);
	return empty == isl_bool_true;
}

/* Tile "node", if it is a band node with at least 2 members.
 * The tile sizes are set from the "tile_size" option.
 * Splits node into two bands: tile loops and point loops.
 * If "tile_maximize_outer_coincidence" option is set, interchange tile loops
 * so that parallel loops come foremost in the tile band.
 */
static __isl_give isl_schedule_node *tile_band(
	__isl_take isl_schedule_node *node, void *user)
{
	struct ppcg_scop *scop = user;
	int n;
	isl_space *space;
	isl_multi_val *sizes;
	int use_same = scop->options->tile_spatial == PPCG_TILE_SPATIAL_SAME;

	if (isl_schedule_node_get_type(node) != isl_schedule_node_band)
		return node;

	n = isl_schedule_node_band_n_member(node);
	if (n <= 1)
		return node;

	space = isl_schedule_node_band_get_space(node);
	sizes = ppcg_multi_val_from_int(space, scop->options->tile_size);
	if (scop->options->tile_spatial == PPCG_TILE_SPATIAL_FIRST) {
		// Reschedule tile loops using a different policy.  Then substitute
		// current tile loop schedule with the newly computed one while
		// keeping point loops and their children intact.

		isl_schedule_node *rescheduled;
		isl_schedule_node *node_snap;

		rescheduled = reschedule_tile_loops(isl_schedule_node_copy(node), scop);
		if (!rescheduled)
			isl_die(isl_schedule_node_get_ctx(node), isl_error_internal,
				"Could not reschedule tile loop band", return node);
		node = tile(node, sizes);
		node_snap = node;
		node = isl_schedule_node_cow(isl_schedule_node_copy(node));

		space = isl_schedule_node_band_get_space(rescheduled);
		sizes = ppcg_multi_val_from_int(space, scop->options->tile_size);
		rescheduled = tile(rescheduled, sizes);

		isl_schedule_node *rescheduled_point_loop_node =
			isl_schedule_node_first_child(rescheduled);
		rescheduled_point_loop_node =
			isl_schedule_node_replace_tree(rescheduled_point_loop_node,
				isl_schedule_node_first_child(isl_schedule_node_copy(node)));
		rescheduled = isl_schedule_node_parent(rescheduled_point_loop_node);

		node = isl_schedule_node_replace_tree(node, rescheduled);

		if (!check_validity(node, scop)) {
			isl_die(isl_schedule_node_get_ctx(node), isl_error_internal,
				"Defaulting to same-schedule tiling", use_same = 1);
			isl_schedule_node_free(node);
			node = node_snap;
		} else {
			isl_schedule_node_free(node_snap);
		}
	} else if (scop->options->tile_spatial == PPCG_TILE_SPATIAL_LAST) {
		isl_schedule_node *rescheduled;
		isl_schedule_node *point_loop_node;

		node = tile(node, sizes);

		point_loop_node = isl_schedule_node_copy(node);
		point_loop_node = isl_schedule_node_first_child(point_loop_node);

		isl_union_set *dom = isl_schedule_node_get_domain(point_loop_node);
		isl_union_map *sched = schedule_node_band_get_ascendant_schedule(node);
		sched = replicate_tag(sched);
		// dom = isl_union_set_apply(dom, isl_union_map_copy(sched));
		dom = bandwise_domain(node, scop);

		isl_union_map *validity = bandwise_dependences(
			isl_schedule_node_copy(node),
			isl_schedule_node_get_domain(node), scop);
		validity = isl_union_map_intersect_domain(validity,
			isl_union_set_copy(dom));
		validity = isl_union_map_intersect_range(validity,
			isl_union_set_copy(dom));
		// validity = isl_union_map_apply_domain(validity,
		// 	isl_union_map_copy(sched));
		// validity = isl_union_map_apply_range(validity,
		// 	isl_union_map_copy(sched));

		isl_union_map *proximity = isl_union_map_copy(scop->dep_flow);
		// proximity = isl_union_map_apply_domain(proximity,
			// isl_union_map_copy(sched));
		// proximity = isl_union_map_apply_range(proximity,
			// isl_union_map_copy(sched));

		isl_union_map *access_map;
		isl_union_set *access_set;
		access_map = isl_union_map_copy(scop->counted_accesses);
		access_map = isl_union_set_unwrap(isl_union_map_domain(access_map));
		access_map = isl_union_map_universe(access_map);
		access_map = isl_union_map_intersect_domain(access_map,
			isl_union_set_copy(dom));
		access_set = isl_union_map_wrap(access_map);

		isl_union_map *spatial_proximity = isl_union_map_copy(scop->retagged_dep);
		spatial_proximity = isl_union_map_intersect_domain(spatial_proximity,
			isl_union_set_copy(access_set));
		spatial_proximity = isl_union_map_intersect_range(spatial_proximity,
			isl_union_set_copy(access_set));

		isl_union_map *counted_accesses = isl_union_map_copy(scop->counted_accesses);
		counted_accesses = isl_union_map_intersect_domain(counted_accesses,
			access_set);

		isl_schedule_constraints *sc = isl_schedule_constraints_on_domain(dom);
		sc = isl_schedule_constraints_set_validity(sc, validity);
		sc = isl_schedule_constraints_set_proximity(sc, proximity);
		sc = isl_schedule_constraints_set_spatial_proximity(sc, spatial_proximity);
		sc = isl_schedule_constraints_set_counted_accesses(sc, counted_accesses);

		fprintf(stderr, "[ppcg] here i am\n");

		// isl_schedule_constraints_dump(sc);
		rescheduled = reschedule_whole_component(sc);
		rescheduled = remove_tile_parameters(rescheduled,
			isl_schedule_node_copy(node));
		isl_schedule_node_dump(rescheduled);

		// isl_schedule_node_dump(rescheduled);

		fprintf(stderr, "[ppcg] here i am\n");

		isl_union_map *new_schedule =
			isl_schedule_node_band_get_partial_schedule_union_map(rescheduled);
		isl_union_map *old_schedule =
			isl_schedule_node_band_get_partial_schedule_union_map(point_loop_node);
		old_schedule = replicate_tag(old_schedule);
		old_schedule = isl_union_map_apply_range(old_schedule, new_schedule);
		point_loop_node = isl_schedule_node_band_reset_schedule(point_loop_node,
			isl_multi_union_pw_aff_from_union_map(old_schedule));

		// TODO: free node correctly
		node = isl_schedule_node_parent(point_loop_node);


	} else if (0 && scop->options->tile_spatial == PPCG_TILE_SPATIAL_LAST) {
		isl_schedule_node *rescheduled;
		isl_schedule_node *node_snap;

		rescheduled = isl_schedule_node_copy(node);
		node = tile(node, sizes);
		rescheduled = reschedule_point_loops(rescheduled,
			isl_schedule_node_copy(node), scop);
		if (!rescheduled)
			isl_die(isl_schedule_node_get_ctx(node), isl_error_internal,
				"Could not reschedule point loop band", return node);
		// node = tile(node, sizes);
		node_snap = node;
		node = isl_schedule_node_cow(isl_schedule_node_copy(node));

		space = isl_schedule_node_band_get_space(rescheduled);
		sizes = ppcg_multi_val_from_int(space, scop->options->tile_size);

		isl_union_map *point_sched;

#if 0
		isl_union_map *new_sched =
			isl_schedule_node_band_get_partial_schedule_union_map(rescheduled);

		// isl_union_set *dom = isl_schedule_node_get_domain(
		// 	isl_schedule_node_first_child(isl_schedule_node_copy(node)));
		// sched = isl_union_map_intersect_domain(sched, dom);

		isl_union_map *point_sched =
			isl_schedule_node_band_get_partial_schedule_union_map(
				isl_schedule_node_first_child(isl_schedule_node_copy(node)));

				isl_union_map_dump(new_sched);
				isl_union_map_dump(point_sched);

		isl_union_map *dmap = isl_union_map_preimage_range_union_pw_multi_aff(
			isl_union_map_copy(point_sched),
			isl_union_pw_multi_aff_from_union_map(new_sched));
		isl_union_map_dump(dmap);
		point_sched = isl_union_map_apply_domain(point_sched, isl_union_map_reverse(dmap));
		isl_union_map_dump(point_sched);
		fprintf(stderr, "[ppcg] here\n");
#endif

#if 0
		// fix all previous dimensions
		unsigned n_previous = schedule_node_prefix_n_member(rescheduled);
		sched = fix_union_map_dims_as_params(sched, n_previous, 0, 0);
		isl_union_map *point_sched =
			isl_schedule_node_band_get_partial_schedule_union_map(
				isl_schedule_node_first_child(isl_schedule_node_copy(node)));
		point_sched = replicate_tag(point_sched);
		isl_union_map_dump(point_sched);
		point_sched = isl_union_map_apply_range(point_sched, sched);
		isl_union_map_dump(point_sched);
#endif

		rescheduled = tile(rescheduled, sizes);

		rescheduled = isl_schedule_node_first_child(rescheduled);

		rescheduled = isl_schedule_node_band_reset_schedule(rescheduled,
			isl_multi_union_pw_aff_from_union_map(point_sched));
		//isl_schedule_node_dump(rescheduled);
		// TODO: go down the schedule tree and get rid of the parameters as well
		rescheduled = remove_tile_parameters(rescheduled, node);

		node = isl_schedule_node_first_child(node);

		isl_schedule_node *continuation =
			isl_schedule_node_first_child(isl_schedule_node_copy(node));

		node = isl_schedule_node_replace_tree(node, rescheduled);

		node = isl_schedule_node_first_child(node);
		node = isl_schedule_node_replace_tree(node, continuation);
		node = isl_schedule_node_parent(node);

		node = isl_schedule_node_parent(node);

		if (!check_validity(node, scop)) {
			isl_die(isl_schedule_node_get_ctx(node), isl_error_internal,
				"Defaulting to same-schedule tiling", use_same = 1);
			isl_schedule_node_free(node);
			node = node_snap;
		} else {
			isl_schedule_node_free(node_snap);
		}
	} else {
		node = tile(node, sizes);
	}

	return node;
}

/* Construct schedule constraints from the dependences in ps
 * for the purpose of computing a schedule for a CPU.
 *
 * The proximity constraints are set to the flow dependences.
 *
 * If live-range reordering is allowed then the conditional validity
 * constraints are set to the order dependences with the flow dependences
 * as condition.  That is, a live-range (flow dependence) will be either
 * local to an iteration of a band or all adjacent order dependences
 * will be respected by the band.
 * The validity constraints are set to the union of the flow dependences
 * and the forced dependences, while the coincidence constraints
 * are set to the union of the flow dependences, the forced dependences and
 * the order dependences.
 *
 * If live-range reordering is not allowed, then both the validity
 * and the coincidence constraints are set to the union of the flow
 * dependences and the false dependences.
 *
 * Note that the coincidence constraints are only set when the "openmp"
 * options is set.  Even though the way openmp pragmas are introduced
 * does not rely on the coincident property of the schedule band members,
 * the coincidence constraints do affect the way the schedule is constructed,
 * such that more schedule dimensions should be detected as parallel
 * by ast_schedule_dim_is_parallel.
 * Since the order dependences are also taken into account by
 * ast_schedule_dim_is_parallel, they are also added to
 * the coincidence constraints.  If the openmp handling learns
 * how to privatize some memory, then the corresponding order
 * dependences can be removed from the coincidence constraints.
 */
static __isl_give isl_schedule_constraints *construct_cpu_schedule_constraints(
	struct ppcg_scop *ps)
{
	isl_schedule_constraints *sc;
	isl_union_map *validity, *coincidence, *proximity;
	int use_coincidence = ps->options->openmp && (!ps->options->tile ||
		(ps->options->tile &&
		 ps->options->tile_spatial != PPCG_TILE_SPATIAL_FIRST));

	sc = isl_schedule_constraints_on_domain(isl_union_set_copy(ps->domain));
	if (ps->options->live_range_reordering) {
		sc = isl_schedule_constraints_set_conditional_validity(sc,
				isl_union_map_copy(ps->tagged_dep_flow),
				isl_union_map_copy(ps->tagged_dep_order));
		validity = isl_union_map_copy(ps->dep_flow);
		validity = isl_union_map_union(validity,
				isl_union_map_copy(ps->dep_forced));
		if (use_coincidence) {
			coincidence = isl_union_map_copy(validity);
			coincidence = isl_union_map_union(coincidence,
					isl_union_map_copy(ps->dep_order));
		}
	} else {
		validity = isl_union_map_copy(ps->dep_flow);
		validity = isl_union_map_union(validity,
				isl_union_map_copy(ps->dep_false));
		if (use_coincidence)
			coincidence = isl_union_map_copy(validity);
	}
	if (use_coincidence)
		sc = isl_schedule_constraints_set_coincidence(sc, coincidence);
	sc = isl_schedule_constraints_set_validity(sc, validity);

	if ((ps->options->spatial_model == PPCG_SPATIAL_MODEL_GROUPS ||
		 ps->options->spatial_model == PPCG_SPATIAL_MODEL_ENDS ||
		 ps->options->spatial_model == PPCG_SPATIAL_MODEL_ENDS_GROUPS)
	    && (!ps->options->tile ||
		    ps->options->tile_spatial != PPCG_TILE_SPATIAL_LAST)) {
		sc = isl_schedule_constraints_set_spatial_proximity(sc,
			isl_union_map_copy(ps->retagged_dep));
		sc = isl_schedule_constraints_set_counted_accesses(sc,
			isl_union_map_copy(ps->counted_accesses));
	}

	if (ps->options->keep_proximity) {
		if (ps->options->remove_nonuniform == PPCG_REMOVE_NONUNIFORM_ALL)
			proximity = isl_union_map_copy(ps->dep_flow_uniform);
		else
			proximity = isl_union_map_copy(ps->dep_flow);
	} else {
		proximity = isl_union_map_copy(ps->retagged_dep);
	}

	sc = isl_schedule_constraints_set_proximity(sc, proximity);

	return sc;
}

/* Compute a schedule for the scop "ps".
 *
 * First derive the appropriate schedule constraints from the dependences
 * in "ps" and then compute a schedule from those schedule constraints,
 * possibly grouping statement instances based on the input schedule.
 */
static __isl_give isl_schedule *compute_cpu_schedule(struct ppcg_scop *ps)
{
	isl_schedule_constraints *sc;
	isl_schedule *schedule;

	if (!ps)
		return NULL;

	sc = construct_cpu_schedule_constraints(ps);

	if (ps->options->debug->dump_schedule_constraints)
		isl_schedule_constraints_dump(sc);
	schedule = ppcg_compute_schedule(sc, ps->schedule, ps->options);

	return schedule;
}

/* Compute a new schedule to the scop "ps" if the reschedule option is set.
 * Otherwise, return a copy of the original schedule.
 */
static __isl_give isl_schedule *optionally_compute_schedule(void *user)
{
	struct ppcg_scop *ps = user;

	if (!ps)
		return NULL;
	if (!ps->options->reschedule)
		return isl_schedule_copy(ps->schedule);
	return compute_cpu_schedule(ps);
}

/* Compute a schedule based on the dependences in "ps" and
 * tile it if requested by the user.
 */
static __isl_give isl_schedule *get_schedule(struct ppcg_scop *ps,
	struct ppcg_options *options)
{
	isl_ctx *ctx;
	isl_schedule *schedule;

	if (!ps)
		return NULL;

	ctx = isl_union_set_get_ctx(ps->domain);
	schedule = ppcg_get_schedule(ctx, options,
				    &optionally_compute_schedule, ps);
	if (ps->options->tile)
		schedule = isl_schedule_map_schedule_node_bottom_up(schedule,
							&tile_band, ps);

	return schedule;
}

/* Generate CPU code for the scop "ps" using "schedule" and
 * print the corresponding C code to "p", including variable declarations.
 */
static __isl_give isl_printer *print_cpu_with_schedule(
	__isl_take isl_printer *p, struct ppcg_scop *ps,
	__isl_take isl_schedule *schedule, struct ppcg_options *options)
{
	int hidden;
	isl_set *context;

	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "/* ppcg generated CPU code */");
	p = isl_printer_end_line(p);

	p = isl_printer_start_line(p);
	p = isl_printer_end_line(p);

	p = ppcg_set_macro_names(p);
	p = ppcg_print_exposed_declarations(p, ps);
	hidden = ppcg_scop_any_hidden_declarations(ps);
	if (hidden) {
		p = ppcg_start_block(p);
		p = ppcg_print_hidden_declarations(p, ps);
	}

	context = isl_set_copy(ps->context);
	context = isl_set_from_params(context);
	schedule = isl_schedule_insert_context(schedule, context);
	if (options->debug->dump_final_schedule)
		isl_schedule_dump(schedule);
	p = print_scop(ps, schedule, p, options);
	if (hidden)
		p = ppcg_end_block(p);

	return p;
}

/* Generate CPU code for the scop "ps" and print the corresponding C code
 * to "p", including variable declarations.
 */
__isl_give isl_printer *print_cpu(__isl_take isl_printer *p,
	struct ppcg_scop *ps, struct ppcg_options *options)
{
	isl_schedule *schedule;

	schedule = isl_schedule_copy(ps->schedule);
	return print_cpu_with_schedule(p, ps, schedule, options);
}

/* Generate CPU code for "scop" and print it to "p".
 *
 * First obtain a schedule for "scop" and then print code for "scop"
 * using that schedule.
 */
static __isl_give isl_printer *generate(__isl_take isl_printer *p,
	struct ppcg_scop *scop, struct ppcg_options *options)
{
	isl_schedule *schedule;

	schedule = get_schedule(scop, options);

	return print_cpu_with_schedule(p, scop, schedule, options);
}

/* Wrapper around generate for use as a ppcg_transform callback.
 */
static __isl_give isl_printer *print_cpu_wrap(__isl_take isl_printer *p,
	struct ppcg_scop *scop, void *user)
{
	struct ppcg_options *options = user;

	return generate(p, scop, options);
}

/* Transform the code in the file called "input" by replacing
 * all scops by corresponding CPU code and write the results to a file
 * called "output".
 */
int generate_cpu(isl_ctx *ctx, struct ppcg_options *options,
	const char *input, const char *output)
{
	FILE *output_file;
	int r;

	output_file = get_output_file(input, output);
	if (!output_file)
		return -1;

	r = ppcg_transform(ctx, input, output_file, options,
					&print_cpu_wrap, options);

	fclose(output_file);

	return r;
}
