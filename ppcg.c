/*
 * Copyright 2011      INRIA Saclay
 * Copyright 2013      Ecole Normale Superieure
 * Copyright 2015      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 * and Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <isl/ctx.h>
#include <isl/id.h>
#include <isl/val.h>
#include <isl/set.h>
#include <isl/union_set.h>
#include <isl/union_map.h>
#include <isl/aff.h>
#include <isl/flow.h>
#include <isl/options.h>
#include <isl/schedule.h>
#include <isl/ast.h>
#include <isl/id_to_ast_expr.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/constraint.h>
#include <pet.h>
#include "ppcg.h"
#include "ppcg_options.h"
#include "cuda.h"
#include "opencl.h"
#include "cpu.h"

#define CACHE_SIZE 32

// #define isl_union_map_debug(a) \
//   fprintf(stderr, "%s:%d in %s, %s\n  ", \
//	     __FILE__, __LINE__, __PRETTY_FUNCTION__, #a); \
//   isl_union_map_dump(a);

#define xDebug(type, a) \
  fprintf(stderr, "%s:%d in %s, (%s) %s\n  ", \
	  __FILE__, __LINE__, __PRETTY_FUNCTION__, #type, #a); \
  isl_ ## type ## _dump(a);

#define isl_union_map_debug(a) xDebug(union_map, a)
#define isl_union_set_debug(a) xDebug(union_set, a)
#define isl_map_debug(a) xDebug(map, a)
#define isl_set_debug(a) xDebug(set, a)
#define isl_basic_map_debug(a) xDebug(basic_map, a)
#define isl_basic_set_debug(a) xDebug(basic_set, a)
#define isl_constraint_debug(a) xDebug(constraint, a)
#define isl_aff_debug(a) xDebug(aff, a)
#define isl_multi_aff(a) xDebug(multi_aff, a)
#define isl_pw_aff(a) xDebug(pw_aff, a)
#define isl_pw_multi_aff(a) xDebug(pw_multi_aff, a)
#define isl_flow_debug(a) xDebug(flow, a)
#define isl_union_flow_debug(a) xDebug(union_flow, a)
#define isl_union_pw_multi_aff_debug(a) xDebug(union_pw_multi_aff, a)
#define isl_space_debug(a) xDebug(space, a)

struct options {
	struct pet_options *pet;
	struct ppcg_options *ppcg;
	char *input;
	char *output;
};

const char *ppcg_version(void);
static void print_version(void)
{
	printf("%s", ppcg_version());
}

ISL_ARGS_START(struct options, options_args)
ISL_ARG_CHILD(struct options, pet, "pet", &pet_options_args, "pet options")
ISL_ARG_CHILD(struct options, ppcg, NULL, &ppcg_options_args, "ppcg options")
ISL_ARG_STR(struct options, output, 'o', NULL,
	"filename", NULL, "output filename (c and opencl targets)")
ISL_ARG_ARG(struct options, input, "input", NULL)
ISL_ARG_VERSION(print_version)
ISL_ARGS_END

ISL_ARG_DEF(options, struct options, options_args)

/* Return a pointer to the final path component of "filename" or
 * to "filename" itself if it does not contain any components.
 */
const char *ppcg_base_name(const char *filename)
{
	const char *base;

	base = strrchr(filename, '/');
	if (base)
		return ++base;
	else
		return filename;
}

/* Copy the base name of "input" to "name" and return its length.
 * "name" is not NULL terminated.
 *
 * In particular, remove all leading directory components and
 * the final extension, if any.
 */
int ppcg_extract_base_name(char *name, const char *input)
{
	const char *base;
	const char *ext;
	int len;

	base = ppcg_base_name(input);
	ext = strrchr(base, '.');
	len = ext ? ext - base : strlen(base);

	memcpy(name, base, len);

	return len;
}

/* Does "scop" refer to any arrays that are declared, but not
 * exposed to the code after the scop?
 */
int ppcg_scop_any_hidden_declarations(struct ppcg_scop *scop)
{
	int i;

	if (!scop)
		return 0;

	for (i = 0; i < scop->pet->n_array; ++i)
		if (scop->pet->arrays[i]->declared &&
		    !scop->pet->arrays[i]->exposed)
			return 1;

	return 0;
}

/* Collect all variable names that are in use in "scop".
 * In particular, collect all parameters in the context and
 * all the array names.
 * Store these names in an isl_id_to_ast_expr by mapping
 * them to a dummy value (0).
 */
static __isl_give isl_id_to_ast_expr *collect_names(struct pet_scop *scop)
{
	int i, n;
	isl_ctx *ctx;
	isl_ast_expr *zero;
	isl_id_to_ast_expr *names;

	ctx = isl_set_get_ctx(scop->context);

	n = isl_set_dim(scop->context, isl_dim_param);

	names = isl_id_to_ast_expr_alloc(ctx, n + scop->n_array);
	zero = isl_ast_expr_from_val(isl_val_zero(ctx));

	for (i = 0; i < n; ++i) {
		isl_id *id;

		id = isl_set_get_dim_id(scop->context, isl_dim_param, i);
		names = isl_id_to_ast_expr_set(names,
						id, isl_ast_expr_copy(zero));
	}

	for (i = 0; i < scop->n_array; ++i) {
		struct pet_array *array = scop->arrays[i];
		isl_id *id;

		id = isl_set_get_tuple_id(array->extent);
		names = isl_id_to_ast_expr_set(names,
						id, isl_ast_expr_copy(zero));
	}

	isl_ast_expr_free(zero);

	return names;
}

/* Return an isl_id called "prefix%d", with "%d" set to "i".
 * If an isl_id with such a name already appears among the variable names
 * of "scop", then adjust the name to "prefix%d_%d".
 */
static __isl_give isl_id *generate_name(struct ppcg_scop *scop,
	const char *prefix, int i)
{
	int j;
	char name[16];
	isl_ctx *ctx;
	isl_id *id;
	int has_name;

	ctx = isl_set_get_ctx(scop->context);
	snprintf(name, sizeof(name), "%s%d", prefix, i);
	id = isl_id_alloc(ctx, name, NULL);

	j = 0;
	while ((has_name = isl_id_to_ast_expr_has(scop->names, id)) == 1) {
		isl_id_free(id);
		snprintf(name, sizeof(name), "%s%d_%d", prefix, i, j++);
		id = isl_id_alloc(ctx, name, NULL);
	}

	return has_name < 0 ? isl_id_free(id) : id;
}

/* Return a list of "n" isl_ids of the form "prefix%d".
 * If an isl_id with such a name already appears among the variable names
 * of "scop", then adjust the name to "prefix%d_%d".
 */
__isl_give isl_id_list *ppcg_scop_generate_names(struct ppcg_scop *scop,
	int n, const char *prefix)
{
	int i;
	isl_ctx *ctx;
	isl_id_list *names;

	ctx = isl_set_get_ctx(scop->context);
	names = isl_id_list_alloc(ctx, n);
	for (i = 0; i < n; ++i) {
		isl_id *id;

		id = generate_name(scop, prefix, i);
		names = isl_id_list_add(names, id);
	}

	return names;
}

/* Is "stmt" not a kill statement?
 */
static int is_not_kill(struct pet_stmt *stmt)
{
	return !pet_stmt_is_kill(stmt);
}

/* Collect the iteration domains of the statements in "scop" that
 * satisfy "pred".
 */
static __isl_give isl_union_set *collect_domains(struct pet_scop *scop,
	int (*pred)(struct pet_stmt *stmt))
{
	int i;
	isl_set *domain_i;
	isl_union_set *domain;

	if (!scop)
		return NULL;

	domain = isl_union_set_empty(isl_set_get_space(scop->context));

	for (i = 0; i < scop->n_stmt; ++i) {
		struct pet_stmt *stmt = scop->stmts[i];

		if (!pred(stmt))
			continue;

		if (stmt->n_arg > 0)
			isl_die(isl_union_set_get_ctx(domain),
				isl_error_unsupported,
				"data dependent conditions not supported",
				return isl_union_set_free(domain));

		domain_i = isl_set_copy(scop->stmts[i]->domain);
		domain = isl_union_set_add_set(domain, domain_i);
	}

	return domain;
}

/* Collect the iteration domains of the statements in "scop",
 * skipping kill statements.
 */
static __isl_give isl_union_set *collect_non_kill_domains(struct pet_scop *scop)
{
	return collect_domains(scop, &is_not_kill);
}

/* This function is used as a callback to pet_expr_foreach_call_expr
 * to detect if there is any call expression in the input expression.
 * Assign the value 1 to the integer that "user" points to and
 * abort the search since we have found what we were looking for.
 */
static int set_has_call(__isl_keep pet_expr *expr, void *user)
{
	int *has_call = user;

	*has_call = 1;

	return -1;
}

/* Does "expr" contain any call expressions?
 */
static int expr_has_call(__isl_keep pet_expr *expr)
{
	int has_call = 0;

	if (pet_expr_foreach_call_expr(expr, &set_has_call, &has_call) < 0 &&
	    !has_call)
		return -1;

	return has_call;
}

/* This function is a callback for pet_tree_foreach_expr.
 * If "expr" contains any call (sub)expressions, then set *has_call
 * and abort the search.
 */
static int check_call(__isl_keep pet_expr *expr, void *user)
{
	int *has_call = user;

	if (expr_has_call(expr))
		*has_call = 1;

	return *has_call ? -1 : 0;
}

/* Does "stmt" contain any call expressions?
 */
static int has_call(struct pet_stmt *stmt)
{
	int has_call = 0;

	if (pet_tree_foreach_expr(stmt->body, &check_call, &has_call) < 0 &&
	    !has_call)
		return -1;

	return has_call;
}

/* Collect the iteration domains of the statements in "scop"
 * that contain a call expression.
 */
static __isl_give isl_union_set *collect_call_domains(struct pet_scop *scop)
{
	return collect_domains(scop, &has_call);
}

/* Given a union of "tagged" access relations of the form
 *
 *	[S_i[...] -> R_j[]] -> A_k[...]
 *
 * project out the "tags" (R_j[]).
 * That is, return a union of relations of the form
 *
 *	S_i[...] -> A_k[...]
 */
static __isl_give isl_union_map *project_out_tags(
	__isl_take isl_union_map *umap)
{
	return isl_union_map_domain_factor_domain(umap);
}

/* Construct a function from tagged iteration domains to the corresponding
 * untagged iteration domains with as range of the wrapped map in the domain
 * the reference tags that appear in any of the reads, writes or kills.
 * Store the result in ps->tagger.
 *
 * For example, if the statement with iteration space S[i,j]
 * contains two array references R_1[] and R_2[], then ps->tagger will contain
 *
 *	{ [S[i,j] -> R_1[]] -> S[i,j]; [S[i,j] -> R_2[]] -> S[i,j] }
 */
static void compute_tagger(struct ppcg_scop *ps)
{
	isl_union_map *tagged, *array_tag;
	isl_union_pw_multi_aff *tagger;

	tagged = isl_union_map_copy(ps->tagged_reads);
	tagged = isl_union_map_union(tagged,
				isl_union_map_copy(ps->tagged_may_writes));
	tagged = isl_union_map_union(tagged,
				isl_union_map_copy(ps->tagged_must_kills));
	tagged = isl_union_map_universe(tagged);

	array_tag = isl_union_map_copy(tagged);
	ps->array_tagger = isl_union_map_domain_factor_range(array_tag);

	//isl_union_map_dump(ps->array_tagger);

	tagged = isl_union_set_unwrap(isl_union_map_domain(tagged));

	tagger = isl_union_map_domain_map_union_pw_multi_aff(tagged);

	ps->tagger = tagger;
}

static void compute_retagged_tagger(struct ppcg_scop *ps)
{
	isl_union_map *tagged;

	tagged = isl_union_map_copy(ps->retagged_reads);
	tagged = isl_union_map_union(tagged,
		isl_union_map_copy(ps->retagged_must_writes));
	tagged = isl_union_map_universe(tagged);
	tagged = isl_union_set_unwrap(
		isl_union_map_domain(tagged));
	ps->retagged_tagger = isl_union_map_domain_map_union_pw_multi_aff(tagged);
}

static void compute_array_tagger(struct ppcg_scop *ps)
{
	isl_union_map *tagged;

	tagged = isl_union_map_copy(ps->cache_array_tagged_reads);
	tagged = isl_union_map_union(tagged, isl_union_map_copy(ps->cache_array_tagged_may_writes));
	tagged = isl_union_map_union(tagged, isl_union_map_copy(ps->cache_array_tagged_must_writes));

	tagged = isl_union_map_universe(tagged);
	tagged = isl_union_set_unwrap(isl_union_map_domain(tagged));
	ps->cache_array_tagger = isl_union_map_domain_map_union_pw_multi_aff(tagged);
}

/* Compute the live out accesses, i.e., the writes that are
 * potentially not killed by any kills or any other writes, and
 * store them in ps->live_out.
 *
 * We compute the "dependence" of any "kill" (an explicit kill
 * or a must write) on any may write.
 * The elements accessed by the may writes with a "depending" kill
 * also accessing the element are definitely killed.
 * The remaining may writes can potentially be live out.
 *
 * The result of the dependence analysis is
 *
 *	{ IW -> [IK -> A] }
 *
 * with IW the instance of the write statement, IK the instance of kill
 * statement and A the element that was killed.
 * The range factor range is
 *
 *	{ IW -> A }
 *
 * containing all such pairs for which there is a kill statement instance,
 * i.e., all pairs that have been killed.
 */
static void compute_live_out(struct ppcg_scop *ps)
{
	isl_schedule *schedule;
	isl_union_map *kills;
	isl_union_map *exposed;
	isl_union_map *covering;
	isl_union_access_info *access;
	isl_union_flow *flow;

	schedule = isl_schedule_copy(ps->schedule);
	kills = isl_union_map_union(isl_union_map_copy(ps->must_writes),
				    isl_union_map_copy(ps->must_kills));
	access = isl_union_access_info_from_sink(kills);
	access = isl_union_access_info_set_may_source(access,
				    isl_union_map_copy(ps->may_writes));
	access = isl_union_access_info_set_schedule(access, schedule);
	flow = isl_union_access_info_compute_flow(access);
	covering = isl_union_flow_get_full_may_dependence(flow);
	isl_union_flow_free(flow);

	covering = isl_union_map_range_factor_range(covering);
	exposed = isl_union_map_copy(ps->may_writes);
	exposed = isl_union_map_subtract(exposed, covering);
	ps->live_out = exposed;
}

/* Compute the tagged flow dependences and the live_in accesses and store
 * the results in ps->tagged_dep_flow and ps->live_in.
 *
 * We allow both the must writes and the must kills to serve as
 * definite sources such that a subsequent read would not depend
 * on any earlier write.  The resulting flow dependences with
 * a must kill as source reflect possibly uninitialized reads.
 * No dependences need to be introduced to protect such reads
 * (other than those imposed by potential flows from may writes
 * that follow the kill).  We therefore remove those flow dependences.
 * This is also useful for the dead code elimination, which assumes
 * the flow sources are non-kill instances.
 */
static void compute_tagged_flow_dep_only(struct ppcg_scop *ps)
{
	isl_union_pw_multi_aff *tagger;
	isl_schedule *schedule;
	isl_union_map *live_in;
	isl_union_access_info *access;
	isl_union_flow *flow;
	isl_union_map *must_source;
	isl_union_map *kills;
	isl_union_map *tagged_flow;

	tagger = isl_union_pw_multi_aff_copy(ps->tagger);
	schedule = isl_schedule_copy(ps->schedule);
	schedule = isl_schedule_pullback_union_pw_multi_aff(schedule, tagger);
	kills = isl_union_map_copy(ps->tagged_must_kills);
	must_source = isl_union_map_copy(ps->tagged_must_writes);
	must_source = isl_union_map_union(must_source,
				isl_union_map_copy(kills));
	access = isl_union_access_info_from_sink(
				isl_union_map_copy(ps->tagged_reads));
	access = isl_union_access_info_set_must_source(access, must_source);
	access = isl_union_access_info_set_may_source(access,
				isl_union_map_copy(ps->tagged_may_writes));
	access = isl_union_access_info_set_schedule(access, schedule);
	flow = isl_union_access_info_compute_flow(access);
	tagged_flow = isl_union_flow_get_may_dependence(flow);
	tagged_flow = isl_union_map_subtract_domain(tagged_flow,
				isl_union_map_domain(kills));

	ps->tagged_dep_flow = tagged_flow;
	live_in = isl_union_flow_get_may_no_source(flow);
	ps->live_in = project_out_tags(live_in);
	isl_union_flow_free(flow);
}

/* Compute ps->dep_flow from ps->tagged_dep_flow
 * by projecting out the reference tags.
 */
static void derive_flow_dep_from_tagged_flow_dep(struct ppcg_scop *ps)
{
	ps->dep_flow = isl_union_map_copy(ps->tagged_dep_flow);
	ps->dep_flow = isl_union_map_factor_domain(ps->dep_flow);
}

/* Compute the flow dependences and the live_in accesses and store
 * the results in ps->dep_flow and ps->live_in.
 * A copy of the flow dependences, tagged with the reference tags
 * is stored in ps->tagged_dep_flow.
 *
 * We first compute ps->tagged_dep_flow, i.e., the tagged flow dependences
 * and then project out the tags.
 */
static void compute_tagged_flow_dep(struct ppcg_scop *ps)
{
	compute_tagged_flow_dep_only(ps);
	derive_flow_dep_from_tagged_flow_dep(ps);
}

/* Compute the order dependences that prevent the potential live ranges
 * from overlapping.
 *
 * In particular, construct a union of relations
 *
 *	[R[...] -> R_1[]] -> [W[...] -> R_2[]]
 *
 * where [R[...] -> R_1[]] is the range of one or more live ranges
 * (i.e., a read) and [W[...] -> R_2[]] is the domain of one or more
 * live ranges (i.e., a write).  Moreover, the read and the write
 * access the same memory element and the read occurs before the write
 * in the original schedule.
 * The scheduler allows some of these dependences to be violated, provided
 * the adjacent live ranges are all local (i.e., their domain and range
 * are mapped to the same point by the current schedule band).
 *
 * Note that if a live range is not local, then we need to make
 * sure it does not overlap with _any_ other live range, and not
 * just with the "previous" and/or the "next" live range.
 * We therefore add order dependences between reads and
 * _any_ later potential write.
 *
 * We also need to be careful about writes without a corresponding read.
 * They are already prevented from moving past non-local preceding
 * intervals, but we also need to prevent them from moving past non-local
 * following intervals.  We therefore also add order dependences from
 * potential writes that do not appear in any intervals
 * to all later potential writes.
 * Note that dead code elimination should have removed most of these
 * dead writes, but the dead code elimination may not remove all dead writes,
 * so we need to consider them to be safe.
 *
 * The order dependences are computed by computing the "dataflow"
 * from the above unmatched writes and the reads to the may writes.
 * The unmatched writes and the reads are treated as may sources
 * such that they would not kill order dependences from earlier
 * such writes and reads.
 */
static void compute_order_dependences(struct ppcg_scop *ps)
{
	isl_union_map *reads;
	isl_union_map *shared_access;
	isl_union_set *matched;
	isl_union_map *unmatched;
	isl_union_pw_multi_aff *tagger;
	isl_schedule *schedule;
	isl_union_access_info *access;
	isl_union_flow *flow;

	tagger = isl_union_pw_multi_aff_copy(ps->tagger);
	schedule = isl_schedule_copy(ps->schedule);
	schedule = isl_schedule_pullback_union_pw_multi_aff(schedule, tagger);
	reads = isl_union_map_copy(ps->tagged_reads);
	matched = isl_union_map_domain(isl_union_map_copy(ps->tagged_dep_flow));
	unmatched = isl_union_map_copy(ps->tagged_may_writes);
	unmatched = isl_union_map_subtract_domain(unmatched, matched);
	reads = isl_union_map_union(reads, unmatched);
	access = isl_union_access_info_from_sink(
				isl_union_map_copy(ps->tagged_may_writes));
	access = isl_union_access_info_set_may_source(access, reads);
	access = isl_union_access_info_set_schedule(access, schedule);
	flow = isl_union_access_info_compute_flow(access);
	shared_access = isl_union_flow_get_may_dependence(flow);
	isl_union_flow_free(flow);

	ps->tagged_dep_order = isl_union_map_copy(shared_access);
	ps->dep_order = isl_union_map_factor_domain(shared_access);
}

/* Compute those validity dependences of the program represented by "scop"
 * that should be unconditionally enforced even when live-range reordering
 * is used.
 *
 * In particular, compute the external false dependences
 * as well as order dependences between sources with the same sink.
 * The anti-dependences are already taken care of by the order dependences.
 * The external false dependences are only used to ensure that live-in and
 * live-out data is not overwritten by any writes inside the scop.
 * The independences are removed from the external false dependences,
 * but not from the order dependences between sources with the same sink.
 *
 * In particular, the reads from live-in data need to precede any
 * later write to the same memory element.
 * As to live-out data, the last writes need to remain the last writes.
 * That is, any earlier write in the original schedule needs to precede
 * the last write to the same memory element in the computed schedule.
 * The possible last writes have been computed by compute_live_out.
 * They may include kills, but if the last access is a kill,
 * then the corresponding dependences will effectively be ignored
 * since we do not schedule any kill statements.
 *
 * Note that the set of live-in and live-out accesses may be
 * an overapproximation.  There may therefore be potential writes
 * before a live-in access and after a live-out access.
 *
 * In the presence of may-writes, there may be multiple live-ranges
 * with the same sink, accessing the same memory element.
 * The sources of these live-ranges need to be executed
 * in the same relative order as in the original program
 * since we do not know which of the may-writes will actually
 * perform a write.  Consider all sources that share a sink and
 * that may write to the same memory element and compute
 * the order dependences among them.
 */
static void compute_forced_dependences(struct ppcg_scop *ps)
{
	isl_union_map *shared_access;
	isl_union_map *exposed;
	isl_union_map *live_in;
	isl_union_map *sink_access;
	isl_union_map *shared_sink;
	isl_union_access_info *access;
	isl_union_flow *flow;
	isl_schedule *schedule;

	exposed = isl_union_map_copy(ps->live_out);
	schedule = isl_schedule_copy(ps->schedule);
	access = isl_union_access_info_from_sink(exposed);
	access = isl_union_access_info_set_may_source(access,
				isl_union_map_copy(ps->may_writes));
	access = isl_union_access_info_set_schedule(access, schedule);
	flow = isl_union_access_info_compute_flow(access);
	shared_access = isl_union_flow_get_may_dependence(flow);
	isl_union_flow_free(flow);
	ps->dep_forced = shared_access;

	schedule = isl_schedule_copy(ps->schedule);
	access = isl_union_access_info_from_sink(
				isl_union_map_copy(ps->may_writes));
	access = isl_union_access_info_set_may_source(access,
				isl_union_map_copy(ps->live_in));
	access = isl_union_access_info_set_schedule(access, schedule);
	flow = isl_union_access_info_compute_flow(access);
	live_in = isl_union_flow_get_may_dependence(flow);
	isl_union_flow_free(flow);

	ps->dep_forced = isl_union_map_union(ps->dep_forced, live_in);
	ps->dep_forced = isl_union_map_subtract(ps->dep_forced,
				isl_union_map_copy(ps->independence));

	schedule = isl_schedule_copy(ps->schedule);
	sink_access = isl_union_map_copy(ps->tagged_dep_flow);
	sink_access = isl_union_map_range_product(sink_access,
				isl_union_map_copy(ps->tagged_may_writes));
	sink_access = isl_union_map_domain_factor_domain(sink_access);
	access = isl_union_access_info_from_sink(
				isl_union_map_copy(sink_access));
	access = isl_union_access_info_set_may_source(access, sink_access);
	access = isl_union_access_info_set_schedule(access, schedule);
	flow = isl_union_access_info_compute_flow(access);
	shared_sink = isl_union_flow_get_may_dependence(flow);
	isl_union_flow_free(flow);
	ps->dep_forced = isl_union_map_union(ps->dep_forced, shared_sink);
}

/* Remove independence from the tagged flow dependences.
 * Since the user has guaranteed that source and sink of an independence
 * can be executed in any order, there cannot be a flow dependence
 * between them, so they can be removed from the set of flow dependences.
 * However, if the source of such a flow dependence is a must write,
 * then it may have killed other potential sources, which would have
 * to be recovered if we were to remove those flow dependences.
 * We therefore keep the flow dependences that originate in a must write,
 * even if it corresponds to a known independence.
 */
static void remove_independences_from_tagged_flow(struct ppcg_scop *ps)
{
	isl_union_map *tf;
	isl_union_set *indep;
	isl_union_set *mw;

	tf = isl_union_map_copy(ps->tagged_dep_flow);
	tf = isl_union_map_zip(tf);
	indep = isl_union_map_wrap(isl_union_map_copy(ps->independence));
	tf = isl_union_map_intersect_domain(tf, indep);
	tf = isl_union_map_zip(tf);
	mw = isl_union_map_domain(isl_union_map_copy(ps->tagged_must_writes));
	tf = isl_union_map_subtract_domain(tf, mw);
	ps->tagged_dep_flow = isl_union_map_subtract(ps->tagged_dep_flow, tf);
}

/* Compute the dependences of the program represented by "scop"
 * in case live range reordering is allowed.
 *
 * We compute the actual live ranges and the corresponding order
 * false dependences.
 *
 * The independences are removed from the flow dependences
 * (provided the source is not a must-write) as well as
 * from the external false dependences (by compute_forced_dependences).
 */
static void compute_live_range_reordering_dependences(struct ppcg_scop *ps)
{
	compute_tagged_flow_dep_only(ps);
	remove_independences_from_tagged_flow(ps);
	derive_flow_dep_from_tagged_flow_dep(ps);
	compute_order_dependences(ps);
	compute_forced_dependences(ps);
}

/* Compute the potential flow dependences and the potential live in
 * accesses.
 */
static void compute_flow_dep(struct ppcg_scop *ps)
{
	isl_union_access_info *access;
	isl_union_flow *flow;

	access = isl_union_access_info_from_sink(isl_union_map_copy(ps->reads));
	access = isl_union_access_info_set_must_source(access,
				isl_union_map_copy(ps->must_writes));
	access = isl_union_access_info_set_may_source(access,
				isl_union_map_copy(ps->may_writes));
	access = isl_union_access_info_set_schedule(access,
				isl_schedule_copy(ps->schedule));
	flow = isl_union_access_info_compute_flow(access);

	ps->dep_flow = isl_union_flow_get_may_dependence(flow);
	ps->live_in = isl_union_flow_get_may_no_source(flow);
	isl_union_flow_free(flow);
}

struct spatial_deps_data {
	struct ppcg_scop *ps;
	isl_union_map *res;
};

static isl_stat compute_cache_block_dep(__isl_take isl_map *cache_block_map, void *user)
{
	struct spatial_deps_data *data = user;
	struct ppcg_scop *ps = data->ps;
	isl_union_access_info *access;
	isl_union_flow *flow;
	isl_union_map *dep, *acc;
	isl_space *space;
	isl_union_pw_multi_aff *tagger;
	isl_schedule *schedule;

	acc = isl_union_map_from_map(cache_block_map);
	//isl_union_map_dump(acc);

	tagger = isl_union_pw_multi_aff_copy(ps->tagger);
	schedule = isl_schedule_copy(ps->schedule);
	schedule = isl_schedule_pullback_union_pw_multi_aff(schedule, tagger);

	access = isl_union_access_info_from_sink(isl_union_map_copy(acc));
	access = isl_union_access_info_set_must_source(access,
			isl_union_map_copy(acc));
	access = isl_union_access_info_set_may_source(access,
			isl_union_map_copy(acc));
	access = isl_union_access_info_set_schedule(access,
				isl_schedule_copy(schedule));
	flow = isl_union_access_info_compute_flow(access);

	dep = isl_union_flow_get_may_dependence(flow);
	//isl_union_map_dump(dep);

	data->res = isl_union_map_union(data->res, dep);
	isl_union_flow_free(flow);

	return isl_stat_ok;
}

/* Compute the spatial locality dependences
  */
static void compute_spatial_locality_deps(struct ppcg_scop *ps)
{
	struct spatial_deps_data data = {ps};
	isl_space *space;
	space = isl_union_map_get_space(ps->dep_flow);

	data.res = isl_union_map_empty(space);
	if(isl_union_map_foreach_map(ps->tagged_cache_block_reads, &compute_cache_block_dep, &data) < 0)
		data.res = isl_union_map_free(data.res);

	if(isl_union_map_foreach_map(ps->tagged_cache_block_must_writes, &compute_cache_block_dep, &data) < 0)
		data.res = isl_union_map_free(data.res);

	ps->tagged_cache_block_dep_flow = data.res;
	ps->cache_block_dep_flow = isl_union_map_copy(ps->tagged_cache_block_dep_flow);
	ps->cache_block_dep_flow = isl_union_map_factor_domain(ps->cache_block_dep_flow);
}

/* Compute the spatial locality dependences
  */
static void compute_spatial_locality_flow_dep(struct ppcg_scop *ps)
{
	isl_union_access_info *access;
	isl_union_flow *flow;

	access = isl_union_access_info_from_sink(isl_union_map_copy(ps->cache_block_reads));
	access = isl_union_access_info_set_must_source(access,
				isl_union_map_copy(ps->cache_block_must_writes));
	access = isl_union_access_info_set_may_source(access,
				isl_union_map_copy(ps->cache_block_may_writes));
	access = isl_union_access_info_set_schedule(access,
				isl_schedule_copy(ps->schedule));
	flow = isl_union_access_info_compute_flow(access);

	ps->cache_block_dep_flow = isl_union_flow_get_may_dependence(flow);
	isl_union_flow_free(flow);
}

/* Compute the spatial locality dependences
  */
static void compute_spatial_locality_rar_dep(struct ppcg_scop *ps)
{
	isl_union_access_info *access;
	isl_union_flow *flow;

	access = isl_union_access_info_from_sink(isl_union_map_copy(ps->cache_block_reads));
	access = isl_union_access_info_set_must_source(access,
				isl_union_map_copy(ps->cache_block_reads));
	access = isl_union_access_info_set_may_source(access,
				isl_union_map_copy(ps->cache_block_reads));
	access = isl_union_access_info_set_schedule(access,
				isl_schedule_copy(ps->schedule));
	flow = isl_union_access_info_compute_flow(access);

	ps->cache_block_dep_rar = isl_union_flow_get_may_dependence(flow);
	isl_union_flow_free(flow);
}

static void compute_adjacent_deps(struct ppcg_scop *ps)
{
	isl_union_access_info *access;
	isl_union_flow *flow;

	// Construct access info from adjacent accesses:
	// reading next cell should depend on anything that accessed the _previous_ cell.
	// similar for WAW, WAR? :
	// writing next cell should depend on anything that accessed the _previous_ cell.
	access = isl_union_access_info_from_sink(isl_union_map_copy(ps->adjacent_reads));
	access = isl_union_access_info_set_must_source(access,
		isl_union_map_copy(ps->must_writes));
	access = isl_union_access_info_set_may_source(access,
		isl_union_map_copy(ps->may_writes));
	access = isl_union_access_info_set_schedule(access,
		isl_schedule_copy(ps->schedule));
	flow = isl_union_access_info_compute_flow(access);
	isl_union_map_dump(ps->adjacent_reads);
	isl_union_map_dump(ps->must_writes);
	isl_union_map_dump(ps->may_writes);
	isl_union_flow_dump(flow);

	ps->adjacent_dep_flow = isl_union_flow_get_may_dependence(flow);

	// RAR
	access = isl_union_access_info_from_sink(isl_union_map_copy(ps->adjacent_reads));
	access = isl_union_access_info_set_may_source(access,
		isl_union_map_copy(ps->reads));
	access = isl_union_access_info_set_schedule(access,
		isl_schedule_copy(ps->schedule));
	flow = isl_union_access_info_compute_flow(access);

	ps->adjacent_dep_rar = isl_union_flow_get_may_dependence(flow);

	isl_union_map_dump(ps->adjacent_reads);
	isl_union_map_dump(ps->reads);
	isl_schedule_dump(ps->schedule);
	isl_union_map_dump(ps->adjacent_dep_flow);

	isl_union_flow_free(flow);
}

static void compute_cache_deps(struct ppcg_scop *ps)
{
	isl_union_access_info *access;
	isl_union_flow *flow;

	// Sinks are original sinks (reads for flow and RAR, writes for false).
	// Let's keep only flow and RAR for now (they are not separable).
	access = isl_union_access_info_from_sink(isl_union_map_copy(ps->reads));
	access = isl_union_access_info_set_may_source(access, isl_union_map_copy(ps->cache_accesses_from_may));
	access = isl_union_access_info_set_must_source(access, isl_union_map_copy(ps->cache_accesses_from_must));
	access = isl_union_access_info_set_schedule(access, isl_schedule_copy(ps->schedule));
	flow = isl_union_access_info_compute_flow(access);

	ps->cache_dep = isl_union_flow_get_may_dependence(flow);
	ps->cache_dep = isl_union_map_union(ps->cache_dep, isl_union_flow_get_must_dependence(flow));
	isl_union_flow_free(flow);
}

/* Compute the dependences of the program represented by "scop".
 * Store the computed potential flow dependences
 * in scop->dep_flow and the reads with potentially no corresponding writes in
 * scop->live_in.
 * Store the potential live out accesses in scop->live_out.
 * Store the potential false (anti and output) dependences in scop->dep_false.
 *
 * If live range reordering is allowed, then we compute a separate
 * set of order dependences and a set of external false dependences
 * in compute_live_range_reordering_dependences.
 */
static void compute_dependences(struct ppcg_scop *scop)
{
	isl_union_map *may_source;
	isl_union_access_info *access;
	isl_union_flow *flow;

	if (!scop)
		return;

	compute_live_out(scop);

	if (scop->options->live_range_reordering)
		compute_live_range_reordering_dependences(scop);
	else if (scop->options->target != PPCG_TARGET_C)
		compute_tagged_flow_dep(scop);
	else
		compute_flow_dep(scop);

	if (scop->options->model_spatial_locality){
		compute_spatial_locality_deps(scop);
//		compute_spatial_locality_flow_dep(scop);
//		compute_spatial_locality_rar_dep(scop);
	}
	//isl_union_map_dump(scop->dep_flow);
	//isl_union_map_dump(scop->tagged_dep_flow);
	//compute_adjacent_deps(scop);
	compute_cache_deps(scop);

	may_source = isl_union_map_union(isl_union_map_copy(scop->may_writes),
					isl_union_map_copy(scop->reads));
	access = isl_union_access_info_from_sink(
				isl_union_map_copy(scop->may_writes));
	access = isl_union_access_info_set_must_source(access,
				isl_union_map_copy(scop->must_writes));
	access = isl_union_access_info_set_may_source(access, may_source);
	access = isl_union_access_info_set_schedule(access,
				isl_schedule_copy(scop->schedule));
	flow = isl_union_access_info_compute_flow(access);

	scop->dep_false = isl_union_flow_get_may_dependence(flow);
	scop->dep_false = isl_union_map_coalesce(scop->dep_false);
	isl_union_flow_free(flow);
}

static __isl_give isl_union_flow *compute_union_flow(
	__isl_take isl_union_map *sink,
	__isl_take isl_union_map *must_source,
	__isl_take isl_union_map *may_source,
	__isl_take isl_schedule *schedule)
{
	isl_union_access_info *ai;
	ai = isl_union_access_info_from_sink(sink);
	ai = isl_union_access_info_set_must_source(ai, must_source);
	ai = isl_union_access_info_set_may_source(ai, may_source);
	ai = isl_union_access_info_set_schedule(ai, schedule);
	return isl_union_access_info_compute_flow(ai);
}

static void add_dependences(
	struct ppcg_scop *ps,
	__isl_take isl_union_map *sink,
	__isl_take isl_union_map *must_source,
	__isl_take isl_union_map *may_source,
	__isl_take isl_schedule *schedule)
{
	isl_union_flow *flow;
	isl_union_map *dep;

	flow = compute_union_flow(sink, must_source, may_source, schedule);
	// isl_union_flow_debug(flow);
	dep = isl_union_flow_get_must_dependence(flow);
	ps->cache_array_tagged_dep = isl_union_map_union(
		ps->cache_array_tagged_dep, dep);

	dep = isl_union_flow_get_may_dependence(flow);
	ps->cache_array_tagged_dep = isl_union_map_union(
		ps->cache_array_tagged_dep, dep);

	isl_union_flow_free(flow);
}

static isl_union_map *union_wrap_with_array_name(isl_union_map *);

static void compute_array_tagged_dependences(struct ppcg_scop *ps)
{
	//isl_union_access_info *ai;
	isl_schedule *sched;
	//isl_union_map *dep;
	//isl_union_flow *fl;
	isl_space *space;
	isl_union_map *all_writes;

	sched = isl_schedule_copy(ps->schedule);
	sched = isl_schedule_pullback_union_pw_multi_aff(sched, ps->cache_array_tagger);

	space = isl_union_set_get_space(ps->domain);
	ps->cache_array_tagged_dep = isl_union_map_empty(isl_space_copy(space));

	// isl_union_map_debug(ps->cache_array_tagged_reads);
	// isl_union_map_debug(ps->cache_array_tagged_must_writes);
	// isl_union_map_debug(ps->cache_array_tagged_may_writes);

	// RAW (flow)
	add_dependences(ps,
					isl_union_map_copy(ps->cache_array_tagged_reads),
					isl_union_map_copy(ps->cache_array_tagged_must_writes),
					isl_union_map_empty(isl_space_copy(space)),//isl_union_map_copy(ps->cache_array_tagged_may_writes),
					isl_schedule_copy(sched));
	// RAR (output)
	add_dependences(ps,
					isl_union_map_copy(ps->cache_array_tagged_reads),
					isl_union_map_copy(ps->cache_array_tagged_reads),
					isl_union_map_empty(isl_space_copy(space)),//isl_union_map_copy(ps->cache_array_tagged_reads),
					isl_schedule_copy(sched));

	// WAR (anti/false)
	all_writes = isl_union_map_union(
			isl_union_map_copy(ps->cache_array_tagged_may_writes),
			isl_union_map_copy(ps->cache_array_tagged_must_writes));
	add_dependences(ps,
					isl_union_map_copy(all_writes),
					isl_union_map_copy(ps->cache_array_tagged_reads),
					isl_union_map_empty(isl_space_copy(space)),//isl_union_map_copy(ps->cache_array_tagged_reads),
					isl_schedule_copy(sched));

	// WAW (input/false)
	add_dependences(ps,
					all_writes,
					isl_union_map_copy(ps->cache_array_tagged_must_writes),
					isl_union_map_empty(isl_space_copy(space)),//isl_union_map_copy(ps->cache_array_tagged_may_writes),
					isl_schedule_copy(sched));

	isl_space_free(space);

	// Let's also compute original proximity dependences
	// option 1: do not extend dimensionality
	// option 2: extend dimensionality, and set all extended dimension values
	//           to 0 (or a value outside the domain) so as to ensure dependences
	//           between accesses exist (but do not overlap with induced deps)
	isl_union_map *tagged_reads = union_wrap_with_array_name(ps->reads);
	isl_union_map *tagged_must_writes =
			union_wrap_with_array_name(ps->must_writes);
	isl_union_map *tagged_may_writes =
			union_wrap_with_array_name(ps->may_writes);
	space = isl_union_map_get_space(tagged_reads);
	// RAW (flow)
	add_dependences(ps,
					isl_union_map_copy(tagged_reads),
					isl_union_map_copy(tagged_must_writes),
					isl_union_map_empty(isl_space_copy(space)),
					isl_schedule_copy(sched));
	// RAR (output)
	add_dependences(ps,
					isl_union_map_copy(tagged_reads),
					isl_union_map_copy(tagged_reads),
					isl_union_map_empty(isl_space_copy(space)),
					isl_schedule_copy(sched));

	add_dependences(ps,
					isl_union_map_copy(tagged_must_writes),
					isl_union_map_copy(tagged_reads),
					isl_union_map_empty(isl_space_copy(space)),
					isl_schedule_copy(sched));

	add_dependences(ps,
					isl_union_map_copy(tagged_must_writes),
					isl_union_map_copy(tagged_must_writes),
					isl_union_map_empty(isl_space_copy(space)),
					isl_schedule_copy(sched));
	isl_space_free(space);
	isl_schedule_free(sched);
}

static void add_retagged_dependences(
	struct ppcg_scop *ps,
	__isl_keep isl_union_map *sink,
	__isl_keep isl_union_map *source,
	__isl_keep isl_schedule *schedule)
{
	isl_union_flow *flow;
	isl_union_access_info *ai = isl_union_access_info_from_sink(
		isl_union_map_copy(sink));
	ai = isl_union_access_info_set_must_source(ai,
		isl_union_map_copy(source));
	ai = isl_union_access_info_set_schedule(ai,
		isl_schedule_copy(schedule));

	flow = isl_union_access_info_compute_flow(ai);
	ps->retagged_dep = isl_union_map_union(ps->retagged_dep,
		isl_union_flow_get_must_dependence(flow));
	isl_union_flow_free(flow);
}

static void add_all_retagged_dependences(struct ppcg_scop *ps,
	__isl_keep isl_union_map *reads, __isl_keep isl_union_map *writes)
{
	isl_schedule *schedule;
	schedule = isl_schedule_copy(ps->schedule);
	schedule = isl_schedule_pullback_union_pw_multi_aff(schedule,
		isl_union_pw_multi_aff_copy(ps->retagged_tagger));

	add_retagged_dependences(ps, reads, writes, schedule);
	add_retagged_dependences(ps, reads, reads, schedule);
	add_retagged_dependences(ps, writes, reads, schedule);
	add_retagged_dependences(ps, writes, writes, schedule);
	isl_schedule_free(schedule);
}

static isl_union_map *map_array_accesses_to_next_elements(isl_union_map *);

static void compute_retagged_dependences(struct ppcg_scop *ps)
{
	isl_union_map *spatial_reads, *spatial_writes;

	ps->retagged_dep = isl_union_map_empty(
		isl_union_set_get_space(ps->domain));

	// Spatial locality dependences ()
	spatial_reads = map_array_accesses_to_next_elements(ps->retagged_reads);
	spatial_writes = map_array_accesses_to_next_elements(
		ps->retagged_must_writes);
	add_all_retagged_dependences(ps, spatial_reads, spatial_writes);
	isl_union_map_free(spatial_reads);
	isl_union_map_free(spatial_writes);

	// Original dependences.
	add_all_retagged_dependences(ps, ps->retagged_reads,
		ps->retagged_must_writes);
}

/* Eliminate dead code from ps->domain.
 *
 * In particular, intersect both ps->domain and the domain of
 * ps->schedule with the (parts of) iteration
 * domains that are needed to produce the output or for statement
 * iterations that call functions.
 * Also intersect the range of the dataflow dependences with
 * this domain such that the removed instances will no longer
 * be considered as targets of dataflow.
 *
 * We start with the iteration domains that call functions
 * and the set of iterations that last write to an array
 * (except those that are later killed).
 *
 * Then we add those statement iterations that produce
 * something needed by the "live" statements iterations.
 * We keep doing this until no more statement iterations can be added.
 * To ensure that the procedure terminates, we compute the affine
 * hull of the live iterations (bounded to the original iteration
 * domains) each time we have added extra iterations.
 */
static void eliminate_dead_code(struct ppcg_scop *ps)
{
	isl_union_set *live;
	isl_union_map *dep;
	isl_union_pw_multi_aff *tagger;

	live = isl_union_map_domain(isl_union_map_copy(ps->live_out));
	if (!isl_union_set_is_empty(ps->call)) {
		live = isl_union_set_union(live, isl_union_set_copy(ps->call));
		live = isl_union_set_coalesce(live);
	}

	dep = isl_union_map_copy(ps->dep_flow);
	dep = isl_union_map_reverse(dep);

	for (;;) {
		isl_union_set *extra;

		extra = isl_union_set_apply(isl_union_set_copy(live),
					    isl_union_map_copy(dep));
		if (isl_union_set_is_subset(extra, live)) {
			isl_union_set_free(extra);
			break;
		}

		live = isl_union_set_union(live, extra);
		live = isl_union_set_affine_hull(live);
		live = isl_union_set_intersect(live,
					    isl_union_set_copy(ps->domain));
	}

	isl_union_map_free(dep);

	ps->domain = isl_union_set_intersect(ps->domain,
						isl_union_set_copy(live));
	ps->schedule = isl_schedule_intersect_domain(ps->schedule,
						isl_union_set_copy(live));
	ps->dep_flow = isl_union_map_intersect_range(ps->dep_flow,
						isl_union_set_copy(live));
	tagger = isl_union_pw_multi_aff_copy(ps->tagger);
	live = isl_union_set_preimage_union_pw_multi_aff(live, tagger);
	ps->tagged_dep_flow = isl_union_map_intersect_range(ps->tagged_dep_flow,
						live);
}

/* Intersect "set" with the set described by "str", taking the NULL
 * string to represent the universal set.
 */
static __isl_give isl_set *set_intersect_str(__isl_take isl_set *set,
	const char *str)
{
	isl_ctx *ctx;
	isl_set *set2;

	if (!str)
		return set;

	ctx = isl_set_get_ctx(set);
	set2 = isl_set_read_from_str(ctx, str);
	set = isl_set_intersect(set, set2);

	return set;
}

static void *ppcg_scop_free(struct ppcg_scop *ps)
{
	if (!ps)
		return NULL;

	isl_set_free(ps->context);
	isl_union_set_free(ps->domain);
	isl_union_set_free(ps->call);
	isl_union_map_free(ps->tagged_reads);
	isl_union_map_free(ps->reads);
	isl_union_map_free(ps->live_in);
	isl_union_map_free(ps->tagged_may_writes);
	isl_union_map_free(ps->tagged_must_writes);
	isl_union_map_free(ps->may_writes);
	isl_union_map_free(ps->must_writes);
	isl_union_map_free(ps->live_out);
	isl_union_map_free(ps->tagged_must_kills);
	isl_union_map_free(ps->must_kills);
	isl_union_map_free(ps->tagged_dep_flow);
	isl_union_map_free(ps->dep_flow);
	isl_union_map_free(ps->dep_false);
	isl_union_map_free(ps->dep_forced);
	isl_union_map_free(ps->tagged_dep_order);
	isl_union_map_free(ps->dep_order);

	if(ps->options->model_spatial_locality){
		isl_union_map_free(ps->cache_block_dep_flow);
		isl_union_map_free(ps->cache_block_dep_rar);
		isl_union_map_free(ps->cache_block_may_writes);
		isl_union_map_free(ps->cache_block_must_writes);
		isl_union_map_free(ps->cache_block_reads);
	}

	// isl_union_map_free(ps->adjacent_reads);
	// isl_union_map_free(ps->adjacent_may_writes);
	// isl_union_map_free(ps->adjacent_must_writes);
	// isl_union_map_free(ps->adjacent_dep_flow);
	// isl_union_map_free(ps->adjacent_dep_rar);

	isl_union_map_free(ps->cache_accesses_from_must);
	isl_union_map_free(ps->cache_accesses_from_may);
	isl_union_map_free(ps->cache_dep);

	isl_union_map_free(ps->retagged_reads);
	isl_union_map_free(ps->retagged_must_writes);
	isl_union_map_free(ps->retagged_dep);
	isl_union_pw_multi_aff_free(ps->retagged_tagger);

	isl_schedule_free(ps->schedule);
	isl_union_pw_multi_aff_free(ps->tagger);
	isl_union_map_free(ps->independence);
	isl_id_to_ast_expr_free(ps->names);

	free(ps);

	return NULL;
}


struct cache_block_map_data {
	int cache_block_size;
	isl_union_map *res;
};

/* Construct and apply a cache block map for a given array access.
 * Cache block map groups the contiguous array elements into a single
 * group. This mapping is applied only to the innermost dimension of the
 * array. Assumes the arrays are stored in row major format.
 */
static isl_stat cache_block_map(__isl_take isl_map *map, void *user)
{
	struct cache_block_map_data *data = user;
	isl_space *space, *access_domain, *cache_map_space;
	isl_local_space *ls;
	unsigned int n_array_dims;
	unsigned int cache_block_size = data->cache_block_size;
	isl_basic_map *bmap;
	isl_map *cache_map;
	isl_constraint *c;
	int i;

	n_array_dims = isl_map_n_out(map);
	if(n_array_dims == 0){
//		data->res = isl_union_map_add_map(data->res, isl_map_copy(map));
		return isl_stat_ok;
	}

	space = isl_map_get_space(map);
	access_domain = isl_space_range(space);
	cache_map_space = isl_space_from_domain(access_domain);
	cache_map_space = isl_space_add_dims(cache_map_space, isl_dim_out, n_array_dims);
	isl_space_set_tuple_name(cache_map_space, isl_dim_out,
			isl_space_get_tuple_name(cache_map_space, isl_dim_in));

	bmap = isl_basic_map_universe(cache_map_space);
	ls = isl_local_space_from_space(cache_map_space);

	for(i=0; i< n_array_dims - 1; i++){
		c = isl_constraint_alloc_equality(isl_local_space_copy(ls));
		c = isl_constraint_set_coefficient_si(c, isl_dim_in, i, -1);
		c = isl_constraint_set_coefficient_si(c, isl_dim_out, i, 1);
		bmap = isl_basic_map_add_constraint(bmap, c);
	}

	c = isl_constraint_alloc_inequality(isl_local_space_copy(ls));
    c = isl_constraint_set_coefficient_si(c, isl_dim_out, i, -1 * cache_block_size);
    c = isl_constraint_set_coefficient_si(c, isl_dim_in, i, 1);
    bmap = isl_basic_map_add_constraint(bmap, c);

	c = isl_constraint_alloc_inequality(isl_local_space_copy(ls));
    c = isl_constraint_set_coefficient_si(c, isl_dim_out, i,  cache_block_size);
    c = isl_constraint_set_coefficient_si(c, isl_dim_in, i, -1);
    c = isl_constraint_set_constant_si(c, cache_block_size - 1);
    bmap = isl_basic_map_add_constraint(bmap, c);

    cache_map = isl_map_from_basic_map(bmap);
    map = isl_map_apply_range(map, cache_map);

    data->res = isl_union_map_add_map(data->res, map);

    return isl_stat_ok;
}

__isl_give isl_union_map *map_array_accesses_to_cache_blocks(isl_union_map *accesses, int cache_block_size)
{

	isl_space *space, *access_domain, *cache_map_space;
	struct cache_block_map_data data = {cache_block_size};

	space = isl_union_map_get_space(accesses);

	data.res = isl_union_map_empty(space);
	if (isl_union_map_foreach_map(accesses, &cache_block_map, &data) < 0 )
		data.res = isl_union_map_free(data.res);

	return data.res;

}

struct array_access_next_data
{
	isl_union_map *result;
	int n_elements_forward;
	int n_elements_backwards;
};
typedef struct array_access_next_data array_access_next_data;

static isl_stat array_access_to_next_elements(__isl_take isl_map *map,
	void *user)
{
	isl_space *access_space, *space;
	isl_local_space *constraint_space;
	isl_map *mapper, *mapped;
	isl_constraint *constraint;
	int i, n_dims;
	array_access_next_data *data = user;
	int n_elements_forward = data->n_elements_forward;
	int n_elements_backwards = data->n_elements_backwards;

	// Build the mapper space (range -> range).
	access_space = isl_map_get_space(map);
	access_space = isl_space_range(access_space);
	space = isl_space_map_from_domain_and_range(isl_space_copy(access_space), access_space);
	constraint_space = isl_local_space_from_space(isl_space_copy(space));

	mapper = isl_map_universe(space);
	// Equate all dimensions but last.
	n_dims = isl_map_n_out(mapper);
	for (i = 0; i < n_dims - 1; i++)
	{
		constraint = isl_constraint_alloc_equality(isl_local_space_copy(constraint_space));
		constraint = isl_constraint_set_coefficient_si(constraint, isl_dim_in, i, 1);
		constraint = isl_constraint_set_coefficient_si(constraint, isl_dim_out, i, -1);
		mapper = isl_map_add_constraint(mapper, constraint);
	}
	// Last dim: i' >= i - n_elements_backwards and i' <= i + n_elements_forward, i.e.
	// i' - i + n_elements_backwards >= 0 and i + n_elements_forward - i' >= 0.
	// TODO: should we exclude the i'==i case?
	// Maybe not, it will account for temporal locality, it holds for spatial, it may even
	// help remove originl proximity constraints for simplifying the problem.
#if 0
	constraint = isl_constraint_alloc_inequality(isl_local_space_copy(constraint_space));
	constraint = isl_constraint_set_coefficient_si(constraint, isl_dim_in, n_dims - 1, -1);
	constraint = isl_constraint_set_coefficient_si(constraint, isl_dim_out, n_dims - 1, 1);
	constraint = isl_constraint_set_constant_si(constraint, n_elements_backwards);
	mapper = isl_map_add_constraint(mapper, constraint);
	constraint = isl_constraint_alloc_inequality(constraint_space);
	constraint = isl_constraint_set_coefficient_si(constraint, isl_dim_in, n_dims - 1, 1);
	constraint = isl_constraint_set_coefficient_si(constraint, isl_dim_out, n_dims - 1, -1);
	constraint = isl_constraint_set_constant_si(constraint, n_elements_forward);
	mapper = isl_map_add_constraint(mapper, constraint);
#endif

#if 0
	// tmp: add negative
	constraint = isl_constraint_alloc_equality(isl_local_space_copy(constraint_space));
	constraint = isl_constraint_set_coefficient_si(constraint, isl_dim_in, n_dims - 1, -1);
	constraint = isl_constraint_set_coefficient_si(constraint, isl_dim_out, n_dims - 1, 1);
	constraint = isl_constraint_set_constant_si(constraint, -n_elements_forward);
	isl_map *mapper2 = isl_map_copy(mapper);
	mapper2 = isl_map_add_constraint(mapper2, constraint);
	mapped = isl_map_apply_range(isl_map_copy(map), mapper2);
	mapped = isl_map_intersect_range(mapped, isl_map_range(isl_map_copy(map)));
	data->result = isl_union_map_add_map(data->result, mapped);
#endif

	constraint = isl_constraint_alloc_equality(constraint_space);
	constraint = isl_constraint_set_coefficient_si(constraint, isl_dim_in, n_dims - 1, -1);
	constraint = isl_constraint_set_coefficient_si(constraint, isl_dim_out, n_dims - 1, 1);
	constraint = isl_constraint_set_constant_si(constraint, n_elements_forward);
	mapper = isl_map_add_constraint(mapper, constraint);

// #if 0
	// tmp: add original
	data->result = isl_union_map_add_map(data->result, isl_map_copy(map));
// #endif

	mapped = isl_map_apply_range(isl_map_copy(map), mapper);
	mapped = isl_map_intersect_range(mapped, isl_map_range(map));
	data->result = isl_union_map_add_map(data->result, mapped);

	// if (mapped && !data->result) return isl_stat_error;

	return isl_stat_ok;
}

static isl_union_map *extend_access_by_1(isl_union_map *);

static isl_union_map *union_map_extend_accesses(isl_union_map *);

static __isl_give isl_union_map *map_array_accesses_to_next_elements(
	__isl_keep isl_union_map *access)
{
	isl_space *space;
	isl_union_map *mapped;

	space = isl_union_map_get_space(access); // get the parameteric space
	mapped = isl_union_map_empty(space);
	array_access_next_data data = {mapped, CACHE_SIZE, 0};

	isl_union_map *extended = union_map_extend_accesses(access);

	if (isl_union_map_foreach_map(extended,
			&array_access_to_next_elements, &data) < 0)
		data.result = isl_union_map_free(data.result);
	isl_union_map_free(extended);

	return data.result;
}

static isl_stat array_access_to_next_element(__isl_take isl_map *map,
	void *user)
{
	isl_union_map **result_ptr;
	isl_space *access_space, *space;
	isl_local_space *constraint_space;
	isl_map *mapper, *mapped;
	isl_constraint *constraint;
	int i, n_dims;

	result_ptr = user;

	access_space = isl_map_get_space(map);
	access_space = isl_space_range(access_space);
	space = isl_space_map_from_domain_and_range(isl_space_copy(access_space), access_space);
	constraint_space = isl_local_space_from_space(isl_space_copy(space));

	mapper = isl_map_universe(space);
	// Equate all dimensions but last.
	n_dims = isl_map_n_out(mapper);
	for (i = 0; i < n_dims - 1; i++)
	{
		constraint = isl_constraint_alloc_equality(isl_local_space_copy(constraint_space));
		constraint = isl_constraint_set_coefficient_si(constraint, isl_dim_in, i, 1);
		constraint = isl_constraint_set_coefficient_si(constraint, isl_dim_out, i, -1);
		mapper = isl_map_add_constraint(mapper, constraint);
	}
	// Last dim: i' = i-1.
	constraint = isl_constraint_alloc_equality(constraint_space);
	constraint = isl_constraint_set_coefficient_si(constraint, isl_dim_in, n_dims - 1, 1);
	constraint = isl_constraint_set_coefficient_si(constraint, isl_dim_out, n_dims - 1, -1);
	constraint = isl_constraint_set_constant_si(constraint, -1);
	mapper = isl_map_add_constraint(mapper, constraint);

	mapped = isl_map_apply_range(isl_map_copy(map), mapper);
	// The next line makes the access relation to only hold for the array cells
	// accessed by the original statement.  Although it's more "correct", it
	// may make more sense for locality to keep the accesses outside the original
	// cells to reflect that the following element may be in the cache already.
	// Will it become live-out?
	mapped = isl_map_intersect_range(mapped, isl_map_range(map));
	*result_ptr = isl_union_map_add_map(*result_ptr, mapped);

	return isl_stat_ok;
}

static __isl_give isl_union_map *map_array_accesses_to_next_element(
    __isl_keep isl_union_map *accesses)
{
	isl_space *space;
	isl_union_map *mapped;

	space = isl_union_map_get_space(accesses); // get the parameteric space
	mapped = isl_union_map_empty(space);

	if (isl_union_map_foreach_map(accesses, &array_access_to_next_element, &mapped) < 0)
		isl_union_map_free(mapped);

	return mapped;
}

isl_stat map_wrap_with_array_name(__isl_take isl_map *map, void *user)
{
	isl_space *space, *map_space;
	isl_set *set, *domain_set;
	isl_map *domain_map;
	isl_id *array_id;
	isl_union_map **result = user;

	if (isl_map_domain_is_wrapping(map))
		return isl_stat_error;

#if 0
	map_space = isl_map_get_space(map);
	array_id = isl_space_get_tuple_id(map_space, isl_dim_out);
	space = isl_space_params(map_space);
	space = isl_space_set_from_params(space);
	space = isl_space_set_tuple_id(space, isl_dim_set, array_id);
	set = isl_set_universe(space);
	domain_map = isl_map_from_domain_and_range(isl_map_domain(isl_map_copy(map)),
											   set);
	domain_set = isl_map_wrap(domain_map);
	isl_union_map_product
	map = isl_map_from_domain_and_range(domain_set, isl_map_range(map));
#endif

	map_space = isl_map_get_space(map);
	array_id = isl_space_get_tuple_id(map_space, isl_dim_out);
	space = isl_space_params(map_space);
	space = isl_space_set_tuple_id(space, isl_dim_in, array_id);
	isl_map *mult = isl_map_identity(space);
	map = isl_map_product(map, mult);
	map = isl_map_range_factor_domain(map);

	*result = isl_union_map_add_map(*result, map);

	return isl_stat_ok;
}

__isl_give isl_union_map *union_wrap_with_array_name(__isl_keep isl_union_map *union_map)
{
	isl_union_map *result;
	isl_space *space;

	space = isl_union_map_get_space(union_map);
	result = isl_union_map_empty(space);

	if (isl_union_map_foreach_map(union_map, &map_wrap_with_array_name, &result) != isl_stat_ok)
		return NULL;
	return result;
}

// {[i,j] -> [o1,o2] : o1 = i and o2 = j} =>
// option 1
// {[i,j,t] -> [o1,o2] : o1 = i and 32*t <= j <= 32*t + 31 and o2 = t}
// option 2 (but if we project out, won't it become a division-access?)
// {[i,j] -> [o1,o2] : Exists t : o1 = i and 32*t <= j <= 32*t + 31 and o2 = t}
isl_stat acess_tile(__isl_take isl_basic_map *bmap)
{
	isl_space *space;
	isl_local_space *local_space;
	isl_constraint_list *constraint_list;
	int nc, i, last_out_dim, last_in_dim, ndim;
	isl_constraint *constraint;
	isl_constraint *lower_bound, *upper_bound, *equality;
	isl_basic_map *new_bmap;

	// [i,j] -> [i,j,t] : i = i and 32*t <= [2*i-j] <= 32*t + 31
	bmap = isl_basic_map_add_dims(bmap, isl_dim_in, 1);
	space = isl_basic_map_get_space(bmap);
	new_bmap = isl_basic_map_universe(isl_space_copy(space));
	local_space = isl_local_space_from_space(space);

	constraint_list = isl_basic_map_get_constraint_list(bmap);
	nc = isl_constraint_list_n_constraint(constraint_list);
	last_out_dim = isl_basic_map_n_out(bmap) - 1;
	last_in_dim = isl_basic_map_n_in(bmap) - 1;
	constraint = NULL;
	for (i = 0; i < nc; i++)
	{
		isl_constraint *constr =
				isl_constraint_list_get_constraint(constraint_list, i);
		if (isl_constraint_is_equality(constr) == isl_bool_true &&
		    isl_constraint_involves_dims(constr, isl_dim_out, last_out_dim, 1)
				== isl_bool_true)
		{
			constraint = constr;
		}
		else
		{
			bmap = isl_basic_map_add_constraint(bmap, constraint);
		}

	}
	if (constraint == NULL)
	{
		return isl_stat_error;
	}
	constraint = isl_constraint_set_coefficient_si(constraint,
			isl_dim_out, last_out_dim, 0);
	// iterate over all constraints until getting one featuring o2
	// make coefficient for o2 zero and use this new constraint as a basis
	//   for making strip-mining constraints with t

	lower_bound = isl_constraint_alloc_inequality(
			isl_local_space_copy(local_space));
	upper_bound = isl_constraint_alloc_inequality(
			isl_local_space_copy(local_space));
	// copy all elements
	ndim = isl_constraint_dim(constraint, isl_dim_all);
	for (i = 0; i < ndim; i++)
	{
		isl_val *v = isl_constraint_get_coefficient_val(constraint, isl_dim_all, i);
		lower_bound = isl_constraint_set_coefficient_val(lower_bound,
				isl_dim_all, i, isl_val_copy(v));
		upper_bound = isl_constraint_set_coefficient_val(upper_bound,
				isl_dim_all, i, isl_val_neg(v));
	}
	isl_val *v = isl_constraint_get_constant_val(constraint);
	lower_bound = isl_constraint_set_constant_val(lower_bound, isl_val_copy(v));
	upper_bound = isl_constraint_set_constant_val(upper_bound,
			isl_val_add_ui(isl_val_neg(v), 31));
	lower_bound = isl_constraint_set_coefficient_si(lower_bound,
			isl_dim_in, last_in_dim, -32);
	upper_bound = isl_constraint_set_coefficient_si(upper_bound,
			isl_dim_in, last_in_dim, 32);
	isl_constraint_free(constraint);

	constraint = isl_constraint_alloc_equality(local_space);
	constraint = isl_constraint_set_coefficient_si(constraint,
			isl_dim_in, last_in_dim, 1);
	constraint = isl_constraint_set_coefficient_si(constraint,
			isl_dim_out, last_out_dim, -1);

	bmap = isl_basic_map_add_constraint(bmap, lower_bound);
	bmap = isl_basic_map_add_constraint(bmap, upper_bound);
	bmap = isl_basic_map_add_constraint(bmap, constraint);
}

/* Find a trivial access function that is linearly independent of other access
 * functions represented by equalities in the basic map "bmap".
 * Trivial access function has the form f(i_*) = i_x, i.e. features only one
 * input dimension with coefficient 1.
 *
 * In particular, assuming "bmap" contains equalities of the form o = f(i)
 * with o output dimensions and i input dimensions, take the matrix of
 * input dimension coefficients, extend it with a row containing 1 as each
 * of the coefficients in turn and check whether the matrix still has a
 * full row rank.
 *
 * Return an equality constraint with input dimensions that form a linearly
 * indeendent access function or NULL if there is no trivial access function
 * linearly independent of the given ones.
 */
static
__isl_give isl_constraint *find_one_independent_access(__isl_keep isl_basic_map *bmap)
{
	int n_in, i, nr, rank;
	isl_local_space *local_space;
	isl_constraint *constraint;

	n_in = isl_basic_map_n_in(bmap);
	isl_mat *eqmat = isl_basic_map_equalities_matrix(bmap,
		isl_dim_in, isl_dim_out, isl_dim_param, isl_dim_cst, isl_dim_div);
	eqmat = isl_mat_drop_cols(eqmat, n_in, isl_mat_cols(eqmat) - n_in);
	eqmat = isl_mat_add_zero_rows(eqmat, 1);

	for (i = 0; i < n_in; ++i)
	{
		nr = isl_mat_rows(eqmat);
		eqmat = isl_mat_set_element_si(eqmat, nr - 1, i, 1);
		rank = isl_mat_rank(eqmat);
		if (rank == nr)
			break;

		eqmat = isl_mat_set_element_si(eqmat, nr - 1, i, 0);
	}
	isl_mat_free(eqmat);

	if (i == n_in)
		return NULL;

	local_space = isl_basic_map_get_local_space(bmap);
	constraint = isl_constraint_alloc_equality(local_space);
	constraint = isl_constraint_set_coefficient_si(constraint, isl_dim_in, i, 1);

	return constraint;
}

/* Add outer dimensions to the accesses to make them have the same
 * dimensionalities as the surronding loop nests.  New accesses are
 * linearly independent from old accesses.
 * Does nothing for scalar accesses.
 */
static
isl_stat basic_map_extend_accesses(__isl_take isl_basic_map *bmap,
	void *user)
{
	int n_in, n_out, extra_dims, i;
	isl_space *space;
	isl_id *tuple_id;
	isl_map *map = *(isl_map **)user;

	n_out = isl_basic_map_n_out(bmap);
	if (n_out == 0)		/* ignore scalar accesses */
		return isl_stat_ok;

	n_in = isl_basic_map_n_in(bmap);
	extra_dims = n_in - n_out;

	space = isl_basic_map_get_space(bmap);
	tuple_id = isl_space_get_tuple_id(space, isl_dim_out);
	isl_space_free(space);
	bmap = isl_basic_map_insert_dims(bmap, isl_dim_out, 0, extra_dims);
	bmap = isl_basic_map_set_tuple_id(bmap, isl_dim_out, tuple_id);

	for (i = 0; i < extra_dims; ++i)
	{
		isl_constraint *constraint = find_one_independent_access(bmap);
		if (!constraint)
		{
			isl_basic_map_free(bmap);
			return isl_stat_error;
		}
		constraint = isl_constraint_set_coefficient_si(constraint,
			isl_dim_out, i, -1);
		bmap = isl_basic_map_add_constraint(bmap, constraint);
	}

	map = isl_map_union(map, isl_map_from_basic_map(bmap));
	*(isl_map **)user = map;
	return isl_stat_ok;
}

static
isl_stat map_extend_accesses(__isl_take isl_map *map, void *user)
{
	isl_stat r;
	isl_map *result;
	isl_id *tuple_id;
	isl_space *space;
	int n_in, n_out, extra_dims;

	if (!user)
		return isl_stat_error;

	if (!map)
		return isl_stat_ok;

	n_in = isl_map_n_in(map);
	n_out = isl_map_n_out(map);

	/* keep accesses with enough dimensionality */
	if (n_in <= n_out)
	{
		result = map;
	}
	else
	{
		extra_dims = n_in - n_out;

		space = isl_map_get_space(map);
		tuple_id = isl_space_get_tuple_id(space, isl_dim_out);
		space = isl_space_insert_dims(space, isl_dim_out, 0, extra_dims);
		space = isl_space_set_tuple_id(space, isl_dim_out, tuple_id);
		result = isl_map_empty(space);
		r = isl_map_foreach_basic_map(map, &basic_map_extend_accesses, &result);
		isl_map_free(map);
		if (r != isl_stat_ok)
		{
			isl_map_free(result);
			return r;
		}
	}
	*(isl_union_map **)user =
		isl_union_map_add_map(*(isl_union_map **)user, result);
	return isl_stat_ok;
}

static __isl_give isl_union_map *union_map_extend_accesses(__isl_keep isl_union_map *umap)
{
	isl_union_map *result;
	isl_stat r;

	if (umap == NULL)
		return NULL;

	result = isl_union_map_empty(isl_union_map_get_space(umap));
	r = isl_union_map_foreach_map(umap, &map_extend_accesses, &result);
	if (r != isl_stat_ok)
	{
		isl_union_map_free(result);
		return NULL;
	}
	return result;
}

#if 0
// {[i,j,k] -> [o1,o2] : o1 = i and o2 = j} =>
// {[i,j,k] -> [o0,o1,o2] : o1 = i and o2 = j and o0 = k}
// in a general case, k should be orthogonal to i and j
isl_stat extend_access_by_1_bmap(__isl_take isl_basic_map *bmap,
						void *user)
{
	isl_constraint *constraint;
	isl_map *map = *(isl_map **)user;
	isl_id *tuple_id;

	isl_space *space = isl_basic_map_get_space(bmap);
	tuple_id = isl_space_get_tuple_id(space, isl_dim_out);
	bmap = isl_basic_map_insert_dims(bmap, isl_dim_out, 0, 1);
	bmap = isl_basic_map_set_tuple_id(bmap, isl_dim_out, tuple_id);

	constraint = find_one_independent_access(bmap);
	bmap = isl_basic_map_add_constraint(bmap, constraint);

	map = isl_map_union(map, isl_map_from_basic_map(bmap));
	*(isl_map **)user = map;
	return isl_stat_ok;
}

isl_stat extend_access_by_1_map(__isl_take isl_map *map,
							    void *user)
{
	isl_stat r;
	isl_map *result;
	isl_id *tuple_id;
	isl_space *space = isl_map_get_space(map);

	tuple_id = isl_space_get_tuple_id(space, isl_dim_out);
	space = isl_space_insert_dims(space, isl_dim_out, 0, 1);
	space = isl_space_set_tuple_id(space, isl_dim_out, tuple_id);
	result = isl_map_empty(space);
	r = isl_map_foreach_basic_map(map, &extend_access_by_1_bmap, &result);
	isl_map_free(map);
	if (r != isl_stat_ok)
	{
		isl_map_free(result);
		return r;
	}
	*(isl_union_map **)user = isl_union_map_add_map(*(isl_union_map **)user, result);
	return isl_stat_ok;
}

__isl_give isl_union_map *extend_access_by_1(__isl_keep isl_union_map *umap)
{
	isl_union_map *result;
	isl_stat r;

	if (umap == NULL)
		return NULL;

	result = isl_union_map_empty(isl_union_map_get_space(umap));
	r = isl_union_map_foreach_map(umap, &extend_access_by_1_map, &result);
	if (r != isl_stat_ok)
	{
		isl_union_map_free(result);
		return NULL;
	}
	return result;
}
#endif

/* increment - put 0 to produce a map {[A[*] -> B[x]] -> [A[*] -> B[y]]: y = 1}
 *             put 1 for {[A[*] -> B[x]] -> [A[*] -> B[y]]: y = x + 1},
 *      where A[*] stands for the original LHS of the map wrapped in the
 *      domain of a given map.
 */
static __isl_give isl_basic_map *counter_increment_or_one(
	__isl_keep isl_map *map, int increment)
{
	isl_basic_map *mapper;
	isl_constraint *constraint;
	isl_space *space = isl_map_get_space(map);
	isl_space *space_copy;
	isl_local_space *local_space;
	int n_dim = isl_space_dim(space, isl_dim_in);

	space = isl_space_domain(space);
	space_copy = isl_space_copy(space);
	space = isl_space_map_from_domain_and_range(space, space_copy);
	local_space = isl_local_space_from_space(isl_space_copy(space));
	mapper = isl_basic_map_identity(space);
	mapper = isl_basic_map_drop_constraints_involving_dims(mapper, isl_dim_in,
		n_dim - 1, 1);
	constraint = isl_constraint_alloc_equality(local_space);
	constraint = isl_constraint_set_constant_si(constraint, 1);
	constraint = isl_constraint_set_coefficient_si(constraint, isl_dim_in,
		n_dim - 1, increment);
	constraint = isl_constraint_set_coefficient_si(constraint, isl_dim_out,
		n_dim - 1, -1);
	mapper = isl_basic_map_add_constraint(mapper, constraint);

	return mapper;
}

static inline __isl_give isl_basic_map *counter_increment(
	__isl_keep isl_map *map)
{
	return counter_increment_or_one(map, 1);
}

static inline __isl_give isl_basic_map *counter_set_one(
	__isl_keep isl_map *map)
{
	return counter_increment_or_one(map, 0);
}

static isl_stat drop_constraints_with_last_dim(
	__isl_take isl_map *map, void *user)
{
	isl_union_map *result = *(isl_union_map **) user;
	int n_in = isl_map_n_in(map);

	map = isl_map_drop_constraints_involving_dims(map, isl_dim_in,
		n_in - 1, 1);
	result = isl_union_map_add_map(result, map);

	*(isl_union_map **) user = result;
	return isl_stat_ok;
}

static __isl_give isl_union_map *union_map_drop_last_in_dim(
	__isl_keep isl_union_map *umap)
{
	isl_union_map *result;
	isl_space *space = space = isl_union_map_get_space(umap);

	result = isl_union_map_empty(space);
	if (isl_union_map_foreach_map(umap, &drop_constraints_with_last_dim,
			&result) < 0)
		return isl_union_map_free(result);
	return result;
}

static __isl_give isl_union_map *union_map_symmetric_difference(
	__isl_take isl_union_map *map1,
	__isl_take isl_union_map *map2)
{
	isl_union_map *left = isl_union_map_subtract(isl_union_map_copy(map1),
		isl_union_map_copy(map2));
	isl_union_map *right = isl_union_map_subtract(map2, map1);
	return isl_union_map_union(left, right);
}

/* user/result has the shape {[Sx[*] -> cnt[r]] -> A[*]} where r is the
 * multiplicity of accesses that gets modified by the current function
 *
 * map has the shape [Sx[*] -> __pet_ref*[]] -> A[*]
 */
static isl_stat tagged_map_to_counted_map(__isl_take isl_map *map, void *user)
{
	isl_union_map *uintersection, *udifference, *umap, *uintersection_nolast;
	isl_map *intersection, *difference, *increment, *set_one;
	isl_union_map *result = *(isl_union_map **) user;
	isl_space *space = isl_map_get_space(map);
	isl_id *id = isl_map_get_tuple_id(map, isl_dim_out);
	int n_in = isl_space_dim(space, isl_dim_in);

	if (n_in == 0)
		space = isl_space_add_dims(space, isl_dim_in, 1);
	else
		space = isl_space_drop_inputs(space, 1, n_in - 1);

	space = isl_space_set_dim_name(space, isl_dim_in, 0, "__ppcg_cnt");
	space = isl_space_set_tuple_id(space, isl_dim_in, id);

	map = isl_map_domain_factor_domain(map);
	map = isl_map_domain_product(map, isl_map_universe(space));
	umap = isl_union_map_from_map(map);
	uintersection = isl_union_map_intersect(isl_union_map_copy(result),
		isl_union_map_copy(umap));

	isl_union_map *symmdiff = union_map_symmetric_difference(
		union_map_drop_last_in_dim(uintersection),
		isl_union_map_copy(umap));

	if (isl_union_map_is_empty(symmdiff))
	{
		isl_union_map_free(umap);
		result = isl_union_map_subtract(result,
			isl_union_map_copy(uintersection));

		// we just saw it is the same as map plus counter value,
		// so it is surely a single map
		map = isl_map_from_union_map(uintersection);
		increment = isl_map_from_basic_map(counter_increment(map));
		map = isl_map_apply_domain(map, increment);
		result = isl_union_map_add_map(result, map);
	}
	else // just add, it should not intersect
	{
		// MUST BE A MAP!
		map = isl_map_from_union_map(umap);
		set_one = isl_map_from_basic_map(counter_set_one(map));
		map = isl_map_apply_domain(map, set_one);
		result = isl_union_map_add_map(result, map);
	}

#if 0
	if (isl_union_map_is_empty(uintersection))
	{
		difference = isl_map_from_union_map(umap);
	}
	else
	{
		result = isl_union_map_subtract(result,
			isl_union_map_copy(uintersection));

		// space = isl_union_map_get_space(umap);
		// uintersection_nolast = isl_union_map_empty(space);
		// isl_union_map_foreach_map(uintersection,
		// 	&drop_constraints_with_last_dim, &uintersection_nolast);
		uintersection_nolast = union_map_drop_last_in_dim(uintersection);
		udifference = isl_union_map_subtract(umap,
			uintersection_nolast);

		// these MUST be single map here, since we intersected with a simple map
		intersection = isl_map_from_union_map(uintersection);
		if (!isl_union_map_is_empty(udifference))
			difference = isl_map_from_union_map(udifference);
		else
		{
			isl_union_map_free(udifference);
			difference = NULL;
		}

		increment = isl_map_from_basic_map(counter_increment(intersection));
		intersection = isl_map_apply_domain(intersection, increment);
		result = isl_union_map_add_map(result, intersection);
	}

	if (difference)
	{
		set_one = isl_map_from_basic_map(counter_set_one(difference));
		difference = isl_map_apply_domain(difference, set_one);
		result = isl_union_map_add_map(result, difference);
	}
#endif

	*(isl_union_map **) user = result;
	return isl_stat_ok;
}

struct union_map_transform_data
{
	__isl_give isl_map *(*f)(__isl_take isl_map *, void *);
	isl_union_map *result;
	void *user;
};

static isl_stat union_map_transform_helper(__isl_take isl_map *map,
	void *user)
{
	struct union_map_transform_data *data = user;
	map = data->f(map, data->user);
	if (!map)
		return isl_stat_error;
	data->result = isl_union_map_add_map(data->result, map);
	if (!data->result)
		return isl_stat_error;
	return isl_stat_ok;
}

static __isl_give isl_union_map *union_map_transform(
	__isl_take isl_union_map *umap,
	__isl_give isl_map *(*f)(__isl_take isl_map *, void *),
	void *user)
{
	isl_union_map *result;
	isl_space *space;

	if (!umap)
		return NULL;

	space = isl_union_map_get_space(umap);
	result = isl_union_map_empty(space);
	struct union_map_transform_data data = {f, result, user};
	if (isl_union_map_foreach_map(umap, &union_map_transform_helper,
			&data) < 0)
		result = isl_union_map_free(result);

	isl_union_map_free(umap);

	return result;
}

static inline __isl_give isl_map *retag_map(__isl_take isl_map *map,
	const char *prefix)
{
	isl_space *space;
	const char *name;
	char new_name[80];

	if (strlen(prefix) > 20)
		return isl_map_free(map);

	space = isl_map_get_space(map);
	space = isl_space_domain_factor_range(space);

	if (!space)
		return isl_map_free(map);

	name = isl_space_get_tuple_name(space, isl_dim_in);
	snprintf(new_name, 80, "__ppcg_%s_%s", prefix, name + 6);
	space = isl_space_set_tuple_name(space, isl_dim_in, new_name);

	map = isl_map_domain_factor_domain(map);
	map = isl_map_domain_product(map, isl_map_universe(space));

	return map;
}

static __isl_give isl_map *retag_map_helper(__isl_take isl_map *map,
	void *user)
{
	const char *prefix = user;
	return retag_map(map, prefix);
}

static __isl_give isl_union_map *tagged_union_map_to_counted(
	__isl_take isl_union_map *counted_accesses,
	__isl_take isl_union_map *tagged_accesses)
{
	if (!tagged_accesses)
		return counted_accesses;

	// TODO: can we do it directly on unions ?
	// we need a way to change __pet_ref* to A* without merging identical unions
	if (isl_union_map_foreach_map(tagged_accesses, &tagged_map_to_counted_map,
			&counted_accesses) < 0)
		counted_accesses = isl_union_map_free(counted_accesses);
	isl_union_map_free(tagged_accesses);

	return counted_accesses;
}

static __isl_give isl_union_map *compute_counted_accesses(
	__isl_keep isl_union_map *tagged_reads,
	__isl_keep isl_union_map *tagged_may_writes,
	__isl_keep isl_union_map *tagged_must_writes)
{
	isl_union_map *all_writes, *counted_accesses;
	isl_space *space;

	if (!tagged_reads || !tagged_may_writes || !tagged_must_writes)
		return NULL;

	space = isl_union_map_get_space(tagged_reads);
	counted_accesses = isl_union_map_empty(space);

	all_writes = isl_union_map_union(isl_union_map_copy(tagged_may_writes),
		isl_union_map_copy(tagged_must_writes));
	counted_accesses = tagged_union_map_to_counted(counted_accesses,
		isl_union_map_copy(tagged_reads));
	counted_accesses = tagged_union_map_to_counted(counted_accesses, all_writes);

	return counted_accesses;
}

/* Extract a ppcg_scop from a pet_scop.
 *
 * The constructed ppcg_scop refers to elements from the pet_scop
 * so the pet_scop should not be freed before the ppcg_scop.
 */
static struct ppcg_scop *ppcg_scop_from_pet_scop(struct pet_scop *scop,
	struct ppcg_options *options)
{
	int i;
	isl_ctx *ctx;
	struct ppcg_scop *ps;

	if (!scop)
		return NULL;

	ctx = isl_set_get_ctx(scop->context);

	ps = isl_calloc_type(ctx, struct ppcg_scop);
	if (!ps)
		return NULL;

	ps->names = collect_names(scop);
	ps->options = options;
	ps->start = pet_loc_get_start(scop->loc);
	ps->end = pet_loc_get_end(scop->loc);
	ps->context = isl_set_copy(scop->context);
	ps->context = set_intersect_str(ps->context, options->ctx);
	if (options->non_negative_parameters) {
		isl_space *space = isl_set_get_space(ps->context);
		isl_set *nn = isl_set_nat_universe(space);
		ps->context = isl_set_intersect(ps->context, nn);
	}
	ps->domain = collect_non_kill_domains(scop);
	ps->call = collect_call_domains(scop);
	ps->tagged_reads = pet_scop_get_tagged_may_reads(scop);
	ps->reads = pet_scop_get_may_reads(scop);
	ps->tagged_may_writes = pet_scop_get_tagged_may_writes(scop);
	ps->may_writes = pet_scop_get_may_writes(scop);
	ps->tagged_must_writes = pet_scop_get_tagged_must_writes(scop);
	ps->must_writes = pet_scop_get_must_writes(scop);
	ps->tagged_must_kills = pet_scop_get_tagged_must_kills(scop);
	ps->must_kills = pet_scop_get_must_kills(scop);
	ps->schedule = isl_schedule_copy(scop->schedule);
	ps->pet = scop;
	ps->independence = isl_union_map_empty(isl_set_get_space(ps->context));
	for (i = 0; i < scop->n_independence; ++i)
		ps->independence = isl_union_map_union(ps->independence,
			isl_union_map_copy(scop->independences[i]->filter));

	ps->retagged_must_writes = union_map_transform(
		isl_union_map_copy(ps->tagged_must_writes),
		&retag_map_helper, "1must_write");
	ps->retagged_reads = union_map_transform(
		isl_union_map_copy(ps->tagged_reads),
		&retag_map_helper, "2read");
	compute_retagged_tagger(ps);
	compute_retagged_dependences(ps);

	isl_union_map_debug(ps->retagged_dep);

	if(options->model_spatial_locality){
		ps->cache_block_reads = map_array_accesses_to_cache_blocks(ps->reads, CACHE_SIZE);
		ps->cache_block_may_writes = map_array_accesses_to_cache_blocks(ps->may_writes, CACHE_SIZE);
		ps->cache_block_must_writes = map_array_accesses_to_cache_blocks(ps->must_writes, CACHE_SIZE);

		ps->tagged_cache_block_reads = map_array_accesses_to_cache_blocks(ps->tagged_reads, CACHE_SIZE);
		ps->tagged_cache_block_may_writes = map_array_accesses_to_cache_blocks(ps->tagged_may_writes, CACHE_SIZE);
		ps->tagged_cache_block_must_writes = map_array_accesses_to_cache_blocks(ps->tagged_must_writes, CACHE_SIZE);
	}

	//ps->adjacent_reads = map_array_accesses_to_next_element(ps->reads);
	//ps->adjacent_may_writes = map_array_accesses_to_next_element(ps->may_writes);
	//ps->adjacent_must_writes = map_array_accesses_to_next_element(ps->must_writes);

#if 0
	ps->cache_accesses_from_must = map_array_accesses_to_next_elements(ps->must_writes);
	ps->cache_accesses_from_may = map_array_accesses_to_next_elements(ps->may_writes);
	ps->cache_accesses_from_may = isl_union_map_union(ps->cache_accesses_from_may,
					map_array_accesses_to_next_elements(ps->reads));
#endif

	ps->cache_array_tagged_reads = union_wrap_with_array_name(
		map_array_accesses_to_next_elements(ps->reads));
	ps->cache_array_tagged_may_writes = union_wrap_with_array_name(
		map_array_accesses_to_next_elements(ps->may_writes));
	ps->cache_array_tagged_must_writes = union_wrap_with_array_name(
		map_array_accesses_to_next_elements(ps->must_writes));

	// compute_array_tagger(ps);
	// compute_array_tagged_dependences(ps);

	ps->counted_accesses = compute_counted_accesses(ps->tagged_reads,
		ps->tagged_may_writes, ps->tagged_must_writes);


#if 0
	if(options->only_cache_block_deps){
		ps->reads = map_array_accesses_to_cache_blocks(ps->reads, CACHE_SIZE);
		ps->may_writes = map_array_accesses_to_cache_blocks(ps->may_writes, CACHE_SIZE);
		ps->must_writes = map_array_accesses_to_cache_blocks(ps->must_writes, CACHE_SIZE);
		ps->must_kills = map_array_accesses_to_cache_blocks(ps->must_kills, CACHE_SIZE);
		ps->tagged_reads = map_array_accesses_to_cache_blocks(ps->tagged_reads, CACHE_SIZE);
		ps->tagged_may_writes = map_array_accesses_to_cache_blocks(ps->tagged_may_writes, CACHE_SIZE);
		ps->tagged_must_writes = map_array_accesses_to_cache_blocks(ps->tagged_must_writes, CACHE_SIZE);
		ps->tagged_must_kills = map_array_accesses_to_cache_blocks(ps->tagged_must_kills, CACHE_SIZE);
	}
#endif

	compute_tagger(ps);
	compute_dependences(ps);
	eliminate_dead_code(ps);

	if (!ps->context || !ps->domain || !ps->call || !ps->reads ||
	    !ps->may_writes || !ps->must_writes || !ps->tagged_must_kills ||
	    !ps->must_kills || !ps->schedule || !ps->independence ||
            !ps->names || !ps->counted_accesses)
		return ppcg_scop_free(ps);

	return ps;
}

/* Internal data structure for ppcg_transform.
 */
struct ppcg_transform_data {
	struct ppcg_options *options;
	__isl_give isl_printer *(*transform)(__isl_take isl_printer *p,
		struct ppcg_scop *scop, void *user);
	void *user;
};

/* Should we print the original code?
 * That is, does "scop" involve any data dependent conditions or
 * nested expressions that cannot be handled by pet_stmt_build_ast_exprs?
 */
static int print_original(struct pet_scop *scop, struct ppcg_options *options)
{
	if (!pet_scop_can_build_ast_exprs(scop)) {
		if (options->debug->verbose)
			fprintf(stdout, "Printing original code because "
				"some index expressions cannot currently "
				"be printed\n");
		return 1;
	}

	if (pet_scop_has_data_dependent_conditions(scop)) {
		if (options->debug->verbose)
			fprintf(stdout, "Printing original code because "
				"input involves data dependent conditions\n");
		return 1;
	}

	return 0;
}

/* Callback for pet_transform_C_source that transforms
 * the given pet_scop to a ppcg_scop before calling the
 * ppcg_transform callback.
 *
 * If "scop" contains any data dependent conditions or if we may
 * not be able to print the transformed program, then just print
 * the original code.
 */
static __isl_give isl_printer *transform(__isl_take isl_printer *p,
	struct pet_scop *scop, void *user)
{
	struct ppcg_transform_data *data = user;
	struct ppcg_scop *ps;

	if (print_original(scop, data->options)) {
		p = pet_scop_print_original(scop, p);
		pet_scop_free(scop);
		return p;
	}

	scop = pet_scop_align_params(scop);
	ps = ppcg_scop_from_pet_scop(scop, data->options);

	p = data->transform(p, ps, data->user);

	ppcg_scop_free(ps);
	pet_scop_free(scop);

	return p;
}

/* Transform the C source file "input" by rewriting each scop
 * through a call to "transform".
 * The transformed C code is written to "out".
 *
 * This is a wrapper around pet_transform_C_source that transforms
 * the pet_scop to a ppcg_scop before calling "fn".
 */
int ppcg_transform(isl_ctx *ctx, const char *input, FILE *out,
	struct ppcg_options *options,
	__isl_give isl_printer *(*fn)(__isl_take isl_printer *p,
		struct ppcg_scop *scop, void *user), void *user)
{
	struct ppcg_transform_data data = { options, fn, user };
	return pet_transform_C_source(ctx, input, out, &transform, &data);
}

/* Check consistency of options.
 *
 * Return -1 on error.
 */
static int check_options(isl_ctx *ctx)
{
	struct options *options;

	options = isl_ctx_peek_options(ctx, &options_args);
	if (!options)
		isl_die(ctx, isl_error_internal,
			"unable to find options", return -1);

	if (options->ppcg->openmp &&
	    !isl_options_get_ast_build_atomic_upper_bound(ctx))
		isl_die(ctx, isl_error_invalid,
			"OpenMP requires atomic bounds", return -1);

	return 0;
}

int main(int argc, char **argv)
{
	int r;
	isl_ctx *ctx;
	struct options *options;

	options = options_new_with_defaults();
	assert(options);

	ctx = isl_ctx_alloc_with_options(&options_args, options);
	ppcg_options_set_target_defaults(options->ppcg);
	isl_options_set_ast_build_detect_min_max(ctx, 1);
	isl_options_set_ast_print_macro_once(ctx, 1);
	isl_options_set_schedule_whole_component(ctx, 0);
	isl_options_set_schedule_maximize_band_depth(ctx, 1);
	isl_options_set_schedule_maximize_coincidence(ctx, 1);
	pet_options_set_encapsulate_dynamic_control(ctx, 1);
	argc = options_parse(options, argc, argv, ISL_ARG_ALL);

	if (check_options(ctx) < 0)
		r = EXIT_FAILURE;
	else if (options->ppcg->target == PPCG_TARGET_CUDA)
		r = generate_cuda(ctx, options->ppcg, options->input);
	else if (options->ppcg->target == PPCG_TARGET_OPENCL)
		r = generate_opencl(ctx, options->ppcg, options->input,
				options->output);
	else
		r = generate_cpu(ctx, options->ppcg, options->input,
				options->output);

	isl_ctx_free(ctx);

	return r;
}
