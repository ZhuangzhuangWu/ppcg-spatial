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

#define CACHE_SIZE 4
#define DISTANCE 1

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
	isl_union_map *tagged;
	isl_union_pw_multi_aff *tagger;

	tagged = isl_union_map_copy(ps->tagged_reads);
	tagged = isl_union_map_union(tagged,
				isl_union_map_copy(ps->tagged_may_writes));
	tagged = isl_union_map_union(tagged,
				isl_union_map_copy(ps->tagged_must_kills));
	tagged = isl_union_map_universe(tagged);

	tagged = isl_union_set_unwrap(isl_union_map_domain(tagged));

	tagger = isl_union_map_domain_map_union_pw_multi_aff(tagged);

	ps->tagger = tagger;
}

static __isl_give isl_union_pw_multi_aff *compute_retagged_tagger(
	__isl_keep isl_union_map *reads, __isl_keep isl_union_map *must_writes)
{
	isl_union_map *tagged;

	tagged = isl_union_map_copy(reads);
	tagged = isl_union_map_union(tagged, isl_union_map_copy(must_writes));
	tagged = isl_union_map_universe(tagged);
	tagged = isl_union_set_unwrap(isl_union_map_domain(tagged));
	return isl_union_map_domain_map_union_pw_multi_aff(tagged);
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
 * Both must-writes and must-kills are allowed to kill dependences
 * from earlier writes to subsequent reads.
 * The must-kills are not included in the potential sources, though.
 * The flow dependences with a must-kill as source would
 * reflect possibly uninitialized reads.
 * No dependences need to be introduced to protect such reads
 * (other than those imposed by potential flows from may writes
 * that follow the kill).  Those flow dependences are therefore not needed.
 * The dead code elimination also assumes
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
	kills = isl_union_map_union(kills, must_source);
	access = isl_union_access_info_from_sink(
				isl_union_map_copy(ps->tagged_reads));
	access = isl_union_access_info_set_kill(access, kills);
	access = isl_union_access_info_set_may_source(access,
				isl_union_map_copy(ps->tagged_may_writes));
	access = isl_union_access_info_set_schedule(access, schedule);
	flow = isl_union_access_info_compute_flow(access);
	tagged_flow = isl_union_flow_get_may_dependence(flow);
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
 *
 * Both must-writes and must-kills are allowed to kill dependences
 * from earlier writes to subsequent reads, as in compute_tagged_flow_dep_only.
 */
static void compute_flow_dep(struct ppcg_scop *ps)
{
	isl_union_access_info *access;
	isl_union_flow *flow;
	isl_union_map *kills, *must_writes;

	access = isl_union_access_info_from_sink(isl_union_map_copy(ps->reads));
	kills = isl_union_map_copy(ps->must_kills);
	must_writes = isl_union_map_copy(ps->must_writes);
	kills = isl_union_map_union(kills, must_writes);
	access = isl_union_access_info_set_kill(access, kills);
	access = isl_union_access_info_set_may_source(access,
				isl_union_map_copy(ps->may_writes));
	access = isl_union_access_info_set_schedule(access,
				isl_schedule_copy(ps->schedule));
	flow = isl_union_access_info_compute_flow(access);

	ps->dep_flow = isl_union_flow_get_may_dependence(flow);
	ps->live_in = isl_union_flow_get_may_no_source(flow);
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

	may_source = isl_union_map_union(isl_union_map_copy(scop->may_writes),
					isl_union_map_copy(scop->reads));
	access = isl_union_access_info_from_sink(
				isl_union_map_copy(scop->may_writes));
	access = isl_union_access_info_set_kill(access,
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
	__isl_keep isl_union_map *reads, __isl_keep isl_union_map *writes,
	__isl_keep isl_union_pw_multi_aff *retagged_tagger)
{
	isl_schedule *schedule;
	schedule = isl_schedule_copy(ps->schedule);
	schedule = isl_schedule_pullback_union_pw_multi_aff(schedule,
		isl_union_pw_multi_aff_copy(retagged_tagger));

	add_retagged_dependences(ps, reads, writes, schedule);
	add_retagged_dependences(ps, reads, reads, schedule);
	add_retagged_dependences(ps, writes, reads, schedule);
	add_retagged_dependences(ps, writes, writes, schedule);
	isl_schedule_free(schedule);
}

struct map_transform_helper_data
{
	isl_map *result;
	void *user;
	__isl_give isl_basic_map *(*f)(__isl_take isl_basic_map *, void *);
};

static isl_stat map_transform_helper(__isl_take isl_basic_map *bmap,
	void *user)
{
	struct map_transform_helper_data *data = user;
	bmap = data->f(bmap, data->user);
	if (!bmap)
		return isl_stat_error;
	data->result = isl_map_union(data->result,
		isl_map_from_basic_map(bmap));
	if (!data->result)
		return isl_stat_error;
	return isl_stat_ok;
}

static __isl_give isl_map *map_transform(__isl_take isl_map *map,
	__isl_give isl_basic_map *(*f)(__isl_take isl_basic_map *, void *),
	void *user)
{
	isl_space *space;
	isl_map *result;
	struct map_transform_helper_data data;
	isl_stat r;

	if (!map)
		return NULL;

	space = isl_map_get_space(map);
	result = isl_map_empty(space);
	data.result = result;
	data.user = user;
	data.f = f;
	r = isl_map_foreach_basic_map(map, &map_transform_helper, &data);
	isl_map_free(map);
	if (r == isl_stat_error)
		data.result = isl_map_free(data.result);
	return data.result;
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

static __isl_give isl_basic_map *basic_map_drop_all_inequalities(
	__isl_take isl_basic_map *bmap)
{
	isl_mat *eq, *ineq;
	isl_ctx *ctx;
	isl_space *space;

	if (!bmap)
		return NULL;

	ctx = isl_basic_map_get_ctx(bmap);
	eq = isl_basic_map_equalities_matrix(bmap,
		isl_dim_cst, isl_dim_in, isl_dim_out, isl_dim_div, isl_dim_param);
	ineq = isl_mat_alloc(ctx, 0, isl_mat_cols(eq));
	space = isl_basic_map_get_space(bmap);
	isl_basic_map_free(bmap);

	bmap = isl_basic_map_from_constraint_matrices(space, eq, ineq,
		isl_dim_cst, isl_dim_in, isl_dim_out, isl_dim_div, isl_dim_param);
	return bmap;
}

static __isl_give isl_basic_map *basic_map_drop_all_inequalities_helper(
	__isl_take isl_basic_map *bmap, void *user)
{
	(void) user;
	return basic_map_drop_all_inequalities(bmap);
}

static __isl_give isl_map *map_drop_all_inequalities_helper(
	__isl_take isl_map *map, void *user)
{
	return map_transform(map, &basic_map_drop_all_inequalities_helper, user);
}

static __isl_give isl_union_map *union_map_drop_all_inequalities(
	__isl_take isl_union_map *umap)
{
	return union_map_transform(umap, &map_drop_all_inequalities_helper, NULL);
}

static isl_union_map *map_array_accesses_to_next_elements(isl_union_map *);
static isl_union_map *map_array_accesses_to_cache_blocks(isl_union_map *);
static isl_union_map *map_array_accesses_to_next_elements_grouped(
	isl_union_map *);
static isl_map *retag_map_helper(isl_map *map, void *user);
static isl_stat const_complete_accesses(
	isl_union_map **, isl_union_map **);

static void compute_retagged_dependences_model(struct ppcg_scop *ps,
	__isl_give isl_union_map * (*spatial_model)(__isl_keep isl_union_map *))
{
	isl_union_map *spatial_reads, *spatial_writes;
	isl_union_map *retagged_reads, *retagged_must_writes;

	retagged_must_writes = union_map_transform(
		isl_union_map_copy(ps->tagged_must_writes),
		&retag_map_helper, "1must_write");
	retagged_reads = union_map_transform(
		isl_union_map_copy(ps->tagged_reads),
		&retag_map_helper, "2read");
	isl_union_pw_multi_aff *retagged_tagger =
		compute_retagged_tagger(retagged_reads, retagged_must_writes);

	ps->retagged_dep = isl_union_map_empty(
		isl_union_set_get_space(ps->domain));

	// Spatial locality dependences ()
	spatial_reads = isl_union_map_copy(retagged_reads);
	spatial_writes = isl_union_map_copy(retagged_must_writes);
	const_complete_accesses(&spatial_reads, &spatial_writes);
	spatial_reads = spatial_model(spatial_reads);
	spatial_writes = spatial_model(spatial_writes);
	add_all_retagged_dependences(ps, spatial_reads, spatial_writes,
		retagged_tagger);
	isl_union_map_free(spatial_reads);
	isl_union_map_free(spatial_writes);

	// Original dependences.
//	add_all_retagged_dependences(ps, retagged_reads,
//		retagged_must_writes, retagged_tagger);

	// ps->retagged_dep = union_map_drop_all_inequalities(ps->retagged_dep);
	ps->counted_accesses = isl_union_map_union(retagged_reads,
		retagged_must_writes);
}

static void compute_retagged_dependences(struct ppcg_scop *ps)
{
	compute_retagged_dependences_model(ps,
		&map_array_accesses_to_next_elements);
}

static void compute_retagged_dependences_groups(struct ppcg_scop *ps)
{
	compute_retagged_dependences_model(ps,
		&map_array_accesses_to_cache_blocks);
}

static void compute_retagged_dependences_ends_grouped(struct ppcg_scop *ps)
{
	compute_retagged_dependences_model(ps,
		&map_array_accesses_to_next_elements_grouped);
}

isl_bool constraint_has_only_zero_coefficients(
	__isl_keep isl_constraint *constraint, enum isl_dim_type type)
{
	int i;
	int n = isl_constraint_dim(constraint, type);
	isl_bool b;

	for (i = 0; i < n; ++i) {
		isl_val *v = isl_constraint_get_coefficient_val(constraint,
								type, i);
		b = isl_val_is_zero(v);
		isl_val_free(v);
		if (b != isl_bool_true)
			return b;
	}

	return isl_bool_true;
}

isl_bool basic_map_is_uniform(__isl_keep isl_basic_map *bmap)
{
	isl_constraint *constraint;
	int n_in, n_out, i, n_min, n_max;
	isl_bool eq = isl_bool_error;
	int has_one_stride = 0;

	n_in = isl_basic_map_n_in(bmap);
	n_out = isl_basic_map_n_out(bmap);

	// if (n_out != n_in)
	// 	return isl_bool_false;

	// if (n_in > n_out)
		// return isl_bool_false; // FIXME: dropping the case (i,j,k)->C[i,j] => (i,j)->C[i,j]

	// what about the case (i,j)->C[i,0] => (i,j)->C[i,0]; constant and paramteric-only accesses seem to be managed already (coef for the variable partial is zero)

	// what about (i,j)->C[i,j,j] => (i,j,k)->C[i,j,k]?
	// dep: i'=i, j'=j, k'=j;  does not suffice to check min(n_in, n_out dimensions)
	// check the remaining deps to be constant/parametric only?
	// (i,j)->C[i,j,N] -> (i,j,k)->C[i,j,k]:
	// dep = i'=i, j'=j, k'=N

	// if (n_out > n_in) // case (i,j)->C[i,j] => (i,j,k)->C[i,j]

	n_min = n_in < n_out ? n_in : n_out;
	n_max = n_in > n_out ? n_in : n_out;

	for (i = 0; i < n_max; ++i) {
		isl_val *vin, *vout, *vcst;
		isl_bool b;

		if (i < n_min) {
			if ((eq = isl_basic_map_has_defining_equality(bmap, isl_dim_out,
					i, &constraint)) != isl_bool_true)
				return eq;

			vin = isl_constraint_get_coefficient_val(constraint,
				isl_dim_in, i);
			vout = isl_constraint_get_coefficient_val(constraint,
				isl_dim_out, i);

			vout = isl_val_neg(vout);
			eq = isl_val_eq(vin, vout);
			isl_val_free(vin);
			isl_val_free(vout);
			if (eq != isl_bool_true)
				goto notfound;

			vcst = isl_constraint_get_constant_val(constraint);
			b = isl_val_is_zero(vcst);
			isl_val_free(vcst);
			if (b == isl_bool_error) {
				eq = isl_bool_error;
				goto notfound;
			}
			if (!b) {
				if (has_one_stride) {
					eq = isl_bool_false;
					goto notfound;
				} else {
					has_one_stride = 1;
				}
			}

			constraint = isl_constraint_set_coefficient_si(constraint,
				isl_dim_in, i, 0);
			constraint = isl_constraint_set_coefficient_si(constraint,
				isl_dim_out, i, 0);
		} else {
			enum isl_dim_type type = n_in > n_out ? isl_dim_in : isl_dim_out;

			eq = isl_basic_map_has_defining_equality(bmap, type,
				i, &constraint);
			if (eq == isl_bool_error)
				return eq;
			if (!eq)
				continue;  // This allows (i,j)->C[i,j] => (i,j,k)->C[i,j] for address-based (non-dataflow) analysis where dep. i'=i, j'=j, lb_k<=k<=ub_k.
						// FIXME: but shoud we account for dependences that have old "cache style", i.e. 32*i <= i' <= 32*i+31 ??  they won't fit current conditions, but may be useful (even if they are not uniform...)
		}

		eq = constraint_has_only_zero_coefficients(constraint,
							   isl_dim_in);
		if (eq != isl_bool_true)
			goto notfound;

		eq = constraint_has_only_zero_coefficients(constraint,
							   isl_dim_out);
		if (eq != isl_bool_true)
			goto notfound;

		isl_constraint_free(constraint);
	}

	return isl_bool_true;

notfound:
	isl_constraint_free(constraint);
	return eq;
}

/* If a basic map "bmap" is uniform, return itself, otherwise return an empty
 * map in the same space.  Return NULL on errors.
 */
__isl_give isl_basic_map *basic_map_filter_uniform(
	__isl_take isl_basic_map *bmap, void *user)
{
	isl_bool uniform;

	(void) user;

	uniform = basic_map_is_uniform(bmap);
	if (uniform == isl_bool_error) {
		return isl_basic_map_free(bmap);
	} else if (uniform == isl_bool_false) {
		isl_space *space = isl_basic_map_get_space(bmap);
		isl_basic_map_free(bmap);
		return isl_basic_map_empty(space);
	}

	return bmap;
}

__isl_give isl_map *map_filter_uniform_helper(__isl_take isl_map *map,
	void *user)
{
	return map_transform(map, &basic_map_filter_uniform, user);
}

__isl_give isl_union_map *union_map_filter_uniform(
	__isl_take isl_union_map *umap)
{
	return union_map_transform(umap, &map_filter_uniform_helper, NULL);
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
	dep = isl_union_map_intersect_domain(dep, isl_union_set_copy(ps->domain));
	dep = isl_union_map_intersect_range(dep, isl_union_set_copy(ps->domain));
	dep = isl_union_map_reverse(dep);
	live = isl_union_set_intersect(live, isl_union_set_copy(ps->domain));

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

	if (ps->options->spatial_model == PPCG_SPATIAL_MODEL_GROUPS ||
		ps->options->spatial_model == PPCG_SPATIAL_MODEL_ENDS) {
		isl_union_map_free(ps->retagged_dep);
		isl_union_map_free(ps->counted_accesses);
	}

	if (ps->options->remove_nonuniform == PPCG_REMOVE_NONUNIFORM_ALL) {
		isl_union_map_free(ps->dep_flow_uniform);
	}

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

static isl_union_map *union_map_extend_accesses(isl_union_map *);

__isl_give isl_union_map *map_array_accesses_to_cache_blocks(
	__isl_keep isl_union_map *accesses)
{
	isl_space *space, *access_domain, *cache_map_space;
	struct cache_block_map_data data = { CACHE_SIZE };

	accesses = union_map_extend_accesses(accesses);
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

static isl_union_map *union_map_extend_accesses(isl_union_map *);

static __isl_give isl_union_map *map_array_accesses_to_next_elements(
	__isl_keep isl_union_map *access)
{
	isl_space *space;
	isl_union_map *mapped;
	isl_ctx *ctx = isl_union_map_get_ctx(access);
	int distance = isl_options_get_schedule_spatial_distance(ctx);

	space = isl_union_map_get_space(access); // get the parameteric space
	mapped = isl_union_map_empty(space);
	array_access_next_data data = {mapped, DISTANCE, 0};

	isl_union_map *extended = union_map_extend_accesses(access);

	if (isl_union_map_foreach_map(extended,
			&array_access_to_next_elements, &data) < 0)
		data.result = isl_union_map_free(data.result);
	isl_union_map_free(extended);

	return data.result;
}

static isl_bool access_is_full_ranked(__isl_take isl_basic_map *access)
{
	int n_in, n_out, n_cols, rank;
	isl_mat *eqs;

	if (!access)
		return isl_bool_error;

	access = isl_basic_map_affine_hull(access);
	access = isl_basic_map_remove_divs(access);
	n_in = isl_basic_map_n_in(access);
	n_out = isl_basic_map_n_out(access);
	access = isl_basic_map_drop_constraints_not_involving_dims(access,
		isl_dim_in, 0, n_in);
	access = isl_basic_map_drop_constraints_not_involving_dims(access,
		isl_dim_out, 0, n_out);
	eqs = isl_basic_map_equalities_matrix(access,
		isl_dim_in, isl_dim_out, isl_dim_div, isl_dim_param, isl_dim_cst);
	isl_basic_map_free(access);

	n_cols = isl_mat_cols(eqs);
	eqs = isl_mat_drop_cols(eqs, n_in + n_out, n_cols - n_in - n_out);
	rank = isl_mat_rank(eqs);

	// FIXME: does not belong to this function
	// this decides to drop spatial proximity deps if the last access function
	// is linearly dependent on the first ones, nothing to do with fullness
	if (n_out > n_in && rank == n_in) {
		int prefix_rank;
		n_cols = isl_mat_cols(eqs);
		eqs = isl_mat_drop_cols(eqs, n_cols - 1, 1);
		prefix_rank = isl_mat_rank(eqs);
		if (prefix_rank == rank) {
			return isl_bool_false;
			isl_mat_free(eqs);
		}
	}

	isl_mat_free(eqs);

	// FIXME: bad hack, possible optimization
	if (n_in <= 1)
		return isl_bool_false;

	return (rank == n_in) ? isl_bool_true : isl_bool_false;
}

struct separate_by_full_ranked_data {
	isl_map *full_ranked;
	isl_map *non_full_ranked;
};

static isl_stat separate_by_full_ranked_bmap(__isl_take isl_basic_map *access,
	void *user)
{
	struct separate_by_full_ranked_data *data = user;
	isl_bool r = access_is_full_ranked(isl_basic_map_copy(access));
	if (r == isl_bool_error) {
		isl_basic_map_free(access);
		return isl_stat_error;
	}
	if (r) {
		data->full_ranked = isl_map_union(data->full_ranked,
			isl_map_from_basic_map(access));
	} else {
		data->non_full_ranked = isl_map_union(data->non_full_ranked,
			isl_map_from_basic_map(access));
	}
	return isl_stat_ok;
}

static isl_stat separate_by_full_ranked(__isl_keep isl_map *access,
	__isl_give isl_map **full_ranked, __isl_give isl_map **non_full_ranked)
{
	isl_stat r;
	struct separate_by_full_ranked_data data;
	isl_space *space = isl_map_get_space(access);

	data.full_ranked = isl_map_empty(isl_space_copy(space));
	data.non_full_ranked = isl_map_empty(space);
	if ((r = isl_map_foreach_basic_map(access,
			&separate_by_full_ranked_bmap, &data)) != isl_stat_ok) {
		data.full_ranked = isl_map_free(data.full_ranked);
		data.non_full_ranked = isl_map_free(data.non_full_ranked);
	}

	if (full_ranked)
		*full_ranked = data.full_ranked;
	else
		isl_map_free(data.full_ranked);

	if (non_full_ranked)
		*non_full_ranked = data.non_full_ranked;
	else
		isl_map_free(data.non_full_ranked);

	return r;
}

struct next_elements_grouped_data {
	int distance;
	int group_size;
	isl_union_map *result;
};

static isl_stat array_access_to_next_elements_grouped(
	__isl_take isl_map *map, void *user)
{
	isl_space *space;
	isl_local_space *local_space;
	isl_constraint *constraint;
	isl_map *mapper, *partial_identity, *mapped;
	isl_map *full_ranked, *non_full_ranked;
	int n_dim, n_param;
	isl_val *val;
	isl_ctx *ctx = isl_map_get_ctx(map);
	struct next_elements_grouped_data *data = user;

	if (!isl_map_n_out(map)) {
		return isl_stat_ok;
	}

	separate_by_full_ranked(map, &full_ranked, &non_full_ranked);
	isl_map_free(map);
	map = full_ranked;

	space = isl_map_get_space(map);
	space = isl_space_range(space);
	space = isl_space_map_from_domain_and_range(isl_space_copy(space), space);
	mapper = isl_map_universe(isl_space_copy(space));

	n_dim = isl_space_dim(space, isl_dim_in);
	n_param = isl_space_dim(space, isl_dim_param);

	mapper = isl_map_add_dims(mapper, isl_dim_param, 1);
	local_space = isl_local_space_from_space(isl_space_copy(space));
	local_space = isl_local_space_add_dims(local_space, isl_dim_param, 1);

	constraint = isl_constraint_alloc_inequality(
		isl_local_space_copy(local_space));
	constraint = isl_constraint_set_coefficient_si(constraint,
		isl_dim_param, n_param, data->group_size);
	constraint = isl_constraint_set_coefficient_si(constraint,
		isl_dim_out, n_dim - 1, -1);
	constraint = isl_constraint_set_constant_si(constraint,
		data->group_size - 1);
	mapper = isl_map_add_constraint(mapper, constraint);

	constraint = isl_constraint_alloc_inequality(
		isl_local_space_copy(local_space));
	constraint = isl_constraint_set_coefficient_si(constraint,
		isl_dim_param, n_param, data->group_size);
	constraint = isl_constraint_set_coefficient_si(constraint,
		isl_dim_in, n_dim - 1, -1);
	constraint = isl_constraint_set_constant_si(constraint,
		data->group_size - 1);
	mapper = isl_map_add_constraint(mapper, constraint);

	constraint = isl_constraint_alloc_inequality(
		isl_local_space_copy(local_space));
	constraint = isl_constraint_set_coefficient_si(constraint,
		isl_dim_param, n_param, -data->group_size);
	constraint = isl_constraint_set_coefficient_si(constraint,
		isl_dim_out, n_dim - 1, 1);
	mapper = isl_map_add_constraint(mapper, constraint);

	constraint = isl_constraint_alloc_inequality(
		isl_local_space_copy(local_space));
	constraint = isl_constraint_set_coefficient_si(constraint,
		isl_dim_param, n_param, -data->group_size);
	constraint = isl_constraint_set_coefficient_si(constraint,
		isl_dim_in, n_dim - 1, 1);
	mapper = isl_map_add_constraint(mapper, constraint);

	constraint = isl_constraint_alloc_equality(local_space);
	constraint = isl_constraint_set_coefficient_si(constraint,
		isl_dim_out, n_dim - 1, -1);
	constraint = isl_constraint_set_coefficient_si(constraint,
		isl_dim_in, n_dim - 1, 1);
	constraint = isl_constraint_set_constant_si(constraint, data->distance);
	mapper = isl_map_add_constraint(mapper, constraint);

	mapper = isl_map_project_out(mapper, isl_dim_param, n_param, 1);

	partial_identity = isl_map_identity(space);
	partial_identity = isl_map_drop_constraints_involving_dims(
		partial_identity, isl_dim_in, n_dim - 1, 1);
	mapper = isl_map_intersect(mapper, partial_identity);

	mapped = isl_map_apply_range(isl_map_copy(map), mapper);
	mapped = isl_map_intersect_range(mapped, isl_map_range(isl_map_copy(map)));
	mapped = isl_map_union(mapped, map);
	mapped = isl_map_union(mapped, non_full_ranked);
	data->result = isl_union_map_add_map(data->result, mapped);
	if (!data->result)
		return isl_stat_error;
	return isl_stat_ok;
}

static __isl_give isl_union_map *map_array_accesses_to_next_elements_grouped(
	__isl_keep isl_union_map *access)
{
	isl_ctx *ctx = isl_union_map_get_ctx(access);
	int distance = isl_options_get_schedule_spatial_distance(ctx);
	isl_union_map *result = isl_union_map_empty(
		isl_union_map_get_space(access));
	struct next_elements_grouped_data data = { DISTANCE, CACHE_SIZE, result };
	access = union_map_extend_accesses(access);

	if (isl_union_map_foreach_map(access,
			&array_access_to_next_elements_grouped, &data) < 0)
		data.result = isl_union_map_free(data.result);
	isl_union_map_free(access);

	return data.result;
}

/* Add outer dimensions to the access relation to make it have the same
 * dimension as the surronding loop nest.  New accesses are linearly
 * independent from old accesses.  Does nothing for scalar accesses or those
 * with at least as many access dimensions as loop dimensions.
 */
static __isl_give isl_basic_map *basic_map_extend_access(
	__isl_take isl_basic_map *bmap)
{
	isl_mat *eq, *ineq;
	isl_space *space;
	int n_in, n_out, i;
	isl_id *tuple_id;

	space = isl_basic_map_get_space(bmap);
	n_in = isl_basic_map_dim(bmap, isl_dim_in);
	n_out = isl_basic_map_dim(bmap, isl_dim_out);
	if (n_out == 0 || n_out >= n_in)
		return bmap;

	eq = isl_basic_map_equalities_matrix(bmap, isl_dim_in, isl_dim_out,
			isl_dim_param, isl_dim_cst, isl_dim_div);
	ineq = isl_basic_map_inequalities_matrix(bmap, isl_dim_in, isl_dim_out,
			isl_dim_param, isl_dim_cst, isl_dim_div);

	int n_row;
	n_row = isl_mat_rows(eq);
	eq = isl_mat_linear_independent_fullrank(eq, n_in);
	int extra_rows;
	extra_rows = isl_mat_rows(eq) - n_row;
	if (extra_rows == 0) {
		isl_mat_free(eq);
		isl_mat_free(ineq);
		return bmap;
	}
	isl_basic_map_free(bmap);

	// offsets existing n_out by extra_row, and assings newly created lines to
	// new output dims
	eq = isl_mat_insert_zero_cols(eq, n_in, extra_rows);
	for (i = 0; i < extra_rows; ++i)
		eq = isl_mat_set_element_si(eq, n_row + i, n_in + i, -1);

	ineq = isl_mat_insert_zero_cols(ineq, n_in, extra_rows);

	tuple_id = isl_space_get_tuple_id(space, isl_dim_out);
	space = isl_space_add_dims(space, isl_dim_out, extra_rows);
	space = isl_space_set_tuple_id(space, isl_dim_out, tuple_id);

	return isl_basic_map_from_constraint_matrices(space, eq, ineq,
			isl_dim_in, isl_dim_out, isl_dim_param, isl_dim_cst,
			isl_dim_div);
}

static isl_stat basic_map_extend_accesses_callback(
	__isl_take isl_basic_map *bmap, void *user)
{
	isl_map *map = *(isl_map **)user;

	bmap = basic_map_extend_access(bmap);

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
	else if (n_out == 0) // ignore scalars FIXME: should we filter them separately?
	{
		isl_map_free(map);
		return isl_stat_ok;
	}
	else
	{
		extra_dims = n_in - n_out;

		space = isl_map_get_space(map);
		tuple_id = isl_space_get_tuple_id(space, isl_dim_out);
		space = isl_space_insert_dims(space, isl_dim_out, 0, extra_dims);
		space = isl_space_set_tuple_id(space, isl_dim_out, tuple_id);
		result = isl_map_empty(space);
		r = isl_map_foreach_basic_map(map,
			&basic_map_extend_accesses_callback, &result);
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

	isl_union_map *uintersection_notag =
		union_map_drop_last_in_dim(uintersection);
	isl_bool intersection_is_map = isl_union_map_is_equal(
		uintersection_notag, umap);
	isl_union_map_free(uintersection_notag);

	if (intersection_is_map == isl_bool_error)
	{
		isl_union_map_free(umap);
		return isl_stat_error;
	}

	if (intersection_is_map == isl_bool_true)
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

__isl_give isl_union_map *tagged_gist_domain(__isl_take isl_union_map *umap,
	__isl_take isl_union_set *uset)
{
	isl_union_set *umap_domain =
		isl_union_map_domain(isl_union_map_copy(umap));
	isl_union_map *umap_domain_umap = isl_union_set_unwrap(umap_domain);
	umap_domain_umap = isl_union_map_intersect_domain(umap_domain_umap, uset);
	uset = isl_union_map_wrap(umap_domain_umap);
	return isl_union_map_gist_domain(umap, uset);
}


// Removes first out dimension because it will be used to put the expansion
// constant (dimension must be added by the caller).
// Removes last dimension because it is interesting for spatial proximity.
// Pattern is computed for the remaining dimensions, with all tags except the
// output tuple (array name) removed.
// Basic maps have wrapped domains, that is [[A->ptr]->B]
static int access_prefix_pattern_id(__isl_take isl_basic_map *bmap,
		__isl_take __isl_give isl_basic_map_list **patterns)
{
	int n_out, n_in;
	int i, n;
	isl_id *id;

	n_out = isl_basic_map_dim(bmap, isl_dim_out);
	if (n_out < 2)
		return -1;

	n_in = isl_basic_map_dim(bmap, isl_dim_in);
	bmap = isl_basic_map_project_out(bmap, isl_dim_in, n_in, 0); // factor_domain
	id = isl_basic_map_get_tuple_id(bmap, isl_dim_out);
	bmap = isl_basic_map_affine_hull(bmap);
	bmap = isl_basic_map_project_out(bmap, isl_dim_out, n_out - 1, 1);
	bmap = isl_basic_map_project_out(bmap, isl_dim_out, 0, 1);
	bmap = isl_basic_map_set_tuple_id(bmap, isl_dim_out, id);

	bmap = isl_basic_map_set_tuple_name(bmap, isl_dim_in, NULL);
	n_out -= 2;
	for (i = 0; i < n_out; ++i)
		bmap = isl_basic_map_set_dim_name(bmap, isl_dim_out, i, NULL);
	for (i = 0; i < n_in; ++i)
		bmap = isl_basic_map_set_dim_name(bmap, isl_dim_in, i, NULL);

	n = isl_basic_map_list_n_basic_map(*patterns);
	for (i = 0; i < n; ++i) {
		isl_basic_map *other =
				isl_basic_map_list_get_basic_map(*patterns, i);
		isl_bool b = isl_basic_map_is_equal(bmap, other);
		if (b < 0) {
			i = -1;
			break;
		}
		if (b)
			break;
	}
	if (i == n) {
		*patterns = isl_basic_map_list_add(*patterns, bmap);
		return i;
	}

	isl_basic_map_free(bmap);
	return i;
}

static __isl_give isl_basic_map *basic_map_const_complete(
	__isl_take isl_basic_map *bmap, void *user)
{
	isl_basic_map_list **patterns = (isl_basic_map_list **) user;
	int pattern_id;
	isl_id *id;
	isl_local_space *local_space;
	isl_constraint *cstr;

	if (!patterns || !(*patterns))
		return isl_basic_map_free(bmap);

	if (isl_basic_map_dim(bmap, isl_dim_out) < 2)
		return bmap;

	pattern_id = access_prefix_pattern_id(isl_basic_map_copy(bmap),
					      patterns);
	if (pattern_id < 0)
		return isl_basic_map_free(bmap);

	local_space = isl_basic_map_get_local_space(bmap);
	cstr = isl_constraint_alloc_equality(local_space);
	cstr = isl_constraint_set_coefficient_si(cstr, isl_dim_out, 0, -1);
	cstr = isl_constraint_set_constant_si(cstr, pattern_id);

	return isl_basic_map_add_constraint(bmap, cstr);
}

static __isl_give isl_map *map_const_complete(__isl_take isl_map *map, void *user)
{
	isl_id *id;
	if (isl_map_dim(map, isl_dim_out) == 0) {
		isl_space *space = isl_map_get_space(map);
		isl_map_free(map);
		return isl_map_empty(space);
	}

	id = isl_map_get_tuple_id(map, isl_dim_out);
	map = isl_map_insert_dims(map, isl_dim_out, 0, 1);
	map = isl_map_set_tuple_id(map, isl_dim_out, id);
	return map_transform(map, &basic_map_const_complete, user);
}

static isl_stat const_complete_accesses(
	__isl_take __isl_give isl_union_map **reads,
	__isl_take __isl_give isl_union_map **writes)
{
	isl_ctx *ctx;

	if (!reads || !writes)
		return isl_stat_error;

	ctx = isl_union_map_get_ctx(*reads);
	isl_basic_map_list *patterns = isl_basic_map_list_alloc(ctx,
		isl_union_map_n_map(*reads) + isl_union_map_n_map(*writes));
	if (!patterns)
		return isl_stat_error;

	*reads = union_map_transform(*reads, &map_const_complete, &patterns);
	*writes = union_map_transform(*writes, &map_const_complete, &patterns);
	isl_basic_map_list_free(patterns);

	if (!*reads || !*writes) {
		*reads = isl_union_map_free(*reads);
		*writes = isl_union_map_free(*writes);
		return isl_stat_error;
	}
	return isl_stat_ok;
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
	ps->tagged_reads = tagged_gist_domain(
		pet_scop_get_tagged_may_reads(scop),
		isl_union_set_copy(ps->domain));
	ps->tagged_may_writes = tagged_gist_domain(
		pet_scop_get_tagged_may_writes(scop),
		isl_union_set_copy(ps->domain));
	ps->tagged_must_writes = tagged_gist_domain(
		pet_scop_get_tagged_must_writes(scop),
		isl_union_set_copy(ps->domain));
	ps->reads = isl_union_map_gist_domain(
		pet_scop_get_may_reads(scop),
		isl_union_set_copy(ps->domain));
	ps->may_writes = isl_union_map_gist_domain(
		pet_scop_get_may_writes(scop),
		isl_union_set_copy(ps->domain));
	ps->must_writes = isl_union_map_gist_domain(
		pet_scop_get_must_writes(scop),
		isl_union_set_copy(ps->domain));
	ps->tagged_must_kills = pet_scop_get_tagged_must_kills(scop);
	ps->must_kills = pet_scop_get_must_kills(scop);
	ps->schedule = isl_schedule_copy(scop->schedule);
	ps->pet = scop;
	ps->independence = isl_union_map_empty(isl_set_get_space(ps->context));
	for (i = 0; i < scop->n_independence; ++i)
		ps->independence = isl_union_map_union(ps->independence,
			isl_union_map_copy(scop->independences[i]->filter));

	if (options->spatial_model == PPCG_SPATIAL_MODEL_ENDS) {
		compute_retagged_dependences(ps);
	} else if (options->spatial_model == PPCG_SPATIAL_MODEL_GROUPS) {
		compute_retagged_dependences_groups(ps);
	} else if (options->spatial_model == PPCG_SPATIAL_MODEL_ENDS_GROUPS) {
		compute_retagged_dependences_ends_grouped(ps);
	}

	compute_tagger(ps);
	compute_dependences(ps);
	eliminate_dead_code(ps);

	isl_union_map_free(ps->tagged_reads);
	isl_union_map_free(ps->reads);
	isl_union_map_free(ps->tagged_may_writes);
	isl_union_map_free(ps->may_writes);
	isl_union_map_free(ps->tagged_must_writes);
	isl_union_map_free(ps->must_writes);
	ps->tagged_reads = pet_scop_get_tagged_may_reads(scop);
	ps->reads = pet_scop_get_may_reads(scop);
	ps->tagged_may_writes = pet_scop_get_tagged_may_writes(scop);
	ps->may_writes = pet_scop_get_may_writes(scop);
	ps->tagged_must_writes = pet_scop_get_tagged_must_writes(scop);
	ps->must_writes = pet_scop_get_must_writes(scop);

	if (options->remove_nonuniform == PPCG_REMOVE_NONUNIFORM_SPATIAL ||
		options->remove_nonuniform == PPCG_REMOVE_NONUNIFORM_ALL)
		ps->retagged_dep = union_map_filter_uniform(ps->retagged_dep);
	if (options->remove_nonuniform == PPCG_REMOVE_NONUNIFORM_ALL)
		ps->dep_flow_uniform = union_map_filter_uniform(
			isl_union_map_copy(ps->dep_flow));

	if (!ps->context || !ps->domain || !ps->call || !ps->reads ||
	    !ps->may_writes || !ps->must_writes || !ps->tagged_must_kills ||
	    !ps->must_kills || !ps->schedule || !ps->independence ||
            !ps->names)
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
	if (!ps) {
		pet_scop_free(scop);
		return isl_printer_free(p);
	}

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
		r = generate_cuda(ctx, options->ppcg, options->input, options->output);
	else if (options->ppcg->target == PPCG_TARGET_OPENCL)
		r = generate_opencl(ctx, options->ppcg, options->input,
				options->output);
	else
		r = generate_cpu(ctx, options->ppcg, options->input,
				options->output);

	isl_ctx_free(ctx);

	return r;
}
