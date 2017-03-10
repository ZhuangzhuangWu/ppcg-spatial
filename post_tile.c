/*
 * Copyright 2017 Inria
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Oleksandr Zinenko, Inria,
 * 2 rue Simone IFF,
 * CS 42112
 * 75589 Paris cedex 12
 * France
 */

 #include <isl/ctx.h>
 #include <isl/map.h>
 #include <isl/constraint.h>
 #include <isl/schedule.h>
 #include <isl/schedule_node.h>

 #include "ppcg.h"
 #include "ppcg_options.h"
 #include "schedule.h"
 #include "util.h"

#define PPCG_PLUTOSTYLE_MAX_STRIDE 4

#if 0
static isl_bool access_has_spatial_locality_dim(__isl_take isl_map *map, int dim)
{
	// access map has spatial locality if it is last output dimension
	// has [1..cst] parameter in front of dim-s input dimension

	// we check this by checking whether the given map is a subset of the
	// map only counstrained by  1*(dim-s in) <= out <= cst*(dim-s in)

	isl_space *space = isl_map_get_space(map);
	isl_local_space *local_space;
	int n_out = isl_space_dim(space, isl_dim_out);
	isl_map *spatial_map;
	isl_constraint *cstr;

	if (dim < 0 || dim >= isl_space_dim(space, isl_dim_in)) {
		isl_map_free(map);
		return isl_bool_error;
	}

	local_space = isl_local_space_from_space(isl_space_copy(space));
	spatial_map = isl_map_universe(space);

	cstr = isl_constraint_alloc_inequality(isl_local_space_copy(local_space));
	cstr = isl_constraint_set_coefficient_si(cstr, isl_dim_out, n_out - 1, 1);
	cstr = isl_constraint_set_coefficient_si(cstr, isl_dim_in, dim, -1);
	spatial_map = isl_map_add_constraint(map, cstr);;

	cstr = isl_constraint_alloc_inequality(local_space);
	cstr = isl_constraint_set_coefficient_si(cstr, isl_dim_out, n_out - 1, -1);
	cstr = isl_constraint_set_coefficient_si(cstr, isl_dim_in, dim,
		PPCG_PLUTOSTYLE_MAX_STRIDE);
	spatial_map = isl_map_add_constraint(map, cstr);

	isl_map_is_subset(map, spatial_map); // FIXME: does not work as I expect..
}
#endif

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

static int is_invariant_up_to(__isl_keep isl_basic_map *bmap, int dim, int n_out)
{
	int i;
	isl_constraint *constraint;
	isl_val *coef;

	for (i = 0; i < n_out; ++i) {
		if (!isl_basic_map_has_defining_equality(bmap, isl_dim_out, i,
		    &constraint))
			return 0;
		coef = isl_constraint_get_coefficient_val(constraint, isl_dim_in, dim);
		if (isl_val_is_zero(coef) != isl_bool_true) {
			isl_val_free(coef);
			return 0;
		}
		isl_val_free(coef);
	}
	return 1;
}

inline static int is_invariant(__isl_keep isl_basic_map *bmap, int dim)
{
	int n_out = isl_basic_map_n_out(bmap);
	return is_invariant_up_to(bmap, dim, n_out);
}

inline static int has_spatial_locality(__isl_keep isl_basic_map *bmap, int dim)
{
	int n_out = isl_basic_map_n_out(bmap);
	isl_constraint *constraint;
	isl_val *coef, *d, *limit;
	isl_bool in_limit, positive;
	isl_ctx *ctx = isl_basic_map_get_ctx(bmap);

	if (n_out == 0)
		return 0;

	if (!is_invariant_up_to(bmap, dim, n_out - 1))
		return 0;

	if (!isl_basic_map_has_defining_equality(bmap, isl_dim_out, n_out - 1,
	    &constraint))
		return 0;

	coef = isl_constraint_get_coefficient_val(constraint, isl_dim_in, dim);
	d = isl_constraint_get_coefficient_val(constraint, isl_dim_out, n_out - 1);
	d = isl_val_neg(d);
	coef = isl_val_div(coef, d);

	limit = isl_val_int_from_si(ctx, PPCG_PLUTOSTYLE_MAX_STRIDE);
	positive = isl_val_gt(isl_val_copy(coef), isl_val_zero(ctx));
	in_limit = isl_val_le(coef, limit);

	return (in_limit == isl_bool_true) && (positive == isl_bool_true);
}

struct spatial_locality_dim_properties {
	int n_member;
	int member;
	int n_temporal_locality;
	int n_spatial_locality;
	int n_access;
};

static isl_stat basic_map_compute_spatial_locality_weight(
	__isl_take isl_basic_map *bmap, void *user)
{
	struct spatial_locality_dim_properties *data = user;
	int n_out = isl_basic_map_n_out(bmap);
	int n_in = isl_basic_map_n_in(bmap);
	isl_constraint *constraint;
	int weight = -16;
	int i;
	isl_val *out_coef, *in_coef, *limit;
	isl_ctx *ctx = isl_basic_map_get_ctx(bmap);
	isl_bool positive, in_limit;

	int dim = n_in - data->n_member + data->member;
	if (is_invariant(bmap, dim)) {
		data->n_temporal_locality += 1;
	} else if (has_spatial_locality(bmap, dim)) {
		data->n_spatial_locality += 1;
	}

#if 0
	// scalars have temporal locality
	if (n_out == 0) {
		isl_basic_map_free(bmap);
		data->n_temporal_locality += 1;
		return isl_stat_ok;
	}

	// if the access function is not defined, no locality
	if (!isl_basic_map_has_defining_equality(bmap, isl_dim_out,
	    n_out - 1, &constraint)) {
		isl_basic_map_free(bmap);
		return isl_stat_ok;
	}
	isl_basic_map_free(bmap);

	// temporal locality (independent of input dims)
	for (i = 0; i < n_in; ++i) {
		isl_bool r;
		isl_val *coef;

		coef = isl_constraint_get_coefficient_val(constraint, isl_dim_in, i);
		if ((r = isl_val_is_zero(coef)) < 0) {
			isl_constraint_free(constraint);
			return isl_stat_error;
		}
		isl_val_free(coef);
		if (!r)
			break;
	}
	if (i == n_in) {
		data->n_temporal_locality += 1;
		isl_constraint_free(constraint);
		return isl_stat_ok;
	}

	// spatial
	out_coef = isl_constraint_get_coefficient_val(constraint,
		isl_dim_out, n_out - 1);
	in_coef = isl_constraint_get_coefficient_val(constraint,
		isl_dim_in, n_in - data->n_member + data->member);
	isl_constraint_free(constraint);

	out_coef = isl_val_neg(out_coef);
	in_coef = isl_val_div(in_coef, out_coef);
	limit = isl_val_int_from_si(ctx, PPCG_PLUTOSTYLE_MAX_STRIDE);
	positive = isl_val_gt(isl_val_copy(in_coef), isl_val_zero(ctx));
	in_limit = isl_val_le(in_coef, limit);

	if (positive < 0 || in_limit < 0)
		return isl_stat_error;
	if (positive && in_limit) {
		data->n_spatial_locality += 1;
	}
#endif

	return isl_stat_ok;
}

static isl_stat map_compute_spatial_locality_weight(__isl_take isl_map *map,
	void *user)
{
	struct spatial_locality_dim_properties *data = user;
	data->n_access += isl_map_n_basic_map(map);
	isl_stat r = isl_map_foreach_basic_map(map,
	    &basic_map_compute_spatial_locality_weight, data);
	isl_map_free(map);
	return r;
}

/* Pluto-style heuristic:
 * for each dimension in a band, using counted accesses, copmute weight
 * * (+2x) each access with spatial locality
 * * (+4x) each access with temporal locality
 * * (+8x) each acess with both spatial and temporal locality
 * * (-16x) each access without spatial nor temporal locality
 * select the dimension with maximum weight and put it last.
 *
 * Pluto also multiplies weight by the number of stmts in each loop of the band,
 * but it remains constant so
 */
static int compute_spatial_locality_weight(__isl_keep isl_union_map *accesses,
	int n_member, int member)
{
	int vectorizable, non_local;
	struct spatial_locality_dim_properties data = {
		n_member, member, 0, 0, 0 };

	isl_stat r;
	if ((r = isl_union_map_foreach_map(accesses,
	    &map_compute_spatial_locality_weight, &data)) < 0) {
		return -100500;
	}

	vectorizable = (data.n_access > 0) &&
		(data.n_spatial_locality + data.n_temporal_locality == data.n_access);
	non_local = data.n_access - data.n_temporal_locality - data.n_spatial_locality;

	return 2 * data.n_spatial_locality + 4 * data.n_temporal_locality +
		8 * vectorizable - 16 * non_local;
}

__isl_give isl_schedule_node *compute_wavefront(
	__isl_take isl_schedule_node *node, struct ppcg_scop *scop)
{
	// FIXME: Using coincidence flag for now, do we need a proper check?
	isl_bool coincident = isl_schedule_node_band_member_get_coincident(node, 0);

	if (coincident < 0)
		return isl_schedule_node_free(node);
	if (coincident)
		return node;

	isl_union_map *partial_schedule =
		isl_schedule_node_band_get_partial_schedule_union_map(node);
	isl_union_set *partial_schedule_uset =
		isl_union_map_range(partial_schedule);
	isl_set *partial_schedule_set =
		isl_set_from_union_set(partial_schedule_uset);
	isl_space *space = isl_set_get_space(partial_schedule_set);
	isl_set_free(partial_schedule_set);

	int n = isl_space_dim(space, isl_dim_set);
	if (n <= 1) {
		isl_space_free(space);
		return node;
	}

	space = isl_space_map_from_domain_and_range(space, isl_space_copy(space));
	isl_basic_map *wavefront_bmap = isl_basic_map_universe(space);

	isl_local_space *ls = isl_basic_map_get_local_space(wavefront_bmap);
	isl_constraint *c = isl_constraint_alloc_equality(ls);
	c = isl_constraint_set_coefficient_si(c, isl_dim_out, 0, -1);
	c = isl_constraint_set_coefficient_si(c, isl_dim_in, 0, 1);
	c = isl_constraint_set_coefficient_si(c, isl_dim_in, 1, 1);
	wavefront_bmap = isl_basic_map_add_constraint(wavefront_bmap, c);

	int i;
	for (i = 1; i < n; ++i) {
		isl_local_space *ls = isl_basic_map_get_local_space(wavefront_bmap);
		isl_constraint *c = isl_constraint_alloc_equality(ls);
		c = isl_constraint_set_coefficient_si(c, isl_dim_out, i, -1);
		c = isl_constraint_set_coefficient_si(c, isl_dim_in, i, 1);
		wavefront_bmap = isl_basic_map_add_constraint(wavefront_bmap, c);
	}

	isl_union_map *wavefront_umap =
		isl_union_map_from_basic_map(wavefront_bmap);
	partial_schedule =
		isl_schedule_node_band_get_partial_schedule_union_map(node);
	partial_schedule = isl_union_map_apply_range(partial_schedule,
		wavefront_umap);

	return isl_schedule_node_band_set_partial_schedule(node,
		isl_multi_union_pw_aff_from_union_map(partial_schedule));
}

__isl_give isl_schedule_node *tile_sink_spatially_local_loops(
	__isl_take isl_schedule_node *node, struct ppcg_scop *scop,
	__isl_take isl_multi_val *sizes,
	__isl_give isl_schedule_node *(*tile)(__isl_take isl_schedule_node *,
		__isl_take isl_multi_val *))
{

	isl_union_set *band_domain, *access_set;
	isl_union_map *access_map, *counted_accesses, *schedule;
	int n_member, i;
	int *order;
	int weight, max_weight, max_weight_member;
	isl_ctx *ctx = isl_schedule_node_get_ctx(node);

	if (isl_schedule_node_get_type(node) != isl_schedule_node_band)
		return node;

	band_domain = isl_schedule_node_get_domain(node);
	// we need "scheduled" accesses
	schedule = schedule_node_band_get_ascendant_schedule(node);
	schedule = isl_union_map_intersect_domain(schedule, band_domain);

	if (scop->options->posttile_reorder == PPCG_POSTTILE_REORDER_SPATIAL) {
		band_domain = isl_schedule_node_get_domain(node);
		access_map = isl_union_map_copy(scop->counted_accesses);
		access_map = isl_union_set_unwrap(isl_union_map_domain(access_map));
		access_map = isl_union_map_universe(access_map);
		access_map = isl_union_map_intersect_domain(access_map, band_domain);
		access_set = isl_union_map_wrap(access_map);

		counted_accesses = isl_union_map_copy(scop->counted_accesses);
		counted_accesses = isl_union_map_intersect_domain(counted_accesses,
			access_set);
		counted_accesses = isl_union_map_curry(counted_accesses);
		counted_accesses = isl_union_map_apply_domain(counted_accesses, schedule);
		counted_accesses = isl_union_map_uncurry(counted_accesses);
	} else if (scop->options->posttile_reorder == PPCG_POSTTILE_REORDER_PLUTO) {
		isl_union_map *accesses = isl_union_map_copy(scop->reads);
		accesses = isl_union_map_union(accesses,
			isl_union_map_copy(scop->must_writes));
		accesses = isl_union_map_union(accesses,
			isl_union_map_copy(scop->may_writes));
		accesses = isl_union_map_apply_domain(accesses, schedule);
		counted_accesses = accesses;
	}

	n_member = isl_schedule_node_band_n_member(node);
	max_weight = -100500;
	for (i = 0; i < n_member; ++i) {
		weight = compute_spatial_locality_weight(counted_accesses, n_member, i);
		if (weight > max_weight) {
			max_weight = weight;
			max_weight_member = i;
		}
	}
	isl_union_map_free(counted_accesses);

	node = tile(node, sizes);

	node = compute_wavefront(node, scop);

	if (max_weight_member == n_member - 1)
		return node;

	order = isl_calloc_array(ctx, int, n_member);
	for (i = 0; i < n_member; ++i) {
		if (i < max_weight_member)
			order[i] = i;
		else if (i == max_weight_member)
			order[i] = n_member - 1;
		else
			order[i] = i - 1;
	}

	node = isl_schedule_node_first_child(node);
	node = isl_schedule_node_band_permute(node, order);
	node = isl_schedule_node_parent(node);
	free(order);

	return node;
}