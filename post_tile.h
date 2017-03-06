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

 #include <isl/schedule_node.h>
 #include <isl/val.h>

 #include "ppcg.h"

__isl_give isl_schedule_node *tile_sink_spatially_local_loops(
	__isl_take isl_schedule_node *node, struct ppcg_scop *scop,
	__isl_take isl_multi_val *sizes,
	__isl_give isl_schedule_node *(*tile)(__isl_take isl_schedule_node *,
		__isl_take isl_multi_val *));