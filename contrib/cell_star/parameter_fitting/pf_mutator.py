# -*- coding: utf-8 -*-
__author__ = 'Adam Kaczmarek, Filip Mróz'

import numpy as np
import copy.copy as copy
import random

from contrib.cell_star.core.point import *
from contrib.cell_star.utils.calc_util import polar_to_cartesian
from contrib.cell_star.parameter_fitting.pf_rank_snake import PFRankSnake

def add_mutations(gt_and_grown, avg_cell_diameter):
    mutants = []
    for (gt, grown) in gt_and_grown:
        mutants += [#(gt, grown.create_mutation(3, rand_range=(-20, 20))), (gt, grown.create_mutation(-3, rand_range=(-20, 20))),
                    (gt, grown.create_mutation(10)), (gt, grown.create_mutation(-10)),
                    #(gt, grown.create_mutation(3, rand_range=(-5, 5))),
                    #(gt, grown.create_mutation(-3, rand_range=(-5, 5)))
                    ]
    return gt_and_grown + mutants

def create_mutation(pf_rank_snake, dilation, rand_range=[0, 0]):
    mutant_snake = copy.copy(pf_rank_snake.grown_snake)
    # zero rank so it recalculates
    mutant_snake.rank = None

    # change to pixels
    dilation /= pf_rank_snake.polar_transform.step
    boundary_change = np.array([dilation + random.randrange(rand_range[0], rand_range[1] + 1)
                                for _ in range(mutant_snake.polar_coordinate_boundary.size)])

    new_boundary = np.array(1)
    while (new_boundary <= 3).all() and abs(boundary_change.max()) > 3:
        new_boundary = np.maximum(np.minimum(
            mutant_snake.polar_coordinate_boundary + boundary_change,
            len(pf_rank_snake.polar_transform.R) - 1), 3)
        boundary_change /= 1.3

    px, py = polar_to_cartesian(new_boundary, mutant_snake.seed.x, mutant_snake.seed.y, pf_rank_snake.polar_transform)

    mutant_snake.polar_coordinate_boundary = new_boundary
    mutant_snake.points = [Point(x, y) for x, y in zip(px, py)]

    # TODO need to update self.final_edgepoints to calculate properties (for now we ignore this property)
    mutant_snake.calculate_properties_vec(pf_rank_snake.polar_transform)

    return PFRankSnake(pf_rank_snake.gt_snake,mutant_snake,pf_rank_snake.avg_cell_diameter,pf_rank_snake.initial_parameters)