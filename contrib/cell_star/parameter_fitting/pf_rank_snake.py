# -*- coding: utf-8 -*-
__author__ = 'Adam Kaczmarek, Filip Mróz'

import copy
import random
random.seed(1)  # make it deterministic
import numpy as np

from contrib.cell_star.core.polar_transform import PolarTransform
from contrib.cell_star.utils.calc_util import polar_to_cartesian
from contrib.cell_star.core.point import Point
from contrib.cell_star.parameter_fitting.pf_snake import PFSnake
import pf_mutator


class PFRankSnake(object):
    def __init__(self, gt_snake, grown_snake, avg_cell_diameter, params):
        self.gt_snake = gt_snake
        self.grown_snake = grown_snake
        self.avg_cell_diameter = avg_cell_diameter
        self.initial_parameters = params
        self.fitness = PFSnake.fitness_with_gt(grown_snake, gt_snake)
        self.rank_vector = grown_snake.properties_vector(avg_cell_diameter)
        self.polar_transform = PolarTransform.instance(params["segmentation"]["avgCellDiameter"],
                                                           params["segmentation"]["stars"]["points"],
                                                           params["segmentation"]["stars"]["step"],
                                                           params["segmentation"]["stars"]["maxSize"])

    @staticmethod
    def create_all(gt_snake, grown_pf_snake, params):
        return [(gt_snake, PFRankSnake(gt_snake, snake, grown_pf_snake.avg_cell_diameter, params)) for snake in grown_pf_snake.snakes]

    def create_mutation(self, dilation, rand_range=[0, 0]):
        return pf_mutator.create_mutation(self, dilation, rand_range)

    @staticmethod
    def merge_rank_parameters(initial_parameters, new_params):
        params = copy.deepcopy(initial_parameters)
        for k, v in new_params.iteritems():
            params["segmentation"]["ranking"][k] = v

        return params

    def merge_parameters_with_me(self, new_params):
        return PFRankSnake.merge_rank_parameters(self.initial_parameters, new_params)


    def calculate_ranking(self, ranking_params):
        return self.grown_snake.star_rank(ranking_params, self.avg_cell_diameter)