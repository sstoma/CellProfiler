# -*- coding: utf-8 -*-
__author__ = 'Adam Kaczmarek, Filip Mróz'

import numpy as np
from numpy import linalg

# parameters_settings = {
#     "brightnessWeight": [min, max, scale_down, scale_up]
# }
#
#

parameters_range = {"borderThickness": (0.001, 1.0),
                    "brightnessWeight": (-0.4, 0.4),
                    "cumBrightnessWeight": (0, 500),
                    "gradientWeight": (-30, 30),
                    "sizeWeight": (10, 300),
                    "smoothness": (4, 10)
}

rank_parameters_range = {"avgBorderBrightnessWeight": (0, 600),
                         "avgInnerBrightnessWeight": (-100, 100),
                         "avgInnerDarknessWeight": (-100, 100),
                         "logAreaBonus": (5, 50),
                         "maxInnerBrightnessWeight": (-10, 50),
                         # "maxRank": (5, 300),
                         # "stickingWeight": (0, 120)  # cannot calculate entropy for mutants -- this was 60 so may be important
}


class OptimisationBounds(object):
    def __init__(self, xmax=1, xmin=0):
        self.xmax = xmax
        self.xmin = xmin

    @staticmethod
    def from_ranges(ranges_dict):
        bounds = OptimisationBounds()
        bounds.xmin = []
        bounds.xmax = []
        for k,v in list(sorted(ranges_dict.iteritems())):
            if k == "borderThickness":
                bounds.xmin.append(0.001)
                bounds.xmax.append(2)
            elif k == "smoothness":
                bounds.xmin.append(4.0)
                bounds.xmax.append(10.0)
            else:
                bounds.xmin.append(-1000000)
                bounds.xmax.append(1000000)
            #bounds.xmin, bounds.xmax = zip(*zip(*list(sorted(ranges_dict.iteritems())))[1])
        return bounds

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin

ContourBounds = OptimisationBounds.from_ranges(parameters_range)
RankBounds = OptimisationBounds(xmax = 100, xmin = -100)

#
#
# PARAMETERS ENCODE DECODE
#
#

def pf_parameters_encode(parameters):
    """
    brightnessWeight: 0.0442 +brightness on cell edges
    cumBrightnessWeight: 304.45 -brightness in the cell center
    gradientWeight: 15.482 +gradent on the cell edges
    sizeWeight: 189.4082 (if list -> avg. will be comp.) +big cells
    smoothness: 7.0 +smoothness fact.

    @param parameters: dictionary segmentation.stars
    """
    parameters = parameters["segmentation"]["stars"]
    point = []
    for name, (vmin, vmax) in sorted(parameters_range.iteritems()):
        val = parameters[name]
        if name == "sizeWeight":
            if not isinstance(val, float):
                val = np.mean(val)
        # trim_val = max(vmin, min(vmax, val))
        # point.append((trim_val - vmin) / float(vmax - vmin))
        point.append(val)
    # should be scaled to go from 0-1
    return point


def pf_parameters_decode(param_vector, org_size_weights_list, step, avg_cell_diameter, max_size):
    """
    sizeWeight is one number (mean of the future list)
    @type param_vector: numpy.ndarray
    @return:
    """
    parameters = {}
    for (name, (vmin, vmax)), val in zip(sorted(parameters_range.iteritems()), param_vector):
        # val = min(1, max(0, val))
        # rescaled = vmin + val * (vmax - vmin)
        rescaled = val
        if name == "sizeWeight":
            rescaled = list(np.array(org_size_weights_list) * (rescaled/np.mean(org_size_weights_list)))
        elif name == "borderThickness":
            rescaled = min(max(0.001, val), 3)
        parameters[name] = rescaled
    return parameters


def pf_rank_parameters_encode(parameters, complete_params=True):
    """
    # Set: config.yaml
    # Usage: snake.py - 2 times
    avgBorderBrightnessWeight: 300 # OPT
    # Set: config.yaml
    # Usage: snake.py - 2 times
    avgInnerBrightnessWeight: 10 # OPT
    # Set: config.yaml
    # Usage: snake.py - 2 times - as multiplier - zeroes avg_inner_darkness in calculation of rank
    avgInnerDarknessWeight: 0 # OPT
    # Set: config.yaml
    # Usage: snake.py - 2 times
    logAreaBonus: 18 # OPT
    # Set: config.yaml
    # Usage: snake.py - 2 times
    maxInnerBrightnessWeight: 10 # OPT
    # Set: config.yaml
    # Usage: snake_filter.py - 1 time - actually 0 is meaningfull (!)
    maxRank: 100 # OPT
    # Set: config.yaml
    # Usage: snake.py - 1 time - as ranking weight
    stickingWeight: 60 # OPT
    @param parameters: dictionary all params
    """
    if complete_params:
        parameters = parameters["segmentation"]["ranking"]
    point = []
    for name, (vmin, vmax) in sorted(rank_parameters_range.iteritems()):
        val = parameters[name]
        trim_val = val #max(vmin, min(vmax, val))
        if vmax - vmin == 0:
            point.append(0)
        else:
            point.append((trim_val - vmin) / float(vmax - vmin))
    return point


def pf_rank_parameters_decode(param_vector, final=False):
    """
    @type param_vector: numpy.ndarray
    @return: only ranking parameters as a dict
    """
    parameters = {}
    for (name, (vmin, vmax)), val in zip(sorted(rank_parameters_range.iteritems()), param_vector):
        #val = min(1, max(0, val))
        rescaled = vmin + val * (vmax - vmin)
        parameters[name] = rescaled
    parameters["stickingWeight"] = 0

    if final:
        normalizer = linalg.norm(parameters.values())
        for a in parameters.keys():
            parameters[a] /= normalizer
        parameters["stickingWeight"] = 60 # / 300.8720658352982  # default normalization

    return parameters
