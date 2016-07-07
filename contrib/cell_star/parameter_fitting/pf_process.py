# -*- coding: utf-8 -*-
__author__ = 'Adam Kaczmarek, Filip Mr�z'

import copy
import random
import time
from multiprocessing import Process, Queue

import numpy as np
import scipy.optimize as opt
from scipy.linalg import norm

random.seed(1)
np.random.seed(1)

import logging
logger = logging.getLogger(__name__)

from contrib.cell_star.utils.params_util import *
from contrib.cell_star.core.seed import Seed
from contrib.cell_star.core.image_repo import ImageRepo
from contrib.cell_star.parameter_fitting.pf_snake import PFSnake, GTSnake
from contrib.cell_star.core.seeder import Seeder
from contrib.cell_star.process.segmentation import Segmentation
from contrib.cell_star.utils.debug_util import image_show, image_save, explore_cellstar
from contrib.cell_star.parameter_fitting.pf_auto_params import pf_parameters_encode, pf_parameters_decode, parameters_range, ContourBounds

import cellprofiler.preferences
get_max_workers = cellprofiler.preferences.get_max_workers

min_number_of_chosen_seeds = 6
max_number_of_chosen_snakes = 20

#
#
# PROGRESS CALLBACKS
#
#
ESTIMATED_CALCULATIONS_NUMBER = 3000.0
callback_progress = None


def show_progress(current_distance, calculation):
    if calculation % 100 == 0:
        logger.debug("Current distance: %f, Best: %f, Calc %d"%(current_distance, best_so_far,calculation))
    if callback_progress is not None and calculation % (ESTIMATED_CALCULATIONS_NUMBER / 50) == 0:
        callback_progress(float(calculation) / ESTIMATED_CALCULATIONS_NUMBER)

#
#
# COST FUNCTION AND FITNESS
#
#
best_so_far = 1
calculations = 0
best_3 = []


def keep_3_best(partial_parameters, distance):
    global best_3
    best_3.append((distance, partial_parameters))
    best_3.sort(key=lambda x: x[0])
    best_3 = best_3[:3]
    if best_3[0][0] == best_3[-1][0]:
        best_3 = [best_3[0]]


def distance_norm(fitnesses):
    global calculations, best_so_far
    # Mean-Squared Error
    distance = norm((np.ones(fitnesses.shape) - fitnesses)) / np.sqrt(fitnesses.size)
    best_so_far = min(best_so_far, distance)
    calculations += 1

    show_progress(distance, calculations)
    return distance


def grow_single_seed(seed, images, init_params, pf_param_vector):
    pfsnake = PFSnake(seed, images, init_params)
    return pfsnake.grow(pf_parameters_decode(pf_param_vector, pfsnake.orig_size_weight_list, init_params["segmentation"]["stars"]["step"], init_params["segmentation"]["avgCellDiameter"], init_params["segmentation"]["stars"]["maxSize"]))


def snakes_fitness(gt_snake_seed_pairs, images, parameters, pf_param_vector, debug=False):
    gt_snake_grown_seed_pairs = [(gt_snake, grow_single_seed(seed, images, parameters, pf_param_vector)) for
                                     gt_snake, seed in gt_snake_seed_pairs]

    return np.array([pf_s.multi_fitness(gt_snake) for gt_snake, pf_s in gt_snake_grown_seed_pairs])


#
#
# PREPARE DATA
#
#


def get_gt_snake_seeds(gt_snake, times=3, radius=5, min_radius=0):
    """
    Create random seeds inside gt snake.
    @type gt_snake: GTSnake
    """
    seed = Seed(gt_snake.centroid_x, gt_snake.centroid_y, "optimize_star_parameters")
    seeds = [seed]
    left = times
    while left > 0:
        random_seeds = Seeder.rand_seeds(radius, left, [seed], min_random_radius=min_radius)
        inside_seeds = [s for s in random_seeds if gt_snake.is_inside(s.x, s.y)]
        seeds += inside_seeds
        left = times - (len(seeds) - 1)
        min_radius /= 1.2 # make sure that it finish

    return seeds


def get_size_weight_list(params):
    l = params["segmentation"]["stars"]["sizeWeight"]
    if isinstance(l, float):
        l = [l]
    return l


def prepare_snake_seed_pairs(gt_snakes, initial_parameters):
    radius = initial_parameters["segmentation"]["seeding"]["randomDiskRadius"] * initial_parameters["segmentation"][
        "avgCellDiameter"]
    gt_snake_seed_pairs = [(gt_snake, seed) for gt_snake in gt_snakes for seed in
                           get_gt_snake_seeds(gt_snake, times=3, radius=radius, min_radius=2*radius/3.0)]
    random.shuffle(gt_snake_seed_pairs)
    return gt_snake_seed_pairs


def pf_get_distances(gt_snakes, images, initial_parameters, callback=keep_3_best):
    gt_snake_seed_pairs = prepare_snake_seed_pairs(gt_snakes, initial_parameters)
    pick_seed_pairs = max(min_number_of_chosen_seeds, max_number_of_chosen_snakes /
                          len(initial_parameters["segmentation"]["stars"]["sizeWeight"]))
    chosen_gt_snake_seed_pairs = gt_snake_seed_pairs[:pick_seed_pairs]

    explore_cellstar(image=images.image, images=images, params=initial_parameters,
                                  seeds=[sp[1] for sp in gt_snake_seed_pairs],
                                  snakes=[])

    def create_distance_function(pairs_to_use):
        def distance(partial_parameters, debug=False):
            current_distance = distance_norm(
                snakes_fitness(pairs_to_use, images, initial_parameters, partial_parameters, debug=debug)
            )
            if callback is not None:
                callback(partial_parameters, current_distance)

            return current_distance

        return distance

    return create_distance_function(gt_snake_seed_pairs), create_distance_function(chosen_gt_snake_seed_pairs)


#
#
# VISUALIZATION
#
#

def test_trained_parameters(image, star_params, ranking_params, precision, avg_cell_diameter, output_name=None):
    seg = Segmentation(segmentation_precision=precision, avg_cell_diameter=avg_cell_diameter)
    for k, v in star_params.iteritems():
        seg.parameters["segmentation"]["stars"][k] = v
    for k, v in ranking_params.iteritems():
        seg.parameters["segmentation"]["ranking"][k] = v

    seg.set_frame(image)
    seg.run_segmentation()
    if output_name is None:
        image_show(seg.images.segmentation, 1)
    else:
        image_save(seg.images.segmentation, output_name)

#
#
# OPTIMIZATION
#
#

def run(image, gt_snakes, precision, avg_cell_diameter, method='brute', initial_params=None, background_image=None, ignore_mask=None):
    global best_3, calculations
    """
    :param image: input image
    :param gt_snakes: gt snakes label image
    :param precision: if initial_params is None then it is used to calculate parameters
    :param avg_cell_diameter: if initial_params is None then it is used to calculate parameters
    :param method: optimization engine
    :param initial_params: overrides precision and avg_cell_diameter
    :return:
    """
    logger.info("Parameter fitting started...")
    if initial_params is None:
        params = default_parameters(segmentation_precision=precision, avg_cell_diameter=avg_cell_diameter)
    else:
        params = copy.deepcopy(initial_params)
    images = ImageRepo(image, params)
    images.background = background_image
    if ignore_mask is not None:
        images.apply_mask(ignore_mask)

    start = time.clock()
    best_3 = []
    calculations = 0
    optimized = optimize(method, gt_snakes, images, params, precision, avg_cell_diameter)

    best_arg = optimized[0]
    best_params = pf_parameters_decode(optimized[0], get_size_weight_list(params), params["segmentation"]["stars"]["step"], avg_cell_diameter, params["segmentation"]["stars"]["maxSize"])
    best_score = optimized[1]

    stop = time.clock()
    smoothness = best_params["smoothness"]
    smoothness_bounded = max(parameters_range["smoothness"][0], min(smoothness, parameters_range["smoothness"][1]))
    if smoothness_bounded != smoothness:
        logger.info("Bounding smoothness: %f -> %f" % (smoothness,smoothness_bounded))
        best_params["smoothness"] = float(smoothness_bounded)

    logger.debug("Best: \n" + "\n".join([k + ": " + str(v) for k, v in sorted(best_params.iteritems())]))
    logger.debug("Time: %d" % (stop - start))
    logger.info("Parameter fitting finished with best score %f" % best_score)
    # test_trained_parameters(image, best_params, precision, avg_cell_diameter)
    return PFSnake.merge_parameters(params, best_params), best_arg, best_score


def optimize(method_name, gt_snakes, images, params, precision, avg_cell_diameter):
    encoded_params = pf_parameters_encode(params)
    complete_distance, fast_distance = pf_get_distances(gt_snakes, images, params)
    initial_distance = fast_distance(encoded_params)
    initial_complete_distance = complete_distance(encoded_params)
    logger.debug("Initial parameters complete-distance is (%f)." % (initial_complete_distance))
    logger.debug("Initial parameters distance is (%f)." % (initial_distance))
    logger.debug("Initial parameters are %s." % params)
    if method_name == "mp" and getattr(sys, "frozen", False) and sys.platform == 'win32':
        # multiprocessing do not work then
        method_name = "brutemaxbasin"
    if method_name == "mp":
        best_params_encoded, distance = multiproc_multitype_fitness(images.image, gt_snakes, precision, avg_cell_diameter, params)
        # test_trained_parameters(images.image, params, precision, avg_cell_diameter)
        # return fitted_params, score
    else:
        if method_name == 'brute':
            best_params_encoded, distance = optimize_brute(encoded_params, fast_distance)
        elif method_name == 'brutemaxbasin':
            best_params_encoded, distance = optimize_brute(encoded_params, fast_distance)
            logger.debug("Best grid parameters distance is (%f)." % distance)
            best_params_encoded, distance = optimize_basinhopping(best_params_encoded, fast_distance, time_percent=100)
        elif method_name == 'brutemax3basin':
            _, _ = optimize_brute(encoded_params, fast_distance)
            logger.debug("Best grid parameters distance are %s." %  str(zip(*best_3)[0]))
            logger.debug("Best grid parameters parameters are %s." %  str(zip(*best_3)[1]))

            best_basins = []
            for candidate in list(best_3):
                best_basins.append(optimize_basinhopping(candidate[1], fast_distance, time_percent = 33))
            best_basins.sort(key=lambda x: x[1])

            best_params_encoded, distance = best_basins[0]
        elif method_name == 'basin':
            best_params_encoded, distance = optimize_basinhopping(encoded_params, fast_distance)
        elif method_name == 'diffevo':
            best_params_encoded, distance = optimize_de(encoded_params, fast_distance)

    complete_distance = complete_distance(best_params_encoded)
    logger.debug("Final parameters complete-distance is (%f)." % (complete_distance))
    if initial_complete_distance <= complete_distance:
        logger.debug("Initial parameters (%f) are not worse than the best found (%f)." % (initial_complete_distance, complete_distance))
        return encoded_params, initial_complete_distance
    else:
        return best_params_encoded, complete_distance


def optimize_brute(params_to_optimize, distance_function):
    broadness = 0.1
    search_range = 10 * broadness

    lower_bound = params_to_optimize - np.maximum(np.abs(params_to_optimize), 0.1)
    #lower_bound[params_to_optimize == 0] = -100 * search_range

    upper_bound = params_to_optimize + np.maximum(np.abs(params_to_optimize), 0.1)
    #upper_bound[params_to_optimize == 0] = 100 * search_range

    # introduce random shift (0,grid step) # max 20%
    number_of_steps = 3
    step = (upper_bound - lower_bound) / float(number_of_steps)
    random_shift = np.array([random.random() * 2 / 10 for _ in range(len(lower_bound))])
    lower_bound += random_shift * step
    upper_bound += random_shift * step

    print lower_bound, upper_bound

    logger.debug("Search range: " + str(zip(lower_bound,upper_bound)))
    result = opt.brute(distance_function, zip(lower_bound, upper_bound), Ns=number_of_steps, disp=True, finish=None, full_output=True)
    logger.debug("Opt finished:" + str(result[:2]))
    return result[0], result[1]


def optimize_de(params_to_optimize, distance_function):
    broadness = 0.1
    search_range = 10 * broadness

    lower_bound = params_to_optimize - np.maximum(np.abs(params_to_optimize), 0.1)
    lower_bound[params_to_optimize == 0] = -100 * search_range

    upper_bound = params_to_optimize + np.maximum(np.abs(params_to_optimize), 0.1)
    upper_bound[params_to_optimize == 0] = 100 * search_range

    bounds = zip(lower_bound, upper_bound)
    result = opt.differential_evolution(distance_function, bounds, maxiter=10, popsize=30, init='latinhypercube')
    logger.debug("Opt finished: " + str(result))
    # fitness(result.x, debug=True)
    return result.x, result.fun


def optimize_basinhopping(params_to_optimize, distance_function, time_percent = 100):
    minimizer_kwargs = {"method": "COBYLA"}
    bounds = ContourBounds
    result = opt.basinhopping(distance_function, params_to_optimize, accept_test=bounds,  minimizer_kwargs=minimizer_kwargs, niter=35*time_percent/100)
    logger.debug("Opt finished: " + str(result))
    return result.x, result.fun



#
#
#   MULTIPROCESSING - MULTIPLE METHODS
#
#

def general_multiproc_fitting(run_wrapper, *args):
    result_queue = Queue()
    update_queue = Queue()
    workers_num = get_max_workers()

    optimizers = [
        Process(target=run_wrapper, args=(result_queue, update_queue) + args)
        for _ in range(workers_num)]

    for optimizer in optimizers:
        optimizer.start()

    optimizers_left = workers_num
    results = []
    while optimizers_left > 0:
        time.sleep(0.1)
        if not update_queue.empty() and callback_progress is not None:
            callback_progress(update_queue.get())

        if not result_queue.empty():
            results.append(result_queue.get())
            optimizers_left -= 1


    for optimizer in optimizers:
        optimizer.join()

    sorted_results = sorted(results, key=lambda x: x[2])
    logger.debug(str(sorted_results[0]))
    return sorted_results[0][1], sorted_results[0][2]


def run_wrapper(queue, update_queue, *args):
    global callback_progress
    random.seed()  # reseed with random
    callback_progress = lambda p: update_queue.put(p)
    result = run(*args)
    queue.put(result)


def multiproc_multitype_fitness(image, gt_snakes, precision, avg_cell_diameter, init_params=None):
    return general_multiproc_fitting(run_wrapper, image, gt_snakes, precision, avg_cell_diameter, "brutemaxbasin", init_params)
