__author__ = 'Adam Kaczmarek, Filip Mroz'

import logging
import os.path as path
import sys

import numpy as np
import scipy as sp

from cellprofiler.preferences import get_max_workers
import contrib.cell_star.parameter_fitting.pf_process as pf_process
from contrib.cell_star.parameter_fitting.pf_process import run, test_trained_parameters
from contrib.cell_star.parameter_fitting.pf_snake import GTSnake
from contrib.cell_star.utils import image_util, debug_util

logger = logging.getLogger(__name__)

global corpus_path
corpus_path = "../cell_star_plugins/yeast_corpus/data/"


def single_mask_to_snake(bool_mask, seed=None):
    return GTSnake(bool_mask, seed)


def gt_label_to_snakes(components):
    num_components = components.max()
    return [single_mask_to_snake(components == label) for label in range(1, num_components + 1)]


def image_to_label(image):
    values = np.unique(image)
    if len(values) == 2:  # it is a mask
        components, num_components = sp.ndimage.label(image, np.ones((3, 3)))
        return components
    else:  # remap labels to [1..] values
        curr = 1
        label_image = image.copy()
        for v in values[1:]:  # zero is ignored
            label_image[image == v] = curr
            curr += 1
        return label_image


def load_from_testset(filepath):
    """
    @param filepath: TestSetX/frame/BF_frame001.tif
    @return: loaded image
    """
    return image_util.load_image(path.join(corpus_path, filepath))


def try_load_image(image_path):
    return image_util.load_frame(corpus_path, image_path)


def run_pf(input_image, background_image, ignore_mask_image, gt_label, parameters, precision, avg_cell_diameter, callback_progress = None):
    """
    :param input_image:
    :param gt_label:
    :param parameters:
    :return: Best complete parameters settings, best distance
    """

    gt_mask = image_to_label(gt_label)
    pf_process.callback_progress = callback_progress

    gt_snakes = gt_label_to_snakes(gt_mask)
    if get_max_workers() > 1:
        best_complete_params, _, best_score = run(input_image, gt_snakes, precision=precision,
                                                  avg_cell_diameter=avg_cell_diameter, initial_params=parameters,
                                                  method='mp', background_image=background_image,
                                                  ignore_mask=ignore_mask_image)
    else:
        best_complete_params, _, best_score = run(input_image, gt_snakes, precision=precision,
                                                  avg_cell_diameter=avg_cell_diameter, initial_params=parameters,
                                                  method='brutemaxbasin', background_image=background_image,
                                                  ignore_mask=ignore_mask_image)

    return best_complete_params, best_score


def test_pf(image_path, mask_path, precision, avg_cell_diameter, method, initial_params=None, options=None):
    frame = try_load_image(image_path)

    if options == 'invert':
        frame = 1 - frame

    gt_image = np.array(try_load_image(mask_path) * 255, dtype=int)

    gt_mask = image_to_label(gt_image)

    gt_snakes = gt_label_to_snakes(gt_mask)
    return run(frame, gt_snakes, precision, avg_cell_diameter, method, initial_params=initial_params)


def test_parameters(image_path, mask_path, precision, avg_cell_diameter, params, output_path=None, options=None):
    frame = try_load_image(image_path)

    if options == 'invert':
        frame = 1 - frame
    #gt_image = np.array(try_load_image(mask_path) * 255, dtype=int)

    #cropped_image, cropped_gt_label = cropped_to_gt(avg_cell_diameter, frame, gt_image)
    #gt_snakes = gt_label_to_snakes(cropped_gt_label)

    output_name = None
    if output_path is not None:
        debug_util.debug_image_path = output_path
        output_name = "trained"

    test_trained_parameters(frame, params["segmentation"]["stars"], params["segmentation"]["ranking"], precision, avg_cell_diameter, output_name)


if __name__ == "__main__":
    if len(sys.argv) < 7:
        print "Usage: <script> base_path image_path mask_path precision avg_cell_diameter method {image_result_path}"
        print "Given: " + " ".join(sys.argv)
        sys.exit(-1)

    pf_process.get_max_workers = lambda: 2
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger = logging.getLogger('contrib.cell_star.parameter_fitting')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    corpus_path = sys.argv[1]
    precision = int(sys.argv[4])
    avg_cell_diameter = float(sys.argv[5])

    image_result_path = None
    if len(sys.argv) >= 8:
        image_result_path = sys.argv[7]

    complete_params, _, _ = test_pf(sys.argv[2], sys.argv[3], precision, avg_cell_diameter, sys.argv[6])

    print "Best_params:", complete_params
    print

    debug_util.DEBUGING = True
    if image_result_path is not None:
        test_parameters(sys.argv[2], sys.argv[3], precision, avg_cell_diameter, complete_params, image_result_path)