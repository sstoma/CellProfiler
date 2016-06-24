__author__ = 'Adam'

import logging
import sys

import numpy as np

import contrib.cell_star.utils.debug_util as debug_util
import contrib.cell_star.parameter_fitting.pf_rank_process as pf_rank
import contrib.cell_star.parameter_fitting.test_pf as test_pf
from cellprofiler.preferences import get_max_workers
from contrib.cell_star.parameter_fitting.test_pf import try_load_image, image_to_label, gt_label_to_snakes
from contrib.cell_star.utils.params_util import default_parameters


def run_rank_pf(input_image, background_image, ignore_mask_image, gt_mask, parameters, callback_progress = None):
    """
    :param input_image:
    :param gt_mask:
    :param parameters:
    :return: Best complete parameters settings, best distance
    """

    gt_mask = image_to_label(gt_mask)
    pf_rank.callback_progress = callback_progress

    gt_snakes = gt_label_to_snakes(gt_mask)
    if get_max_workers() > 1 and not(getattr(sys, "frozen", False) and sys.platform == 'win32'):
        # multiprocessing do not work if frozen on win32
        best_complete_params, _, best_score = pf_rank.run_multiprocess(input_image, gt_snakes,
                                                                       initial_params=parameters,
                                                                       method='brutemaxbasin',
                                                                       background_image=background_image,
                                                                       ignore_mask=ignore_mask_image)
    else:
        best_complete_params, _, best_score = pf_rank.run_singleprocess(input_image, gt_snakes,
                                                                        initial_params=parameters,
                                                                        method='brutemaxbasin',
                                                                        background_image=background_image,
                                                                        ignore_mask=ignore_mask_image)

    return best_complete_params, best_score


def test_rank_pf(image_path, mask_path, precision, avg_cell_diameter, method, initial_params=None, options=None):
    frame = try_load_image(image_path)

    if options == 'invert':
        frame = 1 - frame

    gt_image = np.array(try_load_image(mask_path) * 255, dtype=int)

    gt_mask = image_to_label(gt_image)

    gt_snakes = gt_label_to_snakes(gt_mask)
    if method == "mp":
        return pf_rank.run_multiprocess(frame, gt_snakes, precision, avg_cell_diameter, 'brutemaxbasin', initial_params=initial_params)
    else:
        return pf_rank.run_singleprocess(frame, gt_snakes, precision, avg_cell_diameter, method, initial_params=initial_params)


if __name__ == "__main__":
    if len(sys.argv) < 7:
        print "Usage: <script> base_path image_path mask_path precision avg_cell_diameter method {image_result_path}"
        print "Given: " + " ".join(sys.argv)
        sys.exit(-1)

    pf_rank.get_max_workers = lambda: 2
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger = logging.getLogger('contrib.cell_star.parameter_fitting')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    test_pf.corpus_path = sys.argv[1]

    precision = int(sys.argv[4])
    avg_cell_diameter = float(sys.argv[5])

    image_result_path = None
    if len(sys.argv) >= 8:
        image_result_path = sys.argv[7]

    #complete_params = default_parameters(segmentation_precision=precision, avg_cell_diameter=avg_cell_diameter)
    complete_params, _, _ = test_rank_pf(sys.argv[2], sys.argv[3], precision, avg_cell_diameter, sys.argv[6])

    print "Best_params:", complete_params
    print

    debug_util.DEBUGING = True
    if image_result_path is not None:
        test_pf.test_parameters(sys.argv[2], sys.argv[3], precision, avg_cell_diameter, complete_params, image_result_path)