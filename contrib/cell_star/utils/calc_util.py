# -*- coding: utf-8 -*-
"""
Calculation package contains a number of functions used in contour grow and evaluation.
Date: 2013-2016
Website: http://cellstar-algorithm.org/
"""

# External imports
import math

import numpy as np
import scipy.ndimage as sp_image
from matplotlib.path import Path

from index import Index


def euclidean_norm((x1, y1), (x2, y2)):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def interpolate_radiuses(values_mask, length, values):
    """
    Fill values with linear interpolation using values_mask values.
    @type values_mask: np.ndarray
    @param values_mask: mask of existing values
    @type values: np.ndarray
    @type length: int
    """
    cumlengths = np.where(values_mask)[0]
    if len(cumlengths) > 0:
        cumlengths_loop = np.append(cumlengths, cumlengths[0] + int(length))
        for i in range(len(cumlengths)):
            # Find left and right boundary in existing values.
            left_interval_boundary = cumlengths_loop[i]
            right_interval_boundary = cumlengths_loop[i + 1] % length

            # Length of the interpolated interval.
            interval_length = cumlengths_loop[i + 1] - left_interval_boundary - 1

            # Interpolate for every point in the interval.
            for k in range(1, interval_length + 1):
                interpolated = (left_interval_boundary + k) % length

                new_val = round(values[left_interval_boundary] +
                                (values[right_interval_boundary] - values[left_interval_boundary]) *
                                k / (interval_length + 1)) # TODO? dzielenie całkowitoliczbowe

                # TODO? Zwróć minimum jako wynik interpolacji - interpolacja nie może oddalić konturu od środka komórki
                values[interpolated] = min(values[interpolated], new_val)


def loop_connected_components(mask):
    """
    @type mask: np.ndarray
    @rtype (np.ndarray, np.ndarray, np.ndarray)
    """

    c = np.array([])
    init = np.array([])
    fin = np.array([])

    if mask.sum() > 0:
        labeled = sp_image.label(mask)[0]
        components = sp_image.measurements.find_objects(labeled)
        c_fin = [(s[0].stop - s[0].start, s[0].stop - 1) for s in components]
        if len(c_fin) > 1 and mask[0] and mask[-1]:
            c_fin[0] = c_fin[0][0] + c_fin[-1][0], c_fin[0][1]
            c_fin = c_fin[0:-1]

        c, fin = zip(*c_fin)
        c = np.array(c, dtype=int)
        fin = np.array(fin, dtype=int)
        init = (fin - c) % mask.shape[0] + 1
    return c, init, fin


def unstick_contour(edgepoints, unstick_coeff):
    """
    Removes edgepoints near previously discarded points.
    @type edgepoints: list[bool]
    @param edgepoints: current edgepoint list
    @type unstick_coeff: float
    @param unstick_coeff
    @return: filtered edgepoints
    """
    (n, init, end) = loop_connected_components(np.logical_not(edgepoints))
    filtered = np.copy(edgepoints)
    n_edgepoint = len(edgepoints)
    for size, s, e in zip(n, init, end):
        for j in range(1, int(size * unstick_coeff + 0.5) + 1):
            filtered[(e + j) % n_edgepoint] = 0
            filtered[(s - j) % n_edgepoint] = 0
    return filtered


def sub2ind(dim, (x, y)):
    return x + y * dim


def get_gradient(im, index, border_thickness_steps):
    """
    Fun. calc. radial gradient including thickness of cell edges
    @param im: image (for which grad. will be calc.)
    @param index: indices of pixes sorted by polar coords. (alpha, radius) 
    @param border_thickness_steps: number of steps to cop. grad. - depands on cell border thickness
    @return: gradient matrix for cell
    """
    # index of axis used to find max grad.
    # PL: Indeks pomocniczy osi służący do wyznaczenia maksymalnego gradientu
    max_gradient_along_axis = 2
    # preparing the image limits (called subimage) for which grad. will be computed
    # PL: Wymiary wycinka obrazu, dla którego będzie obliczany gradient
    radius_lengths, angles = index.shape[0], index.shape[1]
    # matrix init
    # for each single step for each border thick. separated grad. is being computed
    # at the end the max. grad values are returned (for all steps and thick.)
    # PL: Inicjacja macierzy dla obliczania gradientów
    # PL: Dla każdego pojedynczego kroku dla zadanej grubości krawędzi komórki obliczany jest osobny gradient
    # PL: Następnie zwracane są maksymalne wartości gradientu w danym punkcie dla wszystkich kroków grubości krawędzi
    gradients_for_steps = np.zeros((radius_lengths, angles, border_thickness_steps), dtype=np.float64)
    # PL: Dla każdego kroku wynikającego z grubości krawędzi komórki:
    # PL: Najmniejszy krok ma rozmiar 1, największy ma rozmiar: ${border_thickness_steps}
    for border_thickness_step in range(1, int(border_thickness_steps) + 1):

        # find beg. and end indices of input matrix for which the gradient will be computed
        # PL: Wyznacz początek i koniec wycinka macierzy, dla którego będzie wyliczany gradient
        matrix_end = radius_lengths - border_thickness_step
        matrix_start = border_thickness_step

        # find beg. and end indices of pix. for which the gradient will be computed
        # PL: Wyznacz początek i koniec wycinka indeksu pikseli, dla którego będzie wyliczany gradient
        starting_index = index[:matrix_end, :]
        ending_index = index[matrix_start:, :]

        # find the spot in matrix where comp. gradient will go
        # PL: Wyznacz początek i koniec wycinka macierzy wynikowej, do którego będzie zapisany obliczony gradient
        intersect_start = int(math.ceil(border_thickness_step / 2.0))
        intersect_end = int(intersect_start + matrix_end)

        # comp. current gradient for selected (sub)image 
        # PL: Wylicz bieżącą wartość gradientu dla wyznaczonego wycinka obrazu
        try:
            current_step_gradient = im[Index.to_numpy(ending_index)] - im[Index.to_numpy(starting_index)]
        except Exception:
            print border_thickness_step
            print radius_lengths
            print matrix_start
            print matrix_end
            print ending_index
            print starting_index

            raise Exception

        current_step_gradient /= np.sqrt(border_thickness_step)
        # Zapisz gradient do wyznaczonego wycinka macierzy wyników
        gradients_for_steps[intersect_start:intersect_end, :, border_thickness_step - 1] = current_step_gradient

    return gradients_for_steps.max(axis=max_gradient_along_axis)


def get_polygon_path(polygon_x, polygon_y):
    vertices = zip(list(polygon_x) + [polygon_x[0]], list(polygon_y) + [polygon_y[0]])
    codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 2) + [Path.CLOSEPOLY]
    p = Path(vertices, codes)
    return p


def get_in_polygon(x1, x2, y1, y2, path):
    x, y = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
    x, y = x.flatten(), y.flatten()
    pts = np.vstack((x, y)).T

    # Find points that belong to snake in minimal rectangle
    grid = path.contains_points(pts)
    grid = grid.reshape(y2 - y1, x2 - x1)
    return grid


def inslice_point(point_yx_in_slice, slices):
    y = point_yx_in_slice[0]
    x = point_yx_in_slice[1]
    max_len = 1000000
    return y - slices[0].indices(max_len)[0], x - slices[1].indices(max_len)[0]


def unslice_point(point_yx_in_slice, slices):
    y = point_yx_in_slice[0]
    x = point_yx_in_slice[1]
    max_len = 1000000
    return y + slices[0].indices(max_len)[0], x + slices[1].indices(max_len)[0]


def get_cartesian_bounds(polar_coordinate_boundary, origin_x, origin_y, polar_transform):
    polygon_x, polygon_y = polar_to_cartesian(polar_coordinate_boundary, origin_x, origin_y, polar_transform)
    x1 = int(max(0, math.floor(min(polygon_x))))
    x2 = int(math.ceil(max(polygon_x)) + 1)
    y1 = int(max(0, math.floor(min(polygon_y))))
    y2 = int(math.ceil(max(polygon_y)) + 1)
    return slice(y1, y2), slice(x1, x2)


def polar_to_cartesian(polar_coordinate_boundary, origin_x, origin_y, polar_transform):
    t = polar_transform.t
    step = polar_transform.step
    px = origin_x + step * polar_coordinate_boundary * np.cos(t.T)
    py = origin_y + step * polar_coordinate_boundary * np.sin(t.T)

    return px, py


def star_in_polygon((max_y, max_x), polar_coordinate_boundary, seed_x, seed_y, polar_transform):
    polygon_x, polygon_y = polar_to_cartesian(polar_coordinate_boundary, seed_x, seed_y, polar_transform)

    x1 = int(max(0, math.floor(min(polygon_x))))
    x2 = int(min(max_x, math.ceil(max(polygon_x)) + 1))
    y1 = int(max(0, math.floor(min(polygon_y))))
    y2 = int(min(max_y, math.ceil(max(polygon_y)) + 1))

    x1 = min(x1, max_x)
    y1 = min(y1, max_y)
    x2 = max(0, x2)
    y2 = max(0, y2)

    small_boolean_mask = get_in_polygon(x1, x2, y1, y2, get_polygon_path(polygon_x, polygon_y))

    boolean_mask = np.zeros((max_y, max_x), dtype=bool)
    boolean_mask[y1:y2, x1:x2] = small_boolean_mask

    yx = [y1, x1]

    return boolean_mask, small_boolean_mask, yx
