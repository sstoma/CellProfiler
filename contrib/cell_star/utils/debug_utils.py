# -*- coding: utf-8 -*-
"""
Utilities with tools that can help with debuging / profiling CellStar
Date: 2016
Website: http://cellstar-algorithm.org/
"""
import os
from os import makedirs
from os.path import exists

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import image_util
from contrib.cell_star.core import image_repo

debug_image_path = "debug"

SHOW = False
if not SHOW:
    matplotlib.use('Agg')
SILENCE = False


def prepare_debug_folder():
    if not exists(debug_image_path):
        makedirs(debug_image_path)


def show_snake(snake, name):
    polygon = image_util.draw_polygons(snake.images.image, [zip(snake.xs, snake.ys)])
    image_out = polygon + (1 - polygon) * snake.images.image
    image_util.image_show(image_out, name)


def draw_seeds(seeds, background, title="some_source"):
    if not SILENCE:
        prepare_debug_folder()
        fig = plt.figure("draw_seeds")
        fig.frameon = False
        plt.imshow(background, cmap=plt.cm.gray)
        plt.plot([s.x for s in seeds], [s.y for s in seeds], 'bo', markersize=3)
        plt.savefig(os.path.join(debug_image_path, "seeds_" + title + ".png"), pad_inches=0.0)
        fig.clf()
        plt.close(fig)


def images_repo_save(images):
    """
    @type images: image_repo.ImageRepo
    """
    image_save(images.background, "background")
    image_save(images.brighter, "brighter")
    image_save(images.brighter_original, "brighter_original")
    image_save(images.darker, "darker")
    image_save(images.darker_original, "darker_original")
    image_save(images.cell_content_mask, "cell_content_mask")
    image_save(images.cell_border_mask, "cell_border_mask")
    image_save(images.foreground_mask, "foreground_mask")
    image_save(images.image_back_difference, "image_back_difference")


def image_save(image, title):
    """
    Displays image with title using matplotlib.pyplot
    @param image:
    @param title:
    """

    if not SILENCE:
        prepare_debug_folder()
        sp.misc.imsave(os.path.join(debug_image_path, title + '.png'), image)


def image_show(image, title):
    """
    Displays image with title using matplotlib.pyplot
    @param image:
    @param title:
    """
    if not SILENCE and SHOW:
        prepare_debug_folder()
        fig = plt.figure(title)
        plt.imshow(image, cmap=plt.cm.gray, interpolation='none')
        plt.show()
        fig.clf()
        plt.close(fig)


def draw_overlay(image, x, y):
    if not SILENCE and SHOW:
        prepare_debug_folder()
        fig = plt.figure()
        plt.imshow(image, cmap=plt.cm.gray, interpolation='none')
        plt.plot(x, y)
        plt.show()
        fig.clf()
        plt.close(fig)


def draw_snakes(image, snakes, outliers=.1, it=0):
    if not SILENCE and len(snakes) > 1:
        prepare_debug_folder()
        snakes = sorted(snakes, key=lambda ss: ss.rank)
        fig = plt.figure("draw_snakes")
        plt.imshow(image, cmap=plt.cm.gray, interpolation='none')

        snakes_tc = snakes[:int(len(snakes) * (1 - outliers))]

        max_rank = snakes_tc[-1].rank
        min_rank = snakes_tc[0].rank
        rank_range = max_rank - min_rank
        if rank_range == 0:  # for example there is one snake
            rank_range = max_rank

        rank_ci = lambda rank: 999 * ((rank - min_rank) / rank_range) if rank <= max_rank else 999
        colors = plt.cm.jet(np.linspace(0, 1, 1000))
        s_colors = [colors[rank_ci(s.rank)] for s in snakes]

        for snake, color in zip(snakes, s_colors):
            plt.plot(snake.xs, snake.ys, c=color, linewidth=4.0)

        plt.savefig(os.path.join(debug_image_path, "snakes_rainbow_" + str(it) + ".png"), pad_inches=0.0)
        if SHOW:
            plt.show()

        fig.clf()
        plt.close(fig)
