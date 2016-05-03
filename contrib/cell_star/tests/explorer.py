# -*- coding: utf-8 -*-
"""
Images repository calculates and stores intermediate images used in segmentation process.
Date: 2013-2016
Website: http://cellstar-algorithm.org/
"""

import sys

import matplotlib as mpl
import wx
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as Canvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2Wx as Toolbar

import contrib.cell_star.utils.debug_util as debug_utils
from contrib.cell_star.core.image_repo import ImageRepo
from contrib.cell_star.parameter_fitting.pf_snake import *
from contrib.cell_star.process.segmentation import Segmentation
from contrib.cell_star.utils.debug_util import draw_snakes_on_axes, draw_seeds_on_axes
from contrib.cell_star.utils.image_util import load_image


# input_path = r"D:\Fafa\Drozdze\CellStarTesting\Data\Benchmark2\TestSet10_1k\frames\Point0000_Seq00000022.tif"


class Explorer:
    def __init__(self, image, images, ui, cell_star):
        """
        @type cell_star: Segmentation
        """
        self.image = image
        self.images = images
        self.ui = ui
        self.ui.onclick = self.manage_click
        self.ui.press = self.manage_press
        self.cell_star = cell_star

    def manage_click(self, button, x, y):
        if button == 2:
            self.grow_and_show(x, y)
        elif button == 3:
            self.ui.axes.lines = []
            self.grow_and_show(x, y)

    def manage_press(self, key):
        if key == 's':
            self.ui.axes.lines = []
            self.cell_star.find_seeds(False)
            seeds = self.cell_star.all_seeds
            self.cell_star.all_seeds = []
            self.cell_star.seeds = []
            draw_seeds_on_axes(seeds, self.ui.axes)

    def grow_and_show(self, x, y):
        pfsnake = PFSnake(Seed(x, y, "click"), self.images, self.cell_star.parameters)
        pfsnake.grow()

        draw_snakes_on_axes(sorted(pfsnake.snakes, key=lambda s: -s.rank), self.ui.axes)
        #draw_snakes_on_axes([pfsnake.best_snake], self.ui.axes)
        pass


class ExplorerFrame(wx.Dialog):
    def __init__(self, images, parent=None, id=wx.ID_ANY, title='CellStar explorer', x=900, y=600):
        """
        @type images: ImageRepo
        """
        style = wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX | wx.MINIMIZE_BOX
        wx.Dialog.__init__(self, parent, id, title, style=style)
        self.Size = (x, y)
        self.figure = mpl.figure.Figure(dpi=300, figsize=(1, 1))
        self.axes = self.figure.add_subplot(111)
        self.axes.margins(0, 0)
        self.canvas = Canvas(self, -1, self.figure)
        self.toolbar = Toolbar(self.canvas)
        self.toolbar.Realize()

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.EXPAND)
        sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        self.SetSizer(sizer)

        self.layer0 = self.axes.imshow(images.image, cmap=mpl.cm.gray)

        def onclick_internal(event):
            if event.ydata is None:
                return

            print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f' % (
                event.button, event.x, event.y, event.xdata, event.ydata)

            if event.button != 1:
                self.onclick(event.button, event.xdata, event.ydata)
                self.figure.canvas.draw()

        def press_internal(event):
            print('press', event.key)
            if event.key in 'qwer':
                self.layer0.remove()
                if event.key == 'q':  # input
                    self.layer0 = self.axes.imshow(images.image, cmap=mpl.cm.gray)
                elif event.key == 'w':  # image clean
                    self.layer0 = self.axes.imshow(images.image_back_difference_blurred, cmap=mpl.cm.gray)
                elif event.key == 'e':  # brighter
                    self.layer0 = self.axes.imshow(images.brighter, cmap=mpl.cm.gray)
                elif event.key == 'r':  # darker
                    self.layer0 = self.axes.imshow(images.darker, cmap=mpl.cm.gray)
                self.figure.canvas.draw()
            else:
                self.press(event.key)
                self.figure.canvas.draw()

        self.figure.canvas.mpl_connect('button_press_event', onclick_internal)
        self.figure.canvas.mpl_connect('key_press_event', press_internal)
        self.Show(True)

    def onclick(self, x, y):
        pass


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print "Usage: <script> input_path"
        print "Given: " + " ".join(sys.argv)

    debug_utils.SILENCE = False
    input_path = sys.argv[1]
    image = load_image(input_path)
    cell_star = Segmentation(12)
    cell_star.set_frame(image)
    cell_star.pre_process()

    app = wx.App(0)
    explorer_ui = ExplorerFrame(cell_star.images)
    explorer = Explorer(image, cell_star.images, explorer_ui, cell_star)
    app.MainLoop()
