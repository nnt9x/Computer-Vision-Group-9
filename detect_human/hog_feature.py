import cv2
import numpy as np
from numpy import linalg as LA


class HOGFeature(object):
    def __init__(self, window_height=128, window_width=64, cell_size=8, block_size=2, bins=9):
        self.window_height = window_height
        self.window_width = window_width
        self.cell_size = cell_size
        self.block_size = block_size
        self.bins = bins
        pass

    def __call__(self, cv_img):
        if len(cv_img.shape) > 2:  # convert to gray image
            img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        else:
            img = cv_img
        h, w = img.shape  # 128, 64

        # resize
        if h != self.window_height or w != self.window_width:
            img = cv2.resize(src=img, dsize=(
                self.window_width, self.window_height))
            h, w = img.shape  # 128, 64

        # gradient
        xkernel = np.array([[-1, 0, 1]])
        ykernel = np.array([[-1], [0], [1]])
        dx = cv2.filter2D(img, cv2.CV_32F, xkernel)
        dy = cv2.filter2D(img, cv2.CV_32F, ykernel)

        # histogram
        magnitude = np.sqrt(np.square(dx) + np.square(dy))
        orientation = np.arctan(np.divide(dy, dx+0.00001))  # radian
        orientation = np.degrees(orientation)  # -90 -> 90
        orientation += 90  # 0 -> 180

        num_cell_x = w // self.cell_size  # 8
        num_cell_y = h // self.cell_size  # 16
        hist_tensor = np.zeros(
            [num_cell_y, num_cell_x, self.bins])  # 16 x 8 x 9
        for cx in range(num_cell_x):
            for cy in range(num_cell_y):
                ori = orientation[cy*self.cell_size:cy*self.cell_size +
                                  self.cell_size, cx*self.cell_size:cx*self.cell_size+self.cell_size]
                mag = magnitude[cy*self.cell_size:cy*self.cell_size+self.cell_size,
                                cx*self.cell_size:cx*self.cell_size+self.cell_size]
                # https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
                hist, _ = np.histogram(ori, bins=self.bins, range=(
                    0, 180), weights=mag)  # 1-D vector, 9 elements
                hist_tensor[cy, cx, :] = hist
            pass
        pass

        # normalization
        redundant_cell = self.block_size-1
        feature_tensor = np.zeros([num_cell_y-redundant_cell, num_cell_x -
                                  redundant_cell, self.block_size*self.block_size*self.bins])
        for bx in range(num_cell_x-redundant_cell):  # 7
            for by in range(num_cell_y-redundant_cell):  # 15
                by_from = by
                by_to = by+self.block_size
                bx_from = bx
                bx_to = bx+self.block_size
                v = hist_tensor[by_from:by_to, bx_from:bx_to,
                                :].flatten()  # to 1-D array (vector)
                feature_tensor[by, bx, :] = v / LA.norm(v, 2)
                # avoid NaN:
                # avoid NaN (zero division)
                if np.isnan(feature_tensor[by, bx, :]).any():
                    feature_tensor[by, bx, :] = v

        return feature_tensor.flatten()  # 3780 features
        pass
