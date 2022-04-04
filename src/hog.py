import numpy as np
from load import Data
import sys
from copy import deepcopy

class HOG:
    """ Computes the histogram of Oriented Gradients"""
    def __init__(self, pixels_per_cell=8, cells_per_block=3, orientations=9):
        super(HOG, self).__init__()
        self.cell_size = pixels_per_cell
        self.H = 32
        self.W = 32
        self.nb_cell_cols = int(self.H // self.cell_size)
        self.nb_cell_rows = int(self.W // self.cell_size)
        self.nb_bins = orientations
        self.cells_per_block = cells_per_block
        self.features = []
        
    def compute_features(self, X):
        NB_IMAGES = X.shape[0]
        for i in range(NB_IMAGES):
            sys.stdout.write("\r Computing feature .... [{} / {}]".format(i, NB_IMAGES))
            image = X[i]
            hog_feature = self.compute_hog_features(image)
            normalized_hog_feature = self.normalize(hog_feature)
            self.features.append(normalized_hog_feature)
        return np.array(self.features)

    def compute_gradients(self, image):
        g_x = np.zeros((3, self.H, self.W))
        g_y = np.zeros((3, self.H, self.W))
        # On the edges of the image
        g_x[:, :, 0] = 0
        g_x[:, :, -1] = 0
        g_y[:, 0, :] = 0
        g_y[:, -1, :] = 0
        # Inside the image
        g_x[:, :, 1:-1] = image[:, :, 2:] - image[:, :, :-2]
        g_y[:, 1:-1, :] = image[:, 2:, :] - image[:, :-2, :]
        return g_x, g_y

    def compute_direction(self, gradients_x, gradients_y, eps=1e-6):
        theta = np.arctan2(gradients_y , gradients_x)
        grad_direction = np.rad2deg(theta)
        grad_direction = grad_direction % 180
        return grad_direction

    def compute_magnitude(self, g_x, g_y):
        magnitude = np.sqrt(g_x**2 + g_y**2)
        return magnitude

    def compute_hog_features(self, image):

        l_0 = self.cell_size / 2
        c_0 = self.cell_size / 2

        g_x, g_y = self.compute_gradients(image)
        magnitude = self.compute_magnitude(g_x, g_y)
        direction = self.compute_direction(g_x, g_y)

        # Extract the channels where the gradient magnitude is the highest
        idcs_max = magnitude.argmax(axis=0)
        rows, cols = np.meshgrid(np.arange(magnitude.shape[1]), np.arange(magnitude.shape[2]), indexing = 'ij', sparse = True)

        magnitude = magnitude[idcs_max, rows, cols]
        direction = direction[idcs_max, rows, cols]
        hist = np.zeros((self.nb_cell_cols, self.nb_cell_rows, self.nb_bins))

        range_rows_stop = (self.cell_size + 1) / 2
        range_rows_start = -(self.cell_size / 2)
        range_columns_stop = (self.cell_size + 1) / 2
        range_columns_start = -(self.cell_size / 2)

        for i in range(self.nb_bins):
            # Bounds of orientations
            start_direction = (i+1) * 180./self.nb_bins
            end_direction = i * 180./self.nb_bins
            l = l_0
            c = c_0
            l_i = 0
            c_i = 0

            while l < self.H:
                c_i = 0
                c = c_0

                while c < self.W:
                    cell_magnitude = magnitude[int(l+range_rows_start):int(l+range_rows_stop), int(c+range_columns_start):int(c+range_columns_stop)]
                    cell_direction =  direction[int(l+range_rows_start):int(l+range_rows_stop), int(c+range_columns_start):int(c+range_columns_stop)]
                    hist[l_i, c_i, i] = self.compute_cell_grey_hog(cell_magnitude, cell_direction, start_direction, end_direction)

                    c_i += 1
                    c += self.cell_size
                l_i += 1
                l += self.cell_size
        return hist


    def compute_cell_grey_hog(self, cell_magnitude, cell_direction, direction_inf, direction_sup):
        tot_hog = 0.
        for cell_i in range(self.cell_size):
            for cell_j in range(self.cell_size):
                if (cell_direction[cell_i, cell_j] >= direction_inf) or (cell_direction[cell_i, cell_j] < direction_sup):
                    continue
                tot_hog += cell_magnitude[cell_i, cell_j]
        return tot_hog / (self.cell_size * self.cell_size)

    def normalize(self, image_histogram):
        eps=1e-6
        n_blocks_row = (self.H // self.cell_size - self.cells_per_block) + 1
        n_blocks_col = (self.W // self.cell_size - self.cells_per_block) + 1
        normalized_blocks = np.zeros((n_blocks_row, 
                                      n_blocks_col, 
                                      self.cells_per_block,
                                      self.cells_per_block, 
                                      self.nb_bins))
        for r in range(n_blocks_row):
            for c in range(n_blocks_col):
                block = deepcopy(image_histogram[r:r + self.cells_per_block, 
                                                 c:c + self.cells_per_block])
                block = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
                block = np.minimum(block, 0.2)
                block = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
                normalized_blocks[r, c] = block

        return normalized_blocks.ravel()


if __name__ == '__main__':

    data = Data()
    Xte = data.Xte_im
    hogb = HOG(pixels_per_cell=8)
    feats = hogb.compute_features(Xte)
