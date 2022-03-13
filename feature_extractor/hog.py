import numpy as np
from load import Data
import sys


class FeatureExtractor:

    def _compute_features(self, X):
        raise NotImplementedError()


class HOG(FeatureExtractor):
    """ Computes the historgam of Oriented Gradients"""
    def __init__(self, pixels_per_cell=4, cells_per_block=3, orientations=9):
        super(HOG, self).__init__()
        self.cell_size = pixels_per_cell
        self.H = 32
        self.W = 32
        self.nb_cell_cols = int(self.H // self.cell_size)
        self.nb_cell_rows = int(self.W // self.cell_size)
        self.nb_bins = orientations
        self.cells_per_block = cells_per_block
        self.features = []

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

    def compute_grey_gradients(self, image):
        g_x = np.zeros((self.H, self.W))
        g_y = np.zeros((self.H, self.W))
        # On the edges of the image
        g_x[:, 0] = 0
        g_x[:, -1] = 0
        g_y[0, :] = 0
        g_y[-1, :] = 0
        # Inside the image
        g_x[:, 1:-1] = image[:, 2:] - image[:, :-2]
        g_y[1:-1, :] = image[2:, :] - image[:-2, :]
        return g_x, g_y


    def compute_direction(self, gradients_x, gradients_y, eps=1e-6):
        theta = np.arctan2(gradients_y , gradients_x)
        grad_direction = np.rad2deg(theta)
        grad_direction = grad_direction % 180
        return grad_direction

    def compute_magnitude(self, g_x, g_y):
        magnitude = np.sqrt(g_x**2 + g_y**2)
        return magnitude


    def _compute_cell_hog(self, cell_magnitude, cell_direction, direction_inf, direction_sup):
        """" A CORRIGER """
        tot_hog = 0.
        for channel in range(3):
            for cell_i in range(self.cell_size):
                for cell_j in range(self.cell_size):
                    if (cell_direction[channel, cell_i, cell_j] >= direction_inf) or (cell_direction[channel, cell_i, cell_j] < direction_sup):
                        continue
                    tot_hog += cell_magnitude[channel, cell_i, cell_j]
        return tot_hog / (self.cell_size * self.cell_size)


    def compute_hog_features(self, image):

        l_0 = self.cell_size / 2
        c_0 = self.cell_size / 2

        g_x, g_y = self.compute_gradients(image)
        magnitude = self.compute_magnitude(g_x, g_y)
        print("magnitude shape", magnitude.shape)
        direction = self.compute_direction(g_x, g_y)
        hist = np.zeros((3, self.nb_cell_cols, self.nb_cell_rows, self.nb_bins))

        range_rows_stop = (self.cell_size + 1) / 2
        range_rows_start = -(self.cell_size / 2)
        range_columns_stop = (self.cell_size + 1) / 2
        range_columns_start = -(self.cell_size / 2)

        for channel in range(3):
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
                        cell_magnitude = magnitude[channel, int(l+range_rows_start):int(l+range_rows_stop), int(c+range_columns_start):int(c+range_columns_stop)]
                        cell_direction =  direction[channel, int(l+range_rows_start):int(l+range_rows_stop), int(c+range_columns_start):int(c+range_columns_stop)]
                        hist[channel, l_i, c_i, i] = self._compute_cell_hog(self, cell_magnitude, cell_direction, start_direction, end_direction)

                        c_i += 1
                        c += self.cell_size
                    l_i += 1
                    l += self.cell_size

        hog_features = np.reshape(hist, -1)
        return hog_features


    def _compute_features(self, X):
        NB_IMAGES = X.shape[0]
        for i in range(NB_IMAGES):
            sys.stdout.write("\r Computing feature .... [{} / {}]".format(i, NB_IMAGES))
            image = X[i]
            self.features.append(self.compute_hog_features(image))
        return np.array(self.features)


    def _compute_cell_grey_hog(self, cell_magnitude, cell_direction, direction_inf, direction_sup):
        """" A CORRIGER """
        tot_hog = 0.
        for cell_i in range(self.cell_size):
            for cell_j in range(self.cell_size):
                if (cell_direction[cell_i, cell_j] >= direction_inf) or (cell_direction[cell_i, cell_j] < direction_sup):
                    continue
                tot_hog += cell_magnitude[cell_i, cell_j]
        return tot_hog / (self.cell_size * self.cell_size)


    def compute_hog_grey_features(self, image):

        l_0 = self.cell_size / 2
        c_0 = self.cell_size / 2

        g_x, g_y = self.compute_grey_gradients(image)
        magnitude = self.compute_magnitude(g_x, g_y)
        direction = self.compute_direction(g_x, g_y)
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
                    range_rows_start = int(range_rows_start)
                    range_rows_stop = int(range_rows_stop)
                    range_columns_start = int(range_columns_start)
                    range_columns_stop = int(range_columns_stop)
                    cell_magnitude = magnitude[int(l+range_rows_start):int(l+range_rows_stop), int(c+range_columns_start):int(c+range_columns_stop)]
                    cell_direction =  direction[int(l+range_rows_start):int(l+range_rows_stop), int(c+range_columns_start):int(c+range_columns_stop)]
                    hist[l_i, c_i, i] = self._compute_cell_grey_hog(cell_magnitude, cell_direction, start_direction, end_direction)

                    c_i += 1
                    c += self.cell_size
                l_i += 1
                l += self.cell_size
        return hist

    """ TO MODIFY (TAKEN FROM SKIMAGE) """
    def _normalize_descriptors(self, image_histogram):

        n_blocks_row = (self.H // self.cell_size - self.cells_per_block) + 1
        n_blocks_col = (self.W // self.cell_size - self.cells_per_block) + 1
        normalized_blocks = np.zeros(
            (n_blocks_row, n_blocks_col, self.cells_per_block,self.cells_per_block, self.nb_bins)
        )
        for r in range(n_blocks_row):
            for c in range(n_blocks_col):
                block = image_histogram[r:r + self.cells_per_block, c:c + self.cells_per_block, :]
                normalized_blocks[r, c, :] = self._normalize_block(block)

        return normalized_blocks.reshape(-1)

    def _compute_grey_features(self, X):
        NB_IMAGES = X.shape[0]
        for i in range(NB_IMAGES):
            sys.stdout.write("\r Computing feature .... [{} / {}]".format(i, NB_IMAGES))
            image = X[i]
            hog_feature = self.compute_hog_grey_features(image)
            normalized_hog_feature = self._normalize_descriptors(hog_feature)
            self.features.append(normalized_hog_feature)
        return np.array(self.features)

    def _normalize_block(self, block, eps=1e-6):
        res = block / np.sqrt(np.sum(block ** 2) + eps ** 2)
        res = np.minimum(res, 0.2)
        res = res / np.sqrt(np.sum(res ** 2) + eps ** 2)
        return res



if __name__ == '__main__':

    data = Data()
    Xte = data.Xte_im
    Xtr = data.grey_Xtr_im

    hogb = HOG(pixels_per_cell=8)

    hogb.compute_hog_features(Xte[0])
