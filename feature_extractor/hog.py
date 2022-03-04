import numpy as np
from load import Data


class FeatureExtractor:

    def _compute_features(self, X):
        raise NotImplementedError()


class HOG(FeatureExtractor):
    """ Computes the historgam of Oriented Gradients"""
    def __init__(self, cell_size=4):
        super(HOG, self).__init__()
        self.cell_size = cell_size
        self.H = 32
        self.W = 32
        self.nb_cell_cols = int(self.H // self.cell_size)
        self.nb_cell_rows = int(self.W // self.cell_size)
        self.nb_bins = 9
        self.features = []


    def compute_gradients_per_channel(self, image, n_channel=0):
        Jx = np.zeros((self.H, self.W))
        Jy = np.zeros((self.H, self.W))
        for i in range(1, self.H - 1):
            for j in range(1, self.W - 1):
                Jy[i, j] = (image[n_channel, i + 1, j] - image[n_channel, i - 1, j]) / 2
                Jx[i, j] = (image[n_channel, i, j + 1] - image[n_channel, i, j - 1]) / 2

        for i in range(self.H):
            Jx[i, 0] = image[n_channel, i, 1] - image[n_channel, i, 0]
            Jx[i, -1] = image[n_channel, i, -1] - image[n_channel, i, -2]

        for j in range(self.W):
            Jy[0, j] = image[n_channel, 1, j] - image[n_channel, 0, j]
            Jy[-1, j] = image[n_channel, -1, j] - image[n_channel, -2, j]

        return Jx, Jy

    def compute_gradients(self, image):
        gradients_x = np.zeros((3, self.H, self.W))
        gradients_y = np.zeros((3, self.H, self.W))
        for channel in range(3):
            gradients_x[channel], gradients_y[channel]= self.compute_gradients_per_channel(image, channel)
        return gradients_x, gradients_y

    def compute_direction(self, gradients_x, gradients_y):
        theta = np.zeros(gradients_x.shape)
        theta = np.arctan(gradients_y / gradients_x)
        return theta

    def compute_magnitude(self, gradients_x, gradients_y):
        magnitude = np.sqrt(gradients_x**2 + gradients_y**2)
        return magnitude

    def compute_cell_hog_feature(self, cell_image, cell_magnitudes, cell_directions):
        hist_bins = np.arange(0, 180, 20)
        histogram = np.zeros(self.nb_bins)
        cell_H, cell_W = cell_image.shape[0], cell_image.shape[1]
        for row_idx in range(cell_H):
            for col_idx in range(cell_W):
                magnitude_value = cell_magnitudes[row_idx, col_idx]
                direction_value = cell_directions[row_idx, col_idx] * 180 / np.pi

                # Get the two closest bins depending on the direction value
                list_ids = np.where(direction_value * np.ones(self.nb_bins) - hist_bins < 0)[0]
                print("Direction value", direction_value)

                if hist_bins[-1] < direction_value or direction_value < hist_bins[0]:
                    closest_ids = (0, len(hist_bins) -1)
                    if abs(direction_value - hist_bins[closest_ids[0]]) < abs(direction_value - hist_bins[closest_ids[1]]):
                        closest_id = closest_ids[0]
                        second_closest_id = closest_ids[1]
                    else:
                        closest_id = closest_ids[1]
                        second_closest_id = closest_ids[0]
                else:
                    print("List ids", list_ids)
                    closest_ids = (list_ids[0]-1, list_ids[0])
                    if direction_value - hist_bins[closest_ids[0]] < direction_value - hist_bins[closest_ids[1]]:
                        closest_id = closest_ids[0]
                        second_closest_id = closest_ids[1]
                    else:
                        closest_id = closest_ids[1]
                        second_closest_id = closest_ids[0]

                ratio = abs(direction_value - hist_bins[closest_id]) / (hist_bins[1] - hist_bins[0])
                histogram[closest_id] += ratio * magnitude_value
                histogram[second_closest_id] += (1 - ratio) * magnitude_value

        return histogram

    def compute_hog_features(self, image):
        gradients_x, gradients_y = self.compute_gradients(image)
        magnitude = self.compute_magnitude(gradients_x, gradients_y)
        direction = self.compute_direction(gradients_x, gradients_y)

        hist = np.zeros((3, self.nb_cell_cols, self.nb_cell_rows, self.nb_bins))

        for channel in range(3):
            for i in range(self.nb_cell_rows):
                for j in range(self.nb_cell_cols):
                    cell_image = image[channel, i*self.cell_size:(i+1)*self.cell_size, j*self.cell_size:(j+1)*self.cell_size]
                    cell_magnitude = magnitude[channel, i*self.cell_size:(i+1)*self.cell_size, j*self.cell_size:(j+1)*self.cell_size]
                    cell_direction = direction[channel, i*self.cell_size:(i+1)*self.cell_size, j*self.cell_size:(j+1)*self.cell_size]
                    hist[channel, i, j, :] = self.compute_cell_hog_feature(cell_image, cell_magnitude, cell_direction)
        hog_features = np.reshape(hist, -1)
        return hog_features

    def _compute_features(self, X):
        NB_IMAGES = X.shape[0]
        for i in range(NB_IMAGES):
            image = X[i]
            print("IMAGE SHAPE", image.shape)
            self.features.append(self.compute_hog_features(image))

if __name__ == '__main__':
    import matplotlib.pyplot as plt


    hog = HOG()
    data = Data()
    Xte = data.Xte_im
    feats = hog._compute_features(Xte)
    print(feats.shape)