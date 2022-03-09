import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Data:
    
    def __init__(self, repository="data/", dim=32):
        
        self.dim = dim
        self.Xtr = pd.read_csv(repository + "Xtr.csv", 
                               header=None, 
                               usecols=np.arange(3072), 
                               index_col=False).to_numpy()
        self.Ytr = pd.read_csv(repository + "Ytr.csv", 
                               usecols=[1]).to_numpy()
        self.Ntr = self.Xtr.shape[0]
        assert len(self.Ytr) == self.Ntr
        assert self.Xtr.shape[1] == self.dim * self.dim * 3
        
        
        self.Xte = pd.read_csv(repository + "Xte.csv", 
                               header=None, 
                               usecols=np.arange(3072), 
                               index_col=False).to_numpy()
        self.Nte = self.Xte.shape[0]
        assert self.Xte.shape[1] == self.dim * self.dim * 3
        
        self.imreshape()
        self.compute_grey_im()

    def imreshape(self):
        self.Xtr_im = np.reshape(self.Xtr, (self.Ntr, 3, self.dim, -1))
        self.Xte_im = np.reshape(self.Xte, (self.Nte, 3, self.dim, -1))


    def compute_grey_im(self):
        self.grey_Xtr_im = 0.2989 * self.Xtr_im[:, 0, :, :] + 0.5870 * self.Xtr_im[:, 1, :, :] + 0.1140 * self.Xtr_im[:, 2, :, :]
        self.grey_Xte_im = 0.2989 * self.Xte_im[:, 0, :, :] + 0.5870 * self.Xte_im[:, 1, :, :] + 0.1140 * self.Xte_im[:, 2, :, :]


    def get_sift_features(self):
        pass

    def plot_images(self):
        plt.imshow(np.transpose(self.Xtr_im[5], (1, 2, 0)))
        plt.show()
        plt.imshow(self.grey_Xtr_im[5])
        plt.show()


if __name__ == "__main__":
    data = Data()
    Xtr = data.Xtr
    Xtr_im = data.Xtr_im
    data.plot_images()
