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

    def imreshape(self):
        self.Xtr_im = np.reshape(self.Xtr, (self.Ntr, 3, self.dim, -1))
        self.Xte_im = np.reshape(self.Xte, (self.Nte, 3, self.dim, -1))

    def plot_images(self):
        for i in range(len(self.Xtr_im)):
            plt.imsave("visualization/img_{i}.png", 
                       np.transpose(self.Xtr_im[i], (1, 2, 0)))

if __name__ == "__main__":
    data = Data()
    Xtr = data.Xtr
    Xtr_im = data.Xtr_im
    data.plot_images()
