import numpy as np


def test_function(arr):
    return arr*2

if __name__ == '__main__':
   y =  test_function(np.ones(4))
   print(y)