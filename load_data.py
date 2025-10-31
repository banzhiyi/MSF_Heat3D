import numpy as np
from scipy.io import loadmat

data = loadmat('./data/IndianPine.mat')
TR = data['TR']
print("TR unique:", np.unique(TR))