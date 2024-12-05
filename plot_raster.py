
#%% basic imports 
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

#%% set paths to data

filepath = ''
filename = ('%srestCondition_spontaneous.mat' % filepath)


#%% load data

data = loadmat(filename, simplify_cells=True)

params = data['sim_params']
spikes = data['spikes']
N_e = params['N_e']

#%% raster

plt.figure()

# cells
indsE = np.nonzero(spikes[1,:] < N_e)[0]
indsI = np.nonzero(spikes[1,:] >= N_e)[0]
plt.plot(spikes[0,indsE],spikes[1,indsE], '.', markersize=0.5, color='navy')
plt.plot(spikes[0,indsI],spikes[1,indsI], '.', markersize=0.5, color='firebrick')
plt.yticks([])
plt.xlabel('time [s]')
plt.ylabel('neurons')
plt.tight_layout()
