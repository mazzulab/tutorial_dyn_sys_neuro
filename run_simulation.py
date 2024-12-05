#%% IMPORTS


# standard
import time
import sys
from scipy.io import savemat

# simulation parameters
from simParams import sim_params

# simulation functions
from fcn_make_network_cluster import fcn_make_network_cluster
from fcn_simulation import fcn_simulate_exact_poisson
from fcn_stimulation import get_stimulated_clusters


#%% SETTING UP...


# path for saving data
outpath = ''


# seed for external input realization
extInput_seed = 1342134

# seed for selecting stimulated clusters
stimClusters_seed = 12421

# seed for stimulation
stimSeed = 23423

# seed for initial conditions and poisson process
ICSeed = 8485

# seed for making the network
networkSeed = 34524


#%% UPDATE SIMULATION PARAMETERS


# initialize parameters
s_params = sim_params()

# update synaptic weights
s_params.update_JplusAB()

# set dependent variables
s_params.set_dependent_vars()

# set external inputs     
s_params.set_external_inputs(extInput_seed) 

# make network
W, popsizeE, popsizeI = fcn_make_network_cluster(s_params, networkSeed)  

# set selective clusters
selectiveClusters = get_stimulated_clusters(s_params, stimClusters_seed)
  
# set selective clusters
s_params.selectiveClusters = selectiveClusters[0]
    
# determine which neurons are stimulated
s_params.get_stimulated_neurons(stimSeed, popsizeE, popsizeI)
    
# determine maximum stimulus strength
s_params.set_max_stim_rate()


#%% RUN THE SIMULATION


t0 = time.time()

if s_params.save_voltage == True:
    sys.exit('dont save voltage data')
else:
    spikes = fcn_simulate_exact_poisson(s_params, W, ICSeed)
               
tf = time.time()

print('simulation done') 
print('sim time = %0.3f seconds' %(tf-t0))

        
#%% 
#--------------------------------------------------------------------------
# SAVE THE DATA
#-------------------------------------------------------------------------- 

        
results_dictionary = {'sim_params':                     s_params, \
                      'spikes':                         spikes, \
                      'W':                              W, \
                      'clust_sizeE':                    popsizeE, \
                      'clust_sizeI':                    popsizeI, \
                      'extInput_seed':                  extInput_seed, \
                      'stimClusters_seed':              stimClusters_seed,\
                      'stimSeed':                       stimSeed, \
                      'ICSeed':                         ICSeed, \
                      'networkSeed':                    networkSeed}
    

# filename              
if s_params.sd_nu_ext_e_pert == 0.:
    
    filename = ('%srestCondition_spontaneous.mat' % (outpath))      
    
else:
    
    filename = ('%srunCondition_spontaneous.mat' % (outpath))      
    
savemat(filename, results_dictionary)
print('results written to file')


