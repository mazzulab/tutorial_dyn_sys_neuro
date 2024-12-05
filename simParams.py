"""
set all simulation parameters
"""


import numpy as np
import sys
from fcn_make_network_cluster import fcn_compute_cluster_assignments

#-----------------------------------------------------------------------------
# CLASS FILE THAT SETS PARAMETERS FOR SIMULATIONS
# TIME UNITS IN SECONDS
#-----------------------------------------------------------------------------

class sim_params:

    
    # INIT METHOD
    def __init__(self):     
        
        #-----------------------------------------------------------------------------
        # DEFAULTS
        #-----------------------------------------------------------------------------  
        
        self.save_voltage = False              # whether or not to save membrane potential, input current 
                    
        self.T0 = 0.                            # simulation start time
        self.TF = 10                           # simulation end time
        self.dt = 1e-3                       # time step
        
        self.N = 1000                           # total number of neurons
        self.ne = 0.8                           # total fraction of excitatory neurons
        
        self.Vth_e = 1.5                        # excitatory threshold
        self.Vth_i = 0.75                       # inhibitory threshold
        self.Vr_e = 0.                          # reset potential E
        self.Vr_i = 0.                          # reset potential I
        
        self.tau_m_e = 20e-3                    # membrane time constant E
        self.tau_m_i = 20e-3                    # membrane time constant I  
        self.tau_s_e = 5e-3
        self.tau_s_i = 5e-3         
        self.tau_r = 5e-3                       # refractory period
        self.t_delay = self.dt                  # delay
        
        self.synType = 'exp'                    # synapse type
        
        
        # catch for t_delay = 0
        if self.t_delay == 0:
            sys.exit('ERROR: TIME DELAY MUST BE >= dt !')
            
            
        # whether or not external inputs are poisson
        self.extCurrent_poisson = True
        
    
        #-----------------------------------------------------------------------------
        # EXTERNAL INPUTS
        #----------------------------------------------------------------------------- 
        
        # external input mean and spatial standard dev 
        self.mean_nu_ext_e = 7.
        self.mean_nu_ext_i = 7.
        
        # perturbations
        
        # avg input
        self.mean_nu_ext_e_offset = 0.                    
        self.mean_nu_ext_i_offset = 0.
        
        # spatial variance of external inputs
        self.sd_nu_ext_e_pert = 0.0 
        self.sd_nu_ext_i_pert = 0.0 
        
        
        #-----------------------------------------------------------------------------
        # NETWORK PROPERTIES
        #-----------------------------------------------------------------------------        
        
        # cluster or hom
        self.net_type = 'cluster'
        
        # network connection type        
        self.connType = 'fixed_P' 
        # self.connType = 'fixed_InDegree' 
        
        # whether or not to depress inter-cluster connections
        self.depress_interCluster = True
               
        self.pext = 0.2
        self.pee = 0.2
        self.pei = 0.5
        self.pii = 0.5
        self.pie = 0.5
        
        self.Jee = 0.68/np.sqrt(self.N)
        self.Jie = 0.63/np.sqrt(self.N)
        self.Jei = 1.8/np.sqrt(self.N)
        self.Jii = 3.8/np.sqrt(self.N)
        self.Jie_ext = 2.3/np.sqrt(self.N)
        self.Jee_ext = 2.3/np.sqrt(self.N)
        
        # clusters
        self.p = 15
        self.bgrE = 0.1
        self.bgrI = 0.1
        # which neurons & weights are clustered
        self.clusters = ['E','I']
        self.clusterWeights = ['EE','EI','IE','II']
        
        # other cluster properties (probably wont change much)
        self.Ecluster_weightSize = False
        # cluster size heterogeneity ('hom' or 'het')
        self.clustE = 'hom' # E clusters 
        self.clustI = 'hom' # I clusters 
        if self.clustE == 'hom' or self.clustI == 'hom':
            # std of cluster size (as a fraction of mean)
            self.clust_std = 0.0 
        else:
            self.clust_std = 0.1
            
        # cluster depression & potentiation
        self.JplusEE = 15.75                 # EE intra-cluster potentiation factor (poisson)
        self.JplusII = 5.0                   # II intra-cluster potentiation factor
        self.JplusEI = 6.25                  # EI intra-cluster potentiation factor
        self.JplusIE = 5.45                  # IE intra-cluster potentiation factor
        # variance in synaptic weights
        self.deltaEE = 0.2
        self.deltaEI = 0.2
        self.deltaIE = 0.2
        self.deltaII = 0.2
        
        #-----------------------------------------------------------------------------
        # STIMULUS PROPERTIES
        #-----------------------------------------------------------------------------    
        # for stimuli, specify:
        self.stim_type = ''                 # type of stimulus ['' or 'noStim']
        self.nStim = 1                      # number of different stimuli to run
        self.mixed_selectivity = True       # allow different stimuli to target same clusters
        self.stim_shape = 'diff2exp'        # type of stimulus
        self.stim_onset = self.T0 + 1.0     # stimulus onset
        self.f_selectiveClus = 0.2          # fraction of clusters that are selective to each stimulus
        self.f_Ecells_target = 0.5          # fraction E cells targeted in selective clusters
        self.f_Icells_target = 0.0          # fraction of I cells targeted in selective clsuters
        self.stim_rel_amp = 0.2          # relative strength (fraction above baseline)
        if self.stim_type == 'noStim':      # set stim strength to zero if stim type is 'noStim'
            self.nStim = 1
            self.stim_rel_amp = 0
        
        # for box and linear
        self.stim_duration = 0.5            # duration of stimulus in seconds
        # for difference of exponentials
        self.stim_taur = 0.075
        self.stim_taud = 0.1
        
        #-----------------------------------------------------------------------------
        # FILENAMES
        #----------------------------------------------------------------------------- 
        self.parameters_fileName = ('simulationData')
        self.network_name = 'network'
        
        # PRINT
        print('sim_params class initialized')
        
#-----------------------------------------------------------------------------
# CLASS FUNCTIONS
#-----------------------------------------------------------------------------
    
    #---------------------------------------------------------------------------------
    # GET ANY VALUES THAT COULD HAVE BEEN PASSED THROUGH ARGPARSE
    #---------------------------------------------------------------------------------
    def get_argparse_vals(self, args):
        
        # get any values from argparser
        
        # perturbations
        self.mean_nu_ext_e_offset = args.mean_nu_ext_e_offset
        self.sd_nu_ext_e_pert = args.sd_nu_ext_e_pert
        
        # network type
        self.net_type = args.net_type

        # stimulation
        self.stim_rel_amp = args.stim_rel_amp
        
        # JplusEE
        self.JplusEE = args.JplusEE
 
    #---------------------------------------------------------------------------------
    # SET ANY VARIABLES THAT ARE COMPLETELY DETERMINED BY MAIN INPUTS
    #---------------------------------------------------------------------------------       
    def set_dependent_vars(self):
                   
        # variables that depend on main inputs
        
        # total numbers of E and I neurons
        self.N_e = int(self.N*self.ne)                   
        self.N_i = int(self.N - self.N_e)       
        
        # number of connections
        self.Cee = self.pee*self.N_e
        self.Cei = self.pei*self.N_i
        self.Cii = self.pii*self.N_i
        self.Cie = self.pie*self.N_e
        self.Cext = self.N_e*self.pext 


    #---------------------------------------------------------------------------------
    # SET JplusAB BASED ON NETWORK TYPE
    #--------------------------------------------------------------------------------- 
    def update_JplusAB(self):
        
        if self.net_type == 'hom':
            
            self.JplusEE = 1.0       # EE intra-cluster potentiation factor
            self.JplusII = 1.0       # II intra-cluster potentiation factor
            self.JplusEI = 1.0       # EI intra-cluster potentiation factor
            self.JplusIE = 1.0       # IE intra-cluster potentiation factor
                        
        
    #---------------------------------------------------------------------------------
    # SET EXTERNAL INPUTS
    #---------------------------------------------------------------------------------    
    def set_external_inputs(self, random_seed):
        
        # set random number generator using the specified seed
        rng = np.random.default_rng(random_seed)

        # input to excitatory
        ze = rng.normal(loc = self.mean_nu_ext_e_offset, \
                        scale = self.sd_nu_ext_e_pert, \
                        size = self.N_e)
        
        self.nu_ext_e = self.mean_nu_ext_e + ze*self.mean_nu_ext_e
        
        
        # find inputs that become negative and set to zero
        neg_input_inds = np.where(self.nu_ext_e < 0)[0]
        self.nu_ext_e[neg_input_inds] = 0
        
        
        # input to inhibitory
        zi = rng.normal(loc = self.mean_nu_ext_i_offset, \
                        scale = self.sd_nu_ext_i_pert, \
                        size = self.N_i)
            
        self.nu_ext_i = self.mean_nu_ext_i + zi*self.mean_nu_ext_i
        
        
        # find inputs that become negative and set to zero
        neg_input_inds = np.where(self.nu_ext_i < 0)[0]
        self.nu_ext_i[neg_input_inds] = 0        
        

    #---------------------------------------------------------------------------------
    # ONLY KEEP STIMULUS PARAMETERS RELEVANT FOR PARTICULAR SHAPE
    #---------------------------------------------------------------------------------     
    def update_stim_params(self):    
    
        if ((self.stim_shape == 'diff2exp')):
            
            self.stim_duration = []
                            
    
    #---------------------------------------------------------------------------------
    # COMPUTE MAX STIMULUS STRENGTH
    # SET TO BE SOME FRACTION OF THE BASELINE EXTERNAL RATE mean_nu_ext
    #--------------------------------------------------------------------------------- 
    def set_max_stim_rate(self):
        
        self.stimRate_E = self.stim_rel_amp*self.mean_nu_ext_e
        self.stimRate_I = self.stim_rel_amp*self.mean_nu_ext_i
        
    #---------------------------------------------------------------------------------
    # COMPUTE WHICH NEURONS ARE STIMULATED 
    #---------------------------------------------------------------------------------    
    def get_stimulated_neurons(self, random_seed, clust_sizeE, clust_sizeI):       
        
        # boolean arrays that denote which neurons receive stimulus
        self.stim_Ecells = np.zeros(self.N_e)
        self.stim_Icells = np.zeros(self.N_i)
        
        # set random number generator using the specified seed
        if random_seed == 'random':
            random_seed = np.random.choice(10000,1)
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng(random_seed)

        # get selective cluster ids
        selectiveClusters = self.selectiveClusters        
        
        # get assignment of neurons to clusters
        Ecluster_inds, Icluster_inds = fcn_compute_cluster_assignments(clust_sizeE,clust_sizeI)
        
        # loop over selective clusters
        for cluInd in selectiveClusters:
            
            #---------- Ecells -----------#
            
            # cells in this cluster
            cells_in_clu = np.where(Ecluster_inds == cluInd)[0]
            
            # number to select
            nstim = np.round(self.f_Ecells_target*np.size(cells_in_clu),0).astype(int)
            
            # randomly select fraction of them
            stim_cells = rng.choice(cells_in_clu, \
                                    size = nstim, \
                                    replace=False)
            
            # update array
            self.stim_Ecells[stim_cells] = True
            
            
            #---------- Icells -----------#
            
            # cells in this cluster
            cells_in_clu = np.where(Icluster_inds == cluInd)[0]
            
            # number to select
            nstim = np.round(self.f_Icells_target*np.size(cells_in_clu),0).astype(int)
            
            # randomly select fraction of them
            stim_cells = rng.choice(cells_in_clu, \
                                    size = nstim, \
                                    replace=False)
            
            # update array
            self.stim_Icells[stim_cells] = True       
           

        