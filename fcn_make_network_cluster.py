import sys
import numpy as np

#-----------------------------------------------------------------------------
# SCRIPT TO COMPUTE A CLUSTERED SYNAPTIC CONNECTIVITY MATRIX J
# BOTH E AND I NEURONS
# THIS FUNCTION DEPRESSES ALL WEIGHTS TO/FROM BACKGROUND POPN 
# (EXCEPT FOR BACKGROUND -- BACKGROUND)

# GETS PARAMETERS FROM SIM_PARAMS AND THEN CREATES NETWORK
# NOTE: WILL GET SMALL DISCEPANICES IN WEIGHT BEFORE AND AFTER CLUSTERS IF
# pab*Nb is not a whole number, where Nc=number of units in cluster
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# REFERENCES
#-----------------------------------------------------------------------------
# 1) WYRICK AND MAZZUCATO 2020
# 2) AMIT AND BRUNEL, CEREBRAL CORTEX, 1997



def fcn_compute_depressFactors(sim_params):
    
    if sim_params.depress_interCluster == False:
        
        gEE = 1.
        gEI = 1.
        gIE = 1.
        gII = 1.
    
    else:
    
        p = sim_params.p                                        # number of clusters
        bgrE = sim_params.bgrE                                  # fraction of background neurons of type E or I
        bgrI = sim_params.bgrI
            
        # depression & potentiation
        JplusEE = sim_params.JplusEE                            # EE intra-cluster potentiation factor
        JplusII = sim_params.JplusII                            # II intra-cluster potentiation factor
        JplusEI = sim_params.JplusEI                            # EI intra-cluster potentiation factor
        JplusIE = sim_params.JplusIE                            # IE intra-cluster potentiation factor
        
        # fraction of E/I neurons per clusters
        fE = (1-bgrE)/p  
        fI = (1-bgrI)/p       
    
    
        # potentiation and depression    
        if fI == 0:
            gII = 1
        else:
            gII = (fI + fI - p*fI*fI - fI*fI*JplusII)/(fI + fI - p*fI*fI - fI*fI)
                   
        if fE == 0:
            gEE = 1
        else:
            gEE = (fE + fE - p*fE*fE - fE*fE*JplusEE)/(fE + fE - p*fE*fE - fE*fE)
                
        if (fI==0 and fE==0):
            gEI = 1
            gIE = 1
        else:
            gEI = (fI + fE - p*fI*fE - fI*fE*JplusEI)/(fI + fE - p*fI*fE - fI*fE)
            gIE = (fI + fE - p*fI*fE - fI*fE*JplusIE)/(fI + fE - p*fI*fE - fI*fE)            
        
        
    return gEE, gEI, gIE, gII



def fcn_make_network_cluster(sim_params, rand_seed=-1):
    
    #-----------------------------------------------------------------------------
    # SEED FOR NETWORK RANDOMNESS
    #-----------------------------------------------------------------------------
    if rand_seed==-1:
        # set random seed
        seed = np.random.randint(0,1000,1)
        rng = np.random.default_rng(seed)
    else:
        # set random number generator using the specified seed
        rng = np.random.default_rng(rand_seed)
 
    #-----------------------------------------------------------------------------
    # NETWORK PARAMS
    #-----------------------------------------------------------------------------              
    N_e = sim_params.N_e                                    # E neurons
    N_i = sim_params.N_i                                    # I neurons

    #-----------------------------------------------------------------------------
    # CLUSTER PARAMS
    #-----------------------------------------------------------------------------
    p = sim_params.p                                        # number of clusters
    bgrE = sim_params.bgrE                                  # fraction of background neurons of type E or I
    bgrI = sim_params.bgrI
    Ecluster_weightSize = sim_params.Ecluster_weightSize    # weight within-cluster E weights by cluster size

    # which neurons & weights are clustered
    clusters = sim_params.clusters
    clusterWeights = sim_params.clusterWeights

    # cluster size heterogeneity ('hom' or 'het')
    clustE = sim_params.clustE                              # E clusters 
    clustI = sim_params.clustI                              # I clusters 
    clust_std = sim_params.clust_std                        # std of cluster size (as a fraction of mean)

    #-----------------------------------------------------------------------------
    # SYNAPTIC WEIGHT PARAMS
    #-----------------------------------------------------------------------------

    # baseline weights
    Jee = sim_params.Jee
    Jii = -sim_params.Jii
    Jie = sim_params.Jie
    Jei = -sim_params.Jei

    # SDs of synpatic weights
    deltaEE = sim_params.deltaEE  
    deltaEI = sim_params.deltaEI   
    deltaIE = sim_params.deltaIE   
    deltaII = sim_params.deltaII   

    # depression & potentiation
    JplusEE = sim_params.JplusEE                            # EE intra-cluster potentiation factor
    JplusII = sim_params.JplusII                            # II intra-cluster potentiation factor
    JplusEI = sim_params.JplusEI                            # EI intra-cluster potentiation factor
    JplusIE = sim_params.JplusIE                            # IE intra-cluster potentiation factor

    #-----------------------------------------------------------------------------
    # CONNECTIVITY PARAMS
    #-----------------------------------------------------------------------------
    pee = sim_params.pee
    pei = sim_params.pei
    pii = sim_params.pii
    pie = sim_params.pie
    
    #-----------------------------------------------------------------------------
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # MAKE NETWORK
    #-----------------------------------------------------------------------------
    

    # check that there's a whole number of backgroud neurons in each popn
    if ( (np.mod(N_e*bgrE,1)!=0.0) or (np.mod(N_i*bgrI,1)!=0.0) ):
        sys.exit('not a whole number of background neurons')

    # check that number of clusters is a factor of number available E and I
    # neurons to be clustered
    if ( abs((N_e*(1-bgrE)/p)-round((N_e*(1-bgrE)/p))) > 1e-8 or abs((N_i*(1-bgrI)/p)-round((N_i*(1-bgrI)/p))) > 1e-8 ):   
        sys.exit('number of clusters is not a factor of the number of'\
                  'E or I neurons to be clustered')
            
    # check that if the weights arent clustered, background frction =1
    if ('I' not in clusters):
        sim_params.bgrI = 1.0
        
    if ('E' not in clusters):
        sim_params.bgrE = 1.0
        
    # number of background neurons
    nbE = round(N_e*bgrE)
    nbI = round(N_i*bgrI)

    # fraction of E/I neurons per clusters
    fE = (1-bgrE)/p  
    fI = (1-bgrI)/p       

    '''
    # check that #cells/cluster x connection probability is whole number of for fixed in degree case
    if sim_params.connType == 'fixed_InDegree':
        
        a = fE*N_e*pee
        b = fE*N_e*pie
        c = fI*N_i*pei
        d = fI*N_i*pii
        
        if ((a).is_integer() == False or (b).is_integer() == False or \
            (c).is_integer() == False or (d).is_integer() == False):
            sys.exit('cluster sizes such that they cant receive whole number of inputs from themselves')
    '''

    # potentiation/depression factor
    gEE, gEI, gIE, gII  = fcn_compute_depressFactors(sim_params) 
    
    #-----------------------------------------------------------------------------
    # E CLUSTERS
    #-----------------------------------------------------------------------------
    NcUnits = int((N_e*(1-bgrE))/p)     # number of Exc units per cluster
    if ('E' in clusters):
        # homogeneous clusters: all have the same size
        if (clustE == 'hom' or clust_std == 0):           
            popsizeE = NcUnits*np.ones(p).astype(int) # row vector of population sizes
            popsizeE = np.hstack((popsizeE,nbE))      # concatenate size of background
        # heterogeneous clusters: different sizes   
        elif (clustE == 'het'): 
            Nc_trial = rng.uniform(0,1,1) # initialize random number of neurons in clusters
            # distribute neurons into clusters with certain mean and std
            # until correct number of neurons have been used and there no
            # negative sizes
            while ( ( np.sum(Nc_trial)-(N_e-nbE)!=0 ) or ( np.any(Nc_trial<0) ) ):
                Nc_trial=np.round(rng.normal(NcUnits, NcUnits*clust_std, p)) 
    
            popsizeE = Nc_trial.astype(int) # array of cluster sizes
            popsizeE = np.hstack((popsizeE,nbE))      # concatenate size of background
        else:
            sys.exit('not a valid option for clustE')
    
        cusumNcE = np.concatenate(( (np.zeros(1).astype(int)), np.cumsum(popsizeE) ))
    
    else:
        popsizeE = [N_e]
        cusumNcE = np.concatenate(( (np.zeros(1).astype(int)), np.cumsum(popsizeE) ))

    #-----------------------------------------------------------------------------
    # ICLUSTERS
    #-----------------------------------------------------------------------------
    NcUnits = int(fI*N_i)        # number of inh units per cluster
    if ('I' in clusters):
        if (clustI == 'hom'):
            popsizeI = NcUnits*np.ones(p).astype(int) # vector of population sizes
            popsizeI = np.hstack((popsizeI,nbI))      # concatenate size of background
        elif (clustI == 'het'):
            Nc_trial = rng.uniform(0,1,1) # initialize random number of neurons in clusters
            while ( ( np.sum(Nc_trial)-(N_i-nbI)!=0 ) or ( np.any(Nc_trial<0) ) ):
                Nc_trial=np.round(rng.normal(NcUnits, NcUnits*clust_std, p))
            
            popsizeI = Nc_trial.astype(int) # array of cluster sizes
            popsizeI = np.hstack((popsizeI,nbI))      # concatenate size of background
        else:
            sys.exit('not a valid option for clustI')
  
        cusumNcI = np.concatenate(( (np.zeros(1).astype(int)), np.cumsum(popsizeI) ))
        
    else:
        popsizeI = [N_i]
        cusumNcI = np.concatenate(( (np.zeros(1).astype(int)), np.cumsum(popsizeI) ))
        
    #--------------------------------------------------------------------------
    # POTENTIATION AND DEPRESSION FACTORS
    #--------------------------------------------------------------------------
    # E --> E
    if ('EE' in clusterWeights):
        jee_out = gEE*Jee
        if jee_out < 0:
            sys.exit('error : got negative weights for JEE')
        if (Ecluster_weightSize == 0):
            jee_in = JplusEE*Jee*np.ones(p) 
        else:
            # re-weight by size of cluster 
            jee_in = JplusEE*Jee
            jee_in = jee_in*(np.divide(np.mean(popsizeE[0:p]),popsizeE[0:p]))
    else:
        jee_out = Jee
        jee_in = Jee

    # I --> I
    if ('II' in clusterWeights):
        jii_out = gII*Jii              # inter-cluster
        jii_in = JplusII*Jii           # intra-cluster
        if jii_out >= 0 or jii_in >= 0:
            sys.exit('error : got positive weights for JII')
    else:
        jii_out = Jii
        jii_in = Jii
        
    # I --> E
    if ('EI' in clusterWeights):
        jei_out = gEI*Jei         # inter-cluster
        jei_in = JplusEI*Jei           # intra-cluster
        if jei_out >=0 or jei_in >=0:
            sys.exit('error : got positive weights for JEI')
    else:
        jei_out = Jei
        jei_in = Jei

    # E --> I
    if ('IE' in clusterWeights):
        jie_out = gIE*Jie         # inter-cluster
        jie_in = JplusIE*Jie           # intra-cluster
        if jie_out < 0 or jie_in < 0:
            sys.exit('error : got negative weights for JIE')
    else:
        jie_out = Jie
        jie_in = Jie
        
    #------------------------------------------------------------------------
    # MAKE DIFFERENT BLOCKS OF SYNPATIC WEIGHT MATRIX
    #------------------------------------------------------------------------
    
    #------------------------------------------------------------------------
    # INITIALIZE BASELINE WEIGHTS FIRST W/0 POTENTIATION OR DEPRESSION
    # THEN ADD IN NEXT LINES
    #------------------------------------------------------------------------
    #generate a distribution of synaptic weights with mean J and variance delta^2 J^2
    #second term takes into account connection probabilities
    weights = Jee*(np.ones((N_e,N_e))+deltaEE*rng.normal(0,1,(N_e,N_e)))
    if sim_params.connType == 'fixed_P':
        connMat = rng.uniform(0,1,(N_e,N_e))<pee
    else:
        connMat = np.zeros((N_e,N_e))
        for i in range(0,N_e,1):
            options = np.arange(N_e)
            options = np.delete(options,i) # avoid self connections
            randConnections = rng.choice(options,int(pee*N_e),replace=False)
            connMat[i,randConnections] = 1
    JEE = np.multiply(weights, connMat)
    
    
    weights = Jei*(np.ones((N_e,N_i))+deltaEI*rng.normal(0,1,(N_e,N_i)))
    if sim_params.connType == 'fixed_P':
        connMat = rng.uniform(0,1,(N_e,N_i))<pei
    else:
        connMat = np.zeros((N_e,N_i))
        for i in range(0,N_e,1):  
            randConnections = rng.choice(N_i,int(pei*N_i),replace=False)
            connMat[i,randConnections] = 1
    JEI = np.multiply(weights, connMat)
    
    
    weights = Jie*(np.ones((N_i,N_e))+deltaIE*rng.normal(0,1,(N_i,N_e)))
    if sim_params.connType == 'fixed_P':
        connMat = rng.uniform(0,1,(N_i,N_e))<pie
    else:
        # ensure each I neuron receives same number of inputs from each E neuron type if E clusters are present
        if('E' in clusters and 'I' not in clusters): 
            connMat = np.zeros((N_i,N_e))
            bgrE_inds = np.arange(cusumNcE[-2],cusumNcE[-1],1)
            # E clusters to I
            for i in range(0,N_i,1): 
                for clu in range(1,p+1,1): # cluster to I
                    options = np.arange(cusumNcE[clu-1],cusumNcE[clu],1)
                    randConnections = rng.choice(options,int(pie*N_e*fE),replace=False)
                    connMat[i,randConnections] = 1
                # background E to I
                options = bgrE_inds
                randConnections = rng.choice(options,int(pie*N_e*bgrE),replace=False)
                connMat[i,randConnections] = 1
        else:
            connMat = np.zeros((N_i,N_e))
            for i in range(0,N_i,1): 
                randConnections = rng.choice(N_e,int(pie*N_e),replace=False)
                connMat[i,randConnections] = 1                               
    JIE = np.multiply(weights, connMat)
    
    
    weights = Jii*(np.ones((N_i,N_i))+deltaII*rng.normal(0,1,(N_i,N_i)))
    if sim_params.connType == 'fixed_P':
        connMat = rng.uniform(0,1,(N_i,N_i))<pii
    else:
        connMat = np.zeros((N_i,N_i))
        for i in range(0,N_i,1):  
            options = np.arange(N_i)
            options = np.delete(options,i) # avoid self connections
            randConnections = rng.choice(options,int(pii*N_i),replace=False)
            connMat[i,randConnections] = 1
    JII = np.multiply(weights, connMat)
    
    #-------------------------------------------------------------------------
    # EE weights
    #-------------------------------------------------------------------------
    if ('E' in clusters):
        # depressed inter-cluster weights (jee_out)
        if sim_params.connType == 'fixed_P':
            weights = jee_out*(np.ones((N_e,N_e))+deltaEE*rng.normal(0,1,(N_e,N_e)))
            connMat = rng.uniform(0,1,(N_e,N_e))<pee
            JEE = np.multiply(weights, connMat)
        else:
            weights = np.ones((N_e,N_e))+deltaEE*rng.normal(0,1,(N_e,N_e))
            JEE = np.zeros((N_e, N_e))
            bgrE_inds = np.arange(cusumNcE[-2],cusumNcE[-1],1)
            for clu in range(1,p+1,1):
                inCluster_inds = np.arange(cusumNcE[clu-1],cusumNcE[clu],1)
                for i in range(0,popsizeE[clu-1],1):
                    ind = inCluster_inds[i]
                    # between different clusters
                    options = np.arange(N_e)
                    removeInds = np.hstack((inCluster_inds,bgrE_inds))
                    options = np.delete(options,removeInds) # avoid self connections
                    randConnections = rng.choice(options,int(pee*N_e*(p-1)*fE),replace=False)
                    JEE[ind,randConnections] = jee_out*weights[ind,randConnections]
                    # background to cluster
                    options = bgrE_inds
                    randConnections = rng.choice(options,int(pee*N_e*bgrE),replace=False)
                    JEE[ind,randConnections] = jee_out*weights[ind,randConnections]      
            # cluster to background
            for i in range(0,len(bgrE_inds),1):
                ind = bgrE_inds[i]
                for clu in range(1,p+1,1):
                    inCluster_inds = np.arange(cusumNcE[clu-1],cusumNcE[clu],1)
                    options = inCluster_inds
                    randConnections = rng.choice(options,int(pee*N_e*fE),replace=False)
                    JEE[ind,randConnections] = jee_out*weights[ind,randConnections]
            
                       
        # keep background-to-background unchanged 
        if sim_params.connType == 'fixed_P':
            weights = Jee*(np.ones((nbE,nbE))+deltaEE*rng.normal(0,1,(nbE,nbE)))
            connMat = rng.uniform(0,1,(nbE,nbE))<pee   
            JEE[cusumNcE[-2]:N_e,cusumNcE[-2]:N_e]= np.multiply(weights, connMat)
        else:
            bgrE_inds = np.arange(cusumNcE[-2],cusumNcE[-1],1)
            for i in range(0,len(bgrE_inds),1):
                ind = bgrE_inds[i]
                options = bgrE_inds
                removeInd = np.where(options==ind)
                options = np.delete(options,removeInd)
                randConnections = rng.choice(options,int(pee*N_e*bgrE),replace=False)
                JEE[ind,randConnections] = Jee*weights[ind,randConnections]
        
              
        # higher intra cluster weights
        # loop over clusters
        for clu in range(1,p+1,1):
            if sim_params.connType == 'fixed_P':
                weights = jee_in[clu-1]*(np.ones((popsizeE[clu-1],popsizeE[clu-1]))+\
                          deltaEE*rng.normal(0,1,(popsizeE[clu-1],popsizeE[clu-1])))
                connMat = rng.uniform(0,1,(popsizeE[clu-1],popsizeE[clu-1]))<pee
                JEE[cusumNcE[clu-1]:cusumNcE[clu],cusumNcE[clu-1]:cusumNcE[clu]]= np.multiply(weights, connMat) 
            else:
                inCluster_inds = np.arange(cusumNcE[clu-1],cusumNcE[clu],1)
                for i in range(0,popsizeE[clu-1],1):
                    ind = inCluster_inds[i]
                    # within same clusters
                    options = inCluster_inds
                    removeInds = np.where(options==ind)
                    options = np.delete(options,removeInds) # avoid self connections
                    randConnections = rng.choice(options,int(pee*N_e*fE),replace=False)
                    JEE[ind,randConnections] = jee_in[clu-1]*weights[ind,randConnections]                 
    
    #-------------------------------------------------------------------------
    # EI weights
    #-------------------------------------------------------------------------
    if ('E' in clusters and 'I' in clusters):
        # depressed inter-cluster weights (jei_out)
        if sim_params.connType == 'fixed_P':
            weights = jei_out*(np.ones((N_e,N_i))+deltaEI*rng.normal(0,1,(N_e,N_i)))
            connMat = rng.uniform(0,1,(N_e,N_i))<pei
            JEI = np.multiply(weights, connMat)
        else:
            weights = np.ones((N_e,N_i))+deltaEI*rng.normal(0,1,(N_e,N_i))
            JEI = np.zeros((N_e, N_i))
            bgrI_inds = np.arange(cusumNcI[-2],cusumNcI[-1],1)
            for clu in range(1,p+1,1):
                inCluster_indsI = np.arange(cusumNcI[clu-1],cusumNcI[clu],1)
                inCluster_indsE = np.arange(cusumNcE[clu-1],cusumNcE[clu],1)
                for i in range(0,popsizeE[clu-1],1):
                    ind = inCluster_indsE[i]
                    # between different clusters
                    options = np.arange(N_i)
                    removeInds = np.hstack((inCluster_indsI,bgrI_inds))
                    options = np.delete(options,removeInds) # avoid self connections
                    randConnections = rng.choice(options,int(pei*N_i*(p-1)*fI),replace=False)
                    JEI[ind,randConnections] = jei_out*weights[ind,randConnections]
                    # background
                    options = bgrI_inds
                    randConnections = rng.choice(options,int(pei*N_i*bgrI),replace=False)
                    JEI[ind,randConnections] = jei_out*weights[ind,randConnections]  
            # cluster to background
            for i in range(0,len(bgrE_inds),1):
                ind = bgrE_inds[i]
                for clu in range(1,p+1,1):
                    inCluster_indsI = np.arange(cusumNcI[clu-1],cusumNcI[clu],1)
                    options = inCluster_indsI
                    randConnections = rng.choice(options,int(pei*N_i*fI),replace=False)
                    JEI[ind,randConnections] = jei_out*weights[ind,randConnections]
                                
        
        # keep background-to-background unchanged
        if sim_params.connType == 'fixed_P':
            weights = Jei*(np.ones((nbE,nbI))+deltaEI*rng.normal(0,1,(nbE,nbI)))
            connMat = rng.uniform(0,1,(nbE,nbI))<pei
            JEI[cusumNcE[-2]:N_e,cusumNcI[-2]:N_i]= np.multiply(weights, connMat)
        else:
            bgrE_inds = np.arange(cusumNcE[-2],cusumNcE[-1],1)
            bgrI_inds = np.arange(cusumNcI[-2],cusumNcI[-1],1)
            for i in range(0,len(bgrE_inds),1):
                ind = bgrE_inds[i]
                options = bgrI_inds
                randConnections = rng.choice(options,int(pei*N_i*bgrI),replace=False)
                JEI[ind,randConnections] = Jei*weights[ind,randConnections]
        
            
        # higher intra cluster weights
        # loop over clusters
        for clu in range(1,p+1,1):
            if sim_params.connType == 'fixed_P':
                weights = jei_in*(np.ones((popsizeE[clu-1],popsizeI[clu-1]))+\
                  deltaEI*rng.normal(0,1,(popsizeE[clu-1],popsizeI[clu-1])))
                connMat = rng.uniform(0,1,(popsizeE[clu-1],popsizeI[clu-1]))<pei
                JEI[cusumNcE[clu-1]:cusumNcE[clu],cusumNcI[clu-1]:cusumNcI[clu]]= np.multiply(weights, connMat)
            else:
                inCluster_indsI = np.arange(cusumNcI[clu-1],cusumNcI[clu],1)
                inCluster_indsE = np.arange(cusumNcE[clu-1],cusumNcE[clu],1)
                for i in range(0,popsizeE[clu-1],1):
                    ind = inCluster_indsE[i]
                    # within same clusters
                    options = inCluster_indsI
                    randConnections = rng.choice(options,int(pei*N_i*fI),replace=False)
                    JEI[ind,randConnections] = jei_in*weights[ind,randConnections]                    
    
    #-------------------------------------------------------------------------
    # IE weights
    #-------------------------------------------------------------------------
    if ('E' in clusters and 'I' in clusters):
        # depressed inter-cluster weights (jie_out)
        if sim_params.connType == 'fixed_P':
            weights = jie_out*(np.ones((N_i,N_e))+deltaIE*rng.normal(0,1,(N_i,N_e)))
            connMat = rng.uniform(0,1,(N_i,N_e))<pie
            JIE = np.multiply(weights, connMat)
        else:
            weights = np.ones((N_i,N_e))+deltaIE*rng.normal(0,1,(N_i,N_e))
            JIE = np.zeros((N_i, N_e))
            bgrE_inds = np.arange(cusumNcE[-2],cusumNcE[-1],1)
            for clu in range(1,p+1,1):
                inCluster_indsI = np.arange(cusumNcI[clu-1],cusumNcI[clu],1)
                inCluster_indsE = np.arange(cusumNcE[clu-1],cusumNcE[clu],1)
                for i in range(0,popsizeI[clu-1],1):
                    ind = inCluster_indsI[i]
                    # between different clusters
                    options = np.arange(N_e)
                    removeInds = np.hstack((inCluster_indsE, bgrE_inds))
                    options = np.delete(options,removeInds) # avoid self connections
                    randConnections = rng.choice(options,int(pie*N_e*(p-1)*fE),replace=False)
                    JIE[ind,randConnections] = jie_out*weights[ind,randConnections]
                    # background
                    options = bgrE_inds
                    randConnections = rng.choice(options,int(pie*N_e*bgrE),replace=False)
                    JIE[ind,randConnections] = jie_out*weights[ind,randConnections]    
            # cluster to background
            for i in range(0,len(bgrI_inds),1):
                ind = bgrI_inds[i]
                for clu in range(1,p+1,1):
                    inCluster_indsE = np.arange(cusumNcE[clu-1],cusumNcE[clu],1)
                    options = inCluster_indsE
                    randConnections = rng.choice(options,int(pie*N_e*fE),replace=False)
                    JIE[ind,randConnections] = jie_out*weights[ind,randConnections]  
                
                    
        # keep background-to-background unchanged
        if sim_params.connType == 'fixed_P':
            weights = Jie*(np.ones((nbI,nbE))+deltaIE*rng.normal(0,1,(nbI,nbE)))
            connMat = rng.uniform(0,1,(nbI,nbE))<pie
            JIE[cusumNcI[-2]:N_i,cusumNcE[-2]:N_e] = np.multiply(weights, connMat)

        else:
            bgrE_inds = np.arange(cusumNcE[-2],cusumNcE[-1],1)
            bgrI_inds = np.arange(cusumNcI[-2],cusumNcI[-1],1)
            for i in range(0,len(bgrI_inds),1):
                ind = bgrI_inds[i]
                options = bgrE_inds
                randConnections = rng.choice(options,int(pie*N_e*bgrE),replace=False)
                JIE[ind,randConnections] = Jie*weights[ind,randConnections]
    
        # higher intra cluster weights
        # loop over clusters
        for clu in range(1,p+1,1):
            if sim_params.connType == 'fixed_P':
                weights = jie_in*(np.ones((popsizeI[clu-1],popsizeE[clu-1]))+\
                  deltaIE*rng.normal(0,1,(popsizeI[clu-1],popsizeE[clu-1])))
                connMat = rng.uniform(0,1,(popsizeI[clu-1],popsizeE[clu-1]))<pie
                JIE[cusumNcI[clu-1]:cusumNcI[clu],cusumNcE[clu-1]:cusumNcE[clu]] = np.multiply(weights, connMat)   
            else:
                inCluster_indsI = np.arange(cusumNcI[clu-1],cusumNcI[clu],1)
                inCluster_indsE = np.arange(cusumNcE[clu-1],cusumNcE[clu],1)
                for i in range(0,popsizeI[clu-1],1):
                    ind = inCluster_indsI[i]
                    # within same clusters
                    options = inCluster_indsE
                    randConnections = rng.choice(options,int(pie*N_e*fE),replace=False)
                    JIE[ind,randConnections] = jie_in*weights[ind,randConnections]                     
         
    #-------------------------------------------------------------------------
    # II weights
    #-------------------------------------------------------------------------
    if ('E' in clusters and 'I' in clusters):
        # depressed inter-cluster weights (jii_out)
        if sim_params.connType == 'fixed_P':
            weights = jii_out*(np.ones((N_i,N_i))+deltaII*rng.normal(0,1,(N_i,N_i)))
            connMat = rng.uniform(0,1,(N_i,N_i))<pii
            JII = np.multiply(weights, connMat)
        else:
            weights = np.ones((N_i,N_i))+deltaII*rng.normal(0,1,(N_i,N_i))
            JII = np.zeros((N_i, N_i))
            bgrI_inds = np.arange(cusumNcI[-2],cusumNcI[-1],1)
            for clu in range(1,p+1,1):
                inCluster_indsI = np.arange(cusumNcI[clu-1],cusumNcI[clu],1)
                for i in range(0,popsizeI[clu-1],1):
                    ind = inCluster_indsI[i]
                    # between different clusters
                    options = np.arange(N_i)
                    removeInds = np.hstack((inCluster_indsI,bgrI_inds))
                    options = np.delete(options,removeInds) # avoid self connections
                    options = options
                    randConnections = rng.choice(options,int(pii*N_i*(p-1)*fI),replace=False)
                    JII[ind,randConnections] = jii_out*weights[ind,randConnections]
                    # background
                    options = bgrI_inds
                    randConnections = rng.choice(options,int(pii*N_i*bgrI),replace=False)
                    JII[ind,randConnections] = jii_out*weights[ind,randConnections] 
            # cluster to background
            for i in range(0,len(bgrI_inds),1):
                ind = bgrI_inds[i]
                for clu in range(1,p+1,1):
                    inCluster_indsI = np.arange(cusumNcI[clu-1],cusumNcI[clu],1)
                    options = inCluster_indsI
                    randConnections = rng.choice(options,int(pii*N_i*fI),replace=False)
                    JII[ind,randConnections] = jii_out*weights[ind,randConnections]        
 
                           
        # keep background-to-background unchanged 
        if sim_params.connType == 'fixed_P':
            weights = Jii*(np.ones((nbI,nbI))+deltaII*rng.normal(0,1,(nbI,nbI)))
            connMat = rng.uniform(0,1,(nbI,nbI))<pii   
            JII[cusumNcI[-2]:N_i,cusumNcI[-2]:N_i]= np.multiply(weights, connMat)
        else:
            bgrI_inds = np.arange(cusumNcI[-2],cusumNcI[-1],1)
            for i in range(0,len(bgrI_inds),1):
                ind = bgrI_inds[i]
                options = bgrI_inds
                removeInd = np.where(options==ind)
                options = np.delete(options,removeInd)
                randConnections = rng.choice(options,int(pii*N_i*bgrI),replace=False)
                JII[ind,randConnections] = Jii*weights[ind,randConnections]           
       
        # higher intra cluster weights
        # loop over clusters
        for clu in range(1,p+1,1):
            if sim_params.connType == 'fixed_P':
                weights = jii_in*(np.ones((popsizeI[clu-1],popsizeI[clu-1]))+\
                      deltaII*rng.normal(0,1,(popsizeI[clu-1],popsizeI[clu-1])))
                connMat = rng.uniform(0,1,(popsizeI[clu-1],popsizeI[clu-1]))<pii
                JII[cusumNcI[clu-1]:cusumNcI[clu],cusumNcI[clu-1]:cusumNcI[clu]] = np.multiply(weights, connMat)
            else:
                inCluster_inds = np.arange(cusumNcI[clu-1],cusumNcI[clu],1)
                for i in range(0,popsizeI[clu-1],1):
                    ind = inCluster_inds[i]
                    # within same clusters
                    options = inCluster_inds
                    removeInds = np.where(options==ind)
                    options = np.delete(options,removeInds) # avoid self connections
                    randConnections = rng.choice(options,int(pii*N_i*fI),replace=False)
                    JII[ind,randConnections] = jii_in*weights[ind,randConnections] 
               
                

    if ('E' not in clusters and 'I' in clusters):
        sys.exit('this code currently doesnt support I clusters only!')
        
    #-------------------------------------------------------------------------
    # FINALIZE NETWORK
    #-------------------------------------------------------------------------      
    # concatenate
    JE = np.hstack((JEE,JEI))
    JI = np.hstack((JIE,JII))
    J = np.vstack((JE,JI))
    # remove self couplings
    np.fill_diagonal(J,0)
    
    #-------------------------------------------------------------------------
    # CHECKS
    #-------------------------------------------------------------------------         
    if sim_params.connType!='fixed_P':
        B=J!=0
        Kin = np.sum(B,1)
        Kin_equal_E = np.all(Kin[:N_e] == Kin[0])
        Kin_equal_I = np.all(Kin[N_e:] == Kin[N_e])
        if (Kin_equal_E == False or Kin_equal_I == False):
            print('error')
            sys.exit('in degrees are not equal for all E/I neurons!')
        
     
    #-------------------------------------------------------------------------
    # RETURN
    #------------------------------------------------------------------------- 
    return J, popsizeE, popsizeI
    

#%% compute which cluster each neuron belongs to given vector of cluster sizes
def fcn_compute_cluster_assignments(popsizeE, popsizeI):
    
    # number of E and I neurons
    Ne = np.sum(popsizeE)
    Ni = np.sum(popsizeI)
    
    # initialize outputs
    Ecluster_ids = np.zeros(Ne)
    Icluster_ids = np.zeros(Ni)
    
    # population start and end indices [E]
    pops_start_end = np.append(0, np.cumsum(popsizeE))
    
    # number of populations
    npops = np.size(pops_start_end)-1
        
    # loop over populations
    for popInd in range(0,npops,1):
    
        # cluster start and end
        startID = pops_start_end[popInd]
        endID = pops_start_end[popInd+1]
        
        Ecluster_ids[startID:endID] = popInd
        
    # population start and end indices [I]
    pops_start_end = np.append(0, np.cumsum(popsizeI))
    
    # number of populations
    npops = np.size(pops_start_end)-1
    
    # loop over populations
    for popInd in range(0,npops,1):
    
        # cluster start and end
        startID = pops_start_end[popInd]
        endID = pops_start_end[popInd+1]
        
        Icluster_ids[startID:endID] = popInd       
        
    return Ecluster_ids, Icluster_ids
        
        
        
        
    
