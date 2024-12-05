import numpy as np
import sys
import fcn_stimulation



#%%

'''
run a simulation of an LIF neuron with exponential synapses

use exact updates between time steps

assumes that baseline external inputs are either
    homogeneous poisson spikes, filtered through an exponential synapse
    constant, goes directly to voltage equation
    
assumes that external stimulus enters voltage equation, not poisson

this function yields results that are exactly consistent with NEST when the 
external inputs are constant (i.e., deterministic case)

equations take the form of Brunel & Sergi 1998


'''


def fcn_simulate_exact_poisson(sim_params, J, rand_seed):

#------------------------------------------------------------------------------
# GET PARAMETERS
#------------------------------------------------------------------------------

    save_voltage = sim_params.save_voltage # whether or not to save voltage array


    T0 = sim_params.T0                  # simulation start time
    TF = sim_params.TF                  # simulation end time
    dt = sim_params.dt                  # time step

    N = sim_params.N                    # total number of neurons
    Ne = sim_params.N_e                 # number excitatory
    Ni = sim_params.N_i                 # number inhibitory
    
    Vth_e = sim_params.Vth_e            # threshold potential E
    Vth_i = sim_params.Vth_i            # threshold potential I
    
    Vr_e = sim_params.Vr_e              # reset potential E
    Vr_i = sim_params.Vr_i              # reset potential E
    
    tau_r = sim_params.tau_r            # refractory period
    tau_m_e = sim_params.tau_m_e        # membrane time constant E
    tau_m_i = sim_params.tau_m_i        # membrane time constant I
    tau_s_e = sim_params.tau_s_e        # synaptic time constant E
    tau_s_i = sim_params.tau_s_i        # synaptic time constant I
    
    t_delay = sim_params.t_delay        # synaptic delay

    
    pext = sim_params.pext              # external connection probability
    Jee_ext = sim_params.Jee_ext        # external E to E weight
    Jie_ext = sim_params.Jie_ext        # external E to I weight
    nu_ext_e = sim_params.nu_ext_e      # avg baseline afferent rate to E & I neurons [spk/s]
    nu_ext_i = sim_params.nu_ext_i      
       
    extCurrent_poisson = sim_params.extCurrent_poisson # whether or not external current is Poisson
    
#------------------------------------------------------------------------------
# SETUP
#------------------------------------------------------------------------------
     
    # THRESHOLD VOLTAGE VECTORS   
    Vth_e_vec = Vth_e*np.ones(Ne)
    Vth_i_vec = Vth_i*np.ones(Ni)
    Vth = np.concatenate((Vth_e_vec, Vth_i_vec))
    # RESET VOLTAGE VECTOR
    Vr_e_vec = Vr_e*np.ones(Ne)
    Vr_i_vec = Vr_i*np.ones(Ni)
    Vr = np.concatenate((Vr_e_vec, Vr_i_vec))
    # MEMBRANE TIME CONSTANT VECTOR
    tau_m_e_vec = tau_m_e*np.ones(Ne)
    tau_m_i_vec = tau_m_i*np.ones(Ni)
    tau_m = np.concatenate((tau_m_e_vec, tau_m_i_vec))    
    # SYNAPTIC WEIGHTS  
    Jij = J
    
    
        
#------------------------------------------------------------------------------
# INITIAL CONDITIONS
#------------------------------------------------------------------------------ 

    # if seed not input to function, initial voltage is randomly distributed between reset and threshold
    rng = np.random.default_rng(rand_seed)
    iV = Vr + (Vth-Vr)*rng.uniform(size=(N)) 

    # time each neuron has left to be refractory
    time_ref = np.zeros(N)

#------------------------------------------------------------------------------
# SIMULATION STUFF
#------------------------------------------------------------------------------  

    nSteps = int((TF-T0)/dt)               # number of time steps is one less than number of time points in simulation
    spikes = np.zeros((2,1))*np.nan   # initialize array for spike times
        
#------------------------------------------------------------------------------
# SET UP PROPAGATOR FOR SUBTHRESHOLD DYNAMICS
#------------------------------------------------------------------------------ 

    propagator = np.zeros((N, 4, 4))
    
    propagator[:,0,0] = np.exp(-dt/tau_m)
    propagator[:,0,1] = (tau_s_e*tau_m)/(tau_s_e - tau_m) * ( np.exp(-dt/tau_s_e) - np.exp(-dt/tau_m) )
    propagator[:,0,2] = (tau_s_i*tau_m)/(tau_s_i - tau_m) * ( np.exp(-dt/tau_s_i) - np.exp(-dt/tau_m) )
    propagator[:,0,3] = tau_m*(1 - np.exp(-dt/tau_m))
    
    propagator[:,1,0] = 0
    propagator[:,1,1] = np.exp(-dt/tau_s_e)
    propagator[:,1,2] = 0
    propagator[:,1,3] = 0
    
    propagator[:,2,0] = 0
    propagator[:,2,1] = 0
    propagator[:,2,2] = np.exp(-dt/tau_s_i)
    propagator[:,2,3] = 0
    
    propagator[:,3,0] = 0
    propagator[:,3,1] = 0
    propagator[:,3,2] = 0
    propagator[:,3,3] = 1
    

        
#------------------------------------------------------------------------------
# STATE VARIABLES
#------------------------------------------------------------------------------ 

    if save_voltage == True:
        
        timePts = np.arange(T0,TF+dt,dt)  # time points of simulation
    
        Istim = np.zeros((N, nSteps+1))
        V = np.zeros((N, nSteps+1))
        I_exc = np.zeros((N, nSteps+1))
        I_inh = np.zeros((N, nSteps+1))
        I_o = np.zeros((N, nSteps+1))
    
        # initial conditions
        V[:, 0] = iV
        I_exc[:, 0] = 0.
        I_inh[:, 0] = 0.
        
        
        # POISSON WITH RATE C_ext*nu_ext
        if extCurrent_poisson == True:
        
            extPoisson_e_vec = np.zeros((Ne, nSteps+1))
            extPoisson_i_vec = np.zeros((Ni, nSteps+1))
            extPoisson_vec = np.zeros((N, nSteps+1))
            
        else:
            
            extPoisson_vec = np.zeros((N, nSteps+1))
            
                
    else:
        
        V = iV
        I_exc = np.zeros(N)
        I_inh = np.zeros(N)

                
#------------------------------------------------------------------------------
# PRINT
#------------------------------------------------------------------------------ 
    print('Vr_e = %0.3f' % Vr_e_vec[0])
    print('Vr_i = %0.3f' % Vr_i_vec[0])
    print('Vth_e = %0.3f' % Vth_e_vec[0])
    print('Vth_i = %0.3f' % Vth_i_vec[0])
    print('tau_m_e = %0.5f s' % tau_m_e_vec[0])
    print('tau_m_i = %0.5f s' % tau_m_i_vec[0])
    print('tau_s_e = %0.5f s' % tau_s_e)
    print('tau_s_i = %0.5f s' % tau_s_i)
    print('tau_r = %0.5f s' % tau_r)
        
#------------------------------------------------------------------------------
# INTEGRATE
#------------------------------------------------------------------------------     

    if extCurrent_poisson == False:
        
        I_const = np.zeros(N)
        I_const[:Ne] = nu_ext_e*Ne*pext*Jee_ext
        I_const[Ne:] = nu_ext_i*Ne*pext*Jie_ext

    
    # time loop
    for tInd in range(0,nSteps,1):
        
        tNow = np.round(T0 + dt*tInd,5)
        tNext = np.round(tNow + dt,5)
        
        # stimulus
        Istim_now = fcn_stimulation_setup(sim_params, tNow)
        
        # poisson
        if extCurrent_poisson == True:
            
            extPoisson_e_now = rng.poisson(nu_ext_e*Ne*pext*dt, Ne)
            extPoisson_i_now = rng.poisson(nu_ext_i*Ne*pext*dt, Ni)
        
            # poisson spike trains for all cells
            extPoisson_now = np.append(extPoisson_e_now, extPoisson_i_now)
            
        else:
            
            extPoisson_now = np.zeros(N)
                 
    
        #------------------ SUBTHRESHOLD STATE UPDATE ------------------------#
        
        if save_voltage == True:
            
            Istim[:, tInd] = Istim_now.copy()
        
            if extCurrent_poisson == True:
                
                I_o[:, tInd] = Istim_now.copy()
                
                extPoisson_e_vec[:, tInd] = extPoisson_e_now.copy()
                extPoisson_i_vec[:, tInd] = extPoisson_i_now.copy()
                extPoisson_vec[:, tInd] = extPoisson_now.copy()
                
            else:
                
                I_o[:, tInd] = Istim_now + I_const  
            
            #------------------ REFRACTORY CONDITION -----------------------------#
        
            # update sensitive ids
            sensitive_id = np.nonzero(np.round(time_ref,5) == 0)[0]
            non_sensitive_id = np.setdiff1d(np.arange(0,N), sensitive_id)
        
            # update time remaining refractory at each step
            time_ref[non_sensitive_id] = time_ref[non_sensitive_id] - dt
        
            # update the voltage of non-refractory
            if np.size(sensitive_id) > 0:
                
                V[sensitive_id, tInd+1] = propagator[sensitive_id, 0, 0] * V[sensitive_id, tInd] + \
                                          propagator[sensitive_id, 0, 1] * I_exc[sensitive_id, tInd] + \
                                          propagator[sensitive_id, 0, 2] * I_inh[sensitive_id, tInd] + \
                                          propagator[sensitive_id, 0, 3] * I_o[sensitive_id, tInd]
                               
                                        
            # update voltage of refractory
            if np.size(non_sensitive_id) > 0:
                
                V[non_sensitive_id, tInd+1] = Vr[non_sensitive_id]
              
        
            # update excitatory synaptic current
            I_exc[:, tInd+1] = propagator[:, 1, 1] * I_exc[:, tInd]
            
            # update inhibitory synaptic current
            I_inh[:, tInd+1] = propagator[:, 2, 2] * I_inh[:, tInd]
            
    
    
            #------------------ GET CELLS THAT FIRED AT CURRENT INDEX -------------#
            
            fired_ind = np.nonzero(V[:, tInd+1]>=Vth)[0]
            
            
            #------------------ STORE THESE NEW SPIKES ----------------------------#      
            
            # store spikes (row 1 = times, row 2 = neuron index)
            new_spikes = np.vstack((timePts[tInd + 1]*np.ones(np.size(fired_ind)), fired_ind))
            spikes = np.append(spikes, new_spikes, 1)
    
    
    
            #------------------ UPDATE SYNAPTIC WEIGHTS --> JUMP ------------------#
            
            # get delayed spikes --> update curent at tInd+1
            # note that with this definition, if t_delay = dt, then a spike at time 
            # t will affect psc at time t+1 and voltage at time t+2
            
            
            tDelay = timePts[tInd + 1 - round(t_delay/dt)]
            if tDelay < 0:
                spikes_for_input = np.zeros(N)
            else:
                spikes_for_input = np.zeros(N)
                fired_delay_ind = np.nonzero(spikes[0,:] == tDelay)[0]
                fired_delay = spikes[1,fired_delay_ind]
                spikes_for_input[fired_delay.astype(int)] = 1
            
            
            # update synaptic current
            
            # external excitatory current 
            # [since homogeneous poisson, don't need to worry about delays]
            I_exc_ext = np.zeros(N)
            I_exc_ext[:Ne] = Jee_ext*extPoisson_vec[:Ne, tInd]/tau_s_e
            I_exc_ext[Ne:] = Jie_ext*extPoisson_vec[Ne:, tInd]/tau_s_e
            
            # recurrent excitatory
            I_exc_rec = np.matmul(Jij[:,:Ne],spikes_for_input[:Ne]*1)/tau_s_e
            
            # total excitatory
            I_exc[:, tInd + 1] = I_exc[:, tInd + 1] + I_exc_ext + I_exc_rec
            
            
            # recurrent inhibitory
            I_inh_rec = np.matmul(Jij[:,Ne:],spikes_for_input[Ne:]*1)/tau_s_i
            
            # total inhibitory
            I_inh[:, tInd + 1] = I_inh[:, tInd + 1] + I_inh_rec
    
            
            
            #------------------ RESET REFRACTORY NEURONS --------------------------#
    
            
            # reset neurons who fired
            V[fired_ind, tInd + 1] = Vr[fired_ind]        
            
            
            # set their time remaining refractory to tau_r
            time_ref[fired_ind] = tau_r
            
            
            
            
        #------------------ SUBTHRESHOLD STATE UPDATE ------------------------#

            
        else:
            
            # stimulation
            Istim = Istim_now.copy()
            
            # external input
            if extCurrent_poisson == True:
                I_o = Istim_now.copy()
            else:
                I_o = Istim_now + I_const     
                
            # poisson
            extPoisson_vec = extPoisson_now.copy()
            
            #print(extPoisson_vec)
            
        
            #------------------ REFRACTORY CONDITION -----------------------------#
        
            # update sensitive ids
            sensitive_id = np.nonzero(np.round(time_ref,5) == 0)[0]
            non_sensitive_id = np.setdiff1d(np.arange(0,N), sensitive_id)
        
            # update time remaining refractory at each step
            time_ref[non_sensitive_id] = time_ref[non_sensitive_id] - dt
            
            # update the voltage of non-refractory
            if np.size(sensitive_id) > 0:
                
                V[sensitive_id] = propagator[sensitive_id, 0, 0] * V[sensitive_id] + \
                                  propagator[sensitive_id, 0, 1] * I_exc[sensitive_id] + \
                                  propagator[sensitive_id, 0, 2] * I_inh[sensitive_id] + \
                                  propagator[sensitive_id, 0, 3] * I_o[sensitive_id]
                               
                                        
            # update voltage of refractory
            if np.size(non_sensitive_id) > 0:
                
                V[non_sensitive_id] = Vr[non_sensitive_id]
              
        
            # update excitatory synaptic current
            I_exc = propagator[:, 1, 1] * I_exc
            
            # update inhibitory synaptic current
            I_inh = propagator[:, 2, 2] * I_inh

            
    
    
            #------------------ GET CELLS THAT FIRED AT CURRENT INDEX -------------#
            
            fired_ind = np.nonzero(V>=Vth)[0]
            
            
            #------------------ STORE THESE NEW SPIKES ----------------------------#      
            
            # store spikes (row 1 = times, row 2 = neuron index)
            new_spikes = np.vstack((tNext*np.ones(np.size(fired_ind)), fired_ind))
            spikes = np.append(spikes, new_spikes, 1)
    
    
    
            #------------------ UPDATE SYNAPTIC WEIGHTS --> JUMP ------------------#
            
            # get delayed spikes --> update curent at tInd+1
            # note that with this definition, if t_delay = dt, then a spike at time 
            # t will affect psc at time t+1 and voltage at time t+2
            
            
            tDelay = np.round(tNext - t_delay,5)
            if tDelay < 0:
                spikes_for_input = np.zeros(N)
            else:
                spikes_for_input = np.zeros(N)
                fired_delay_ind = np.nonzero(spikes[0,:] == tDelay)[0]
                fired_delay = spikes[1,fired_delay_ind]
                spikes_for_input[fired_delay.astype(int)] = 1
            
            
            # update synaptic current
            
            # external excitatory current 
            # [since homogeneous poisson, don't need to worry about delays]
            I_exc_ext = np.zeros(N)
            I_exc_ext[:Ne] = Jee_ext*extPoisson_vec[:Ne]/tau_s_e
            I_exc_ext[Ne:] = Jie_ext*extPoisson_vec[Ne:]/tau_s_e
            
            # recurrent excitatory
            I_exc_rec = np.matmul(Jij[:,:Ne],spikes_for_input[:Ne]*1)/tau_s_e
            
            # total excitatory
            I_exc = I_exc + I_exc_ext + I_exc_rec
            
            
            # recurrent inhibitory
            I_inh_rec = np.matmul(Jij[:,Ne:],spikes_for_input[Ne:]*1)/tau_s_i
            
            # total inhibitory
            I_inh = I_inh + I_inh_rec
    
            
            
            #------------------ RESET REFRACTORY NEURONS --------------------------#
    
            
            # reset neurons who fired
            V[fired_ind] = Vr[fired_ind]       
            
        
            # set their time remaining refractory to tau_r
            time_ref[fired_ind] = tau_r
        
        
        
    # delete first column used to initialize spikes    
    spikes = np.delete(spikes, 0, 1)


    # output
    if save_voltage == True:

        return timePts, spikes, V, I_exc, I_inh, I_o
    
    else:
        
        return spikes




#------------------------------------------------------------------------------
# STIMULATION SETUP [THIS COULD GO INTO A FUNCTION WITH SIM_PARAMS AS INPUT]
#------------------------------------------------------------------------------

def fcn_stimulation_setup(sim_params, t):

    # stimulation parameters
    Ne = sim_params.N_e
    Ni = sim_params.N_i
    pext = sim_params.pext
    Jee_ext = sim_params.Jee_ext
    Jie_ext = sim_params.Jie_ext
    stim_shape = sim_params.stim_shape
    stim_onset = sim_params.stim_onset
    stim_Ecells = sim_params.stim_Ecells
    stim_Icells = sim_params.stim_Icells
    
    # stimulated cells
    stim_cells = np.concatenate((stim_Ecells, stim_Icells))
    

    if ( (stim_shape == 'box') ):
        
        stim_duration = sim_params.stim_duration
        stimRate_e = sim_params.stimRate_E
        stimRate_i = sim_params.stimRate_I
        stim_amp_e = stimRate_e*(Ne*pext*Jee_ext)*np.ones((Ne))
        stim_amp_i = stimRate_i*(Ne*pext*Jie_ext)*np.ones((Ni))
        stim_amplitude = np.concatenate((stim_amp_e, stim_amp_i))
        
        
        Istim_at_t = fcn_stimulation.fcn_box_stimulus(stim_onset, stim_duration, stim_amplitude, t)
        Istim = Istim_at_t*stim_cells
            
    elif ( (stim_shape == 'linear') ):
        
        stim_duration = sim_params.stim_duration
        stimRate_e = sim_params.stimRate_E
        stimRate_i = sim_params.stimRate_I
        stim_amp_e = stimRate_e*(Ne*pext*Jee_ext)*np.ones((Ne))
        stim_amp_i = stimRate_i*(Ne*pext*Jie_ext)*np.ones((Ni))
        stim_amplitude = np.concatenate((stim_amp_e, stim_amp_i))
        

        Istim_at_t = fcn_stimulation.fcn_linear_stimulus(stim_onset, stim_duration, stim_amplitude, t)
        Istim = Istim_at_t*stim_cells
        
    elif stim_shape == 'diff2exp':
        
        taur = sim_params.stim_taur
        taud = sim_params.stim_taud
        stimRate_e = sim_params.stimRate_E
        stimRate_i = sim_params.stimRate_I
        stim_amp_e = stimRate_e*(Ne*pext*Jee_ext)*np.ones((Ne))
        stim_amp_i = stimRate_i*(Ne*pext*Jie_ext)*np.ones((Ni))
        stim_amplitude = np.concatenate((stim_amp_e, stim_amp_i))
        

        Istim_at_t = fcn_stimulation.fcn_diff2exp_stimulus(stim_onset, stim_amplitude, taur, taud, t)
        Istim = Istim_at_t*stim_cells
        
    else:
        sys.exit('unspecified stimulus type')
        
    return Istim
        