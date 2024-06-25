import numpy as np
import ssm
from sklearn.linear_model import LinearRegression


def mysteryData():

          # Set the parameters of the HMM
          time_bins = 1000   # number of time bins
          num_states = 3    # number of discrete states
          obs_dim = 1       # dimensionality of observation
          input_dim=1     # input dimension

          # Create an HMM
          true_hmm = ssm.HMM(num_states, obs_dim, M=input_dim,observations="input_driven_obs_gaussian", transitions="standard")
          # set weights Wk, biases mus, and noise covariance sigmas by hand

          gen_weights=3*np.ones((num_states,obs_dim,input_dim))
          gen_weights[1]=-gen_weights[0]
          gen_weights[2]=0*gen_weights[0]
          # gen_weights=3*np.random.rand(num_states,obs_dim,input_dim)

          mus=-np.ones((num_states, obs_dim))
          mus[1]=-mus[0]
          mus[2]=0*mus[0]
          # mus=np.random.rand(num_states, obs_dim)

          stdnoise=np.array([0.5,0.5,1])
          sigma=np.eye(obs_dim) # diagonal noise correlations with variance stdnoise**2
          sigmas = np.dstack([sigma]*num_states).transpose((2,0,1))
          for i in range(num_states): sigmas[i,:,:]=stdnoise[i]**2*sigma

          true_hmm.observations.mus = mus
          true_hmm.observations.Sigmas = sigmas
          true_hmm.observations.Wks =  gen_weights

          # set transition probabilities as well
          trans_eps=0.01 # off diag transition prob to another state
          trans0=trans_eps*np.ones((num_states,num_states))
          for i in range(num_states):
                    trans0[i,i]=1-(trans_eps*(num_states-1))   
          true_hmm.transitions.log_Ps=np.log(trans0)

          # Create an exogenous input dim T x M
          inpt = np.random.rand(time_bins) # generate random inputs from uniform distribution
          # inpt = 0.1*(np.arange(time_bins)+1) # generate linearly increasing input for a simpler model
          if inpt.ndim == 1: # if input is vector of size self.M (one time point), expand dims to be (1, M)
                    inpt = np.expand_dims(inpt, axis=1)
          inpt=np.tile(inpt,input_dim)

          # Sample some data from the HMM
          true_states, obs = true_hmm.sample(time_bins, input=inpt)
          true_ll = true_hmm.log_likelihood(obs,inputs=inpt)

          return obs, inpt, true_hmm, true_states

def autocovariance(x):
    """
    Compute the autocov of the signal
    Parameters:
    - x: A 1D array of the signal.
    Returns:
    - A 1D array containing the autocorrelation of the input signal.
    """
    
    # Detrend the signal by removing the mean
    xp = x - np.mean(x)
    
    # Compute the FFT of the detrended signal
    f = np.fft.fft(xp)
    
    # Compute the power spectrum density (PSD)
    p = np.real(f) * np.real(f) + np.imag(f) * np.imag(f)
    
    # Inverse FFT to get the autocorrelation function
    pi = np.fft.ifft(p)
    
    # Normalize and return the real part of the autocorrelation
    out=np.real(pi)[:len(x) // 2] / np.sum(xp ** 2)
    return out

def compute_hwhm(acf):
    """Estimate the timescale as the half-width at half-maximum of the ACF's envelope."""
    half_max = np.max(acf) / 2
    for i, val in enumerate(acf):
        if val < half_max:
            return i
    return len(acf) - 1  # Return the max timescale if HWHM is not found

def find_elbow_point(y_values):
    n_points = len(y_values)
    all_coords = np.vstack((range(n_points), y_values)).T
    first_point = all_coords[0]
    line_vec = all_coords[-1] - all_coords[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    vec_from_first = all_coords - first_point
    scalar_product = np.sum(vec_from_first * line_vec_norm, axis=1)
    vec_to_line = vec_from_first - np.outer(scalar_product, line_vec_norm)
    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
    elbow_index = np.argmax(dist_to_line)
    return elbow_index + 1  # +1 as index starts from 0

def xval_func(data_in,num_states0):
    
    training_data=data_in['training_data']
    test_data=data_in['test_data']
    training_inpts=data_in['training_inpts']
    test_inpts=data_in['test_inpts']
    N_iters=data_in['N_iters']
    TOL=data_in['TOL']
    
    obs_dim = len(training_data[0])             # number of observed dimensions: outcome
    input_dim = len(training_inpts[0])         # input dimensions: [consReward,consFailure,value,count,bias]
    nTrain=len(training_data)
    nTest=len(test_data)
    
    out={}
    mle_hmm = ssm.HMM(num_states0, obs_dim, M=input_dim, 
        observations="input_driven_obs_gaussian", transitions="standard")
    #fit on training data
    hmm_lls = mle_hmm.fit(training_data, inputs=training_inpts, method="em", num_iters=N_iters, tolerance=TOL)                
    #Compute log-likelihood for each dataset
    out['ll_training'] = mle_hmm.log_likelihood(training_data, inputs=training_inpts)/nTrain
    out['ll_heldout'] = mle_hmm.log_likelihood(test_data, inputs=test_inpts)/nTest
    
    #Create HMM object to fit: MAP
    # Instantiate GLM-HMM and set prior hyperparameters
    prior_sigma = 2
    prior_alpha = 2
    map_hmm = ssm.HMM(num_states0, obs_dim, M=input_dim, 
        observations="input_driven_obs_gaussian", 
                observation_kwargs=dict(prior_sigma=prior_sigma),
                transitions="sticky", transition_kwargs=dict(alpha=prior_alpha,kappa=0))
    #fit on training data
    hmm_lls = map_hmm.fit(training_data, inputs=training_inpts, method="em", num_iters=N_iters, tolerance=TOL)                
    #Compute log-likelihood for each dataset
    out['ll_training_map'] = map_hmm.log_likelihood(training_data, inputs=training_inpts)/nTrain
    out['ll_heldout_map'] = map_hmm.log_likelihood(test_data, inputs=test_inpts)/nTest

    return out

class EMAlgorithm:
    def __init__(self, n_components, max_iter=100, tol=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.models_ = [LinearRegression() for _ in range(n_components)]
        self.pi_ = np.ones(n_components) / n_components
        self.sigma2_ = np.ones(n_components)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        responsibilities = np.zeros((n_samples, self.n_components))

        # Randomly initialize models to ensure diversity
        for i in range(self.n_components):
            indices = np.random.choice(n_samples, n_samples // self.n_components, replace=False)
            self.models_[i].fit(X[indices], y[indices])
            residuals = y - self.models_[i].predict(X)
            self.sigma2_[i] = np.var(residuals)

        for iteration in range(self.max_iter):
            # E-step: Compute responsibilities
            for i, model in enumerate(self.models_):
                residuals = y - model.predict(X)
                responsibilities[:, i] = self.pi_[i] * np.exp(-0.5 * residuals**2 / self.sigma2_[i]) / np.sqrt(2 * np.pi * self.sigma2_[i])
            responsibilities /= responsibilities.sum(axis=1, keepdims=True)

            # M-step: Update parameters
            for i in range(self.n_components):
                # Update mixing coefficients
                self.pi_[i] = responsibilities[:, i].mean()

                # Update regression coefficients
                W = np.diag(responsibilities[:, i])
                self.models_[i].fit(X, y, sample_weight=responsibilities[:, i])

                # Update variances
                residuals = y - self.models_[i].predict(X)
                self.sigma2_[i] = np.sum(responsibilities[:, i] * residuals**2) / np.sum(responsibilities[:, i])

            # Check for convergence
            if iteration > 0 and np.linalg.norm(responsibilities - prev_responsibilities) < self.tol:
                break
            prev_responsibilities = responsibilities.copy()

        self.responsibilities_ = responsibilities
        return self

    def compute_responsibilities(self, X, y):
        responsibilities = np.zeros((X.shape[0], self.n_components))
        for i, model in enumerate(self.models_):
            predictions = model.predict(X)
            residuals = y - predictions
            responsibilities[:, i] = self.pi_[i] * np.exp(-0.5 * residuals**2 / self.sigma2_[i]) / np.sqrt(2 * np.pi * self.sigma2_[i])
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def predict(self, X,y):
        predictions = np.zeros((X.shape[0], self.n_components))
        for i, model in enumerate(self.models_):
            predictions[:, i] = model.predict(X)
        responsibilities = self.compute_responsibilities(X, y)
        weighted_predictions = (predictions * responsibilities).sum(axis=1)
        return weighted_predictions

    def predict_labels(self, X,y):
        # predictions = self.predict(self, X,y)
        responsibilities = self.compute_responsibilities(X, y)
        return np.argmax(responsibilities, axis=1)

class MixtureOfLinearRegressions:
    def __init__(self, n_components, max_iter=100, tol=1e-6):
        self.em_algorithm = EMAlgorithm(n_components, max_iter, tol)

    def fit(self, X, y):
        self.em_algorithm.fit(X, y)
        return self

    def predict(self, X,y):
        return self.em_algorithm.predict(X,y)

    def predict_labels(self, X,y):
        return self.em_algorithm.predict_labels(X,y)
    
