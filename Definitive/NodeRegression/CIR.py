from scipy.stats import ncx2, norm, chi2, erlang
import matplotlib.pyplot as plt
import numpy as np
import torch

def get_CIR(ts, alpha, b, sigma, initial_value, n_samples=1, seed=False, seed_number=0):
    '''CIR process.
    
    The process is a square-root diffusion (aka CIR process) of the form:
    
        dX(t) = alpha*(b - X(t)) dt + sigma \sqrt{X(t)} dW(t)    (1)
    
    Estimation is exact on grid points (see ref [2]).
    
    Parameters:
    -----------
    ts : numpy.ndarray, size (n,)
        Grid points in time at which process is evaluated.
    
    alpha : float
        Mean-reversion parameter in CIR diffusion (see Eq. (1) above).
        
    b : float
        Long-run mean parameter in CIR diffusion (see Eq. (1) above).
        
    sigma : float
        Volatility parameter in CIR diffusion (see Eq. (1) above).
        
    initial_value : float
        Value of X(0) (has to be deterministic)
    
    n_samples : int
        Number of samples.

    seed : bool
        Flag that tells whether to use a predetermined input seed or not.

    seed_number : int
        The predetermined seed number
        
        
    Returns:
    --------
    proc : numpy.ndaaray, size (n_samples, n)
        Array of simulated paths.
        
        
    References:
    -----------
    The model was first introducted in [1]. This implementation is based on [2, Sec. 3.4].
    
    [1] Duffie, Garleanu - Risk and valuation of collateralized debt obligations. 
        Financial analysts journal, 2001, 57. Jg., Nr. 1, S. 41-59.    
    [2] Glasserman - Monte Carlo methods in financial engineering. 
        New York: springer, 2004.
    '''
    #set seed
    if seed:

        np.random.seed(seed_number)

    #CIR simulation
    proc = np.zeros((n_samples, len(ts)))
    proc[:, 0] = initial_value*np.ones(n_samples)

    dt = ts[1] - ts[0]   #as they are evenly spaced
    c = (sigma**2)*((1-np.exp(-alpha*dt))/(4*alpha))
    d = (4*b*alpha)/(sigma**2)

    # print(f'd: {d}')
    # print(f'2ab: {2*alpha*b}, sigma^2: {sigma**2}')

    for i in range(1, len(ts)):
        
        nc = proc[:, i-1]*((np.exp(-alpha*dt))/c)        
        proc[:, i] = c*(ncx2.rvs(df=d, nc=nc))

    return proc

### Example code

# # time-grid
# ts = np.linspace(0., 1., 1000)

# alpha = 4.
# b = 0.5
# sigma = 1.
# v_0 = 0.

# # Simulate CIR process
# proc = get_CIR(ts, alpha, b, sigma, v_0, n_samples=3)

# # Plot trajectories
# plt.plot(proc.T)
# plt.show()
