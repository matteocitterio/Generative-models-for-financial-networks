import numpy as np
from simulation import Simulation 
import matplotlib.pyplot as plt
from CIR import get_CIR
from tqdm import tqdm

alpha = 0.6
b = 0.04
sigma = 0.14
years = 10

grid = np.linspace(0., years, 365*years)


# PriceMatrix = np.zeros((sim.TotPoints - 365, 365))
# print(PriceMatrix.shape)
# loop = tqdm(range(sim.TotPoints - 365), desc='t')


r_grid = np.linspace(0.0001, 0.16,1200)
PriceMatrix = np.zeros((1200, 366))
n_samples = 1000


loop = tqdm(range(1200))

for i in loop:
    
    for tau in range(366):

        loop.set_postfix(tau = tau)
        
        if tau == 0:
               PriceMatrix[i, tau] = 1

        else:
            #Ora siamo nella situazione r[t] = r_grid[i], T-t = tau
            #Tau + 1?
            MC_sims = get_CIR(grid[0:tau +1],
                                     alpha,
                                     b,
                                     sigma,
                                     r_grid[i],
                                     n_samples = n_samples,
                                     seed = True,
                                     seed_number=0)
            prod = np.prod(1+MC_sims[1:]*(1./365), axis=1)
            PriceMatrix[i,tau] = np.mean(prod**(-1))

np.save('/u/mcitterio/data/PriceMatrix_Duffie_Updated.npy', PriceMatrix)