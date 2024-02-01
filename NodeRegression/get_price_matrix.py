import numpy as np
from simulation import Simulation 
import matplotlib.pyplot as plt
from tqdm import tqdm

alpha = 0.6
b = 0.02
sigma = 0.14
v_0 = 0.04
years = 20

# This has a fixed seed
sim = Simulation(alpha, b, sigma, v_0, years, seed=True)

PriceMatrix = np.zeros((sim.TotPoints, 365))

loop = tqdm(range(sim.TotPoints), desc='t')

for i in loop:

    t = i

    for j in range(365):
        loop.set_postfix(j=j)
        PriceMatrix[t,j] = sim.MontecarloPrice(t, t + j, n_samples=10000)

np.save('/u/mcitterio/data/PriceMatrix_Duffie_diecimila.npy', PriceMatrix)