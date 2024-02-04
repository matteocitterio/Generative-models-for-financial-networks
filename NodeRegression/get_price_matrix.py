import numpy as np
from simulation import Simulation 
import matplotlib.pyplot as plt
from tqdm import tqdm

alpha = 0.6
b = 0.04
sigma = 0.14
v_0 = 0.04
years = 20

# This has a fixed seed
sim = Simulation(alpha, b, sigma, v_0, years, seed=True)

PriceMatrix = np.zeros((sim.TotPoints - 365, 365))
print(PriceMatrix.shape)
loop = tqdm(range(sim.TotPoints - 365), desc='t')

for i in loop:

    t = i
    price = np.ones((365))

    for j in range(1,365):
        loop.set_postfix(j=j)
        price[j] = sim.MontecarloPrice(t, t + j, n_samples=1000)

    PriceMatrix[t] = price

np.save('/u/mcitterio/data/PriceMatrix_Duffie_Updated.npy', PriceMatrix)