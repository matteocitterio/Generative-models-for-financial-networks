import numpy as np
from simulation import Simulation 
import matplotlib.pyplot as plt

alpha = 0.6
b = 0.02
sigma = 0.14
v_0 = 0.04
years = 3

# This has a fixed seed
sim = Simulation(alpha, b, sigma, v_0, years, seed=True)

TermStructureMatrix = np.zeros((sim.TotPoints, 365))

for i in range(sim.TotPoints):

    print(i)
    t = i

    # The first element is the 'spot rate' r(t_i)
    TermStructureMatrix[t , 0] = sim.CIRProcess[t]

    for j in range(1,365):

        #The other elements are the swap rates R(t, t+ 1), R(t, t+2), .... R(t, t+365)
        if (t+j) > sim.TotPoints:
            TermStructureMatrix[t,j] = sim.SwapRate(t, sim.TotPoints)
        else:
            TermStructureMatrix[t,j] = sim.SwapRate(t, t + j)

np.save('/u/mcitterio/data/TermStructure_Duffie.npy', TermStructureMatrix)