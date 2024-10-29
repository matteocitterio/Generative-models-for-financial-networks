# Simulation

This folder contains the code used to generate the network's synthetic data, as extensively explained in `Sec. 5.2`. Whenever possible, the parameters are explained and described making reference to the Eq. number in the Master's thesis document.

- `CIR.py` contains the code needed to simulate multiple CIR processes. Please note that the code is not CUDA-compatible as the improvements were neglectible.
- `simulation.py` is a sort of 'library' that contains the `Simulation` class used for the Montecarlo simulations of the so-called "Synthetic model" (See Sec. 5.2). Within this class there are implemented a wide variety of methods that were used at some point in the process. Today, not all of these methods are used to produce the Results as depicted in Chapter 6.


