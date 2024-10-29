# Simulation

This folder contains the code used to generate the network's synthetic data, as extensively explained in `Sec. 5.2`. Whenever possible, the parameters are explained and described making reference to the Eq. number in the Master's thesis document.

- `CIR.py` contains the code needed to simulate multiple CIR processes. Please note that the code is not CUDA-compatible as the improvements were neglectible.

- `simulation.py` is a sort of 'library' that contains the `Simulation` class used for the Montecarlo simulations of the so-called "Synthetic model" (See Sec. 5.2). Within this class there are implemented a wide variety of methods that were used at some point in the process. Today, not all of these methods are used to produce the Results as depicted in Chapter 6.

- `SimulateNetwork.py` is the piece of code used to run the actual simulation. It takes a few command line inputs as described in this example usage: `python SimulateNetwork.py --do_simulation --device cuda --nodes 10 --gamma 1`; where:

    - `--do_simulation` is `True` by default; if false it doesnt simulate the contracts arrival process but it loads the pre-computed matrix
    - `--device` the device you want the simulation to run on (cuda, cpu, mps, ...)
    - `--nodes` number of nodes in the network to include in the simulation
    - `--gamma` float, it tunes the stochastic intensity of the COX process that governs the birth of new contracts within the network.

- `edge_features.npy` stores the simulation of the network in terms of the contract process (no pricing yet). The '.npy' format allows to straightforwardly load the matrix into a numpy format.

- `get_price_matrix.py` computes the matrix containing all the p(t,T), computed for the case seed=0. Speeds up simulation a lot.





