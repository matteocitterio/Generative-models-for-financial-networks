This folder contains the code for predicting the Margin Value of a financial network, which falls into the category of **Node regression task** in the context of GNNs literature.

## Network simulation
- `simulation.py`: Contains the class `Simulation` which is used for simulating the FinNet and other useful methods for data processing
- `SimulateNetwork.py`: perferforms the actual simulation of the network. Usage:
 - `--do_simulation`: if present, performs a net simulation and doesn't use a preloaded matrix containing the contracts
 - `--device`: `str` could be `cuda`, `cpu`, `mps`
 - `--nodes`: `int` number of nodes in the network
 - `--gamma`: `float` parameter that tunes the intensity of the contract arrival process according to $\lambda(t) = \gamma \cdot \exp{\eta -\theta r(t)}$ where $r(t)$ is a CIR process