This folder contains the code for predicting the Margin Value of a financial network, which falls into the category of **Node regression task** in the context of GNNs literature.

- `requirements.txt`: list of all the dependencies

## Network simulation
- `simulation.py`: Contains the class `Simulation` which is used for simulating the FinNet and other useful methods for data processing
- `SimulateNetwork.py`: perferforms the actual simulation of the network. Usage:
   - `--do_simulation`: if present, performs a net simulation and doesn't use a preloaded matrix containing the contracts
   - `--device`: `str` could be `cuda`, `cpu`, `mps`
   - `--nodes`: `int` number of nodes in the network
   - `--gamma`: `float` parameter that tunes the intensity of the contract arrival process according to $\lambda(t) = \gamma \cdot \exp{(\eta -\theta r(t))}$ where $r(t)$ is a CIR process
- `CIR.py`: for simulating the Cox-Ingersoll-Ross process
- `get_price_matrix.py`: computes a matrix with entrances $p(t,T)$ using a MonteCarlo simulation so that it doesn't compute an MC expectation every time the price function is called.
- `PriceMatrix_Duffie.npy`: Matrix of prices computed through a MC simulation (see `get_price_matrix.py`) using the CIR parameters found in Duffie et al.

## Margin prediction
- `Analytic_model.py`: contains a class `PerfectPredition` that performs the reconstruction of $M(t)$ given the contracts and the other quantities. This was used to check whether all the information was passed in the proper way
- `test_pred_1_node.py`: performs the prediction of the Margin value M^{i}(t) for a single node for a single contract with maturity 365
- `PredictingPrice.py`: performs the prediction of p(t,T) using the CIR process r(t). This was used to see whether the model is able of learning the expectation of the process.
