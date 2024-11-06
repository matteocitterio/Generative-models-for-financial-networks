import numpy as np
from CIR import get_CIR
from tqdm import tqdm
from torch_geometric.utils import degree
import torch
from sklearn.preprocessing import MinMaxScaler

class Simulation:
    """
    Handles the entire simulation
    """

    def __init__(self, alpha, b, sigma, v_0, years, seed = False, num_nodes = 1, gamma = 1, seed_number=0):
        """
        Sets coefficients and generates CIR process.
        The CIR process is a squared root diffusion process described by:
        
            dX(t) = alpha*(b - X(t)) dt + sigma \sqrt{X(t)} dW(t)
            
        Parameters:
        ------------------
        - alpha : `float` 
        - b : `float`
        - sigma : `float`
        - v_0 : `float`
        - years: `int`
           Number of years to be taken as the simulation time horizon. The simulation will be based on a time grid of a
           number of points 365 * year
        - seed :`bool`
           If True it will set the seed of the simulation to 0
        - num_nodes: `int`
           Number of nodes in the simulated graph
        
        """
        
        self.alpha = alpha 
        self.b = b
        self.sigma = sigma
        self.v_0 = v_0

        #Number of the points of time time grid
        self.TotPoints = int(years * 365)

        #Default Cox-process coefficients (See Eq. 5.17):
        self.eta = - 4
        self.theta = 20
        self.beta = 5
        self.gamma = gamma

        # These grid is used to bin the outcome of the simulation as later shown in `Price`. The grid extreme values are empirically identified. 
        self.r_grid = np.linspace(0.0002, 0.157, 600)

        #This matrix contains all the p(t,T) computed for the case seed=0, speeds up simulation a lot
        self.PriceMatrix = np.load('../data/PriceMatrix_Duffie_Updated.npy')
        #Time grid used for the simulation
        self.time_grid = np.linspace(0., years, 365*years)

        #This is the reference interest rate we take for the entire simulation and all edges
        self.CIRProcess = get_CIR(
            self.time_grid,
            self.alpha,
            self.b,
            self.sigma,
            self.v_0,
            n_samples = 1,
            seed = seed,
            seed_number=seed_number

        )[0]
        self.seed = seed

        #Number of nodes in the graph
        self.num_nodes = num_nodes
        #Matrix that will contain all the historical records of the transactions between (i,j)
        self.E = np.zeros((self.num_nodes, self.num_nodes), dtype=dict)

    def Price(self, t, T, r=None):
        """
        Returns the bond price at time t with maturity at time T.
        This is actually based on simulations made for the CIR process with seed = 0

        Parameters
        -----------
        - t : `int`
           Current time at which we want to know the bond price
        - T : `int`
           Maturity of the bond contract
        - r : `float` [optional]
           Used to pass a costum interest rate 
           
        Returns
        -------
        - P : `float`
           Price of the bond at time t with maturity T (p(t,T))
        """
        tau = int(T-t)

        if r == None:
            r = self.CIRProcess[t]
        
        #This bins the r process into a predefined grid (=>computational efficiency)
        closest_index = np.argmin(np.abs(self.r_grid-r))
        return self.PriceMatrix[closest_index, tau]

    def B(self, t):
        """
        Returns the money-market account value B_t at time t.
        
        Parameters
        -----------
        - t : `int`
           Current time at which we want to know the bond price
           
        Returns
        -------
        - B_t : `float`
           Price of the money-market account B_t at time t.
        """

        t = int(t)
        B_t = np.prod(1. + self.CIRProcess[1:t+1]*(1/365.))
        return B_t
    
    def g(self, x_i,x_j):
        
        return ( -(x_i * x_j) + abs(x_i - x_j) + (x_i + x_j) ) / 3

    def GetArrivalTimes(self, x_i=0,x_j=0):
        """
        Generates the Cox process and the arrival times

        Parameters
        --------
        - x_i, x_j: `int`
            they are +1 for HUBS, -1 for SMALL_PLAYERS

        Returns
        --------
        - arrival_times: `np.array(int)`
           array containing the contracts arrival times
        """

        #The intensity process is computed as an affine process of the simulated CIR (See Eq. 5.17)
        self.lambda_t = self.gamma * np.exp(self.eta + (self.theta + self.beta *self.g(x_i,x_j)) * self.CIRProcess)

        #For the following process please refer to Eq. 5.25 - 5.26 - 5.27 of the Master's Thesis.

        #The approximated integral in discrite-time:
        cumulative_lambda = np.cumsum(self.lambda_t)

        #We generate variables \xi \sim \exp{scale=100} as required in the arrival times algorithm with time-inversion
        xis = np.random.exponential(scale=100, size=2000)
        cum_xis = np.cumsum(xis)

        arrival_times = []
        for n in range(1, 2000 + 1):
            T_n = np.argmax(cumulative_lambda >= cum_xis[n - 1])
            if T_n == 0:
                break
            else:
                arrival_times.append(T_n)
                
        return np.asarray(arrival_times)
    
    def ProvideBenchmark(self, t_l, steps_ahead, contracts, n_simulations = 10000):
        """
        AS EXPLAINED IN Eq. 5.24
        This method computes the variation margin expectation for an m steps ahead prediction made at time t_{l}, i.e. E[M_{t_{l+m}}| ...]. 
        The expectation is conditioned over the contract information available at time t_{l}, so basically the contracts that were available
        at time t_l, together with the additional information associated to the CIR process up to the prediction time t_{l+m}, which we
        here denote with \mathcal{F}_{t_{l+m}}.
        Therefore, the expectation is made of two parts: the first one is the expectation of the margin for the contracts that were active
        at time t_{l}, the second one includes the contracts that appear in the interval [t_{l}, t_{l+m-1}] and involves the simulation of the
        Cox Process over the given CIRProcess sequence.

        Parameters
        ----------
        - t_l : `int`
            time \in \mathcal{T} that represents the time at which the prediction is performed and to which the contract filtration refers to
        - steps_ahead : `int`
            number of steps ahead involved in the prediction
        - contracts : `list`
            list of class `Contract` objects with the proper starting date
        - n_simulations : `int`
            Number of simulations of the Cox Process

        Returns
        ----------
        """

        #Prediction time steps
        t_l = t_l
        t_l_m = t_l + steps_ahead

        #FIRST PART: expectation of the active contracts
        #Retrieve active contracts at time t_l (\mathcal{G}_{t_{l}})
        active_contracts = [contract for contract in contracts if contract.is_active(t_l)]
        active_contracts_in_the_future = [contract for contract in contracts if contract.is_active(t_l_m)]
        #Compute the variation margin for this contracts at time t_{l+m}
        margin_for_already_existing_contracts = 0
        for contract in active_contracts:
            if contract in active_contracts_in_the_future:
               margin_for_already_existing_contracts += contract.GetVariationMargin(t_l_m)

        #SECOND PART: simulation of contracts arrival during the conditioning sequence
        margin_for_newly_arrived_contracts = 0
        #Loop over a number of simulations to compute the expectation
        for i in range(n_simulations):
            
            #Sample the COX process
            temp_arrival = self.GetArrivalTimes()
            #Right extreme excluded, so times up to t_l_m - 1 for the variation margin definition
            selected_times = temp_arrival[ (temp_arrival >= t_l) & (temp_arrival < t_l_m)]
            #Create these contracts
            arrived_contracts = [Contract(time, self) for time in selected_times]
            #Compute the variation margin at time t_l_m for these newly arrived contracts
            for contract in arrived_contracts:
                margin_for_newly_arrived_contracts += contract.GetVariationMargin(t_l_m)
        
        #Normalize the computed margin over the number of simulations to get the expectation
        margin_for_newly_arrived_contracts /= n_simulations

        return margin_for_already_existing_contracts + margin_for_newly_arrived_contracts

    def SimulateAllEdges(self, steps_ahead = 1):
        """
        It simulates the contract arrival process on every edge of the financial network
        """
        self.node_features = torch.tensor(np.random.choice([-1,1], self.num_nodes)).to(torch.float32)
        self.GetContracts()
            

    def GetActiveContractList(self):
        """
        Contructs a list of len = days_with_active_contracts containing at each element a list of contracts that are active at that time step
        Returns
        -------
        - active_contracts: `list`:
            list of len= days where throughout the network at least a contract is active containing all the active contracts at time t
        - active_days : `list`
            list containing all the active days
        """
        loop = tqdm(range(self.TotPoints), desc='Active contracts @t')
        active_contracts = []
        active_days = []

        for t in loop:
            alive_at_t = []
            flattened_alive_at_t = []            

            for i in range(self.num_nodes):
         
                for j in range(i+1, self.num_nodes):

                    active_cont = [contract for contract in self.E[i,j] if contract.is_active(t)]
                    if len(active_cont)> 0:
                        alive_at_t.append( active_cont + [contract for contract in self.E[j,i] if contract.is_active(t)])

            #Flatten the list so that it contains an element at each element instead of having a nested list
            flattened_alive_at_t = [item for sublist in alive_at_t for item in sublist]

            #Append it if its not empty
            if flattened_alive_at_t:
                active_contracts.append(flattened_alive_at_t)
                active_days.append(t)

        return active_contracts, active_days

    def GetContracts(self):
        """
        Construct sim.E with contracts for each edge (undirected)
        """

        for i in range(self.num_nodes):

            for j in range(i+1, self.num_nodes):
                
                arrival_times = self.GetArrivalTimes(self.node_features[i], self.node_features[j])

                #Build the contracts using Simulation and arrival times
                contracts = []
                for arrival_time in arrival_times:
                  contract = Contract(arrival_time, self, src =i, dst = j)
                  contracts.append(contract)
                
                self.E[i,j] = contracts
                self.E[j,i] = self.GetOpposite(contracts)

    def GetOpposite(self, contracts):
        """
        This gets the other side of the transaction
        """
        opposite_contracts = []
        for contract in contracts:
            
            opposite_contract = Contract(contract.t_0, self, src=contract.dst, dst=contract.src)
            opposite_contract.delta = contract.delta * -1
            opposite_contracts.append(opposite_contract)

        return opposite_contracts         
    
    def GetMaximumNActiveContracts(self, active_contracts):
      """
      Compute maximum number of simultaneously active contracts

      Params:
      ---------
      - active_contracts:
            list of concrats that are active for every day where trades are observed
      Returns
      ---------
      - max_len : `int`
        The maximum number of active contracts that has been observed over the simulation horizon period for all nodes
      """
      max_len = 0
      for t in range(len(active_contracts)):
            edges = torch.stack([torch.tensor([contract.src, contract.dst]) for contract in active_contracts[t]], dim=1)
            edge_index, unique_indices, counts = torch.unique(edges.cpu(), dim=1, return_inverse=True, return_counts=True)
    
            #Convert edge_index to int type
            edge_index = edge_index.to(torch.int64)

            #This selects the source nodes for active contracts at time time
            source_nodes = torch.unique(edge_index[0,:])
            for i in source_nodes:
                
                #take active contracts just for node i
                contracts_for_node_i = [contract for contract in active_contracts[t] if contract.src==i]
                if len(contracts_for_node_i)>max_len:
                    max_len=len(contracts_for_node_i)
    
      return max_len

class Contract:

    def __init__(self, t_0, sim, src=1, dst=1):
        """
        Instantiate a contract object

        Parameters
        ----------
        - t_0 : `int`
            starting time of the contract
        - sim : `Simulation object`
            simulation of the financial network

        Data members
        ----------
        - t_0 : `int`
            starting time of the contract
        - T : `int`
            Maturity of the contract, computed as the starting time + 365
        - delta : `int`
            +1 / -1 identifying the leg of the swap contract
        - notional : `int` [default = 1]
            principal notional for the swap contract
        - sim : `Simulation`
            Simulation object for the financial network
        """
        # contract start time
        self.t_0 = int(t_0)
        # contract end time
        self.T = self.t_0 + 365
        self.delta = np.random.choice ([-1,1])
        self.notional = 1.
        self.sim = sim
        #Source and destination node of the contract
        self.src = src
        self.dst = dst
    
    def __repr__(self):
        return f'(t_0:{self.t_0}, T: {self.T}, delta:{self.delta}, src:{self.src}, dst:{self.dst})'

    def get_contract_features(self, t):
        """
        Retrieves the contracts features at time t
        
        Parameters
        ----------
        - t : `int`
            time at which the contracts features are getting retrieved
      
        Returns
        ----------
        - contract : `torch.tensor`
            Tensor containing the contract features {(T-t)/365, log(p(t_0,T)), log(p(t,T)), B(t_0), B(t)}

        """
        t = int(t)
        contract = torch.tensor([(self.T - t)/365.,                                       # (T-t)/365
                                 np.log(self.sim.Price(self.t_0, self.T)),                # p(t_0, T)
                                 np.log(self.sim.Price(t, self.T)),
                                 #self.sim.CIRProcess[t],
                                 np.log(self.sim.B(self.t_0)),                            # B_t_0
                                 np.log(self.sim.B(t)),                                   # B_t
                                 self.delta])                                             # Delta
                                
        return contract

    def is_active(self, t):
        """
        Checks whether a contract is active @time time.
        --- TO CHECK LEFT CONTINUITY ---

        Parameters
        ----------

        - t : `int`
            time at which the contract is checked
      
        Returns
        ----------
        - is_active: `bool`
        """
        t = int(t)
        if t >= self.t_0 and t <= self.T:
            return True
        else:
            return False
        
    def GetFixedLeg(self, t):
        """
        Returns the fixed leg value of the OIS

        Parameters
        ----------
        - t : `int`
           Time at which we want to evaluate the leg

        Returns
        --------
        - `float`
           Value of the fixed leg process at time t
        """
        
        return self.notional * (self.sim.Price(t, self.T) / self.sim.Price(self.t_0, self.T))
    
    def GetFloatingLeg(self, t):
        """
        Evaluates the Floating leg of the OIS
        
        Parameters
        ----------
        - t : `int`
           Time at which we want to evaluate the process

        Returns
        --------
        - `float`
           Value of the floating leg process at time t
        """

        return self.notional * np.prod(1+(self.sim.CIRProcess[self.t_0+1:t+1]*1/365))
    
    def MarkToMarketPrice(self, t):
        """
        Computes the contract's mark-to-market value

        Parameters
        ----------
        - t : `int`
           Time at which we want to evaluate the contract

        Returns
        --------
        - `float`
           Value of the OIS contract for at time `t`
        """
        #As we are considering only active nodes i think there is no need for that
        if t > self.T or t<0 or t<self.t_0:
            return 0

        return self.delta * self.notional * (self.GetFixedLeg(t) - self.GetFloatingLeg(t))
    
    def GetVariationMargin(self,t):
        """
        Returns the Margin (M_k^ij(t)) of a specific contract

        Parameters
        -----------
        - t : `int`
           Time at which we evaluate the contract

        Returns
        -------
        - float
           M_k^{ij}(t) of contract k
        """

        V_t = self.MarkToMarketPrice(t)
        V_t_1 = self.MarkToMarketPrice(t - 1)

        return (V_t - (1 + (self.sim.CIRProcess[t] * 1 / 365) ) * V_t_1)
   