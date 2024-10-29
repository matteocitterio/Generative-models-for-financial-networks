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

    def __init__(self, alpha, b, sigma, v_0, years, seed = False, num_nodes = 10, gamma = 1, seed_number=0):
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

        #Default Cox-process coefficients:
        self.eta = - 4
        self.theta = 80
        self.gamma = gamma
        self.r_grid = np.linspace(0.0002, 0.157, 600)
        #This matrix contains all the p(t,T) computed for the case seed=0, speeds up simulation a lot
        self.PriceMatrix = np.load('/u/mcitterio/data/PriceMatrix_Duffie_Updated.npy')
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
           
        Returns
        -------
        - P : `float`
           Price of the bond at time t with maturity T (p(t,T))
        """
        tau = int(T-t)

        if r == None:
            r = self.CIRProcess[t]
        
        closest_index = np.argmin(np.abs(self.r_grid-r))
        return self.PriceMatrix[closest_index, tau]

    def MontecarloPrice(self, t, T, n_samples = 1000):
         """
         Performs Montecarlo simulation of CIR, generating `n_samples` paths and then computes expectation out of it from 
         which we can get the bond price at time t with maturity at time T

         Parameters
         -----------
         - t : `int`
            Current time at which we want to know the bond price
         - T : `int`
            Maturity of the bond contract
         - n_samples : `int` (default : 10 000)
            Number of paths to generate for the computing the MC expectation
         Returns
         -------
         - P : `float`
            Price of the bond at time t with maturity T (p(t,T))
         """

         if t == T:
               # By definition, p(t,t) = 1 \forall t
               return 1

         # Monte Carlo simulations
         #NB in numpy.prod l'estremo di desta non è incluso nella produttoria quindi se voglia t<=T dobbiamo prendere
         #fino a T+1
         #NB l'estremo di sinistra è invece incluso. Nel calcolo di B_t lo vogliamo prendere incluso dal momento che 
         #poi con il rapporto il t_0 si cancella ma nel calcolo diretto di p(t_0, T) non lo vogliamo quindi prendiamo
         #anche qui t+1 s.t. we get t_0 < t <=T
         MC_sims = get_CIR(self.time_grid[t:T+1],
                                 self.alpha,
                                 self.b,
                                 self.sigma,
                                 self.CIRProcess[t],
                                 n_samples = n_samples,
                                 seed = True,
                                 seed_number=0)
         
         prod = np.prod(1+MC_sims[1:]*(1./365), axis=1)
         return np.mean(prod**(-1))

    def SwapRate(self, t_0, T):
        """
        Computes OIS swap rate, the fixed rate that makes the contract value equals to 0 at time t_0

        Parameters
        -----------
        - t_0 : `int`
           Initial time of the contract
        - T : `int`
           Maturity
           
        Returns
        --------
        - K : `Float` 
           OIS swap rate
        """

        tau = np.array((T - t_0) / 365)

        condition = (tau <= 0)

        if self.seed:
            price = self.Price
      
        else:
            price = self.MontecarloPrice
        
        return np.where(condition, 0, (tau>0)*(-1 / tau) * ((price(t_0, T) - 1) / price(t_0, T)) )

    def GetArrivalTimes(self):
        """
        Generates the Cox process and the arrival times

        Returns
        --------
        - arrival_times: `np.array(int)`
           array containing the contracts arrival times
        """

        #The intensity process is computed as an affine process of the simulated CIR
        self.lambda_t = self.gamma * np.exp(self.eta + self.theta * self.CIRProcess)

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

    def GetFixedLeg(self, t_0, t, T, notional = 1.):
        """
        Returns the fixed leg process involved in the OIS

        Parameters
        ----------
        - t_0 : `int`
           Arrival time of the contract
        - t : `int`
           Time at which we want to evaluate the process
        - T : `int`
           Maturity time of the contracr
        - notional : `float` (default = 1.) 
           Notional of the OIS

        Returns
        --------
        - `float`
           Value of the fixed leg process at time t
        """

        K = self.SwapRate(t_0, T)
        R = self.SwapRate(t, T)
        
        tau_0 = (T-t_0) / 365
        tau_t = (T-t) / 365
        
        return notional * ( (1 + K * tau_0) / (1 + R * tau_t)) 

    def GetFloatingLeg(self, t_0, t, notional = 1):
        """
        Evaluates the Floating leg process of the OIS
        
        Parameters
        ----------
        - t_0 : `int`
           Arrival time of the contract
        - t : `int`
           Time at which we want to evaluate the process
        - T : `int`
           Maturity time of the contracr
        - notional : `float` (default = 1.) 
           Notional of the OIS

        Returns
        --------
        - `float`
           Value of the floating leg process at time t
        """
        #Qui è corretto prendere t_0+1 perchè vogliamo fare il prodotto 0<= t <= t_0 poi facendone il rapporto
        #t_0 si semplifica
        B_t_0 = np.prod(1+(self.CIRProcess[0:t_0+1]*1/365))
        B_t = np.prod(1+(self.CIRProcess[0:t+1]*1/365))
        
        return notional * B_t / B_t_0

    def MarkToMarketPrice(self, delta, t_0, t, T, notional = 1):
        """
        Computes the value at time t of an OIS that was traded at time t_0 with maturity T

        Parameters
        ----------
        - delta : `int`
           could only assume values -1, +1. And it defines the role of the agent, where it is paying the fixed leg or the 
           floating one
        - t_0 : `int`
           Arrival time of the contract
        - t : `int`
           Time at which we want to evaluate the process
        - T : `int`
           Maturity time of the contracr
        - notional : `float` (default = 1.)
           Notional of the OIS

        Returns
        --------
        - `float`
           Value of OIS contract for agent described by `delta` at time `t`
        """
        #As we are considering only active nodes i think there is no need for that
        if t > T or t<0 or t<t_0:
            return 0

        return delta * notional * (self.GetFixedLeg(t_0, t, T) - self.GetFloatingLeg(t_0, t))

    def SimulateAllEdges(self):
        """
        It simulates the contract arrival process on every edge of the financial network
        """
        loop = tqdm(range(self.num_nodes), desc='Simulation')
        
        for i in loop:
            loop.set_postfix(Node=i)
            for j in range(i+1, self.num_nodes):
                self.SimulateEdge(i,j)

    def SimulateEdge(self,i,j):
        """
        It simulates the contract arrival process on the edge (i,j)

        Parameters
        -----------
        - i : `int`
           One of the node in the edge (i,j)
        - j : `int`
           The other node in the edge (i,j)
        """

        # Check i!=j as an institution cant trade with itself
        assert (i!=j)

        #Generate arrival times of contracts for edge (ij)
        arrival_times = self.GetArrivalTimes()
        
        #Generate maturities and deltas for every k-th transition (uniformly)
        maturities = np.zeros((len(arrival_times)), dtype=int) + int(364)
        """
        PLEASE NOTE: right now we are fixing maturity to one year
        """
        #maturities = np.random.randint(1, 365, size = len(arrival_times))
        deltas = np.random.choice ([-1,1], size = len(arrival_times))

        #Compute ending times like t_0 + T
        #ending_times = np.minimum(arrival_times + maturities, self.TotPoints)
        ending_times = arrival_times + maturities

        try:
        #Sometimes you don't have any contract between i and j
            
            #Anche qui per lo stesso motivo di Floatingleg prendiamo in B_t_0 fino a t+1
            swap_rates = self.SwapRate(arrival_times, ending_times)
            B_t_0s = np.asarray(([np.prod(1+(self.CIRProcess[0:t+1]*1/365)) for t in arrival_times]))

            contracts_array_i = np.stack([arrival_times, ending_times, deltas, swap_rates, B_t_0s], axis =1) 
            contracts_array_j = np.stack([arrival_times, ending_times,-1 * deltas, swap_rates, B_t_0s], axis = 1)

            # Convert specific columns to integers
            contracts_array_i[:, :2] = contracts_array_i[:, :2].astype(int)
            contracts_array_j[:, :2] = contracts_array_j[:, :2].astype(int)

            self.E[i,j] = {
                'contracts' : contracts_array_i
            }
            self.E[j,i] = {
                'contracts' : contracts_array_j
            }
        
        except:
            pass
    
    def GetInstantContractValue(self, t, contract):
        """
        Returns the instant Value (V_k^ij(t)) of a specific contract

        Parameters
        -----------
        - t : `int`
           Time at which we evaluate the contract
        - contract : np.array(`int`)
            Array of shape (5,) containg contract information \delta, t_0, T, K

        Returns
        -------
        - float
           V_k^{ij}(t) of contract k
        """
        
        t_0 = int(contract[0])
        T = int(contract[1])
        delta = int(contract[2])

        return self.MarkToMarketPrice(delta, t_0, t, T)


    def GetInstantContractMarginValue(self,t,contract):
        """
        Returns the instant Margin (M_k^ij(t)) of a specific contract

        Parameters
        -----------
        - t : `int`
           Time at which we evaluate the contract
        - contract : np.array(`int`)
            Array of shape (4,) containg contract information \delta, t_0, T, K

        Returns
        -------
        - float
           M_k^{ij}(t) of contract k
           bababa
        """

        V_t = self.GetInstantContractValue(t, contract)
        V_t_1 = self.GetInstantContractValue(t - 1 , contract)
      #   if t==1954 or t==1955:
      #            print('\n INSIDE MT for contract')
      #            print('t: ',t)
      #            print('------------------------------------')
      #            print('contract: ', contract)
      #            print(f't={t}, V(t)={V_t}')
      #            print(f't={t}, V(t-1)={V_t_1}')
      #            print(f't={t}, M(t)={(V_t - (1 + (self.CIRProcess[t] * 1 / 365) ) * V_t_1)}')
      #            print('-------------------------------------')

      #   if t == 1955:
      #       raise NotImplementedError

        return (V_t - (1 + (self.CIRProcess[t] * 1 / 365) ) * V_t_1)

    def GetInstantEdgeMarginValue(self,i,j,t):
        """
        Returns the instant Margin of an edge (M^ij(t)) by netting over the active k contracts

        Parameters
        -----------
        - i : `int`
           One of the node in the edge (i,j)
        - j : `int`
           The other node in the edge (i,j)
        - t : `int`
           Time at which we evaluate the contract

        Returns
        -------
        - float
           M^{ij}(t) of edge (i,j)
        """


        MarginValue = 0
        call = 0

        #Ciclo su tutti i contratti anche quelli non attivi che mi daranno Mt = 0 ... un po' inefficiente
        #Sopratutto fa edge per edge
        for cont in self.E[i,j]['contracts']:

            call+=1
            MarginValue += self.GetInstantContractMarginValue(t,cont)

        return MarginValue
    
    def GetMtForNodeAtActiveContracts(self, active_contracts_for_node_i, t):
        """
        Returns the instant margin for node i, given the set of active contracts

        Params
        -------
        - active_contracts_for_node_i : `torch.Tensor`
           tensor of contracts in the form (src, dst, t_0, T, delta, R, B_t_0)
           represents the active contracts for node i
        - t : `int`
           time index

        Returns
        --------
        - M^i(t) : `float`
           netted margin for node i at time t
        """
        M_t = 0

        #Questo cicla solo su i contratti attivi, molto più efficiente
        for j in range(active_contracts_for_node_i.shape[0]):
            
            M_t += self.GetInstantContractMarginValue(t, active_contracts_for_node_i[j,2:])
         
        return M_t


    def GetMtForNode(self, node_i, edge_index_at_time_t, t):
        """
        Returns the instant margin for node i, given the set of active edges is has at time t

        Params
        -------
        - node_i : `int`
           label of the node over which we perform the netting
        - edge_index_at_time_t : `torch.Tensor`
           Tensor in format COO of the active edges in the network at time t (has shape (2, num_edges))
        - t : `int`
           time index

        Returns
        --------
        - M^i(t) : `float`
           netted margin for node i at time t
        """

        #Lets get the edges in which node i appears:
        indices = (edge_index_at_time_t[0,:] == node_i).nonzero(as_tuple=True)
        target_nodes = edge_index_at_time_t[1][indices]

        #Let's sum over the target_nodes
        M_i_t = 0

        for j in target_nodes:

            M_i_t += self.GetInstantEdgeMarginValue(node_i.cpu(), j.cpu(), t)

        return M_i_t
    
def GetActiveContractsIndices(sim, contracts):
    """
    Computes the indices of the active contracts are time t

    Parameters
    -----------
    - time_array : torch.Tensor(int)
       arrival times throughout the entire graph
    - maturity_array : torch.Tensor(int)
      maturity times throughout the entire graph
    
    Returns
    --------
    - list(list)
       list of list of indices for the contracts that are active at time t
    """

    time_array = contracts[:,2]
    maturity_array = contracts[:,3]

    alive_indices_list=[]
    loop = tqdm(range(sim.TotPoints), desc='Active contracts @t')
    for t in loop:

        loop.set_postfix(Time=t)
        #Find the contracts that began in the past or at actual time
        formed_contract_indices = (time_array <= t).nonzero(as_tuple = True)[0]
        #Find the contracts whose maturity hasn't yet exceeded
        not_yet_expired_contract_indices = (maturity_array >= t).nonzero(as_tuple = True)[0]
        #Find the alive contracts
        alive_contract_indices = np.intersect1d(formed_contract_indices.cpu(), not_yet_expired_contract_indices.cpu())

        alive_indices_list.append(alive_contract_indices)
      #   if t ==1954 or t==1955:
      #       print('t: ',t)
      #       print('Formed contract indices: ', formed_contract_indices)
      #       print('not_yet expired: ', not_yet_expired_contract_indices)
      #       print('Alive_contract: ', alive_contract_indices)
    
    return alive_indices_list

def GetSimulationQuantitiesTensors(simulation, device):
    """
    Retrieves simulation quantities and outputs them as torch.Tensors

    Parameters
    ----------
    - simulation : `Simulation`
       Simulation object
    - device : `str`
       string object describing the device on which to upload the tensors
    
    Returns
    -------
    - tensors for the desired quantities
    """

    #edge index
    src_list = []
    dst_list = []

    #time
    time_list = []

    #Features
    delta_list = []
    maturity_list = []
    swaprate_list = []
    B_t_0_list = []

    # Iterate through the matrix
    for i, row in enumerate(simulation.E):
        
        for j, entry in enumerate(row):
            
            if isinstance(entry, dict) and 'contracts' in entry:
                
                contracts_array = entry['contracts']

                for contract_row in contracts_array:

                    # Append to the lists
                    src_list.append(i)
                    dst_list.append(j)
                    time_list.append(contract_row[0])
                    delta_list.append(contract_row[2])
                    maturity_list.append(contract_row[1])
                    swaprate_list.append(contract_row[3])
                    B_t_0_list.append(contract_row[4])


    # Convert lists to NumPy arrays and put them on GPU
    src_array = torch.Tensor(np.array(src_list)).to(device)
    dst_array = torch.Tensor(np.array(dst_list)).to(device)
    time_array = torch.Tensor(np.array(time_list)).to(device)
    delta_array = torch.Tensor(np.array(delta_list)).to(device)
    maturity_array = torch.Tensor(np.array(maturity_list)).to(device)
    swaprate_array = torch.Tensor(np.array(swaprate_list)).to(device)
    B_t_0_array = torch.Tensor(np.array(B_t_0_list)).to(device)

    return torch.cat([src_array.view(-1,1), dst_array.view(-1,1), time_array.view(-1,1), maturity_array.view(-1,1), delta_array.view(-1,1), swaprate_array.view(-1,1), B_t_0_array.view(-1,1)], dim=1)

def get_max_degree(dataset):
   """
   Returns the maximum degree observed for a TemporalData graph divided according to 'time_frequency'
   """

   max_degree = 0
   for i,data in enumerate(dataset):
      max_degree = max(max_degree, max(degree(data.edge_index[0].to(torch.int64))))

   return max_degree

def return_max_num_contracts(contracts, active_contracts_indices):
   """
   This function returns the maximum number of active contracts ever observed for an edge throughout the entire
   simulation horizon [0,T]
   """
   max_counts = []

   for t in range(len(active_contracts_indices)):
         index = active_contracts_indices[t]
         if index.size==0:
             pass
         else:
             uniques, counts = torch.unique(contracts[:,0][index], return_counts = True)
             #edge_index = torch.stack([contracts[:,0][index], contracts[:,1][index]], dim=0)
             #_, _, counts = torch.unique(edge_index.cpu(), dim=1, return_inverse=True, return_counts=True)
             maxCounts = torch.max(counts).item()
             max_counts.append(maxCounts)

   return np.max(np.asarray(max_counts))

def scale_feature_matrix(dataset):
    """
    Scales the feature matrix's feautures according to their type
    """
    #Get the maximum number of contracts
    num_contracts = int(dataset[0].x.shape[1]/3)
    num_nodes = dataset[0].x.shape[0]

    #Combine all the feature matrices by stacking them vertically
    combined_array = np.vstack([tensor.x.cpu().numpy() for tensor in dataset])

    #Now slice them in 3 by 3, so that for every row we only have a contract
    combined_of_combined = np.vstack([combined_array[:, 0 + (i*3): 3 + (i*3)] for i in range(num_contracts)])

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler on the combined data and transform it
    scaled_array = scaler.fit_transform(combined_of_combined)

    # splits the vertical stack into a list of num_contracts elements
    pre_scaled_tensors = np.split(scaled_array, num_contracts, axis=0)
    #In order to recover the original tensor we need to stack horizontally the list's elements
    finally_tensors = torch.hstack([torch.tensor(pre_scaled_tensors[i]) for i in range(num_contracts)])

    #The last step is revert the first vertical stacking as well:

    scaled_tensors = [finally_tensors[0 + i*num_nodes :num_nodes + i*num_nodes].squeeze() for i in range(len(dataset))]
    return scaled_tensors

def scale_matrix(X):
    """
    Scales the feature matrix's feautures according to its type
    """
    #Get the maximum number of contracts
    num_contracts = int(X.shape[1]/3)
    num_nodes = X.shape[0]

    #Now slice them in 3 by 3, so that for every row we only have a contract
    combined_array = np.vstack([X[:, 0 + (i*3): 3 + (i*3)] for i in range(num_contracts)])

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler on the combined data and transform it
    scaled_array = scaler.fit_transform(combined_array)

    # splits the vertical stack into a list of num_contracts elements
    pre_scaled_tensors = np.split(scaled_array, num_contracts, axis=0)
    #In order to recover the original tensor we need to stack horizontally the list's elements
    finally_tensors = torch.hstack([torch.tensor(pre_scaled_tensors[i]) for i in range(num_contracts)])

    return finally_tensors

def scale_targets(dataset):
    """
    This one does the scaling for targets y
    """
    
    #Get the maximum number of contracts
    num_days = len(dataset)

    #Combine all the feature matrices by stacking them vertically
    combined_array = np.vstack([tensor.y.view(-1,1).cpu().numpy() for tensor in dataset])

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler on the combined data and transform it
    scaled_array = scaler.fit_transform(combined_array)

    # splits the vertical stack into a list of num_contracts elements
    pre_scaled_tensors = np.split(scaled_array, num_days, axis=0)
    #In order to recover the original tensor we need to stack horizontally the list's elements
    finally_tensors = [torch.tensor(pre_scaled_tensors[i]).squeeze() for i in range(num_days)]

    return finally_tensors, scaler