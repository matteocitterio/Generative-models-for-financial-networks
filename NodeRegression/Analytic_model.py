import numpy as np
from CIR import get_CIR
from tqdm import tqdm
from torch_geometric.utils import degree
import torch
from sklearn.preprocessing import MinMaxScaler
from simulation import Simulation

class PerfectPrediction:
    """
    Analitically reconstructs the margins
    """

    def __init__(self, nodes, gamma, years):
        """
        Start by instantiating a simulation
        """
        
        alpha = 0.6
        b = 0.02
        sigma = 0.14
        v_0 = 0.04
        years = years
        num_nodes = nodes

        self.sim = Simulation(alpha, b, sigma, v_0, years, seed = True, num_nodes = num_nodes, gamma = gamma)

    def ExtractContract(self,contract):
        """
        Takes a contract as input and returns the quantity. Ok I am just lazy
        """

        return contract[0], contract[1], contract[2], contract[3], contract[4]

    def GetFixedLegIJK(self,contract, t):
        """
        This takes a contract (a single one as input) and time t and computes the fixed leg
        """

        t_0, T, delta, R_0, B_t_0 = self.ExtractContract(contract)

        T=int(T)
        t = int(t)

        notional = 1
        
        tau_0 = (T-t_0) / 365
        tau_t = (T-t) / 365

        return notional * ( (1 + R_0 * tau_0) / (1 + self.sim.SwapRate(t,T) * tau_t)) 


    def GetFloatingLegIJK(self, contract, B_t):
        """
        Returns the floating leg of a single contract at time defined by B_t
        """

        t_0, T, delta, R_0, B_t_0 = self.ExtractContract(contract)

        notional = 1
        
        return notional * B_t/ B_t_0
    
    def GetContractValueIJK(self, contract, B_t, t):
        """
        Takes a contract (a single one), B_t and t and computes the contract value
        """

        t_0, T, delta, R_0, B_t_0 = self.ExtractContract(contract)
        
        if torch.all(contract == torch.zeros_like(contract)).item() or t>T:
            return torch.tensor(0)

        notional = 1

        return notional * delta * (self.GetFixedLegIJK(contract, t) - self.GetFloatingLegIJK(contract, B_t))
    
    def GetContractValue(self, contracts, B_t, t):
        """
        This takes all the active contracts between node i,j and sums them
        """
        
        V = 0

        #For each of the 34 contracts we want to compute the value of the single contract
        for i in range(contracts.shape[0]):

            value = self.GetContractValueIJK(contracts[i], B_t, t)
            #print(f'time: {t} V(t)= {value}, Fixed={self.GetFixedLegIJK(contracts[i], t)}, Floating={self.GetFloatingLegIJK(contracts[i], B_t)}')
            V += value.to(torch.float64)

        return torch.tensor(V)

    
    def GetUltimateContractLine(self, X):
        """
        In the training set we have a shape: [N_batches, lookback, N_features]
        """

        contracts_last_time = []

        #Loop over the batches
        for i in range(X.shape[0]):

            #Those are a data inside a SINGLE batch, size [lookup, features]
            data_without_batch = X[i]     

            #Pick up just the last time, size [1, features]
            last_time = data_without_batch[-1] 

            #Select only the contract features and not the stuff you put in the concat
            features = last_time[:-2]    

            #5 is because for every contract we have 5 elements: (t_0, T, delta, K, B_t_0)
            contracts = [features[0 + (j*5) : 5 + (j*5)] for j in range(int(len(features) / 5))]
            contracts = torch.stack(contracts)

            contracts_last_time.append(contracts)

        contracts_last_time = torch.stack(contracts_last_time)
        #This returns something of shape [N_batches, N_max_contracts, 5]
        return contracts_last_time
    
    def GetUltimateBt(self, X):
        """
        In the training set we have a shape: [N_batches, lookback, N_features]
        """
        #OCIO CHE IL -1 DIPENDE DA COME COSTRUISCO LA MATRICE X_data IN TEST_1_NODE..
        B_ts = []
        for i in range(X.shape[0]):
            B_ts.append(X[i,-1, -1])
            
        return torch.stack(B_ts)
    
    def GetUltimatert(self, r):
        """
        In the training set we have a shape: [N_batches, lookback, N_features]
        """
        r_ts = []
        for i in range(r.shape[0]):
            r_ts.append(r[i,-1, 0])

        return torch.stack(r_ts)
    
    def GetUltimateTime(self,X):

        """
        In the training set we have a shape: [N_batches, lookback, N_features]
        """
        #-2 depends on the way in which I do the torch.cat when building the features matrix
        t_s = []
        for i in range(X.shape[0]):
            t_s.append(X[i,-1, -2])
        
        return torch.stack(t_s)

    
    #Per batch we want the predictions tool.
    def GetVt(self, X_batch, r_batch):
        """
        This takes a training batch of shape: [N_batch, lookback, n_features] and give the Vt of shape [N_batch]
        """

        contract_timed_for_batch = self.GetUltimateContractLine(X_batch)
        
        B_t_s_for_batch = self.GetUltimateBt(X_batch)
        t_s_for_batch = self.GetUltimateTime(X_batch)
        r_s_for_batch = self.GetUltimatert(r_batch)

        oldVt = []
        newVT = []

        for i in range(contract_timed_for_batch.shape[0]):

            #print(f'Time: {t_s_for_batch[i]}, r_t[i+1]={sim.CIRProcess[int(t_s_for_batch[i]+1)].item()}, r_s_for_batch={(r_s_for_batch[i]).item()}')
            oldVt.append(self.GetContractValue(contract_timed_for_batch[i], B_t=B_t_s_for_batch[i].item(), t=t_s_for_batch[i] ))
            newVT.append(self.GetContractValue(contract_timed_for_batch[i], B_t=B_t_s_for_batch[i].item()*(1+r_s_for_batch[i].item()/365), t=t_s_for_batch[i]+1 ))


        oldVt = torch.stack(oldVt)
        newVT = torch.stack(newVT)

        MT = []

        for i in range(contract_timed_for_batch.shape[0]):

            MT.append(newVT[i] - ((1+r_s_for_batch[i]/365))*oldVt[i])

        MT = torch.stack(MT)

        return MT
