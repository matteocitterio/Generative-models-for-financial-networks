# Torch
import torch
import torch.nn.functional as F

# PyG libraries
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.utils import degree

# Models
from GCLSTM import GCLSTM
from EvolveGCNH import EvolveGCNH
from EvolveGCNO import EvolveGCNO
from DyGrEncoder import DyGrEncoder
from torch_geometric_temporal import GConvLSTM, GConvGRU
from torch_geometric.nn import GAT, GCN, GraphSAGE, GIN, GraphUNet, EdgeCNN

# Other stuff
import numpy as np

class BaseModel(torch.nn.Module):
    """
    This is the Base class for every model
    """
    def __init__(self, args):

        super(BaseModel, self).__init__()
        self.args = args
        self.num_hist_steps = args.lookback
        #self.regressor = Regressor(args)
        #self.dropout = torch.nn.Dropout(p = args.dropout_p)
        #self.decoder = torch.nn.LSTM(input_size = args.out_channels, hidden_size = args.out_channels, num_layers = 2, dropout=0.5)

    def forward(self, InputList):
        raise NotImplementedError("Subclasses must implement the forward method")

    def reset_parameters(self):
        pass

    # def extrapolate(self, node_embedding):
    #     """
    #     Evolves the first prediction node embedding using a RNN
    #     """
    #     #List for storing all the node embeddings
    #     list_of_predictions = []

    #     #Add the first step prediction
    #     list_of_predictions.append(node_embedding)

    #     H = node_embedding
    #     h = None
    #     c = None
    #     lstm_state = None
        
    #     #Add all the predictions from the second step onwards
    #     for t in range(self.args.number_of_predictions - 1):

    #         print('Extrapolate?')
    #         H, lstm_state = self.decoder(H, lstm_state)
    #         list_of_predictions.append(H)
            
    #     return list_of_predictions
    
    # def predict(self,args, InputList, condition):
    #     """
    #     Predict method
    #     """

    #     # Put it into evaluation mode
    #     self.eval()
        
    #     #Create embedding out of Inputs
    #     predicted_embedding = self.forward(InputList)
    #     #create prediction out of embedding
    #     return predict_node_regression(args, predicted_embedding[0], self.classifier, condition)
    

    
#######################################################################################################
"""
No history models (1 step predictions)
"""

class Evolve_GCN_H_model(BaseModel):
    """
    Principal EvolveGCNH model
    """
    def __init__(self, args):

        super(Evolve_GCN_H_model, self).__init__(args)

 
        self.recurrent_1 = EvolveGCNH(args.num_nodes, in_channels=args.batch_size, out_channels=args.lstm_hidden_size)

        self.recurrent_1.reset_parameters()

    def forward(self, InputList):
        """
        Forward method.
        Here the temporal factor is taken into account only to learn the GRU Weights, the model is fed every time
        step with the current design matrix (not an extrapolation) and the last output will later be compared to 
        a time step in the future.
        """

        assert (len(InputList) == self.num_hist_steps)
        
        for t in range(len(InputList)):

            X = InputList[t]

            out = self.recurrent_1(X.x, X.edge_index)  

        return self.extrapolate(out)
    
    def reset_parameters(self):

        for t in range(len(self.recurrent)):
            self.recurrent[t].reset_parameters()        # Reset parameters for the GCN Layers



class GC_LSTM_model(BaseModel):
    """
    GC-LSTM model (from PyG temporal)
    """
    def __init__(self, args):

        super(GC_LSTM_model, self).__init__(args)
        self.recurrent_1 = GCLSTM(in_channels = 1,#args.input_size, 
                                   out_channels = args.lstm_hidden_size,
                                   K = 2)
        self.reset_parameters()
        self.args = args
        self.name=str(args.regressor_hidden_size) + str(args.lookback)

    def forward(self, InputList):

        assert (len(InputList) == self.num_hist_steps)

        H = None
        C = None
        
        for t in range(len(InputList)):

            X = InputList[t]
            H, C = self.recurrent_1(X.x, X.edge_index, H=H, C=C)    # Qui potremo aggiungere anche i pesi
            #H = self.dropout(H)
            
        return H



    # def forward(self, InputList):
    #     """
    #     Forward method.
    #     Here the temporal factor is taken into account only to learn the GRU Weights, the model is fed every time
    #     step with the current design matrix (not an extrapolation) and the last output will later be compared to 
    #     a time step in the future.
    #     """

    #     Hs = []
    #     for batch_idx in range(self.args.batch_size):

    #         H = None
    #         C = None

    #         for t in range(len(InputList)):

    #             X = InputList[t]

    #             mask = (X.batch == batch_idx)
    #             edge_mask = mask[X.edge_index[0]] & mask[X.edge_index[1]]

    #             graph = Data(x = X.x[mask], edge_index = X.edge_index[:, edge_mask] - torch.min(X.edge_index[:,edge_mask]))

    #             H, C = self.recurrent_1( graph.x, graph.edge_index, H=H, C=C)    # Qui potremo aggiungere anche i pesi

    #         Hs.append(H)

    #     return torch.stack(Hs)
    
    def reset_parameters(self):

        #self.lstm.reset_parameters()
        for layer in [self.recurrent_1]:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


"""
On-top models that use node embeddings for different tasks
"""

class LSTMModel(torch.nn.Module):
    """
    This is the LSTM model part
    """
    def __init__(self,args):
        
        super(LSTMModel, self).__init__()

        self.lstm = torch.nn.LSTM(input_size = args.contract_size, 
                                   hidden_size = args.lstm_hidden_size,
                                   num_layers = 1)
        self.regressor = Regressor(args)
        self.relu=torch.nn.ReLU()
        self.fc = torch.nn.Linear(args.contract_size,1)

        self.args = args

        self.lstm.reset_parameters()
        self.lstm.flatten_parameters()
    
    def forward(self, x, hidden_intensity, r):

        """
        x -> batch of contracts (NBatches, lookback, N, input_size * max_contracts)
        hidden_intensity-> result (NBatches,N,hidden) of the GCLSTM
        r-> conditioning
        """
        
        # print('contract batch_shape: ',x.shape)
        pred = torch.zeros(( x.shape[0], self.args.num_nodes, self.args.steps_ahead)).to(torch.float32).to(self.args.device)

        #Cycle over the number of nodes
        for n in range(self.args.num_nodes):

            #Cycle over the maximum number of contracts
            for i in range(x.shape[3]//self.args.contract_size):
            
                #select the contract sequence for the node
                contract = x[:,:,n, i*self.args.contract_size : (i+1)*self.args.contract_size] 
                #if the contract is non empty:
                if not torch.all(torch.eq(contract, torch.zeros_like(contract))):

                    for j in range(self.args.steps_ahead):

                        lstm_hidden, _ = self.lstm(contract)
                        prediction = self.regressor(torch.squeeze(torch.cat([lstm_hidden[:,-1,:], hidden_intensity[:,n,:], r[:,j].reshape(-1,1)],dim=1)))
                        pred[:,n, j] += self.fc(self.relu(prediction)).squeeze()
                        #prediction = prediction.unsqueeze(0).unsqueeze(0) #CHANGE TRAINING / PREDICTION
                        prediction = prediction.unsqueeze(1)
                        contract = torch.cat([contract[:, 1:, :], prediction], dim = 1)
                        
                else:
                    break
         
        return pred
    
    def forward_pred(self, x, hidden_intensity, r):

        """
        x -> batch of contracts (NBatches, lookback, N, input_size * max_contracts)
        hidden_intensity-> result (NBatches,N,hidden) of the GCLSTM
        r-> conditioning
        """
        
        pred = torch.zeros(( x.shape[0], self.args.num_nodes, self.args.steps_ahead)).to(torch.float32).to(self.args.device)

        #Cycle over the number of nodes
        for n in range(self.args.num_nodes):
            
            #Cycle over the maximum number of contracts
            for i in range(x.shape[3]//self.args.contract_size):
            
                #select the contract sequence for the node
                contract = x[:,:,n, i*self.args.contract_size : (i+1)*self.args.contract_size] 
                #if the contract is non empty:
                if not torch.all(torch.eq(contract, torch.zeros_like(contract))):

                    for j in range(self.args.steps_ahead):

                        lstm_hidden, _ = self.lstm(contract)
                        prediction = self.regressor(torch.squeeze(torch.cat([lstm_hidden[:,-1,:], hidden_intensity[:,n,:], r[:,j].reshape(-1,1)],dim=1)))
                        pred[:,n, j] += self.fc(self.relu(prediction)).squeeze()
                        prediction = prediction.unsqueeze(0).unsqueeze(0) #CHANGE TRAINING / PREDICTION
                        #prediction = prediction.unsqueeze(1)
                        contract = torch.cat([contract[:, 1:, :], prediction], dim = 1)
                        
                else:
                    break
         
        return pred
    
class Regressor(torch.nn.Module):
    """
    Classifier for the link prediction task
    """
    def __init__(self, args):

        super().__init__()

        layers = [torch.nn.Linear(args.lstm_hidden_size*2 + 1, args.regressor_hidden_size),
                  torch.nn.ReLU(),
                  torch.nn.Linear(args.regressor_hidden_size, args.regressor_hidden_size_2),
                  torch.nn.ReLU(),
                  torch.nn.Linear(args.regressor_hidden_size_2, args.contract_size)]
        
        self.regressor = torch.nn.Sequential(*layers)
        self.args = args
        
        self.reset_parameters()
    
    def forward(self,x):
        return self.regressor(x)

    def reset_parameters(self):

        #self.lstm.reset_parameters()
        for layer in [self.regressor]:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
class CompleteModel(torch.nn.Module):

    def __init__(self,args):
            
        super(CompleteModel, self).__init__()

        self.gclstm = GC_LSTM_model(args)
        self.lstm = LSTMModel(args)
        self.name=str(args.num_nodes)+str(args.lookback)+str(args.batch_size)+str(args.steps_ahead)#    'ammammate'

        #self.regressor = Regresso(args) already inside lstm
        #self.reset_params() already inside the two objects.

    def forward(self, InputList, x, r):

        #gc_lstm is fed with the historical (X,A) list where X are the node features
        hidden_intensity = self.gclstm(InputList)
        pred = self.lstm(x, hidden_intensity, r)

        return pred   