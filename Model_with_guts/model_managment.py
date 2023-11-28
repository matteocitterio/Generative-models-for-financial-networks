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
from torch_geometric_temporal import GConvLSTM, GConvGRU, DyGrEncoder
from torch_geometric.nn import GAT, GCN, GraphSAGE, GIN, GraphUNet, EdgeCNN

# Other stuff
import utils

class BaseModel(torch.nn.Module):
    """
    This is the Base class for every model
    """
    def __init__(self, args):

        super(BaseModel, self).__init__()
        self.num_hist_steps = args.num_hist_steps
        self.classifier = Classifier(args, in_features= 2 * args.out_channels)
        self.dropout = torch.nn.Dropout(p = args.dropout_p)
        self.decoder = torch.nn.LSTM(input_size = args.out_channels, hidden_size = args.out_channels, num_layers = 2, dropout=0.5)

    def forward(self, InputList):
        raise NotImplementedError("Subclasses must implement the forward method")

    def reset_parameters(self):
        pass

    def extrapolate(self, args, node_embedding):
        list_of_predictions = []

        H = node_embedding
        h = None
        c = None
        lstm_state = None
        
        for t in range(args.number_of_predictions - 1):

            H, lstm_state = self.decoder(H, lstm_state)
            list_of_predictions.append(H)
            
        return list_of_predictions
    
class EqualTimeModel(BaseModel):
    """
    Class for equal time predictions
    """
    def __init__(self,args):
        super(EqualTimeModel, self).__init__(args)
        self.GNN = None

    def reset_parameters(self):
        self.GNN.reset_parameters()

    def forward(self, InputList):
        """
        This must be a sequence with length 1 in the past
        """
        assert (len(InputList) == 1)
        X = InputList[0]

        return self.GNN(X.x, X.edge_index)
 
class GAT_model(EqualTimeModel):
    """
    For equal time predictions
    """
    def __init__(self, args, in_channels):
        super(GAT_model, self).__init__(args)
        self.GNN = GAT(in_channels= in_channels, hidden_channels=args.hidden_channels[0], num_layers=4, out_channels=args.out_channels)
        self.reset_parameters()
    
class GCN_model(EqualTimeModel):
    """
    For equal time predictions
    """
    def __init__(self, args, in_channels):
        super(GCN_model, self).__init__(args)
        self.GNN = GCN(in_channels= in_channels, hidden_channels=args.hidden_channels[0], num_layers=4, out_channels=args.out_channels)
        self.reset_parameters()
    
class GraphSAGE_model(EqualTimeModel):
    """
    For equal time predictions
    """
    def __init__(self, args, in_channels):
        super(GraphSAGE_model, self).__init__(args)
        self.GNN = GraphSAGE(in_channels= in_channels, hidden_channels=args.hidden_channels[0], num_layers=4, out_channels=args.out_channels)
        self.reset_parameters()
    
class GIN_model(EqualTimeModel):
    """
    For equal time predictions
    """
    def __init__(self, args, in_channels):
        super(GIN_model, self).__init__(args)
        self.GNN = GIN(in_channels= in_channels, hidden_channels=args.hidden_channels[0], num_layers=4, out_channels=args.out_channels)
        self.reset_parameters()
    
class GraphUNet_model(EqualTimeModel):
    """
    For equal time predictions
    """
    def __init__(self, args, in_channels):
        super(GraphUNet_model, self).__init__(args)
        self.GNN = GraphUNet(in_channels= in_channels, hidden_channels=args.hidden_channels[0], depth=4, out_channels=args.out_channels)
        self.reset_parameters()

class EdgeCNN_model(EqualTimeModel):
    """
    for equal time predictions
    """
    def __init__(self, args, in_channels):
        super(EdgeCNN_model, self).__init__(args)
        self.GNN = EdgeCNN(in_channels= in_channels, hidden_channels=args.hidden_channels[0], num_layers=4, out_channels=args.out_channels)
        self.reset_parameters()

class DyGrEncoder_model(BaseModel):
    """
    DyGrEncoder model (from PyG temporal)
    """

    def __init__(self, args, conv_out_channels):
        
        super(DyGrEncoder_model, self).__init__(args)
        self.recurrent_1 = DyGrEncoder(conv_out_channels, args.conv_num_layers, args.conv_aggr, lstm_out_channels=args.out_channels , lstm_num_layers = args.lstm_num_layers)
        self.reset_parameters()

    def forward(self, InputList):
        """
        Forward method.
        Here the temporal factor is taken into account only to learn the GRU Weights, the model is fed every time
        step with the current design matrix (not an extrapolation) and the last output will later be compared to 
        a time step in the future.
        """

        assert (len(InputList) == self.num_hist_steps)

        H = None
        C = None
        
        for t in range(len(InputList)):

            X = InputList[t]
            H_tilde, H, C = self.recurrent_1(X.x, X.edge_index, H=H, C=C)    # Qui potremo aggiungere anche i pesi
            H_tilde = self.dropout(H)
            
        return H_tilde
    
    def reset_parameters(self):
        
        self.recurrent_1.conv_layer.reset_parameters()

class GConvLSTM_model(BaseModel):
    """
    GConvLSTM model (from PyG temporal)
    """

    def __init__(self, args, in_channels):

        super(GConvLSTM_model, self).__init__(args)
        self.recurrent_1 = GConvLSTM(in_channels, args.out_channels, K=args.k_chebyshev)
        self.reset_parameters()

    def forward(self, InputList):
        """
        Forward method.
        Here the temporal factor is taken into account only to learn the GRU Weights, the model is fed every time
        step with the current design matrix (not an extrapolation) and the last output will later be compared to 
        a time step in the future.
        """

        assert (len(InputList) == self.num_hist_steps)

        H = None
        C = None
        
        for t in range(len(InputList)):

            X = InputList[t]
            H, C = self.recurrent_1(X.x, X.edge_index, H=H, C=C)    # Qui potremo aggiungere anche i pesi
            H = self.dropout(H)
            
        return H
    
class GConvGRU_model(BaseModel):
    """
    GConvGRU model (from PyG temporal)
    """

    def __init__(self, args, in_channels):

        super(GConvGRU_model, self).__init__(args)

        self.recurrent_1 = GConvGRU(in_channels, args.out_channels, K=args.k_chebyshev)
        self.reset_parameters()

    def forward(self, InputList):
        """
        Forward method.
        Here the temporal factor is taken into account only to learn the GRU Weights, the model is fed every time
        step with the current design matrix (not an extrapolation) and the last output will later be compared to 
        a time step in the future.
        """

        assert (len(InputList) == self.num_hist_steps)

        H = None
        
        for t in range(len(InputList)):

            X = InputList[t]
            H = self.recurrent_1(X.x, X.edge_index, H=H)    # Qui potremo aggiungere anche i pesi
            H = self.dropout(H)
            
        return H

class GC_LSTM_model(BaseModel):
    """
    GC-LSTM model (from PyG temporal)
    """
    def __init__(self, args, in_channels):

        super(GC_LSTM_model, self).__init__(args)
        self.recurrent_1 = GCLSTM(in_channels, args.out_channels, K=args.k_chebyshev)
        self.reset_parameters()

    def forward(self, InputList):
        """
        Forward method.
        Here the temporal factor is taken into account only to learn the GRU Weights, the model is fed every time
        step with the current design matrix (not an extrapolation) and the last output will later be compared to 
        a time step in the future.
        """

        assert (len(InputList) == self.num_hist_steps)

        H = None
        C = None
        
        for t in range(len(InputList)):

            X = InputList[t]
            H, C = self.recurrent_1(X.x, X.edge_index, H=H, C=C)    # Qui potremo aggiungere anche i pesi
            H = self.dropout(H)
            
        return H

class Evolve_GCN_H_model(BaseModel):
    """
    Principal EvolveGCNH model
    """
    def __init__(self, args, node_count, in_channels):

        super(Evolve_GCN_H_model, self).__init__(args)

        # Create list of matrices dimensions for the model layers
        ins, outs = utils.get_channels_lists(args, in_channels)

        #Check if there is a proper number of dimensions given
        assert len(ins) == args.number_of_layers

        # Append model layers
        self.recurrent = []
        for i in range(args.number_of_layers):
            self.recurrent.append(EvolveGCNH(node_count, in_channels=ins[i], out_channels=outs[i]))

        self.reset_parameters()

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
            out = X.x
            for h in range(len(self.recurrent)):

                out = self.recurrent[h](out, X.edge_index)  

            out = self.dropout(out)

        return out
    
    def reset_parameters(self):

        for t in range(len(self.recurrent)):
            self.recurrent[t].reset_parameters()        # Reset parameters for the GCN Layers

        self.classifier.reset_parameters()

    
class Classifier(torch.nn.Module):
    """
    Classifier for the link prediction task
    """
    def __init__(self, args, in_features):

        super().__init__()
        self.in_fc = torch.nn.Linear(in_features, args.hidden_feats)
        self.out_fc = torch.nn.Linear(args.hidden_feats, 1)
        self.reset_parameters()
        
    def reset_parameters(self, ):
        self.in_fc.reset_parameters()
        self.out_fc.reset_parameters()

    def forward(self, coupled_embedding):
       
        h = self.in_fc(coupled_embedding)
        h = torch.nn.functional.relu(h)
        return self.out_fc(h)
    
# class EdgeRegressor(torch.nn.Module):

# Others useful model methods
        
def build_model(args, node_count = None, in_channels = None):
    """
    Util which helps to build the model
    """
    
    if args.model_name == 'GC-LSTM':
        print('GC-LSTM model selected')
        print(f'Number of hist steps: {args.num_hist_steps}\n\n')
        return GC_LSTM_model(args, in_channels)
    
    elif args.model_name == 'GAT':
        print('GAT model selected')
        print(f'Number of hist steps: {args.num_hist_steps}\n\n')
        return GAT_model(args, in_channels)
    
    elif args.model_name == 'GCN':
        print('GCN model selected')
        print(f'Number of hist steps: {args.num_hist_steps}\n\n')
        return GCN_model(args, in_channels)
    
    elif args.model_name == 'GraphSAGE':
        print('GraphSAGE model selected')
        print(f'Number of hist steps: {args.num_hist_steps}\n\n')
        return GraphSAGE_model(args, in_channels)
    
    elif args.model_name == 'GIN':
        print('GIN model selected')
        print(f'Number of hist steps: {args.num_hist_steps}\n\n')
        return GIN_model(args, in_channels)
    
    elif args.model_name == 'PNA':
        print('PNA model selected')
        print(f'Number of hist steps: {args.num_hist_steps}\n\n')
        return PNA_model(args, in_channels)
    
    elif args.model_name == 'GraphUNet':
        print('GraphUNet model selected')
        print(f'Number of hist steps: {args.num_hist_steps}\n\n')
        return GraphUNet_model(args, in_channels)
    
    elif args.model_name == 'EdgeCNN':
        print('EdgeCNN model selected')
        print(f'Number of hist steps: {args.num_hist_steps}\n\n')
        return EdgeCNN_model(args, in_channels)
    
    elif args.model_name == 'GConvLSTM':
        print('GConvLSTM model selected')
        print(f'Number of hist steps: {args.num_hist_steps}\n\n')
        return GConvLSTM_model(args, in_channels)
    
    elif args.model_name == 'GConvGRU':
        print('GConvGRU model selected')
        print(f'Number of hist steps: {args.num_hist_steps}\n\n')
        return GConvGRU_model(args, in_channels)

    elif args.model_name == 'EvolveGCN-H':

        print('EvolveGCN-H model selected')
        print(f'Number of hist steps: {args.num_hist_steps}\n\n')
        return Evolve_GCN_H_model(args, node_count, in_channels)
    
    elif args.model_name == 'DyGrEncoder':
        print('DyGrEncoder model selected')
        print(f'Number of hist steps: {args.num_hist_steps}\n\n')
        return DyGrEncoder_model(args, conv_out_channels = in_channels)
    
    print('Please input a valid model name!')
    return 0

def gather_node_embeddings(node_embeddings, node_indices):
        """
        concatenates the node embeddings for the two nodes involved in a link
        INPUT:
        - node_embeddings: the node embedding matrix produced through the model
        - node_indices: edge set (with positive and negative samples)
        """

        cls_input = []

        for node_set in node_indices:

            cls_input.append(node_embeddings[node_set])
        
        return torch.cat(cls_input, dim=1)
    
def predict(node_embeddings, classifier, node_indices):
    """
    Returns the predictions of the classifier
    INPUTS:
    - node_embeddings: node representation computed by the model
    - classifier: classifier model
    - node_indices: indices of the nodes we are basing our link prediction on 
    """

    predict_batch_size = 100000
    gather_predictions = []

    for i in range(1 + (node_indices.size(1) // predict_batch_size)):
        """
        The for loop is just because it is not efficient for large tensors especially without a gpu to do predictions
        all at once so what we do is just diving the tensor in bunches of `predict_batch_size` and then just 
        pasting them together
        """

        cls_input = gather_node_embeddings(node_embeddings, node_indices[:, i*predict_batch_size:(i+1)*predict_batch_size])
        predictions = classifier(cls_input)
        gather_predictions.append(predictions)

    gather_predictions = torch.cat(gather_predictions, dim=0)
    return gather_predictions 