import torch
from EvolveGCNH import EvolveGCNH

class DoubleLayerModel(torch.nn.Module):
    """
    Principal EvolveGCNH model
    """
    def __init__(self, args, node_count, in_channels):

        super().__init__()
        hidden_channels = args.hidden_channels
        out_channels = args.out_channels

        self.recurrent_1 = EvolveGCNH(node_count, in_channels, hidden_channels)
        self.recurrent_2 = EvolveGCNH(node_count, hidden_channels, out_channels)
        self.num_hist_steps = args.num_hist_steps
        self.classifier = Classifier(args, in_features = 2 * out_channels, out_features=2)

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
            HiddenEmbedding = self.recurrent_1(X.x, X.edge_index)    # Qui potremo aggiungere anche i pesi
            out = self.recurrent_2(HiddenEmbedding, X.edge_index)

        return out
    
    def reset_parameters(self):

        self.recurrent_1.reset_parameters()  # Reset parameters for your EvolveGCNH modules
        self.recurrent_2.reset_parameters()
        self.classifier.reset_parameters()
    
class Classifier(torch.nn.Module):
    """
    Classifier model built on top of the EvolveGCNH Model.
    It takes the node embeddings of two nodes and outputs logits of shape (num_nodes,2) as predictions for the
    edge to be inside one of the two classes (i.e. 0 if it shouldn exist, 1 if it should).
    """
    def __init__(self,args, in_features, out_features=2 ):

        super(Classifier,self).__init__()
        hidden_feats = args.hidden_feats
        activation = torch.nn.ReLU()
        num_feats = in_features
        print ('CLS num_feats',num_feats)

        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_features = num_feats,
                                                       out_features = hidden_feats),
                                       activation,
                                       torch.nn.Linear(in_features= hidden_feats,
                                                       out_features = out_features))

    def forward(self,x):
        return self.mlp(x)
    
    def reset_parameters(self):
        for layer in self.mlp:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
    
class DoubleLayerModelExtrapolation(torch.nn.Module):
    """
    Still to implement and test
    """

    def __init__(self, args, node_count, in_channels):

        super().__init__()
        hidden_channels = args.hidden_channels
        out_channels = args.out_channels

        self.recurrent_1 = EvolveGCNH(node_count, in_channels, hidden_channels)
        self.recurrent_2 = EvolveGCNH(node_count, hidden_channels, out_channels)
        self.num_hist_steps = args.num_hist_steps
        self.reshaper_layer = torch.nn.Linear(out_channels, in_channels)
        self.classifier = Classifier(args, in_features = 2 * out_channels, out_features=2)

    def forward(self, InputList):

        assert (len(InputList) == self.num_hist_steps)
        threshold = 0.5
        
        for t in range(len(InputList)):

            print('Timestep:', t)

            if t == len(InputList) - 1:
                """
                We need to make extrapolations in here
                """
                #print('extravaganza bitch')
                logits = predict(out, self.classifier, InputList[t].edge_index)
                #print('pre sigmoid')
                #print(logits[0:4])
                logits = torch.sigmoid(logits)
                #print('post sigmoid')
                #print(logits[0:4])
                
                # Make binary predictions based on the threshold.
                predictions = (logits >= threshold).int()
                out = self.reshaper_layer(out)
                HiddenEmbedding = self.recurrent_1(out, predictions)    # Qui potremo aggiungere anche i pesi
                out = self.recurrent_2(HiddenEmbedding, X.edge_index)
                

            print('Normal shit')
            X = InputList[t]
            HiddenEmbedding = self.recurrent_1(X.x, X.edge_index)    # Qui potremo aggiungere anche i pesi
            out = self.recurrent_2(HiddenEmbedding, X.edge_index)
                
        return out
    
    def reset_parameters(self):

        self.recurrent_1.reset_parameters()  # Reset parameters for your EvolveGCNH modules
        self.recurrent_2.reset_parameters()
        self.classifier.reset_parameters()
    
def build_model(args, node_count, in_channels):
    """
    Util which helps to build the model
    """

    if args.extrapolation:
        return DoubleLayerModelExtrapolation()
    
    return DoubleLayerModel(args, node_count, in_channels)

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
    
def predict(node_embeddings, classifier, gt_node_indices):
    """
    Returns the predictions of the classifier
    """

    predict_batch_size = 100000
    gather_predictions = []

    for i in range(1 + (gt_node_indices.size(1) // predict_batch_size)):
        """
        The for loop is just because it is not efficient for large tensors especially without a gpu to do predictions
        all at once so what we do is just diving the tensor in bunches of `predict_batch_size` and then just 
        pasting them together
        """

        cls_input = gather_node_embeddings(node_embeddings, gt_node_indices[:, i*predict_batch_size:(i+1)*predict_batch_size])
        predictions = classifier(cls_input)
        gather_predictions.append(predictions)

    gather_predictions = torch.cat(gather_predictions, dim=0)
    return gather_predictions 