from torch_geometric.datasets import BitcoinOTC
from torch_geometric.datasets import JODIEDataset
import torch_geometric.transforms as T
from torch_geometric.utils import degree
from torch_geometric.transforms import one_hot_degree, Compose, ToDevice
import torch
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import DataLoader 
from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch_sparse import coalesce
from torch_sparse import SparseTensor
import numpy as np

def get_dataset(args):
    
    print(f'Getting {args.dataset_name} dataset')

    if args.dataset_name == 'BTC':
        return build_dataset_btc(args)
    
    elif args.dataset_name == 'synthetic':
        return build_synthetic_dataset(args)
    
    return build_continuous_dataset(args)

def build_synthetic_dataset(args):
    dataset = torch.load('/u/mcitterio/data/synthetic/400_600_sin2data_.pt')
    dataset = dataset[200:]

    max_degree = 0
    for i,data in enumerate(dataset):
        max_degree = max(max_degree, max(degree(data.edge_index[0])))
    
    one_hot_degree_transformation = T.OneHotDegree(int(max_degree))

    subgraphs = []
    for i in range(len(dataset)):

        subgraph = Data(edge_index = dataset[i].edge_index, num_nodes = dataset[i].num_nodes)
        subgraph = one_hot_degree_transformation(subgraph)
        # print(dataset[i].x.shape)
        # dataset[i].x = torch.tensor(dataset[i].x).view(-1,1).to(torch.float) #da capire
        # dataset[i].x = torch.tensor(dataset[i].x).view(-1,1)
        # dataset[i].to(args.device)
        features = torch.tensor(dataset[i].x).view(-1, 1)

        # Concatenate the tensors along dimension 1
        subgraph.x = torch.cat([subgraph.x, features], dim=1)
        #print(subgraph)
        subgraph.to(args.device)
        subgraphs.append(subgraph)
        # subgraphs.append(dataset[i])
    
    return subgraphs


def build_dataset_btc(args):
    """
    Build the BitCoin OTC dataset with one-hot-encoding according to the maximum degree found throughout the time
    steps
    """

    # One Hot Degree transformation
    ohd = T.OneHotDegree(args.maximum_degree_btc)  
    # Dataset initialization                                           
    dataset = BitcoinOTC('.',edge_window_size=args.edge_window_size, transform=T.Compose([ToDevice(args.device)]))
    subgraphs=[]

    if args.safe_gpu_load:
        """
        We use sparse matrices for the node features
        """

        # Use a sparse tensor for one-hot degree encoding
        nodes = torch.arange(dataset.num_nodes, device=args.device)

        for data in dataset:

            degrees = degree(data.edge_index[0], num_nodes = dataset.num_nodes)
            values = torch.ones_like(degrees).to(args.device)

            size = torch.Size([dataset.num_nodes, args.maximum_degree_btc+1])
            X = torch.sparse_coo_tensor( torch.stack((nodes, degrees), dim=0) , values, size)  
            subgraphs.append(Data(edge_index=data.edge_index, num_nodes = dataset.num_nodes, x=X))

    else:
        """
        We dont use sparse matrices for the node features
        """
        # Dataset initialization                                           
        dataset = BitcoinOTC('.',edge_window_size=args.edge_window_size, transform=T.Compose([ohd,ToDevice(args.device)]))
        for data in dataset:
            subgraphs.append(data)

    return subgraphs

def build_continuous_dataset(args):
    """
    Build one dataset between MOOC - Wikipedia - LastFM
    """

    # Download TemporalData dataset object
    dataset = JODIEDataset('.', args.dataset_name, transform=T.Compose([ToDevice(args.device)]))[0]
    
    # Get unique time values
    time_frequency = args.time_frequency
    unique_times = torch.unique(dataset.t // time_frequency, sorted = True)

    # First we need to compute the maximum node degree this time frequency generate within graphs
    max_degree = get_max_degree(dataset, time_frequency).to(torch.int)
    one_hot_degree_transformation = T.OneHotDegree(max_degree)
    subgraphs = []

    # Iterate over unique times and create subgraphs
    for time in unique_times:
        # Select indices corresponding to the current time
        indices = (dataset.t// time_frequency == time).nonzero(as_tuple=True)[0]

        if args.safe_gpu_load:

            """
            This will load the node features as a sparse tensor [FOR ONE-HOT-ENCODED features coming from the node ranks]
            """

            # Use a sparse tensor for one-hot degree encoding
            nodes = torch.arange(dataset.num_nodes, device=args.device)

            # Extract relevant data for the subgraph
            edge_index = torch.stack([dataset.src[indices], dataset.dst[indices]], dim=0)
            degrees = degree(edge_index[0], num_nodes = dataset.num_nodes)
            values = torch.ones_like(degrees).to(args.device)

            size = torch.Size([dataset.num_nodes, max_degree+1])
            X = torch.sparse_coo_tensor( torch.stack((nodes, degrees), dim=0) , values, size)  
            subgraph = Data(edge_index = edge_index, x = X, num_nodes = dataset.num_nodes)      

        else:

            """
            Old-fashioned entire matrix loading
            """

            edge_index = torch.stack([dataset.src[indices], dataset.dst[indices]], dim=0)
            subgraph = Data(edge_index = edge_index, num_nodes = dataset.num_nodes)
            subgraph = one_hot_degree_transformation(subgraph)

        # Append the subgraph to the list
        subgraphs.append(subgraph)

    return subgraphs

def get_max_degree(dataset, time_frequency):
    """
    Returns the maximum degree observed for a TemporalData graph divided according to 'time_frequency'
    """

    unique_times = torch.unique(dataset.t // time_frequency, sorted = True)

    temp = []
    for time in unique_times:

        indices = (dataset.t// time_frequency == time).nonzero(as_tuple=True)[0]

        # Extract relevant data for the subgraph
        edge_index = torch.stack([dataset.src[indices], dataset.dst[indices]], dim=0)

        # Create a Data object for the subgraph
        subgraph = Data(edge_index=edge_index, num_nodes = dataset.num_nodes)

        # Append the subgraph to the list
        temp.append(subgraph)

    max_degree = 0
    for i,data in enumerate(temp):
        max_degree = max(max_degree, max(degree(data.edge_index[0])))

    del temp
    
    return max_degree

def get_sample(args, dataset, num_hist_steps, idx, condition_matrix = None, **kwargs):
        """
        Creates a `DataPoint`
        INPUTS:
        - `torch_geometric.data.Data` dataset: The dataset from which we want to drawn our sample (Train/Val/Test)
        - `int` num_hist_step: the number of previous timesteps we are basing our prediction on
        - `int` idx: index used to identify the portion of dataset
        - **kwargs: additional keywords like `all_edges`
        RETURNS:
        - `dict`: contains the index idx, a list of len=NumHistSteps of datasets, a list of positive edges, a list of negative
           edges that will be used as ground truth.
        """
    
        hist_adj_list = [dataset[i] for i in range(idx - num_hist_steps, idx)]    #contains both edges and features

        list_of_positive_edges = []
        list_of_negative_edges = []
        list_of_conditioning = []

        num_of_predictions = args.number_of_predictions                 #If 1 no extrapolation performed
        # print(f'idx: {idx}')
        for i in range(num_of_predictions):

            num_nodes = dataset[idx + i].num_nodes
            positive_edges = dataset[idx + i].edge_index

            neg_mult = args.neg_mult_training
            if 'all_edges' in kwargs.keys() and kwargs['all_edges'] == True:
                neg_mult = args.neg_mult_eval

            # Sample a number `neg_mult`*number of existing edges of negative samples
            negative_edges = negative_sampling(edge_index = positive_edges,
                                                    num_nodes = num_nodes,
                                                    num_neg_samples = positive_edges.shape[1] * neg_mult)
            if args.conditioning:
                
                if args.weird_conditions:
                    list_of_conditioning.append(idx - 1 +i)
                else:
                    list_of_conditioning.append(condition_matrix[idx - 1 +i])       # I am giving the graph stats of the day before the predition, so no data in the future

            list_of_positive_edges.append(positive_edges)
            list_of_negative_edges.append(negative_edges)
        
        return {'idx': idx, 'hist_adj_list': hist_adj_list, 'positive_edges': list_of_positive_edges, 'negative_edges': list_of_negative_edges, 'conditions':list_of_conditioning}

class DataSplit(Dataset):
    """
    Manages the data split
    """

    def __init__(self,args, dataset, start, end, condition_matrix, **kwargs):
        """
        Start and end are indices indicating what item belongs to this split
        """
        self.start = start
        self.end = end
        self.kwargs = kwargs
        self.NumHistSteps = args.num_hist_steps
        self.dataset = dataset                              # Is this thing going to cost me a lot?
        self.args = args
        self.condition_matrix = condition_matrix

    def __len__(self):
        return self.end - self.start
    
    def __getitem__(self, idx):

        idx = self.start + idx
        t = get_sample(self.args, self.dataset, self.NumHistSteps, idx, self.condition_matrix, **self.kwargs)
        return t

class Splitter():
    """
    Creates train - val 
    """

    def __init__(self, args, dataset, condition_matrix = None):

        NumHistSteps = args.num_hist_steps
        TrainSplit = args.train_split

        #Used for extrampolation, if == 1, no extrapolation will be performed
        number_of_predictions = args.number_of_predictions

        assert  TrainSplit < 1, 'There is no space for test sampling'

        # Total number of Datapoints that is possible to make
        total_len = len(dataset) - 2 * NumHistSteps - 2 * (number_of_predictions-1)

        # Set start and end training indices
        start = NumHistSteps
        end = int(np.floor(total_len*TrainSplit) + NumHistSteps)

        # Create training split
        
        TrainSet = DataSplit(args, dataset, start, end, condition_matrix)
        TrainSet = DataLoader(TrainSet, num_workers = 0)

        start = end + NumHistSteps + (number_of_predictions-1)
        end = len(dataset) - (number_of_predictions -1)

        # Create test set
        ValSet = DataSplit(args, dataset,  start, end, condition_matrix, all_edges=True)
        ValSet = DataLoader(ValSet, num_workers = 0)

        total_datapoints = len(TrainSet) + len(ValSet)
        
        print(f'Dataset splits sizes:  train {len(TrainSet)} ({len(TrainSet)/total_datapoints:.2f}%) val {len(ValSet)} ({len(ValSet)/total_datapoints:.2f}%)')

        self.train = TrainSet
        self.val = ValSet

