from torch_geometric.datasets import BitcoinOTC
import torch_geometric.transforms as T
from torch_geometric.utils import degree
from torch_geometric.transforms import one_hot_degree, Compose, ToDevice
import torch
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import DataLoader 
from torch_geometric.data import Data
from torch.utils.data import Dataset
import numpy as np


def build_dataset(args):
    """
    Build the BitCoin OTC dataset with one-hot-encoding according to the maximum degree found throughout the time
    steps
    """

    # One Hot Degree transformation
    ohd = T.OneHotDegree(args.maximum_degree)  
    # Dataset initialization                                           
    dataset = BitcoinOTC('.',edge_window_size=args.edge_window_size, transform=T.Compose([ohd,ToDevice('cpu')]))
    return dataset

def get_sample(args, dataset, num_hist_steps, idx, **kwargs):
        """
        Creates a `DataPoint`
        INPUTS:
        - `torch_geometric.data.Data` dataset: The dataset from which we want to drawn our sample (Train/Val/Test)
        - `int` num_hist_step: the number of previous timesteps we are basing our prediction on
        - `int` idx: index used to identify the portion of dataset
        - **kwargs: additional keywords like `all_edges`
        RETURNS:
        - `dict`: contains the index idx, a list of len=NumHistSteps of datasets, a `label_sp` dataset used as GT 
        """
    
        hist_adj_list = [dataset[i] for i in range(idx - num_hist_steps, idx)]    #contains both edges and features

        num_nodes = dataset[idx].num_nodes
        label_adj = dataset[idx]
        label_adj.edge_attr = torch.ones(label_adj.edge_attr.size(0), dtype = torch.int64)    #Consider them as GT (==1)

        neg_mult = args.neg_mult

        if 'all_edges' in kwargs.keys() and kwargs['all_edges'] == True:
            
            # Consider all possible edges
            non_existing_adj = negative_sampling(edge_index = label_adj.edge_index, 
                                                 num_nodes= num_nodes,
                                                 num_neg_samples = (num_nodes**2))  #I dont know why but im not getting all of them  
        
        else:
            # Sample a number `neg_mult`*number of existing edges of negative samples
            non_existing_adj = negative_sampling(edge_index = label_adj.edge_index,
                                                 num_nodes = num_nodes,
                                                 num_neg_samples = label_adj.edge_index.shape[1] * neg_mult)

        #Concat the label adj with 0s and 1s
        label_adj.edge_index = torch.cat([label_adj.edge_index, non_existing_adj], dim = 1)
        label_adj.edge_attr = torch.cat([label_adj.edge_attr, torch.zeros(non_existing_adj.shape[1], dtype = torch.int64)]).to(args.device)
    
        return {'idx': idx, 'hist_adj_list': hist_adj_list, 'label_sp': label_adj}

class DataSplit(Dataset):
    """
    Manages the data split
    """

    def __init__(self,args, dataset, NumHistSteps, start, end, **kwargs):
        """
        Start and end are indices indicating what item belongs to this split
        """
        self.start = start
        self.end = end
        self.kwargs = kwargs
        self.NumHistSteps = NumHistSteps
        self.dataset = dataset                              # Is this thing going to cost me a lot?
        self.args = args

    def __len__(self):
        return self.end - self.start
    
    def __getitem__(self, idx):

        idx = self.start + idx
        t = get_sample(self.args, self.dataset, self.NumHistSteps, idx, **self.kwargs)
        return t

class Splitter():
    """
    Creates train - val - test split
    """

    def __init__(self, args, dataset):

        NumHistSteps = args.num_hist_steps
        TrainSplit = args.train_split
        ValSplit = args.val_split


        # Total number of Datapoints that is possible to make
        total_len = len(dataset) - 3 * NumHistSteps

        # Set start and end training indices
        start = NumHistSteps
        end = int(np.floor(total_len*TrainSplit) + NumHistSteps)

        # Create training split
        
        TrainSet = DataSplit(args, dataset,NumHistSteps, start, end)
        TrainSet = DataLoader(TrainSet, num_workers = 0)

        start = end + NumHistSteps

        number_of_indices = np.floor(ValSplit * total_len)
        end = int(start + number_of_indices)

        # Create validation split
        ValSet = DataSplit(args, dataset,NumHistSteps,  start, end, all_edges=True)          #link prediction
        ValSet = DataLoader(ValSet, num_workers=0)

        start = end + NumHistSteps
        end = len(dataset)

        # Create test set
        TestSet = DataSplit(args, dataset,NumHistSteps,  start, end, all_edges=True)
        TestSet = DataLoader(TestSet, num_workers = 0)

        total_datapoints = len(TrainSet) + len(ValSet) + len(TestSet)
        
        print(f'Dataset splits sizes:  train {len(TrainSet)} ({len(TrainSet)/total_datapoints:.2f}%) val {len(ValSet)} ({len(ValSet)/total_datapoints:.2f}%) test {len(TestSet)} ({len(TestSet)/total_datapoints:.2f}%)')

        self.train = TrainSet
        self.val = ValSet
        self.test = TestSet

class SplitterNoTest():
    """
    Creates train - val 
    """

    def __init__(self, args, dataset):

        NumHistSteps = args.num_hist_steps
        TrainSplit = args.train_split

        assert  TrainSplit < 1, 'There is no space for test sampling'

        # Total number of Datapoints that is possible to make
        total_len = len(dataset) - 2 * NumHistSteps

        # Set start and end training indices
        start = NumHistSteps
        end = int(np.floor(total_len*TrainSplit) + NumHistSteps)

        # Create training split
        
        TrainSet = DataSplit(args, dataset,NumHistSteps, start, end)
        TrainSet = DataLoader(TrainSet, num_workers = 0)


        start = end + NumHistSteps
        end = len(dataset)

        # Create test set
        ValSet = DataSplit(args, dataset,NumHistSteps,  start, end, all_edges=True)
        ValSet = DataLoader(ValSet, num_workers = 0)

        total_datapoints = len(TrainSet) + len(ValSet)
        
        print(f'Dataset splits sizes:  train {len(TrainSet)} ({len(TrainSet)/total_datapoints:.2f}%) val {len(ValSet)} ({len(ValSet)/total_datapoints:.2f}%)')

        self.train = TrainSet
        self.val = ValSet

def create_splitter(args, dataset):

    if args.no_test:

        return SplitterNoTest(args, dataset)
    
    return Splitter(args, dataset)

