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


def get_sample(args, dataset, num_hist_steps, idx, **kwargs):
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


        #PER IL MULTI-STEP AHEAD
        # list_of_conditioning = torch.zeros((args.num_nodes, args.steps_ahead)).to(args.device)
        # list_of_y =  torch.zeros((args.num_nodes, args.steps_ahead)).to(args.device)
        list_of_conditioning = torch.zeros((args.num_nodes)).to(args.device)
        list_of_y =  torch.zeros((args.num_nodes)).to(args.device)

        num_of_predictions = args.steps_ahead                 #If 1 no extrapolation performed

        for i in range(num_of_predictions):

            y = dataset[idx + i].y
            r = dataset[idx + i].r - dataset[idx + i -1].r
            list_of_y = y
            # print('shape r; ', r.expand(args.num_nodes).shape)
            # print('shape cond: ', list_of_conditioning[:,i].shape)

            list_of_conditioning=r.expand(args.num_nodes)    

        return {'hist_adj_list': hist_adj_list}
        return {'idx': idx, 'hist_adj_list': hist_adj_list, 'y':list_of_y, 'conditions':list_of_conditioning}

def getty(dataset, idx, args):
    
    hist_list = [Data(x=dataset[i].node_feat.reshape(-1,1), edge_index=dataset[i].edge_index) for i in range(idx, idx + args.lookback)]
    return hist_list


class DataSplit(Dataset):
    """
    Manages the data split
    """

    def __init__(self,args, dataset):
        """
        Start and end are indices indicating what item belongs to this split
        """
        self.NumHistSteps = args.lookback
        self.dataset = dataset                              # Is this thing going to cost me a lot?
        self.args = args
        self.current_index = 0

    def __len__(self):
        return len(self.dataset) - self.args.lookback - self.args.steps_ahead + 1
    
    def __getitem__(self, idx):

        i_sample = self.current_index + idx
        #t = get_sample(self.args, self.dataset, self.NumHistSteps, idx, **self.kwargs)
        t = getty(self.dataset, i_sample, self.args)

        return t

class Splitter():
    """
    Creates train - val 
    """

    def __init__(self, args, dataset, val_batch_size):

        NumHistSteps = args.lookback
        TrainSplit = args.train_split

        #Used for extrampolation, if == 1, no extrapolation will be performed
        number_of_predictions = args.steps_ahead

        assert  TrainSplit < 1, 'There is no space for test sampling'

        # Total number of Datapoints that is possible to make
        total_len = len(dataset) - 2 * NumHistSteps - 2 * (number_of_predictions-1)

        # Set start and end training indices
        start = NumHistSteps
        end = int(np.floor(total_len*TrainSplit) + NumHistSteps)

        # Create training split
        
        TrainSet = DataSplit(args, dataset, start, end)
        
        TrainSet = DataLoader(TrainSet, batch_size=args.batch_size, shuffle=True)
        for s in TrainSet:
            print(s)
            raise NotImplementedError

        start = end + NumHistSteps + (number_of_predictions-1)
        end = len(dataset) - (number_of_predictions -1)

        # Create test set
        ValSet = DataSplit(args, dataset,  start, end)
        ValSet = DataLoader(ValSet, batch_size = val_batch_size, shuffle = False)

        total_datapoints = len(TrainSet) + len(ValSet)
        
        print(f'Dataset splits sizes:  train {len(TrainSet)} ({len(TrainSet)/total_datapoints:.2f}%) val {len(ValSet)} ({len(ValSet)/total_datapoints:.2f}%)')

        self.train = TrainSet
        self.val = ValSet

