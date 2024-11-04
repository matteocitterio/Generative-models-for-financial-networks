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

