import numpy as np
from CIR import get_CIR
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from torch_geometric.utils import degree
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from simulation import Simulation
import argparse

parser = argparse.ArgumentParser(description='Parser')
parser.add_argument('--layers', type=int, help='num of layrs')
parser.add_argument('--h', type=int, help='hidden size')
parser.add_argument('--l', type=int, help='FFNN width')

args = parser.parse_args()

class AllSequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, l, num_layers, out_shape):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.5, batch_first=True)
        self.linear = nn.Linear(hidden_size, l)
        self.fc = nn.Linear(l,int(l/2))
        self.fc2 = nn.Linear(int(l/2),1)
        self.relu = nn.LeakyReLU()
        self.reset_parameters()

    def reset_parameters(self):

        for layer in [self.lstm, self.linear, self.fc,self.fc2]:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


    def forward(self, x, r):
        x, _ = self.lstm(x)
        x = self.linear(x)
        x = self.fc(self.relu(x))
        x = self.fc2(self.relu(x))
        return x

class Model1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, linear_witdh):
        super(Model1, self).__init__()

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers= num_layers, batch_first = True)

        self.fc_1_conditioning = nn.Linear(hidden_size+1, linear_witdh)
        self.fc_1_unconditioned = nn.Linear(hidden_size, linear_witdh)

        self.fc_2 = nn.Linear(linear_witdh, 1)

        self.relu = nn.LeakyReLU()

        self.reset_parameters()

    def forward(self, X, r, conditioning):

        output, (hn, cn) = self.lstm(X)

        if conditioning:
            hn = torch.cat([hn[-1],r[:,-1,:]], dim=1)
        else:
            hn = hn[-1]

        out = self.relu(hn)
        
        if conditioning:
            out = self.fc_1_conditioning(out) #first Dense
        else:
            out = self.fc_1_unconditioned(out)
            
        out = self.relu(out) #relu
        out = self.fc_2(out) #Final Output
        return out

    def reset_parameters(self):

        for layer in [self.lstm, self.fc_1_conditioning, self.fc_1_unconditioned, self.fc_2]:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

def predict(model, loader, loader_test, conditioning):
    model.eval()
    train_predictions = []
    train_labels = []
    test_predictions = []
    test_labels = []

    with torch.no_grad():

        loop = tqdm(loader, desc='Train prediction')
    
        for X_batch, y_batch in loop:
    
            y_pred = model(X_batch, r=None, conditioning=None)
    
            for j in range(y_pred.shape[0]):
                
                train_predictions.append(y_pred[j,-1].detach())
                train_labels.append(y_batch[j,-1].detach())
                
                
        loop = tqdm(loader_test, desc='Test prediction')
    
        for X_batch, y_batch in loop:
    
            y_pred = model(X_batch, r=None, conditioning=None)
    
            for j in range(y_pred.shape[0]):
                
                test_predictions.append(y_pred[j,-1].detach())
                test_labels.append(y_batch[j,-1].detach())

    return torch.stack(train_predictions), torch.stack(train_labels), torch.stack(test_predictions), torch.stack(test_labels),

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print('Using device: ', device)

alpha = 0.6
b = 0.02
sigma = 0.14
v_0 = 0.04
years = 10
num_nodes = 2
gamma=50.0

sim = Simulation(alpha, b, sigma, v_0, years, seed = True, num_nodes = num_nodes, gamma = gamma)

PriceMatrix = np.load('/u/mcitterio/data/ForTestingMatrix.npy')

X = sim.CIRProcess[:1460]
y = PriceMatrix

training_index = int( 0.8 * len(X) )

scaler_X = StandardScaler()
X = scaler_X.fit_transform(X.reshape(-1,1))
scaler_y = MinMaxScaler()
y = scaler_y.fit_transform(y.reshape(-1,1))

X_train, y_train = torch.tensor(X[:training_index]).reshape(-1,1), torch.tensor(y[:training_index]).reshape(-1,1)
X_test, y_test = torch.tensor(X[training_index:]).reshape(-1,1), torch.tensor(y[training_index:]).reshape(-1,1)

def create_windowing(X_data,Y_data, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []

    
    for i in range(len(X_data)-lookback):

        
        feature = X_data[i : i + lookback]
        
        #anche dato r(t) io voglio calcolare p(t,T)
        target = Y_data[i  : i + lookback ]

        X.append(feature.to(torch.float32))
        y.append(target.to(torch.float32))

    return torch.stack(X).to(device), torch.stack(y).to(device)

#Lets get the windowing


lookback = 200
X_train, y_train = create_windowing(X_train, y_train, lookback=lookback)
X_test, y_test = create_windowing(X_test, y_test, lookback=lookback)

print(f'Training: X {X_train.shape},  y:{y_train.shape}')
print(f'Test X {X_test.shape}, y:{y_test.shape}')

torch.manual_seed(0)
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train.to(device), y_train.to(device)), shuffle=True, batch_size=8)
loader_test = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test.to(device), y_test.to(device)), shuffle=True, batch_size = 8)

num_epochs = 1000 #1000 epochs
learning_rate = 0.0005 #0.001 lr

input_size = 1 #number of features
hidden_size = args.h #number of features in hidden state
num_layers = args.layers #number of stacked lstm layers
linear_witdh = args.l

model1= Model1(input_size, hidden_size, num_layers, linear_witdh).to(device)
#all_squence_model = AllSequenceModel(input_size, hidden_size, linear_witdh, num_layers, out_shape=1).to(device)

criterion = torch.nn.MSELoss(reduction='mean')    # mean-squared error for regression
optimizer = torch.optim.Adam(model1.parameters(), lr=learning_rate) 
#optimizer = torch.optim.Adam(all_squence_model.parameters(), lr=learning_rate) 

train_loss = []
val_loss = []

best_val_loss = float('inf')  # Initialize with a large value
patience = 100  # Number of consecutive epochs to wait for improvement
counter = 0  # Counter for consecutive epochs without improvement

loop = tqdm(range(num_epochs), desc='Epoch')
for epoch in loop:
    
    model1.train()
    temp_loss_train = 0
    temp_loss_val = 0
    
    for X_batch, y_batch in loader:

        y_pred = model1.forward(X_batch, r=None, conditioning=None) #forward pass
        optimizer.zero_grad() #caluclate the gradient, manually setting to 0
        #raise NotImplementedError
        loss = criterion(y_pred, y_batch[:,-1,:])
        #print(loss)
        temp_loss_train += loss.item()
        loss.backward() #calculates the loss of the loss function
        optimizer.step() 

    if epoch % 10 == 0:
        model1.eval()
        with torch.no_grad():
            for X_batch, y_batch in loader_test:
        
                y_pred = model1.forward(X_batch, r=None, conditioning=None) #forward pass
                loss = criterion(y_pred, y_batch[:,-1,:])
                
                temp_loss_val += loss.item()

        train_loss.append(temp_loss_train/len(loader))
        val_loss.append(temp_loss_val/len(loader_test))
        # Early stopping
        if val_loss[-1] < best_val_loss:
            best_val_loss = val_loss[-1]
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping at epoch {epoch}. Best Validation RMSE: {np.sqrt(best_val_loss)}")
            break
    loop.set_postfix(RMSE = np.sqrt(train_loss[-1]), VAL_RMSE = np.sqrt(val_loss[-1]))

    
#Save val loss & train loss
train_loss = np.asarray(train_loss)
np.save(f'/u/mcitterio/data/train_loss_{args.layers}_{args.h}_{args.l}.npy', train_loss)
val_loss = np.asarray(val_loss)
np.save(f'/u/mcitterio/data/val_loss_{args.layers}_{args.h}_{args.l}.npy', val_loss)

loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train.to(device), y_train.to(device)), shuffle=False, batch_size=8)
loader_test = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test.to(device), y_test.to(device)), shuffle=False, batch_size = 8)

#Save predictions
train_pred, train_label, test_pred, test_label= predict(model1, loader, loader_test, conditioning=False)
train_pred = np.asarray(train_pred.cpu())
np.save(f'/u/mcitterio/data/train_pred_{args.layers}_{args.h}_{args.l}.npy', train_pred)
test_pred = np.asarray(test_pred.cpu())
np.save(f'/u/mcitterio/data/test_pred_{args.layers}_{args.h}_{args.l}.npy', test_pred)