import numpy as np
from CIR import get_CIR
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from simulation import Simulation

from sklearn.preprocessing import MinMaxScaler, StandardScaler

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print('Using device: ', device)

#################################################################
#Models

class AllSequenceModel(nn.Module):
    """
    This one trains the LSTM using the entire sequence of hidden states for each input variable. This specific implementation doesnt support conditioning
    """
    
    def __init__(self, input_size, hidden_size, l, num_layers):
        """
        - input_size: int, features dimensions
        - hidden_size: size of the LSTM hidden state
        - l: width of the FCNN
        - num_layers: num layers of the LSTM
        
        """
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
        """
        Here r is not used but still taken as input (to implement)
        """
        x, _ = self.lstm(x)
        x = self.linear(x)
        x = self.fc(self.relu(x))
        x = self.fc2(self.relu(x))
        return x

class Model1(nn.Module):
    """
    This model trains the LSTM using only the last hidden state h_T as common in sequence-to-one timeseries forecasting. This implementation supports conditioning.
    """
    def __init__(self, input_size, hidden_size, num_layers, linear_witdh):
        """
        - input_size: int, features dimensions
        - hidden_size: size of the LSTM hidden state
        - linear_width: width of the FCNN
        - num_layers: num layers of the LSTM
        """
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
    """
    Utils to produce predictions array and visualize them.
    Please uncomment according to the chosen model
    """
    
    model.eval()
    train_predictions = []
    train_labels = []
    test_predictions = []
    test_labels = []

    with torch.no_grad():

        loop = tqdm(loader, desc='Train prediction')
    
        for X_batch, y_batch, r_batch in loop:

            #Uncomment if using model1 and doing conditioning:
            y_pred = model(X_batch, r=r_batch, conditioning=True)
            #Uncomment if using model1 and doing the UNconditioned prediction:
            #y_pred = model(X_batch, r=r_batch, conditioning=False)
            #Uncomment if using AllSequenceModel
            #y_pred = model(X_batch, r=None, conditioning=None)

            #loop over batches
            for j in range(y_pred.shape[0]):

                #Uncomment if using model1
                train_predictions.append(y_pred[j,-1].detach())
                train_labels.append(y_batch[j,-1].detach())
                #Uncomment if using AllSequenceModel:
                #train_predictions.append(y_pred[j,-1].detach())
                #train_labels.append(y_batch[j,-1].detach())
        
                
        loop = tqdm(loader_test, desc='Test prediction')
    
        for X_batch, y_batch, r_batch in loop:
    
            #Uncomment if using model1 and doing conditioning:
            y_pred = model(X_batch, r=r_batch, conditioning=True)
            #Uncomment if using model1 and doing the UNconditioned prediction:
            #y_pred = model(X_batch, r=r_batch, conditioning=False)
            #Uncomment if using AllSequenceModel
            #y_pred = model(X_batch, r=None, conditioning=None)
    
            for j in range(y_pred.shape[0]):
                
                test_predictions.append(y_pred[j,-1].detach())
                test_labels.append(y_batch[j,-1].detach())

    return torch.stack(train_predictions), torch.stack(train_labels), torch.stack(test_predictions), torch.stack(test_labels),


######################################
#Dataset utils

def scale_targets(dataset, index):

    """
    This one does the scaling for targets y
    """

    idx = index
    
    #Get the maximum number of contracts
    num_days = len(dataset)

    training_set = dataset[:idx]

    #Combine all the feature matrices by stacking them vertically
    combined_array_training = np.vstack([tensor.y.view(-1,1).cpu().numpy() for tensor in training_set])
    combined_array = np.vstack([tensor.y.view(-1,1).cpu().numpy() for tensor in dataset])

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()
    
    #Fit the scaler over the training set
    scaler.fit(combined_array_training)

    #I want to have it scaled so that the 1 is np.max * 1.5 and the 0 is np.min*0.5
    scaler.scale_ /= 1.25
        
    # trainsform the scaler on the combined data and transform it
    scaled_array = scaler.transform(combined_array)

    # splits the vertical stack into a list of num_contracts elements
    pre_scaled_tensors = np.split(scaled_array, num_days, axis=0)
    #In order to recover the original tensor we need to stack horizontally the list's elements
    finally_tensors = [torch.tensor(pre_scaled_tensors[i]).squeeze() for i in range(num_days)]

    return finally_tensors

def scale_feature_matrix(dataset, index):
    """
    Scales the feature matrix's feautures according to their type
    """
    #Get the maximum number of contracts
    num_contracts = int(dataset[0].x.shape[1]/5)
    num_nodes = dataset[0].x.shape[0]

    idx = index

    training_set = dataset[0:idx]

    #Combine all the feature matrices by stacking them vertically
    combined_array_training = np.vstack([tensor.x.cpu().numpy() for tensor in training_set])
    combined_array_complete = np.vstack([tensor.x.cpu().numpy() for tensor in dataset])

    #Now slice them in 3 by 3, so that for every row we only have a contract
    combined_of_combined_training = np.vstack([combined_array_training[:, 0 + (i*5): 5 + (i*5)] for i in range(num_contracts)])
    combined_of_combined = np.vstack([combined_array_complete[:, 0 + (i*5): 5 + (i*5)] for i in range(num_contracts)])

    max_T = int(max(combined_of_combined_training[:,1]))

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler on the combined data and transform it
    scaler.fit(combined_of_combined_training)

    #I want to have it scaled so that the 1 is np.max * 1.5 and the 0 is np.min*0.5
    scaler.scale_ *= 1.25

    #Actually transform the data
    scaled_array = scaler.transform(combined_of_combined)

    # splits the vertical stack into a list of num_contracts elements
    pre_scaled_tensors = np.split(scaled_array, num_contracts, axis=0)
    #In order to recover the original tensor we need to stack horizontally the list's elements
    finally_tensors = torch.hstack([torch.tensor(pre_scaled_tensors[i]) for i in range(num_contracts)])

    #The last step is revert the first vertical stacking as well:

    scaled_tensors = [finally_tensors[0 + i*num_nodes :num_nodes + i*num_nodes].squeeze() for i in range(len(dataset))]
    return scaled_tensors, max_T

def contract_4_sim(contract):
    """
    Inside the simulation.py code the contract convention have shape: (t, delta, T, R(t_0,T), B_t_0 ) but then SimulateNetwork.py used the convention (t,T,delta,R,K). This one
    simply switches it
    """
    return torch.tensor([contract[0], contract[2], contract[1], contract[3], contract[4]])

def create_dataset_all_sequence(dataset,r,targets, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: An array like of time series
        reference: reference CIR interest rate process
        lookback: Size of window for prediction
    """
    X, y, reference = [], [], []
    
    for i in range(len(dataset)-lookback):

        feature = dataset[i : i + lookback, :]
        #Qui si riferisce gia in avanti quindi non ho bisogno di spostarlo
        target = targets[i  : i + lookback ]
        r_temp = r[i + 1 : i + lookback + 1]

        X.append(feature.to(torch.float32))
        y.append(target.to(torch.float32))
        reference.append(r_temp.to(torch.float32))

    return torch.stack(X).to(device), torch.stack(y).to(device), torch.stack(reference).to(device)

####################################################################
#Main

#Define simulation
alpha = 0.6
b = 0.02
sigma = 0.14
v_0 = 0.04
years = 1
num_nodes = 2
gamma=49.0

sim = Simulation(alpha, b, sigma, v_0, years, seed = True, num_nodes = num_nodes, gamma = gamma)

"""
This code will use a fake single contract with maturity T=365 days instead of a produced simulation.
"""

#Retrieve dataset produced from `SimulateNetwork.py`
#dataset = torch.load(f'/u/mcitterio/data/subgraphs_Duffie_{num_nodes}nodes_{gamma}gamma.pt', map_location=torch.device(device))

#Create a contract with t_0 = 10, T = 364. and delta +1. R and K are computed using sim utils
contract = torch.tensor([10., 364., +1.,sim.SwapRate(10, 364), 1.0012891292572021  ]).to(torch.float32)

X = []
for i in range(0, 10):
    X.append(torch.zeros_like(contract))
for i in range(10, 365):
    X.append(contract)

X = torch.stack(X)

#Decide the training portion of the dataset
training_portion =  0.8
training_index = int( training_portion * X.shape[0])

# Identify feautures and targets and reshape them correctly
features = X
targets = [sim.GetInstantContractMarginValue(t, contract_4_sim(X[t])) for t in range(364)]
#features = torch.stack(features)
targets = torch.tensor(targets)
targets = targets.reshape(-1,1)

scaler_features = MinMaxScaler()
scaler_features.fit(features[:training_index,:])
scaler_features.scale_ /= 1.25
features = torch.tensor(scaler_features.transform(features)).to(torch.float32).to(device)

#Rescales the dataset according to the training set
scaler_targets = MinMaxScaler()
scaler_targets.fit(targets[:training_index,:])
scaler_targets.scale_ /= 1.25
targets = torch.tensor(scaler_targets.transform(targets)).to(torch.float32).to(device)

#Decide the training portion of the dataset
training_portion =  0.8
training_index = int( training_portion * len(dataset))

#Create the seed0 CIR process and B_t process :
CIRProcess = sim.CIRProcess[:364]
B_t = np.asarray([np.prod(1+(CIRProcess[0:t]*1/365)) for t in range(1,len(CIRProcess)+1)])
CIRProcess = sim.CIRProcess[:365]
CIRProcess = torch.tensor(CIRProcess.reshape(-1,1)).to(torch.float32)
B_t = torch.tensor(B_t.reshape(-1,1)).to(torch.float32)
time = torch.arange(B_t.shape[0]).reshape(-1,1).to(torch.float32).to(device) / training_index

#Normalize the two processes

scaler_CIR = StandardScaler()
scaler_CIR.fit(CIRProcess[0:training_index].reshape(-1,1))
scaler_CIR.scale_ *= 1.25
CIRProcess = torch.tensor(scaler_CIR.transform(CIRProcess.reshape(-1,1))).to(torch.float32).to(device)

scaler_Bt = StandardScaler()
scaler_Bt.fit(B_t[0:training_index].reshape(-1,1))
scaler_Bt.scale_ *= 1.25
B_t = torch.tensor(scaler_Bt.transform(B_t.reshape(-1,1))).to(torch.float32).to(device)

#Uncomment to see how data is normalized:
# plt.figure(figsize=(10,10), dpi=300)
# plt.subplot(4,4,1)
# plt.hist(CIRProcess.cpu().squeeze(), bins=150, density=True)
# plt.title('CIR')
# plt.subplot(4,4,2)
# plt.hist(B_t.cpu().squeeze(), bins=150, density=True)
# plt.title('B_t')
# plt.subplot(4,4,3)
# plt.hist(time.cpu().squeeze(), bins=150, density=True)
# plt.title('time')

# for i in range(5):
#     plt.subplot(4,4,4+i)
#     feat = []
#     for j in range(int(features.shape[1]/5)):
#         feat.append(np.asarray(features[:,j*5+i].cpu()))
#     plt.hist(np.asarray(feat).ravel(),bins=150, density=True)
#     plt.title(f'Feature {i}')

# plt.tight_layout()
# plt.savefig('NormalizationCheck.pdf')

#Concat in the data matrix, Targets is not actually useful but im gonna keep it for the moment
X_data = torch.cat([features[:-1, :], time, B_t], axis = 1).to(device)

# Create train and test sets
train, CIR_train, targets_train = X_data[:training_index,:], CIRProcess[:training_index], targets[:training_index]
test, CIR_test, targets_test = X_data[training_index:,:], CIRProcess[training_index:], targets[training_index:]

#Create the windows of data for training
lookback = 15
X_train, y_train, r_train = create_dataset_all_sequence(train, CIR_train, targets_train, lookback=lookback)
X_test, y_test, r_test = create_dataset_all_sequence(test, CIR_test, targets_test, lookback=lookback)

print(f'Training: X {X_train.shape}, r: {r_train.shape}, y:{y_train.shape}')
print(f'Test X {X_test.shape}, r: {r_test.shape}, y:{y_test.shape}')

#Create the batch loader
torch.manual_seed(0)
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train.to(device), y_train.to(device), r_train.to(device)), shuffle=False, batch_size=64)
loader_test = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test.to(device), y_test.to(device), r_test.to(device)), shuffle=False, batch_size = 64)

#training parameters
num_epochs = 100 #1000 epochs
learning_rate = 0.001 #0.001 lr

input_size = X_train.shape[2] #number of features
hidden_size = 32 #number of features in hidden state
num_layers = 2 #number of stacked lstm layers
linear_witdh = 256

#all_sequence_model = AllSequenceModel(input_size, hidden_size, num_layers).to(device)
model1= Model1(input_size, hidden_size, num_layers, linear_witdh).to(device)

criterion = torch.nn.MSELoss(reduction='mean')    # mean-squared error for regression
optimizer = torch.optim.Adam(model1.parameters(), lr=learning_rate) 

"""
Training routine
"""
train_loss = []
val_loss = []

#Early stopping parameters
best_val_loss = float('inf')  # Initialize with a large value
patience = 15  # Number of consecutive epochs to wait for improvement
counter = 0  # Counter for consecutive epochs without improvement

loop = tqdm(range(num_epochs), desc='Epoch')
for epoch in loop:
    
    model1.train()
    temp_loss_train = 0
    temp_loss_val = 0
    
    for X_batch, y_batch, r_batch in loader:

        y_pred = model1.forward(X_batch, r_batch, conditioning=True) #forward pass
        optimizer.zero_grad() #caluclate the gradient, manually setting to 0

        loss = criterion(y_pred, y_batch[:,-1,:])
        temp_loss_train += loss.item()
        loss.backward() #calculates the loss of the loss function
        optimizer.step() 

    #Validation
    model1.eval()
    with torch.no_grad():
        for X_batch, y_batch, r_batch in loader_test:
    
            y_pred =  model1.forward(X_batch, r_batch, conditioning=True)
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
        
    loop.set_postfix(loss = train_loss[-1], val_loss = val_loss[-1])

#Perform prediction after training:
train_pred, train_label, test_pred, test_label= predict(model1, loader, loader_test, conditioning=True)

#Visualize them:
plt.figure(figsize=(8,6), dpi=300)
plt.subplot(2,1,1)
plt.plot(test_pred.cpu(), label='Prediction')
plt.plot(test_label.cpu(), label='Ground truth')
plt.grid()
plt.legend(fontsize=14)
plt.title('Test')

plt.subplot(2,1,2)
plt.title('Train')
plt.plot(train_pred.cpu(), label='Prediction')
plt.plot(train_label.cpu(), label='Ground truth')
plt.legend(fontsize=14)
plt.grid()

plt.tight_layout()
plt.show()
