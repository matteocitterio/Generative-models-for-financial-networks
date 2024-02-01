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

from Analytic_model import PerfectPrediction

from sklearn.preprocessing import MinMaxScaler, StandardScaler

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print('Using device: ', device)

def predict(model, loader, loader_test, conditioning):
    model.eval()
    train_predictions = []
    train_labels = []
    test_predictions = []
    test_labels = []

    with torch.no_grad():

        loop = tqdm(loader, desc='Train prediction')
    
        for X_batch, y_batch, r_batch in loop:
    
            y_pred = model(X_batch, y_batch, conditioning)
    
            for j in range(y_pred.shape[0]):
                
                train_predictions.append(y_pred[j].detach())
                train_labels.append(y_batch[j,-1].detach())
                
                
        loop = tqdm(loader_test, desc='Test prediction')
    
        for X_batch, y_batch, r_batch in loop:
    
            y_pred = model(X_batch, y_batch, conditioning)
    
            for j in range(y_pred.shape[0]):
                
                test_predictions.append(y_pred[j].detach())
                test_labels.append(y_batch[j,-1].detach())

    return torch.stack(train_predictions), torch.stack(train_labels), torch.stack(test_predictions), torch.stack(test_labels),

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
    #scaler = MinMaxScaler()
    scaler = StandardScaler()

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

def create_dataset_all_sequence(dataset,r,targets, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y, reference = [], [], []
    
    for i in range(len(dataset)-lookback):

        feature = dataset[i : i + lookback, :]
        #Qui si riferisce gia in avanti quinfi non ho bisogno di spostarlo
        target = targets[i  : i + lookback ]
        r_temp = r[i + 1 : i + lookback + 1]

        X.append(feature.to(torch.float32))
        y.append(target.to(torch.float32))
        reference.append(r_temp.to(torch.float32))

    return torch.stack(X).to(device), torch.stack(y).to(device), torch.stack(reference).to(device)

class AllSequenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, out_shape):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.5, batch_first=True)
        self.linear = nn.Linear(hidden_size+1, int(hidden_size+1/2))
        self.fc = nn.Linear(int(hidden_size+1/2),out_shape)
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):

        for layer in [self.lstm, self.linear, self.fc]:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


    def forward(self, x, r):
        x, _ = self.lstm(x)
        #x = x[:,-1,:]
        print('outputs shape',x.shape)
        print('r shape', r.shape)
        x = torch.cat([x,r], dim=2)#.to(torch.float32)
        print('x.shape', x.shape)
        x = self.linear(x)
        print('x.shape')
        x = self.fc(self.relu(x))
        return x
    
class Model1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, linear_witdh):
        super(Model1, self).__init__()

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers= num_layers, batch_first = True)

        self.fc_1_conditioning = nn.Linear(hidden_size+1, linear_witdh)
        self.fc_1_unconditioned = nn.Linear(hidden_size, linear_witdh)

        self.fc_2 = nn.Linear(linear_witdh, 1)

        self.relu = nn.ReLU()

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

    
class AllSequenceMode_nocond(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.2, batch_first=True)
        self.linear = nn.Linear(hidden_size, int(hidden_size/2))
        self.fc = nn.Linear(int(hidden_size/2),1)
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):

        for layer in [self.lstm, self.linear, self.fc]:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


    def forward(self, x, r):
        x, _ = self.lstm(x)
        #x = x[:,-1,:]
        #x = torch.cat([x,r], dim=2)#.to(torch.float32)
        x = self.linear(x)
        x = self.fc(self.relu(x))
        return x

alpha = 0.6
b = 0.02
sigma = 0.14
v_0 = 0.04
years = 20
num_nodes = 2
gamma=49.0

dataset = torch.load(f'/u/mcitterio/data/subgraphs_Duffie_{num_nodes}nodes_{gamma}gamma.pt', map_location=torch.device(device))


sim = Simulation(alpha, b, sigma, v_0, years, seed = True, num_nodes = num_nodes, gamma = gamma)
#Portion it
#dataset = dataset[365:len(dataset)-365]

#Decide the training portion of the dataset
training_portion =  0.8
training_index = int( training_portion * len(dataset))

#Rescales the dataset according to the training set
rescaled_Xs, max_T = scale_feature_matrix(dataset, training_index)
rescaled_Ys = scale_targets(dataset, training_index)
for i,data in enumerate(dataset):
        dataset[i].x = rescaled_Xs[i].to(torch.float32).to(device)
        dataset[i].y = rescaled_Ys[i].to(torch.float32).to(device)

# Identify feautures and targets and reshape them correctly
features = [dataset[i].x[0] for i in range(len(dataset))]
targets = [dataset[i].y[0] for i in range(len(dataset))]
features = torch.stack(features)
targets = torch.tensor(targets)
targets = targets.reshape(-1,1)


#Create the seed0 CIR process and B_t process be carefull i have to take CIR[:-1] because i cannot use the last point in the future :
CIRProcess = sim.CIRProcess[:-1]
B_t = np.asarray([np.prod(1+(CIRProcess[0:t]*1/365)) for t in range(1,len(CIRProcess)+1)])
CIRProcess = torch.tensor(CIRProcess.reshape(-1,1)).to(torch.float32)
B_t = torch.tensor(B_t.reshape(-1,1)).to(torch.float32)
time = torch.arange(CIRProcess.shape[0]).reshape(-1,1).to(torch.float32).to(device) / (training_index*1.25)


#Normalize the two processes

scaler_CIR = StandardScaler()#MinMaxScaler()
scaler_CIR.fit(CIRProcess[0:training_index].reshape(-1,1))
scaler_CIR.scale_ *= 1.25
CIRProcess = torch.tensor(scaler_CIR.transform(CIRProcess.reshape(-1,1))).to(device)

scaler_Bt = StandardScaler()#MinMaxScaler()
scaler_Bt.fit(B_t[0:training_index].reshape(-1,1))
scaler_Bt.scale_ *= 1.25
B_t = torch.tensor(scaler_Bt.transform(B_t.reshape(-1,1))).to(device)

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
X_data = torch.cat([features, time, B_t], axis = 1).to(device)
print('X_data.shape: ', X_data.shape)

# Create train and test sets
train, CIR_train, targets_train = X_data[:training_index,:], CIRProcess[:training_index], targets[:training_index]
test, CIR_test, targets_test = X_data[training_index:,:], CIRProcess[training_index:], targets[training_index:]

#Create windows ALL-SEQUENCE
"""
Each training window contains features[t - T, t], targets[t - T + 1, t + 1], CIRProcess[t - T + 1 , t + 1 ]
"""

lookback = 100
X_train, y_train, r_train = create_dataset_all_sequence(train, CIR_train, targets_train, lookback=lookback)
X_test, y_test, r_test = create_dataset_all_sequence(test, CIR_test, targets_test, lookback=lookback)

#Instantiate model
all_squence_model = AllSequenceModel(input_size=X_train.shape[2], hidden_size=128, num_layers=1, out_shape=lookback).to(device)

optimizer = optim.Adam(all_squence_model.parameters())
loss_fn = nn.MSELoss()

# learning_rate = 0.01 #0.001 lr
# input_size = X_train.shape[2] #number of features
# hidden_size = 216 #number of features in hidden state
# num_layers = 1 #number of stacked lstm layers
# linear_witdh = 128
# conditioning = True

# model1= Model1(input_size, hidden_size, num_layers, linear_witdh).to(device)

# criterion = torch.nn.MSELoss()    # mean-squared error for regression
# optimizer = torch.optim.Adam(model1.parameters(), lr=learning_rate) 



torch.manual_seed(0)
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train, r_train), shuffle=False, batch_size=32)
loader_test = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, y_test, r_test), shuffle=False, batch_size=32)

# perfect_prediction_model = PerfectPrediction(nodes=num_nodes, years=years, gamma=gamma)

# predictions = []
# labels = []

# i = 0

# mt_pred = []
# m_t_label = []

# loop = tqdm(loader)

# for X_batch, y_batch, r_batch in loop:

    
#     for j in range(y_batch.shape[0]):
#         mt_pred.append(perfect_prediction_model.GetVt(X_batch,r_batch)[j].item())
#         m_t_label.append(y_batch[j,-1,0].item())

#     i+=1
    

# plt.figure(figsize=(10,8), dpi=200)
# plt.title('Perfect prediction', fontsize=16)
# plt.plot(mt_pred[2000:2200], label='PerfectPred')
# plt.plot(m_t_label[2000:2200], label='FromData')
# plt.legend(fontsize=14)
# plt.ylabel('$M(t+1)$', fontsize=14)
# plt.grid()
# plt.tight_layout()
# plt.savefig('PerfectPrediction_train.pdf')

# mt_pred = []
# m_t_label = []

# loop2 = loader_test

# for X_batch, y_batch, r_batch in loop2:

#     #print(y_batch[-1,0])
#     #print(f'model: {perfect_prediction_model.GetVt(X_batch,r_batch)}, label: {y_batch[:,-1,0]}')
#     for j in range(y_batch.shape[0]):
#         mt_pred.append(perfect_prediction_model.GetVt(X_batch,r_batch)[j].item())
#         m_t_label.append(y_batch[j,-1,0].item())

#     i+=1
    

# plt.figure(figsize=(10,8), dpi=200)
# plt.title('Perfect prediction', fontsize=16)
# plt.plot(mt_pred[1000:1400], label='PerfectPred')
# plt.plot(m_t_label[1000:1400], label='FromData')
# plt.legend(fontsize=14)
# plt.ylabel('$M(t+1)$', fontsize=14)
# plt.grid()
# plt.tight_layout()
# plt.savefig('PerfectPrediction_test.pdf')

# print('DONE')

# raise NotImplementedError

num_epochs = 100

train_loss = []
val_loss = []

loop = tqdm(range(num_epochs), desc='Epoch')
for epoch in loop:
    
    model1.train()
    temp_loss_train = 0
    temp_loss_val = 0
    
    for X_batch, y_batch, r_batch in loader:

        y_pred = model1.forward(X_batch, r_batch, conditioning=conditioning) #forward pass
        optimizer.zero_grad() #caluclate the gradient, manually setting to 0
        loss = criterion(y_pred, y_batch[:,-1,:])
        temp_loss_train += loss.item()
        loss.backward() #calculates the loss of the loss function
        optimizer.step() 

    model1.eval()
    with torch.no_grad():
        for X_batch, y_batch, r_batch in loader_test:
    
            y_pred = model1.forward(X_batch, r_batch, conditioning=conditioning) #forward pass
            loss = criterion(y_pred, y_batch[:,-1,:])
            
            temp_loss_val += loss.item()

    train_loss.append(temp_loss_train/len(loader))
    val_loss.append(temp_loss_val/len(loader_test))
    loop.set_postfix(loss = train_loss[-1], val_loss = val_loss[-1])

# RMSE_training, RMSE_validation = [], []

# for epoch in range(n_epochs):
#     model1.train()

#     loop = tqdm(loader, desc=f'Training epoch {epoch+1}')

#     epoch_rmse_training = []

#     for X_batch, y_batch, r_batch in loop:
    
#         y_pred = all_squence_model(X_batch, r_batch)
#         loss = loss_fn(y_pred, y_batch)
#         loop.set_postfix(loss=torch.sqrt(loss).item())
#         epoch_rmse_training.append(torch.sqrt(loss).item())
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     print(f'Epoch {epoch+1}: training RMSE = {np.sum(epoch_rmse_training) / len(loop)}')
#     RMSE_training.append(np.sum(epoch_rmse_training) / len(loop))

#     # Validation
#     # if epoch % 10 != 0:
#     #     continue
#     all_squence_model.eval()
#     with torch.no_grad():
#         loop = tqdm(loader_test, desc='Validation')
#         epoch_rmse_test = []
#         for X_batch, y_batch, r_batch in loop:

#             y_pred = all_squence_model(X_batch, r_batch)
#             loss = loss_fn(y_pred, y_batch)
#             loop.set_postfix(loss=torch.sqrt(loss).item())
#             epoch_rmse_test.append(torch.sqrt(loss).item())

#         print(f'Epoch {epoch+1}: validation RMSE = {np.sum(epoch_rmse_test) / len(loop)}')
#         RMSE_validation.append(np.sum(epoch_rmse_test) / len(loop))

# print('Targets mean: ', targets.mean())


torch.save(model1.state_dict(), "./results/model_standard_time_scalbatch256_2layer_400look.pth")

plt.figure()
plt.title(f'AllSequenceModel N={lookback}')
plt.plot(train_loss, label='Training')
plt.plot(val_loss, label = 'Validation')
plt.ylabel('RMSE')
plt.xlabel('Epoch')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('RMSE_standard_time_scalbatch256_2layer_400look.pdf')

plt.figure(dpi=300)
# with torch.no_grad():
#     # shift train predictions for plotting
#     train_plot = np.ones_like(targets.cpu()) * np.nan
#     train_plot[lookback:training_index] = all_squence_model(X_train, r_train)[:, -1, :].cpu()
#     # shift test predictions for plotting
#     test_plot = np.ones_like(targets.cpu()) * np.nan
#     test_plot[training_index+lookback:] = all_squence_model(X_test, r_test)[:, -1, :].cpu()
# plot
# plt.plot(targets.cpu(), c='tab:blue', label='Labels')
# plt.axhline(targets.mean().cpu())
# plt.plot(train_plot, c='r',lw=1, label='Training')
# plt.plot(test_plot, c='g', label='Test')
train_pred, train_label, test_pred, test_label= predict(model1, loader, loader_test, conditioning=conditioning)

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

# plt.xlim(200,600)
# plt.grid()
# plt.legend()
# plt.tight_layout()
plt.savefig('PRediction_standard_time_scalbatch256_2layer_400look.pdf')