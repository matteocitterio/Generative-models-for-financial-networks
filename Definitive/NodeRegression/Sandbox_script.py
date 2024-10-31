import numpy as np
import os
from CIR import get_CIR
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from simulation import Simulation, Contract
from torch.utils.data import TensorDataset, Dataset, DataLoader, Subset
import Sandbox_utils as utils

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Fix current device

device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print('Current device:', device)

parser = utils.create_parser()
args = utils.parse_args(parser)

path_name = './Definitive/NodeRegression/'
data_file_name = path_name + f'X_tensor_{args.steps_filename}_steps.pt'

if os.path.exists(data_file_name):
    print(f'Retrieving data...')
    X = torch.load(path_name + f'X_tensor_{args.steps_filename}_steps.pt').to(device)
    y_margin = torch.load(path_name + f'y_margin_{args.steps_filename}_steps.pt').to(device)
    conditioning = torch.load(path_name + f'conditioning_{args.steps_filename}_steps.pt').to(device)
    y_benchmark = torch.load(path_name + f'y_benchmark_{args.steps_filename}_steps.pt)').to(device)

else:
    print('Simulating data...')

    # Generate CIR process & simulation
    sim = Simulation(args.alpha, args.b, args.sigma, args.v_0, args.years, gamma = args.gamma, seed=True)
    CIRProcess = torch.tensor(sim.CIRProcess.reshape(-1,1)).to(torch.float32).to(device)

    ### Create contracts

    #Generate arrival times for the contracts
    arrival_times = sim.GetArrivalTimes()

    #Build the contracts using Simulation and arrival times
    contracts = utils.CreateContracts(arrival_times, sim)

    # Compute maximum number of simultaneously active contracts
    max_n_active_contracts, days_with_active_contracts = utils.GetMaximumNActiveContracts(contracts, sim)
    print(f"Max number of simultaneously active contracts: {max_n_active_contracts}")
    print(f"Days with active contracts: {days_with_active_contracts}")

    ### Create dataset (X = features, y = targets)

    #Retrieve the number of feautures per contract
    n_contract_features = len(contracts[0].get_contract_features(contracts[0].t_0))

    #I will fill this one with the indexis of the active contracts
    #This list will contain only active contracts
    X = []
    y_margin = []
    conditioning = []
    y_benchmark = torch.zeros((days_with_active_contracts, args.steps_ahead))

    h=0

    loop = tqdm(range(sim.TotPoints - 1 - args.steps_ahead + 1))

    for i_time, t in enumerate(loop):
        active_contracts = [contract for contract in contracts if contract.is_active(t)]
        
        #This will contain the contracts at a certain time
        temp_contract_row = torch.zeros((max_n_active_contracts*n_contract_features))
        temp_y_margin = 0

        for i_contract, contract in enumerate(active_contracts):
            
            #Retrieve contract features
            contract_features = contract.get_contract_features(t)    
            #Fill the row    
            temp_contract_row[i_contract*len(contract_features) : (i_contract+1)*len(contract_features)] = contract_features
            #Fill the target accordingly
            temp_y_margin += contract.GetVariationMargin(t)
            
        #Append the computed quantities if contract array is different from just zeros
        if not torch.all(torch.eq(temp_contract_row, torch.zeros_like(temp_contract_row))):
            
            X.append(temp_contract_row)
            y_margin.append(temp_y_margin)
            conditioning.append(CIRProcess[i_time])

            for eta in range(args.steps_ahead):
            
                y_benchmark[h, eta] = sim.ProvideBenchmark(t_l=t, steps_ahead=eta+1, contracts=contracts, n_simulations=100)
        
            h+=1

    #Transform quantities in tensors
    X = torch.vstack(X)
    y_margin= torch.tensor(y_margin)
    conditioning = torch.tensor(conditioning)
    print('Len dataset: ', X.shape[0])

    torch.save(X, f'./X_tensor_{args.steps_ahead}_steps.pt')
    torch.save(y_margin, f'./y_margin_{args.steps_ahead}_steps.pt')
    torch.save(conditioning, f'./conditioning_{args.steps_ahead}_steps.pt')
    torch.save(y_benchmark, f'./y_benchmark_{args.steps_ahead}_steps.pt)')

# Train-test split
training_index = int(0.8 * X.shape[0])

print('Shape of X: ', X.shape[1])

#TRAIN
y_margin_train = y_margin[:training_index]
y_benchmark_train = y_benchmark[:training_index]
X_train = X[:training_index, :]
CIR_train = conditioning[:training_index]

#TEST
y_margin_test = y_margin[training_index:]
y_benchmark_test = y_benchmark[training_index:]
X_test = X[training_index:, :]
CIR_test = conditioning[training_index:]
       
# Slice dataset into windows
X_train, y_margin_train, r_train, y_benchmark_train = utils.create_windows(args, device, X_train, y_margin_train, y_benchmark_train, CIR_train)
X_test,y_margin_test, r_test, y_benchmark_test = utils.create_windows(args, device, X_test, y_margin_test, y_benchmark_test, CIR_test)

#Build the pytorch datasets and dataloaders
train_dataset = TensorDataset(X_train.to(device), y_margin_train.to(device), r_train.to(device), y_benchmark_train.to(device))
test_dataset = TensorDataset(X_test.to(device), y_margin_test.to(device), r_test.to(device), y_benchmark_test.to(device))

#Transform it into a dataloader
torch.manual_seed(0)
loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
loader_test = DataLoader(test_dataset, shuffle=True, batch_size = args.batch_size)
         
#define the model
class model(nn.Module):

    def __init__(self, args, n_contract_features):
        super(model, self).__init__()

        self.steps_ahead = args.steps_ahead
        self.input_size = n_contract_features
        self.name = str(args.regressor_hidden_size) + str(args.steps_ahead)

        self.lstm = torch.nn.LSTM(input_size = self.input_size, 
                                   hidden_size = args.lstm_hidden_size,#args.lstm_hidden_size, 
                                   num_layers = 1)
        self.batchnormer = torch.nn.Sequential(utils.LSTMPermuter(),
                                               nn.BatchNorm1d(args.lstm_hidden_size),
                                               utils.LSTMPermuter())
        
        # # Initialize the regressor layers
        layers = [nn.Linear(args.lstm_hidden_size + 1, args.regressor_hidden_size),
                  #nn.BatchNorm1d(args.regressor_hidden_size),
                  nn.ReLU(),
                   ]
        
        # # #Add an additional layer
        for _ in range(args.number_regressor_layers - 1):

            layers.extend([nn.Linear(args.regressor_hidden_size, args.regressor_hidden_size_2),
                           #nn.BatchNorm1d(args.regressor_hidden_size_2),
                           nn.ReLU(),
                          ])
            
        # Add the output layer
        layers.append(nn.Linear(args.regressor_hidden_size_2 if args.number_regressor_layers > 1 else args.regressor_hidden_size, self.input_size))
        self.fc = nn.Linear(self.input_size,1)
        self.relu = nn.ReLU()

        # # Create the regressor using nn.Sequential
        self.regressor = nn.Sequential(*layers)
    
        #Reset parameters of the model
        self.reset_parameters()

    def forward(self, x, r):
        # Flatten LSTM parameters
        #self.lstm.flatten_parameters()

        pred = torch.zeros((x.shape[0], args.steps_ahead)).to(torch.float32).to(device)

        for i in range(x.shape[2]//self.input_size):
            
            contract = x[:,:, i*self.input_size : (i+1)*self.input_size] 
            #if the contract is non empty:
            if not torch.all(torch.eq(contract, torch.zeros_like(contract))):

                for j in range(self.steps_ahead):
                
                    lstm_hidden, _ = self.lstm(contract)
                    #lstm_hidden = self.batchnormer(lstm_hidden)
                    prediction = self.regressor(torch.squeeze(torch.cat([lstm_hidden[:,-1,:], r[:,j].reshape(-1,1)],dim=1)))
                    
                    pred[:, j] += self.fc(self.relu(prediction)).squeeze()
                    #prediction = prediction.unsqueeze(0).unsqueeze(0) #CHANGE TRAINING / PREDICTION
                    prediction = prediction.unsqueeze(1)
                    contract = torch.cat([contract[:, 1:, :], prediction], dim = 1)
            else:
                break
         
        return pred
    
    def forward_pred(self, x, r):

        pred = torch.zeros((x.shape[0], args.steps_ahead)).to(torch.float32).to(device)

        for i in range(x.shape[2]//self.input_size):
            
            contract = x[:,:, i*self.input_size : (i+1)*self.input_size] 
            #if the contract is non empty:
            if not torch.all(torch.eq(contract, torch.zeros_like(contract))):

                for j in range(self.steps_ahead):
                
                    lstm_hidden, _ = self.lstm(contract)
                    #lstm_hidden = self.batchnormer(lstm_hidden)
                    prediction = self.regressor(torch.squeeze(torch.cat([lstm_hidden[:,-1,:], r[:,j].reshape(-1,1)],dim=1)))
                    
                    pred[:, j] += self.fc(self.relu(prediction)).squeeze()
                    prediction = prediction.unsqueeze(0).unsqueeze(0) #CHANGE TRAINING / PREDICTION
                    #prediction = prediction.unsqueeze(1)
                    contract = torch.cat([contract[:, 1:, :], prediction], dim = 1)
            else:
                break
         
        return pred
    
    def reset_parameters(self):

        #self.lstm.reset_parameters()
        for layer in [self.lstm, self.regressor, self.fc]:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
                

#Let's get towards the training:
criterion = nn.MSELoss()           #Loss function

#Define the model
n_contract_features = 6
mymodel = model(args, n_contract_features).to(device)    
print(mymodel)

#optimizer & scheduler
optimizer = torch.optim.Adam(mymodel.parameters(), lr = args.initial_lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.scheduler_milestone_frequency * i for i in range(args.scheduler_number_of_updates)], gamma=args.scheduler_gamma)

#Define an early stopping object
early_stopping = utils.EarlyStopping(args.patience)

#Define two lists to store training record
training_loss = []
validation_loss = []
lrs = []

#Define a tqdm loop to get a nice progress bar
loop = tqdm(range(args.n_epochs))


for epoch in loop:

    #Set model in training mode
    mymodel.train()

    #Perform an epoch of training
    train_loss = utils.do_epoch(args, mymodel, loader, criterion, optimizer)

    #Scheduler step
    scheduler.step()

    #Track the average over the batches
    training_loss.append(train_loss)
 
    #Validation every 50 epochs
    if epoch % args.validation_every ==0:

        with torch.no_grad():
            #Set the model in eval mod
            mymodel.eval()

            test_loss = utils.do_epoch(args, mymodel, loader_test, criterion, optimizer, training=False)

            #Track the average loss over the batches
            validation_loss.append(test_loss)

        #Check early stopping
        early_stopping(validation_loss[-1], mymodel)
        if early_stopping.early_stop:

            #Load the best set of parameters
            mymodel.load_state_dict(torch.load(f'temp_state_dict{mymodel.name}.pt'))
            print(f"Early stopping at epoch {epoch}.")
            break

    lrs.append(optimizer.param_groups[0]["lr"])
            
    #Give informations in the loop
    loop.set_postfix(loss = train_loss, val_loss = test_loss, best_val_loss = early_stopping.best_score, counter=early_stopping.counter, lr= lrs[-1])

# #Get the predictions
print('Performing prediction')
train_margin_labels, train_benchmark, train_margin_predictions = utils.predictions(args, train_dataset, mymodel)
test_margin_labels, test_benchmark, test_margin_predictions = utils.predictions(args, test_dataset, mymodel)

np.save(f'./results/train_label_{args.steps_ahead}.npy',train_margin_labels)
np.save(f'./results/train_pred_{args.steps_ahead}.npy',train_margin_predictions)
np.save(f'./results/train_benchmark_{args.steps_ahead}.npy', train_benchmark)
np.save(f'./results/test_label_{args.steps_ahead}.npy',test_margin_labels)
np.save(f'./results/test_pred_{args.steps_ahead}.npy',test_margin_predictions)
np.save(f'./results/test_benchmark_{args.steps_ahead}.npy', test_benchmark)

