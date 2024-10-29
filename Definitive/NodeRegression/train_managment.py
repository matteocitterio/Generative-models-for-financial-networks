from tqdm import tqdm
import sklearn.metrics as mc
import model_managment
import torch
import numpy as np
import matplotlib.pyplot as plt


def train(args, splitter, model, optimizer, criterion):
    """
    Performs training loop and gets the measuruments
    """

    early_stopping = model_managment.EarlyStopping(args)

    for e in range(args.num_of_epochs):
    
        loop = tqdm(splitter.train, desc=f'Epoch {e+1}')  
        
        # Training Loop
        losses = []

        model.train()

        for s in loop:

            

            # Gets the node embeddings, compute the loss
            loss = run_epoch(args, model, s, criterion)
            # Print out UI updates
            loop.set_postfix(loss=loss.item())
            # Save computed metrics
            losses.append(loss)

            # Perform optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # At the end of the epoch print out the computed loss
        with open(args.output_training_file_name, "a") as file:

            losses = torch.stack(losses)
            file.write(f'{e+1} {losses.mean()}\n')

        if ((e+1)%args.eval_every == 0):

            # Evaluation at the end of an epoch
            val_loss = eval(args, e, model, splitter, criterion)
            #Early Stopping
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    # with open(args.output_training_file_name, "a") as file:
    #     file.write('Done\n')

def run_epoch(args, model, s, criterion):
    """
    run the model over a batch s
    INPUTS:
    - args: arguments read through the .yaml file
    - model: the instantiated model
    - s: batch of data containing the historical list, positive labels, negative labels and conditions
    - criterion: the instantiated loss
    - dataset: original dataset (used only if args.weird_predictions=True)
    - final: if True it computes all the metrics

    OUTPUTS:
    - loss: torch.float with require_gradient == True (if model.train())
    - if Final == True it returns the set of metrics for the processed batch (metrics for every extrapolation point)

    """

    loss = 0
    print(s)
    raise NotImplementedError
    # Get node_embeddings
    node_embeddings = model(s['hist_adj_list'])
    #Get the model predictions

    #ATTENTION PLEASE magari shapes juste ma sto confrontando robe a cazzo
    
    pred = model.regressor(node_embeddings, s['conditions'])

    loss = criterion(pred, s['y'].reshape(-1,1))
        
    return loss


# Explicitly state that this function is not involved in GD
@torch.no_grad()
def eval(args, epoch, model, splitter, criterion):
    """
    Performs model evaluation
    """
    # Set model to eval mode
    model.eval()  
    losses = [] 
    
    loop = tqdm(splitter.val,desc=f'Validation epoch {epoch+1}')
    
    for s in loop:

        loss = run_epoch(args, model, s, criterion)

        # Print out UI updates
        loop.set_postfix(loss=loss.item())
        losses.append(loss)
        
    losses = torch.stack(losses)

    with open(args.output_validation_file_name, "a") as file:
            
        file.write(f'{epoch+1} {losses.mean()}\n')

    return losses.mean()

# Explicitly state that this function is not involved in GD
@torch.no_grad()
def predict(args, model, splitter, training = False):
    """
    SO FAR
    """
    if args.number_of_predictions > 1:
        raise NotImplementedError('Currently implemented for 1 step ahead only')

    if training:
        #Print predictions over the training set
        loop = tqdm(splitter.train,desc=f'Traing set prediction')
    else:
        loop = tqdm(splitter.val, desc=f'Validation set prediction')

    prediction_matrix = torch.zeros((args.dataset_nodes, len(loop))).to(args.device)
    label_matrix = torch.zeros((args.dataset_nodes, len(loop))).to(args.device)

    for i,s in enumerate(loop):

        condition = None
        if args.conditioning:
        
            condition = s['conditions'][0][0]

        prediction = model.predict(args, s['hist_adj_list'], condition).squeeze()
        label = s['y'][0].squeeze()

        prediction_matrix[:, i] = prediction
        label_matrix[:, i] = label

    return prediction_matrix, label_matrix

#######################################################################################################
"""
Some utils to light up a bit the code
"""

def get_predictions(args, node_embeddings, model, s, criterion, condition, prediction_number):
    """
    Performs predictions and computes the loss
    """

    #Get the model predictions
    pred = model_managment.predict_node_regression(args, node_embeddings, model.classifier, condition)

    return criterion(pred.squeeze(), s['y'][prediction_number].squeeze())