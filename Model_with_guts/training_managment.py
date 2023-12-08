from tqdm import tqdm
import sklearn.metrics as mc
import model_managment
import torch
import numpy as np
import matplotlib.pyplot as plt


def train(args, splitter, model, optimizer, criterion, dataset=None):
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
            loss = run_epoch(args, model, s, criterion, dataset, final=False)
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
            file.write(f'{e+1} {losses.mean():.5f}\n')

        if ((e+1)%args.eval_every == 0):

            # Evaluation at the end of an epoch
            val_loss = eval(args, e, model, splitter, criterion, dataset)
            #Early Stopping
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    with open(args.output_training_file_name, "a") as file:
        file.write('Done\n')

def run_epoch(args, model, s, criterion, dataset, final=False):
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
    # Get node_embeddings
    node_embeddings = model(s['hist_adj_list'])

    if final:
        #These list will have len == `args.number_of_extrapolation`
        aps = []                        # each element will contain a float (for each prediction step)
        aucs = []                       # each element will contain a float (for each prediction step)
        confusion_matrices = []         # each element will contain a list of len==`number_of_thresholds` (for each prediction step)

    for l in range(args.number_of_predictions):

        #Get the proper condition
        condition = None
        if args.conditioning:
            condition = s['conditions'][l][0]

        # Work positive samples
        pos_loss, pos_pred = get_predictions(args, node_embeddings[l], model, s, criterion, dataset, condition, positive=True, prediction_number=l)
        # Work negative samples
        neg_loss, neg_pred = get_predictions(args, node_embeddings[l], model, s, criterion, dataset, condition, positive=False, prediction_number=l)

        #Sum the loss
        loss += pos_loss + neg_loss

        # get metrics for the first prediction step (if final==True)
        if final:
            ap, auc, confusion_matrix = metrics(pos_pred, neg_pred)

            aps.append(ap)
            aucs.append(auc)
            confusion_matrices.append(confusion_matrix)

    if final:        
        # if final==True
        return loss, aps, aucs, confusion_matrices
        
    return loss

def metrics(pos_pred, neg_pred):
    """
    Computes evaluation metrics
    """
    y_pred = torch.cat([pos_pred, neg_pred], dim=0).sigmoid().cpu().detach()
    y_true = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)], dim=0).cpu().detach()

    #Define a proper set of thresholds for computing the confusion matrix
    number_of_thresholds = 3
    thresholds = np.linspace(torch.min(y_pred), torch.max(y_pred), number_of_thresholds)
    
    #Compute confusion matrix
    cm = [mc.confusion_matrix(y_true, (y_pred.numpy()>=threshold))for threshold in thresholds]

    #Store them as a list where each element is computed over a different threshold
    tn, fp, fn, tp = zip(*[t.ravel() for t in cm])
    conf_matrix = np.vstack((tn, fp, fn, tp))

    # AUC and MAP
    auc = mc.roc_auc_score(y_true, y_pred)
    ap = mc.average_precision_score(y_true, y_pred)
    
    return ap, auc, conf_matrix

# Explicitly state that this function is not involved in GD
@torch.no_grad()
def eval(args, epoch, model, splitter, criterion, dataset=None, final=False):
    """
    Performs model evaluation
    """
    # Set model to eval mode
    model.eval()  
    losses = []    

    if final:                  

        batches_of_aps = []
        batches_of_aucs = []
        batches_of_cms = []
    
    loop = tqdm(splitter.val,desc=f'Validation epoch {epoch+1}')
    
    for s in loop:

        if final:
            # Get the metrics as well
            loss, aps, aucs, confusion_matrices = run_epoch(args, model, s, criterion, dataset, final=True)
            batches_of_aps.append(aps)
            batches_of_aucs.append(aucs)
            batches_of_cms.append(confusion_matrices)

        else: 
            loss = run_epoch(args, model, s, criterion, dataset, final=False)

        # Print out UI updates
        loop.set_postfix(loss=loss.item())
        losses.append(loss)
            
    if final:

        #We want to compute the everage over the batches and output these averages for each extrapolation point
        averaged_confusion_matrix = np.average(batches_of_cms, axis = 0)  #average over the batches
        print('Shape conf mat: ', averaged_confusion_matrix.shape, 'We have 4 thresholds, 6 extrap points')
        averaged_ap = np.average(batches_of_aps, axis = 0)  #average over the batches
        print('Shape ap: ', averaged_ap.shape, ' 6 extrap points')
        averaged_auc = np.average(batches_of_aucs, axis = 0)  #average over the batches
        print('Shape auc: ', averaged_auc.shape, ' 6 extrap points')

        #Save metrics
        np.savetxt(f'{args.output_validation_file_name}_CF.txt', np.vstack([averaged_confusion_matrix[i] for i in range(len(averaged_confusion_matrix))]))
        np.savetxt(f'{args.output_validation_file_name}_AP_AUC.txt', np.vstack((averaged_ap, averaged_auc)))
        
    losses = torch.stack(losses)

    with open(args.output_validation_file_name, "a") as file:
            
        file.write(f'{epoch+1} {losses.mean():.5f}\n')

    return losses.mean()

#######################################################################################################
"""
Some utils to light up a bit the code
"""

def get_predictions(args, node_embeddings, model, s, criterion, dataset, condition, positive, prediction_number):
    """
    Performs predictions and computes the loss
    """

    #Positive prediction
    if positive:
        edges_string = 'positive_edges'
        class_weights_index = 0

    #Negative prediction
    else:
        edges_string = 'negative_edges'
        class_weights_index = 1

    #Get the model predictions
    pred = model_managment.predict(args, node_embeddings, model.classifier, s[edges_string][prediction_number][0],dataset, condition)
    #Get the weights if you are using a class weighted loss
    weights = torch.ones_like(pred) * args.loss_class_weights[class_weights_index]
    #Get the loss
    if positive:
        #The labels are a `torch.ones_like`
        return (criterion(pred, torch.ones_like(pred)) * weights).mean(), pred
    
    #The labels are a `torch.zeros_like` in the negative case
    return (criterion(pred, torch.zeros_like(pred)) * weights).mean(), pred