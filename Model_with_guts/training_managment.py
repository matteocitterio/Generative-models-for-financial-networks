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

    for e in range(args.num_of_epochs):
    
        loop = tqdm(splitter.train, desc=f'Epoch {e+1}')  
        
        # Training Loop
        losses = []
        aps = []
        aucs = []

        model.train()

        for s in loop:
            
            # Gets the node embeddings, performs predictions and compute metrics
            if args.extrapolation:
                ap, auc, loss, extrap_aps, extrap_aucs = run_epoch(args, model, s, criterion)
            else: 
                ap, auc, loss = run_epoch(args, model, s, criterion)
            # Print out UI updates
            loop.set_postfix(loss=loss.item(), AP = ap, AUC = auc)
            # Save computed metrics
            losses.append(loss)
            aps.append(ap)
            aucs.append(auc)
            # Perform optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # At the end of the epoch print out the computed metrics
        with open(args.output_training_file_name, "a") as file:

            AUCs_tensors = [torch.tensor(auc) for auc in aucs]
            APs_tensors = [torch.tensor(ap) for ap in aps]

            losses = torch.stack(losses)
            AUCs = torch.stack(AUCs_tensors) 
            APs = torch.stack(APs_tensors)
            
            file.write(f'{e+1} {losses.mean():.5f} {APs.mean():.5f} {AUCs.mean():.5f}\n')

        if ((e+1)%args.eval_every == 0):

            # Evaluation at the end of an epoch
            eval(args, e, model, splitter, criterion)

def run_epoch(args, model, s, criterion):

    # Get node_embeddings
    node_embeddings = model(s['hist_adj_list'])
    #print(s['hist_adj_list'][0].edge_attr)

    # Work positive samples
    pos_pred = model_managment.predict(node_embeddings, model.classifier, s['positive_edges'][0][0])
    pos_weights = torch.ones_like(pos_pred)*args.loss_class_weights[0]
    pos_loss = (criterion(pos_pred, torch.ones_like(pos_pred)) * pos_weights).mean()

    # Work negative samples
    neg_pred = model_managment.predict(node_embeddings, model.classifier, s['negative_edges'][0][0])
    neg_weights = torch.ones_like(neg_pred)*args.loss_class_weights[1]
    neg_loss = (criterion(neg_pred, torch.zeros_like(neg_pred)) * neg_weights).mean()

    # get metrics
    ap, auc = metrics(pos_pred, neg_pred)
    loss = pos_loss + neg_loss

    if args.extrapolation:

        extrapolated = model.extrapolate(args, node_embeddings)
        extrap_aps = []
        extrap_aucs = []
        extrap_aps.append(ap)
        extrap_aucs.append(auc)
        for l in range(args.number_of_predictions-1):
                    
            pos_pred = model_managment.predict(extrapolated[l], model.classifier, s['positive_edges'][l+1][0])
            neg_pred = model_managment.predict(extrapolated[l], model.classifier, s['negative_edges'][l+1][0])
            pos_weights = torch.ones_like(pos_pred)*1
            neg_weights = torch.ones_like(neg_pred)*1
            pos_loss = (criterion(pos_pred, torch.ones_like(pos_pred)) * pos_weights).mean()
            neg_loss = (criterion(neg_pred, torch.zeros_like(neg_pred)) * neg_weights).mean()
            ap_e, auc_e = metrics(pos_pred, neg_pred)
            extrap_aps.append(ap_e)
            extrap_aucs.append(auc_e)
            loss += pos_loss + neg_loss

        ap = (ap + np.cumsum(extrap_aps)[-1]) /(len(extrap_aps) +1)
        auc = (auc + np.cumsum(extrap_aucs)[-1]) /(len(extrap_aucs) +1)

        return ap, auc, loss, extrap_aps, extrap_aucs

    return ap, auc, loss

def metrics(pos_pred, neg_pred):
    """
    Computes evaluation metrics
    """
    y_pred = torch.cat([pos_pred, neg_pred], dim=0).sigmoid().cpu().detach()
    y_true = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)], dim=0).cpu().detach()

    # AUC and MAP
    auc = mc.roc_auc_score(y_true, y_pred)
    ap = mc.average_precision_score(y_true, y_pred)
    
    return ap, auc

# Explicitly state that this function is not involved in GD
@torch.no_grad()
def eval(args, epoch, model, splitter, criterion):
    """
    Performs model evaluation
    """
    model.eval()                        # Set model to eval mode

    AUCs = []
    APs = []
    LOSSESs = []
    Extrap_APS = []
    Extrap_AUCS = []
    
    loop = tqdm(splitter.val,desc=f'Validation epoch {epoch+1}')
    
    for s in loop:

        if args.extrapolation:
                ap, auc, loss, extrap_aps, extrap_aucs = run_epoch(args, model, s, criterion)
        else: 
                ap, auc, loss = run_epoch(args, model, s, criterion)
        # Print out UI updates
        loop.set_postfix(loss=loss.item(), AP = ap, AUC = auc)
        # Save computed metrics
        AUCs.append(auc)
        APs.append(ap)
        LOSSESs.append(loss)

        if args.extrapolation:
            Extrap_APS.append(extrap_aps)
            Extrap_AUCS.append(extrap_aucs)
        
    #Convert them in tensors so that torch is happy
    AUCs_tensors = [torch.tensor(auc) for auc in AUCs]
    APs_tensors = [torch.tensor(ap) for ap in APs]
    if args.extrapolation:
        transposed_extrapolated_APs = list(zip(*Extrap_APS))
        transposed_extrapolated_AUCs = list(zip(*Extrap_AUCS))
        
    LOSSESs = torch.stack(LOSSESs)
    AUCs = torch.stack(AUCs_tensors)
    APs = torch.stack(APs_tensors)
    if args.extrapolation:
        Extrap_APS = [sum(values) / len(values) for values in transposed_extrapolated_APs]
        Extrap_AUCS = [sum(values) / len(values) for values in transposed_extrapolated_AUCs]

    #Save them in a file
    with open(args.output_validation_file_name, "a") as file:
        
        print(f'Epoch {epoch+1} done | Mean Val Loss: {LOSSESs.mean():.5f} | Mean AUC: {AUCs.mean():.5f} | MAP: {APs.mean():.5f}')
        if args.extrapolation:
            file.write(f'{epoch+1} {LOSSESs.mean():.5f} {APs.mean():.5f} {AUCs.mean():.5f} {" ".join(f"{E_APS:.5f}" for E_APS in Extrap_APS)} {" ".join(f"{E_AUCS:.5f}" for E_AUCS in Extrap_AUCS)}\n')
        else:
            file.write(f'{epoch+1} {LOSSESs.mean():.5f} {APs.mean():.5f} {AUCs.mean():.5f}\n')
