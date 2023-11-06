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
        f1s = []
        accs = []
        histogram_data_list=[]

        for s in loop:

            model.train()
            node_embeddings = model(s['hist_adj_list'])
            predictions = model_managment.predict(node_embeddings, model.classifier, s['label_sp'].edge_index)
            ap, auc, f1, acc, hist = metrics(args, predictions, s['label_sp'].edge_attr)
            loss = criterion(predictions, s['label_sp'].edge_attr)
            losses.append(loss)
            aps.append(ap)
            aucs.append(auc)
            f1s.append(f1)
            accs.append(acc)
            histogram_data_list.append((hist))
            loop.set_postfix(loss=loss.item(), AP = ap, AUC = auc, acc=acc, f1=f1)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with open(args.output_training_file_name, "a") as file:

            AUCs_tensors = [torch.tensor(auc) for auc in aucs]
            APs_tensors = [torch.tensor(ap) for ap in aps]
            F1s_tensors = [torch.tensor(f1) for f1 in f1s]
            ACCs_tensors = [torch.tensor(acc) for acc in accs]

            losses = torch.stack(losses)
            AUCs = torch.stack(AUCs_tensors) 
            APs = torch.stack(APs_tensors)
            F1s = torch.stack(F1s_tensors) 
            ACCs = torch.stack(ACCs_tensors) 
            
            file.write(f'{e+1} {losses.mean():.5f} {APs.mean():.5f} {AUCs.mean():.5f} {F1s.mean():.5f} {ACCs.mean():.5f}\n')

            with open(f'{args.histogram_train_file_name}histogram_data_{e+1}.txt', 'wb') as file_hist:
                np.save(file_hist, histogram_data_list)

        if ((e+1)%args.eval_every == 0):

            # Evaluation at the end of an epoch
            eval(args, e, model, splitter, criterion)

def metrics(args, predictions, labels):
    """
    Computes evaluation metrics
    """
    probabilities = torch.sigmoid(predictions).detach().to('cpu')
    labels = labels.detach().to('cpu')
    gathered_probs = probabilities.gather(1,labels.view(-1,1))

    # AUC and MAP
    auc = mc.roc_auc_score(labels, gathered_probs, average='micro')
    ap = mc.average_precision_score(labels, gathered_probs, average='micro')
    
    if args.f1:
        
        # Compute F1 score also
        threshold = 0.5
        class_1_probabilities = probabilities[:, 1]
        hist, bin = np.histogram(class_1_probabilities, bins=np.linspace(0,1,args.num_hist), density=True)
        binary_predictions = (class_1_probabilities >= threshold).to(torch.int)
        acc = accuracy(labels, binary_predictions)
        f1 = mc.f1_score(labels, binary_predictions, average='micro')
        return ap, auc, f1, acc, hist

    return ap, auc

def accuracy(pred_y, y):
    return ((pred_y == y).sum() / len(y)).item()

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
    
    if args.f1:
        F1s = []
        ACCs = []
        histogram_data_list=[]
    
    loop = tqdm(splitter.val,desc=f'Validation epoch {epoch+1}')
    
    for s in loop:

        node_embeddings= model(s['hist_adj_list'])
        predictions = model_managment.predict(node_embeddings, model.classifier,  s['label_sp'].edge_index)
        loss = criterion(predictions, s['label_sp'].edge_attr)

        if args.f1:
        
            ap, auc, f1, acc, hist = metrics(args, predictions, s['label_sp'].edge_attr)
            F1s.append(f1)
            ACCs.append(acc)
            

        else:
            ap, auc  = metrics(args, predictions, s['label_sp'].edge_attr)
            
        AUCs.append(auc)
        APs.append(ap)
        LOSSESs.append(loss)
        histogram_data_list.append((hist))
        loop.set_postfix(loss=loss.item(), AP = ap, AUC = auc, acc=acc, f1=f1)
        

    #Convert them in tensors so that torch is happy
    AUCs_tensors = [torch.tensor(auc) for auc in AUCs]
    APs_tensors = [torch.tensor(ap) for ap in APs]

    if args.f1:
        F1s_tensors = [torch.tensor(f1) for f1 in F1s]
        ACCs_tensors = [torch.tensor(acc) for acc in ACCs]
        F1s = torch.stack(F1s_tensors)
        ACCs = torch.stack(ACCs_tensors)
        
    LOSSESs = torch.stack(LOSSESs)
    AUCs = torch.stack(AUCs_tensors)
    APs = torch.stack(APs_tensors)

    with open(args.output_validation_file_name, "a") as file:
        
        if args.f1:
        
            print(f'Epoch {epoch+1} done | Mean Val Loss: {LOSSESs.mean():.5f} |  Mean AUC: {AUCs.mean():.5f} | MAP: {APs.mean():.5f} |  Mean F1 score: {F1s.mean():.5f} | Accuracy: {ACCs.mean():.5f}')
            file.write(f'{epoch+1} {LOSSESs.mean():.5f} {AUCs.mean():.5f} {APs.mean():.5f} {F1s.mean():.5f} {ACCs.mean():.5f}\n')
            with open(f'{args.histogram_val_file_name}histogram_data_{epoch+1}.txt', 'wb') as file_hist:
                np.save(file_hist, histogram_data_list)
    
        else:
    
            print(f'Epoch {epoch+1} done | Mean Val Loss: {LOSSESs.mean():.5f} | Mean AUC: {AUCs.mean():.5f} | MAP: {APs.mean():.5f}')
            file.write(f'{epoch+1} {LOSSESs.mean():.5f} {AUCs.mean():.5f} {APs.mean():.5f}\n')

