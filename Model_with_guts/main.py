import torch

import random

import utils
import dataset_managment
import logits_cross_entropy
import model_managment
import training_managment
import numpy as np

if __name__ == '__main__':

    # Parse arguments from command line
    parser = utils.create_parser()
    args = utils.parse_args(parser)

    # Ein bisschen of UI messages
    print(f'\n\nUsing device: {args.device}')
    print(f'Using safe gpu load: {args.safe_gpu_load}')
    print(f'Printing results @{args.output_validation_file_name}')
    print(f'Currently doing extrapolation: ', args.extrapolation)
    print(f'Number of extrapolation points {args.number_of_predictions}')
    print(f'Doing Early stopping on validation loss with patience {args.patience}')
    print(f'Doing conditioning: {args.conditioning}')

    # Define dataset
    dataset = dataset_managment.get_dataset(args)
    node_count = dataset[0].x.size(dim=0)                  # We assume a fixed number of nodes throughout the timesteps
    in_channels = dataset[0].x.size(dim=1)                # Number of convolutional filter

    no_edges = []
    for i in range(len(dataset)):
        no_edges.append(len(dataset[i].edge_index[0]))

    test = dataset

    print('\nNO edges average', np.average(np.asarray(no_edges)), '\n')
    
    # Load the conditioning if supposed to
    condition_matrix = utils.manage_conditions(args)

    # Define the splitter class which will handle data windowing
    splitter = dataset_managment.Splitter(args, dataset, condition_matrix)

    # Define loss function [custum label weighted cross entropy with logits]
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none').to(args.device)

    # Define the model
    model = model_managment.build_model(args, node_count, in_channels).to(args.device)
    model.train()
    # Avoid using the parameters of the previous run
    model.reset_parameters()             

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-4)

    # TRAINING PROCESS
    training_managment.train(args, splitter, model, optimizer, criterion, dataset = test)