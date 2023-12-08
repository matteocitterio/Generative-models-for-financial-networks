import torch

import random

import utils
import dataset_managment
import logits_cross_entropy
import model_managment
import training_managment
import numpy as np
import sys

if __name__ == '__main__':

    #WE NEED TO ADD THE LINES THAT DELETE THE FIRST ARGS.PYAML LINE

    parser = utils.create_parser_custom_file()
    args = utils.parse_args(parser, save=False)

    #Names of the interesting files
    model_file = args.server_string+ '/results/metrics_test_'+str(args.number)+'.txtcheckpoint.pt'
    
    # Define dataset
    dataset = dataset_managment.get_dataset(args)
    node_count = dataset[0].x.size(dim=0)                  # We assume a fixed number of nodes throughout the timesteps
    in_channels = dataset[0].x.size(dim=1)                # Number of convolutional filter
    
    # Load the conditioning if supposed to
    condition_matrix = utils.manage_conditions(args)
    test = None
    if args.weird_conditions:
        test = dataset

    # Define the splitter class which will handle data windowing
    splitter = dataset_managment.Splitter(args, dataset, condition_matrix)
    # Define loss function [custum label weighted cross entropy with logits]
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none').to(args.device)

    #Define the instance of the used model
    model = model_managment.build_model(args, node_count, in_channels).to(args.device)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    #Perform the evaluation
    training_managment.eval(args, 1000, model, splitter, criterion, dataset=test, final=True)

    print(model)