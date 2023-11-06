import torch
import utils
import dataset_managment
import logits_cross_entropy
import model_managment
import training_managment

if __name__ == '__main__':

    # Parse arguments from command line
    parser = utils.create_parser()
    args = utils.parse_args(parser)

    # Define dataset
    dataset = dataset_managment.build_dataset(args)
    node_count = dataset[0].x.shape[0]                  # We assume a fixed number of nodes throughout the timesteps
    in_channels = dataset[0].x.shape[1]                 # Number of convolutional filter

    # Define the splitter class which will handle data windowing
    splitter = dataset_managment.create_splitter(args, dataset)

    # Define loss function [custum label weighted cross entropy with logits]
    criterion = logits_cross_entropy.Cross_Entropy(args)

    # Define the model
    model = model_managment.build_model(args, node_count, in_channels).to('cpu')
    model.train()
    # Avoid using the parameters of the previous run
    model.reset_parameters()             

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # TRAINING PROCESS
    training_managment.train(args, splitter, model, optimizer, criterion)
    # Final evaluation, the epochs parameter is fake, just set so that we are sure it is a final evaluation
    training_managment.eval(args, args.num_of_epochs+2, model, splitter, criterion)
