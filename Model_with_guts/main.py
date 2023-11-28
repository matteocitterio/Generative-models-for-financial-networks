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

    # Ein bisschen of UI messages
    print(f'\n\nUsing device: {args.device}')
    print(f'Printing results @{args.output_validation_file_name}')
    print(f'Currently doing extrapolation: ', args.extrapolation)
    print(f'Number of extrapolation points {args.number_of_predictions}')

    # Define dataset
    dataset = dataset_managment.get_dataset(args)
    node_count = dataset[0].x.size(dim=0)                  # We assume a fixed number of nodes throughout the timesteps
    in_channels = dataset[0].x.size(dim=1)                # Number of convolutional filter

    # Define the splitter class which will handle data windowing
    splitter = dataset_managment.Splitter(args, dataset)

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
    training_managment.train(args, splitter, model, optimizer, criterion)