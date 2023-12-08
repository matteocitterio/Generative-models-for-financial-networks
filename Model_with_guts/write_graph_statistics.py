import torch
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
    stats_matrix = utils.compute_temporal_graph_statistics(dataset)
    np.savetxt('./'+args.dataset_name+'_stats_matrix.csv', stats_matrix.numpy(), delimiter=',')