import argparse
import yaml
import re
import os
import sys

def create_parser():
    """
    Selects the parameter file from a command line input
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--config_file',default='params.yaml', type=argparse.FileType(mode='r'), help='optional, yaml file containing parameters to be used, overrides command line parameters')
    return parser

def parse_args(parser):
    """
    Takes a parser in input which tells it which file we'll be working on and returns a args.
    RETURN
    args
    """
    args = parser.parse_args()
    if args.config_file:
        data = yaml.safe_load(args.config_file)
        delattr(args, 'config_file')
        arg_dict = args.__dict__

        # Get the args from the params.yaml file
        for key, value in data.items():
            arg_dict[key] = value

        # Get the test names
        path_folder = './parameters/'
        try:
            file_names = [f for f in os.listdir(path_folder) if os.path.isfile(os.path.join(path_folder, f))]
            matches = []
            for file in file_names:
                try: 
                    match = re.search(r'\d+', file).group() 
                    matches.append(int(match))
                except: pass

            matches.sort()
            new_test_number = matches[-1]+1

        except: 
            # First file in the folder
            new_test_number = 1

        new_params_filename = "./parameters/test_"+str(new_test_number)+".yaml"

        # Check if the new file already exists
        if os.path.exists(new_params_filename):
            print(f"Error: File '{new_params_filename}' already exists. Updating test_name")
            sys.exit(1)  # Exit the program with a non-zero exit code

        # Make the validation name coherent to the training name
        args.output_validation_file_name = './results/metrics_test_'+str(new_test_number)+'_valid.txt'
        args.output_training_file_name = './results/metrics_test_'+str(new_test_number)+'.txt'

        # Save the data to a new YAML file with the new filename
        with open(new_params_filename, 'w') as new_file:
            yaml.dump(args, new_file, default_flow_style=False)

    return args

def get_channels_lists(args, in_channels):
    """
    Creates the lists of proper inputs/outputs for a multilayer GCN model
    """
    list_in_channels = []
    list_out_channels = []

    list_in_channels.append(in_channels)
    for i in range(len(args.hidden_channels)):
        list_in_channels.append(args.hidden_channels[i])
        list_out_channels.append(args.hidden_channels[i])

    list_out_channels.append(args.out_channels)
    
    return list_in_channels, list_out_channels