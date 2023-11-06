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
    parser.add_argument('--config_file',default='parameters/params.yaml', type=argparse.FileType(mode='r'), help='optional, yaml file containing parameters to be used, overrides command line parameters')
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
        # print(data)
        arg_dict = args.__dict__

        for key, value in data.items():
            arg_dict[key] = value

        # I want to create a params file name which will store the running informations
        match = re.search(r'\d+', args.output_training_file_name).group()
        new_params_filename = "./parameters/test_"+str(match)+".yaml"

         # Check if the new file already exists
        if os.path.exists(new_params_filename):
            print(f"Error: File '{new_params_filename}' already exists. Update test name.")
            sys.exit(1)  # Exit the program with a non-zero exit code

        # Make the validation name coherent to the training name
        args.output_validation_file_name = './results/metrics_test_'+str(match)+'_valid.txt'
        args.histogram_val_file_name='./results/test_'+str(match)+'_valid_histo/'
        args.histogram_train_file_name='./results/test_'+str(match)+'_train_histo/'

        os.mkdir(args.histogram_val_file_name)
        os.mkdir(args.histogram_train_file_name)

        # Save the data to a new YAML file with the new filename
        with open(new_params_filename, 'w') as new_file:
            yaml.dump(args, new_file, default_flow_style=False)

    return args