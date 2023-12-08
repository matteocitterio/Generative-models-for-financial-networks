import argparse
import yaml
import re
import os
import sys
from torch_geometric.utils import degree, to_dense_adj, to_networkx
import torch_geometric.transforms as T
import networkx as nx
import powerlaw
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import warnings

def create_parser():
    """
    Selects the parameter file from a command line input
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--config_file',default='params.yaml', type=argparse.FileType(mode='r'), help='optional, yaml file containing parameters to be used, overrides command line parameters')
    return parser

def create_parser_custom_file():
    """
    Selects the parameter file from a command line input
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--number', type=int, help='Provide the test number')
    server_file_string = '/u/mcitterio/Model_with_guts/'
    parser.add_argument('--server_string', default=server_file_string, type=str, help='Provide the server file location')
    parameters_file = server_file_string + 'parameters/test_'+str(parser.parse_args().number)+'.yaml'
    parser.add_argument('--config_file',default=parameters_file, type=argparse.FileType(mode='r'), help='optional, yaml file containing parameters to be used, overrides command line parameters')
    return parser

def parse_args(parser, save=True):
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

        if save:

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

# Graph statistics helper functions
def get_largest_connected_component(nx_graph_undir):
    return len(max(nx.connected_components(nx_graph_undir), key=len))
    
def get_triangle_count(nx_graph_undirected):
    return np.sum(list(nx.triangles(nx_graph_undirected).values()))/3
    
def get_square_count(nx_graph_undir):
    square_count = 0
    for cycle in nx.cycle_basis(nx_graph_undir):
    # Check if the cycle is a square (4-cycle)
        if len(cycle) == 4 and all(nx_graph_undir.has_edge(cycle[i], cycle[(i+1)%4]) for i in range(4)):
            square_count += 1
    return square_count

def get_powerlaw_alpha(deg):
    return powerlaw.Fit(deg.to('cpu').numpy(), xmin=max(np.min(deg.to('cpu').numpy()),1)).power_law.alpha

def get_gini_coefficient(deg):
    sorted_degrees = np.sort(deg.to('cpu').numpy())
    cum_freq = np.cumsum(sorted_degrees)
    # Calculate Gini coefficient
    n = len(sorted_degrees)
    return (np.sum((2 * np.arange(1, n + 1) - n - 1) * sorted_degrees) / (n * np.sum(sorted_degrees)))

def get_edge_entropy(deg, A_in):
    degrees = deg.to('cpu').numpy()
    m = 0.5 * np.sum(np.square(A_in[0].to('cpu').numpy()))
    n = A_in[0].shape[0]
    return 1 / np.log(n) * np.sum(-degrees / (2 * float(m)) * np.log((degrees+.0001) / (2 * float(m))))

def compute_single_graph_statistic(graph):
    stats = []

    #Compute degrees
    deg = degree(graph.edge_index[0])
    stats.append(max(deg).item())
    stats.append(min(deg).item())
    stats.append(np.average(deg.to('cpu')))

    #Transform it into a networkx graph
    nx_graph = to_networkx(graph, to_undirected=False)
    nx_graph_undirected = to_networkx(graph, to_undirected=True)
    A_in = to_dense_adj(graph.edge_index)
    
    #Compute lcc
    stats.append(get_largest_connected_component(nx_graph_undirected))
    
    #Compute clustering coeff
    stats.append(nx.average_clustering(nx_graph))
    
    #Compute transitivity
    stats.append(nx.transitivity(nx_graph))
    
    #Compute triangle count
    stats.append(get_triangle_count(nx_graph_undirected))
    
    #Compute number of square counts
    stats.append(get_square_count(nx_graph_undirected))
    
    #Compute powerlaw alpha exponent of the degree distribution
    warnings.filterwarnings("error", category=RuntimeWarning)
    try:
        stats.append(get_powerlaw_alpha(deg))
    except RuntimeWarning:
    # Handle the warning by replacing the result with zero
        stats.append(0)
    finally:
        # Reset the warning filter to its original state
        warnings.filterwarnings("default")
        
    
    #Compute Gini coefficient
    stats.append(get_gini_coefficient(deg))
    
    #Compute edge distribution entropy
    stats.append(get_edge_entropy(deg, A_in))
    
    #Compute_assortativity
    warnings.filterwarnings("error", category=RuntimeWarning)
    try:
        stats.append(nx.degree_assortativity_coefficient(nx_graph))
    except RuntimeWarning:
        stats.append(0)
    finally:
        # Reset the warning filter to its original state
        warnings.filterwarnings("default")
    
    #Compute number of connected components
    stats.append(nx.number_connected_components(nx_graph_undirected))

    stats_tensor = torch.tensor(stats)

    return stats_tensor

def compute_temporal_graph_statistics(temporal_graph):
    """
    Compute the vector of statistics for each snapshot within a given temporal graph
    """

    all_stats = []
    i=0
    for graph in temporal_graph:
        i+=1
        print(f'Retrieving stats of graph {i} / {len(temporal_graph)}')
        stats = compute_single_graph_statistic(graph)
        all_stats.append(stats)


    # Stack all the statistics into a matrix
    stats_matrix = torch.stack(all_stats) 

    data_matrix_np = stats_matrix.numpy()

    # Create MinMaxScaler instance
    scaler = MinMaxScaler()

    # Fit and transform the data using sklearn's MinMaxScaler
    normalized_data_matrix_np = scaler.fit_transform(data_matrix_np)

    # Convert NumPy array back to PyTorch tensor
    stats_matrix = torch.from_numpy(normalized_data_matrix_np)

    return stats_matrix

def load_stats_matrix(args):
    """
    Load the computed stats for temporal graphs (done in write graph statistics)
    """
    df = pd.read_csv('./'+args.dataset_name+'_stats_matrix.csv', header=None)
    if args.dataset_name == 'synthetic':
        df = df[200:]
    # Convert the DataFrame to a PyTorch tensor
    stats_matrix = torch.tensor(df.values).to(torch.double)

    return stats_matrix

def manage_conditions(args):

    if args.conditioning:
        condition_matrix = load_stats_matrix(args).to(args.device)
        args.conditioning_size = len(condition_matrix[0])
        print('Args.conditioning_size: ', args.conditioning_size)
        print(f'Args condition: {args.dataset_name+"_stats_matrix.csv"}')
    else:
        condition_matrix = None
        args.conditioning_size = None
    
    return condition_matrix

# def sythetic_condition():
    
#     scaler = MinMaxScaler()
#     t = np.arange(1,100)
#     condition = np.sin(10*t)
#     # Fit and transform the data using sklearn's MinMaxScaler
#     normalized_data_matrix_np = scaler.fit_transform(condition.reshape(-1,1))
#     print(normalized_data_matrix_np)

#     np.savetxt('condition.csv', normalized_data_matrix_np)

