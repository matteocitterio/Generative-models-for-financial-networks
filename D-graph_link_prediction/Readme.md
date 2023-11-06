Please have a look at the [EvolveGCN paper](https://arxiv.org/pdf/1902.10191.pdf) and the [Chapter 13 of the Maxime Labonne book](https://github.com/PacktPublishing/Hands-On-Graph-Neural-Networks-Using-Python).

Here you can find the code I used for the EvolveGCN-H model i a temporal link prediction task.
- `main.py`: main file
- `dateset_managment.py`: builds the dataset and takes care of the data windowing process, including negative sampling procedure.
- `logits_cross_entropy.py`: original implementation of the cross entropy function used in the paper
- `EvolveGCNH.py`: pytorch geometric implementation of EvolveGCNH with a slight custom adjustment.
- `EvolveGCNO.py`: pytorch geometric implementation of EvolveGCNO
- `model_managment.py`: takes care of model classes definitions and their methods
- `train_managment.py`: implementation of the training and evaluation routine
- `utils.py`: mainly used for argument parsing.
