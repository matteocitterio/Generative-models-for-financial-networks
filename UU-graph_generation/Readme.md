# Generation of Unweighted and Undirected graphs

Here we explore the easiest task within graph generation: the generation of unweighted and undirected graphs, which are represented as binary and symmetric adjacency matrices.
The models that are taken into account are the `NetGAN` model and the `CELL` model, both tested on datasets of increasing size:

- `SmallGraph`: small UU-graph consisting of 332 nodes.
- `Cora-ML`: citation network with 2810 nodes
- `CiteSeer`: citation network with 4230 nodes and multiple connected components.

For a more detailed explaination of the models, see `./Models/`

## How to evaluate the models

Models evaluation in the context of graph generation has always been considered a quite trublesome task. This is beacuse this particular task inherits the difficulties already experimented in the field of data generation, such as image generation, but without having at least the possibility of discerning by sight wheter a certain sample properly generalizes the input data.
Thorughout the scientific community, it has been widely adopted a rather topological approach: i.e. comparing the generated samples with the input data using several coefficients describing the graph from a topological point of view

## List of graph statistics

- Assortativity: $\frac{cov{(X,Y)}}{\sigma_{X}\sigma_{Y}}$ It is the Pearson's correlation of degrees of connencted nodes, where the $(X,Y)$ pairs represent the degrees of connected nodes
- Power law exponent: $1+n\left(\sum_{v\in V}{\log{\frac{d(u)}{d_{min}}}}\right)^{-1}$ It is the exponent of the power law distribution, where $d_{min}$ denotes the minimum degree in a network
- Gini Coefficient: $\frac{2\sum_{i=1}^{N}{i\hat{d}_{i}}}{N\sum_{i=1}^{N}{\hat{d}_{i}}}$

## Results

The following are the results of the models over the datasets:

| Graph           | d_max   | d_min | d      | LCC | triangle count | power law exp | gini  | real edge distribution entropy | assortativity | clustering coefficient | #components | cpl   | time[s] |
| --------------- | ------- | ----- | ------ | --- | -------------- | ------------- | ----- | ------------------------------ | ------------- | ---------------------- | ----------- | ----- | ------- |
| **Small graph**     | 139     | 1     | 12.807 | 332 | 12181          | 1.583         | 0.641 | 0.865                          | \-0.207       | 0.016                  | 1           | 2.738 | \-      |
| \-              | \-      | \-    | \-     | \-  | \-             | \-            | \-    | \-                             | \-            | \-                     | \-          | \-    | \-      |
| **Erdós-Renyi**    | 25      | 5     | 13.21  | 332 | 386            | 2.07          | 0.145 | 0.993                          | 0.008         | 0.009                  | 1           | 2.533 | 0       |
| **NetGAN (53% EO)** | 119.666 | 1     | 12.807 | 332 | 4819           | 1.491         | 0.527 | 0.910                          | \-0,214       | 0.010                  | 1           | 2.549 | 2782.8  |
| **CELL (53% EO)**   | 106.6   | 1     | 12.807 | 332 | 6094           | 1.522         | 0.569 | 0.896                          | \-0,214       | 0.012                  | 1           | 2.716 | <1



| Graph           | d_max | d_min | d     | LCC    | triangle count | power law exp | gini  | real edge distribution entropy | assortativity | clustering coefficient | #components | cpl   | time[s]    |
| --------------- | ----- | ----- | ----- | ------ | -------------- | ------------- | ----- | ------------------------------ | ------------- | ---------------------- | ----------- | ----- | ---------- |
| **Cora-ML**         | 246   | 1     | 5.680 | 2810   | 5247           | 1.767         | 0.495 | 0.938                          | \-0,076       | 0.004                  | 1           | 5.271 | \-         |
| \-              | \-    | \-    | \-    | \-     | \-             | \-            | \-    | \-                             | \-            | \-                     | \-          | \-    | \-         |
| **Erdós-Renyi**     | 15    | 0     | 5.64  | 2804   | 36             | 1.611         | 0.231 | 0.988                          | 0.0005        | 0.001                  | 7           | 4.790 | 10         |
| **NetGAN (53% EO)** | 189.2 | 1     | 5.680 | 2809   | 2128           | 1.703         | 0.423 | 0.955                          | \-0,08        | 0.003                  | 1.4         | 4.712 | 63.485.598 |
| **CELL (53% EO)**   | 213.4 | 1     | 5.680 | 2798.4 | 2820           | 1.744         | 0.469 | 0.946                          | \-0,08        | 0.003                  | 5.6         | 4.911 | 7          |


| Graph           | d_max | d_min | d     | LCC    | triangle count | power law exp | gini  | real edge distribution entropy | assortativity | clustering coefficient | #components | cpl   | time[s] |
| --------------- | ----- | ----- | ----- | ------ | -------------- | ------------- | ----- | ------------------------------ | ------------- | ---------------------- | ----------- | ----- | ------- |
| **CiteSeer**        | 85    | 1     | 2.523 | 1681   | 1041           | 2.766         | 0.460 | 0.942                          | \-0,077       | 0.00866                | 515         | 7.367 | \-      |
| \-              | \-    | \-    | \-    | \-     | \-             | \-            | \-    | \-                             | \-            | \-                     | \-          | \-    | \-      |
| **Erdós-Renyi**     | 10    | 0     | 2.518 | 3824   | 4              | 2.161         | 0.344 | 0.973                          | 0.002         | 0.001                  | 371         | 8.834 | 5       |
| **NetGAN (27% EO)** | 51.4  | 1     | 2.523 | 4202.2 | 154.6          | 2.364         | 0.333 | 0.973                          | \-0,117       | 0.00737                | 7.8         | 10.18 | Days    |
| **CELL (53% EO)**   | 63.4  | 1     | 2.523 | 3864   | 181            | 2.548         | 0.407 | 0.958                          | \-0,012       | 0.00336                | 120.8       | 9.052 | 10      |


## Bibliography:

- GraphRNN (generation evaluation): (https://arxiv.org/abs/1802.08773)
- GraphVAE: (https://arxiv.org/abs/1611.07308v1)(Kipf Welling 2016) (https://arxiv.org/abs/1802.03480)(2018)
- NetGAN:(paper [https://arxiv.org/abs/1803.00816]), (implementation: [https://github.com/mmiller96/netgan_pytorch])
- Cell: (paper [https://proceedings.mlr.press/v119/rendsburg20a.html]), implementation: [https://github.com/hheidrich/CELL]
- eros renyi graph:[https://snap.stanford.edu/class/cs224w-readings/erdos59random.pdf]
- cora ml dataset: graph extraction: https://arxiv.org/abs/1707.03815 raw data: https://link.springer.com/article/10.1023/A:1009953814988
- Citeseer dataset: [https://arxiv.org/abs/1603.08861]
