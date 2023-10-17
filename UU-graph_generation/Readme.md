Here we explore the easiest task within graph generation: the generation of unweighted and undirected graphs, which are represented as binary and symmetric adjacency matrices.
The models that are taken into account are the `NetGAN` model and the `CELL` model, both tested on datasets of increasing size:

- `SmallGraph`: small UU-graph consisting of 332 nodes.
- `Cora-ML`: citation network with 2810 nodes
- `CiteSeer`: citation network with 4230 nodes and multiple connected components.


| d_max              | d_min | d                 | LCC | triangle_count    | power_law_exp     | gini               | rel_edge_distr_entropy | assortativity        | clustering_coefficient | n_components | cpl                |
| ------------------ | ----- | ----------------- | --- | ----------------- | ----------------- | ------------------ | ---------------------- | -------------------- | ---------------------- | ------------ | ------------------ |
| 119.66666666666667 | 1     | 12.80722891566265 | 332 | 4819.333333333333 | 1.491578323383265 | 0.5274536528050112 | 0.9103931254335839     | -0.21426240349439865 | 0.010978695060603163   | 1            | 2.5496184132299593 |
