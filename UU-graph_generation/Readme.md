Here we explore the easiest task within graph generation: the generation of unweighted and undirected graphs, which are represented as binary and symmetric adjacency matrices.
The models that are taken into account are the `NetGAN` model and the `CELL` model, both tested on datasets of increasing size:

- `SmallGraph`: small UU-graph consisting of 332 nodes.
- `Cora-ML`: citation network with 2810 nodes
- `CiteSeer`: citation network with 4230 nodes and multiple connected components.

| Graph           | d_max   | d_min | d      | LCC | triangle count | power law exp | gini  | real edge distribution entropy | assortativity | clustering coefficient | #components | cpl   | time[s] |
| --------------- | ------- | ----- | ------ | --- | -------------- | ------------- | ----- | ------------------------------ | ------------- | ---------------------- | ----------- | ----- | ------- |
| Small graph     | 139     | 1     | 12.807 | 332 | 12181          | 1.583         | 0.641 | 0.865                          | \-0.207       | 0.016                  | 1           | 2.738 | \-      |
| \-              | \-      | \-    | \-     | \-  | \-             | \-            | \-    | \-                             | \-            | \-                     | \-          | \-    | \-      |
| Erdós-Renyi     | 25      | 5     | 13.21  | 332 | 386            | 2.07          | 0.145 | 0.993                          | 0.008         | 0.009                  | 1           | 2.533 | 0       |
| NetGAN (53% EO) | 119.666 | 1     | 12.807 | 332 | 4819           | 1.491         | 0.527 | 0.910                          | \-0,214       | 0.010                  | 1           | 2.549 | 2782.8  |
| CELL (53% EO)   | 106.6   | 1     | 12.807 | 332 | 6094           | 1.522         | 0.569 | 0.896                          | \-0,214       | 0.012                  | 1           | 2.716 | <1
