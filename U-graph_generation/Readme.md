# Generation of Unweighted (but Directed) graphs

## Performance evaluation

write about the considered graph statistics

here we normalized the metrics using $\left|\sum_{S_{i}\in S}{\frac{S_{i}}{O_{i}}-1}\right|\cdot 100$ where $S,O$ are the sets of graphs statistics for the average of the generated samples and the orginal graph, respectively. By doing so, we obtain a single scalar that we want as closer as possible to 0.

## Results:

- DirectedCell, best model: 25.0817% [SampleGraphFromPaths, LocalityLossFunction] [Facebook]
- DirectedCell, best model:          [,] [WikiVote]
