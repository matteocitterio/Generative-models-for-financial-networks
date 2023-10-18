# Models

This folder contains explanatory notebook concerning the particular models we used for UU-graph generation.
- **GraphVAE**: one of the first model. The first version is currently obsolete in the sense that is able of generating graphs with ~40 nodes, we reported here an explanation as a case study even though we never actually used it for graph generation
- **NetGAN**: this model uses a combination of RWs sampling, Generative Adversial Networks (GAN) and LSTM architectures to generalize input graphs. It was one of the first generative models to exhibit acceptable performances despite the intensive computational cost.
- **Cell**: this model represents an updated version of NetGAN: the core of the is unaltered, however the authors mainly focused over the low-rank approximation performed in NetGAN. By doing so and eliminating the GAN, the RWs sampling and the LSTM architecture, they achieved similar performances while drastically reducing the computational costs.
