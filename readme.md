# SeDyT: A General Framework for Multi-Step Event Forecasting via Sequence Modeling on Dynamic Entity Embeddings

# Dependencies

* python >= 3.8.8
* pytorch >= 1.8.1
* dgl >= 0.6.1
* colorama >= 0.4.4
* numpy >= 1.19.2

# Configuration Files

All configuration files are stored at **/config/\<DATASET>/*.yml**. The meaning for each hyper-parameter is

- emb-net: embedding network
  - dim: dimension of the output embedding
  - dim_e: dimension of the learnable entity embedding
  - dim_t: dimension of the time encoding
  - history: number of history for the self connection
  - layer: number of GNN layers
  - sample: number of neighbors, '-1' for no neighbors
  - granularity: time duration for each sliding window
  - r_limit: maximum types of relationships, the other relationships are treated as an additional type
- gen-net: sequence model
  - dim_r: dimension of the learnable relationship embedding
  - arch: architecture of each layer, separated by '-'
  - dim: output dimension of each layer, separated by '-'
  - att_head: number of attention heads of each layer, separated by '-'
  - history: selected input history embeddings, separated by ' '
- train: training parameter
  - fwd: ignore this number of timestamps at beginning (for fast training)
  - epoch: number of epochs
  - batch_size: batch size
  - lr: learning rate
  - dropout: dropout rate
  - weight_decay: weight decay rate
  - norm_loss: currently deprecated

# Run
To run the code

```
python train.py --data <DATASET> --config <path_to_config>
```

The first run will generate the graph, which will need more time.

# Citation

Here's the bibtex in case you want to cite our work

```
@inbook{10.1145/3459637.3482177,
    author = {Zhou, Hongkuan and Orme-Rogers, James and Kannan, Rajgopal and Prasanna, Viktor},
    title = {SeDyT: A General Framework for Multi-Step Event Forecasting via Sequence Modeling on Dynamic Entity Embeddings},
    year = {2021},
    isbn = {9781450384469},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3459637.3482177},
    booktitle = {Proceedings of the 30th ACM International Conference on Information &amp; Knowledge Management},
    pages = {3667–3671},
    numpages = {5}
}
```
