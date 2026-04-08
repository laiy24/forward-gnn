# forward-gnn
Reproducing paper "Forward Learning of Graph Neural Networks" https://arxiv.org/pdf/2403.11004

## How to run
Follow the following steps to run this repo

### 1. Create Conda ENV
Running [install_packages.sh](install/install_packages.sh) to sets up the conda environment named `ForwardLearningGNN` and installs required packages.

### 2. Datasets
Use `python datasets/download.py` to run [download.py](datasets/download.py) to download all datasets used in the paper. Data will be downloaded in [`data`](./data/) folder.

### 3. Data Splits
It is recommanded that, to use the node and edge splits used for the experiments in the paper,
download them from 
[this repository](https://github.com/NamyongPark/forwardgnn-datasplits), and 
place them in the [`datasplits`](./datasplits/) folder.

On the other hand, you can generate with your own config using [datasplit.py](datasets/datasplit.py), simply edit the file `num_splits` parameter and run `python datasets/download.py`. Node and edge splits to be used for node classification and link prediction, respectively, 
will be generated in the [`datasplits`](./datasplits/) folder when the code runs for the first time.


### 4. Run
Scripts in [`exp/nodeclass/`](./exp/nodeclass/) and [`exp/linkpred/`](./exp/linkpred/) can be used 
to train GNNs using the proposed forward learning algorithms of ForwardGNN or backpropagation 
for node classification and link prediction, respectively. Simply do `sh {script_path}`.

Sample Slurm Job script to runall experiment is in [`exp/runall-fgnn.sh`](./exp/runall-fgnn.sh). This submit 9 jobs that run each config respectively.

### 5. Generate Plots
By default the results are all store in ./results. To generate plots. 
1. Run `python generate_plots/build_structural_csv.py ./results --output generate_plots/results.csv`
2. Run `python generate_plots/plot-from-csvs.py ./generate_plots`

## Folder Structure
```
├── datasets/: *Utils to download, splits and load datasets
├── exp/ : *Scripts to run experiments
├── install
│   └── install_packages.sh : *Set up conda env
├── layers : Control train and inference of each layer
│   ├── link
│   │   ├── common.py
│   │   └── link_ff.py
│   ├── node
│   │   ├── common.py
│   │   ├── node_ff.py
│   │   └── node_sf.py
│   ├── conv_layer.py
│   ├── gcn_conv.py
│   └── sage_conv.py
├── models : Control train and inference of the entire network
│   ├── bp_trainer.py
│   ├── fw_trainer.py
│   ├── link
│   │   ├── common.py
│   │   └── link_ff.py
│   ├── node
│   │   ├── common.py
│   │   ├── node_ff.py
│   │   └── node_sf.py
│   └── shared
│       ├── loss.py
│       └── utils.py
├── utils : * Utility helper functions to store results, log activities, and evaluate the model
    ├── eval_utils.py
    ├── log_utils.py
    └── train_utils.py
├── settings.py : configuration
├── experiment.py : main()
└── README.md
```
- `*` Files are modified/reused based on the original [repo](https://github.com/facebookresearch/forwardgnn/tree/main) from the authors of the paper. Those code do not touch the main novel algorithem of the paper, so it is not in the scope of this reproducing. 

## Citation
Park, N., Wang, X., Simoulin, A., Yang, S., Yang, G., Rossi, R., Trivedi, P., & Ahmed, N. (2024). Forward learning of graph neural networks. The Twelfth International Conference on Learning Representations.
