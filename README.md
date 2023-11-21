# Bridge-Aware Clustering
Repository for Bridge-Aware Clustering algorithm, developed in python3.


### Installation instructions
##### pip
```bash
pip install -r requirements.txt
```
##### Anaconda
```bash
conda env create -f environment.yml
```

### BorderPeeling implementation
The BorderPeeling implementation is obtained from authors' official repository:

https://github.com/nadavbar/BorderPeelingClustering

Minor changes were made to adapt the original source code to python3.

### DADC implementation
The DADC implementation is obtained from authors' official repository:

https://github.com/JianguoChen2015/DADC

Minor changes were made: plotting functionalities and export information as csv files were disabled.

### DenMune clustering
The DenMune clustering algorithm was installed via `pip install denmune` command.

The authors' implementation is available at the following GitHub repository: https://github.com/egy1st/denmune-clustering-algorithm.

### Data
The 25 synthetic dataset were downloaded from

https://github.com/deric/clustering-benchmark

The datasets are stored in `datasets` folder in `.arff` format.

### Scripts description
* `comparisons.py`: computes the adjusted rand index scores for each of the considered techniques and perform the statistical tests. Generates figures and a final report;
* `statistical_tests.py`: performs the Friedman and Nemenyi statistical tests given the input `.csv` file;
* `utils.py`: contains utility functions;
* `generate_arrow_images.py`: generates images with connectivity graph;
* `plot_bridges_figures.py`: generates plots containing datasets, clusters, bridges and outliers;
* `grid_search.py`: performs the grid search and save results using pickle;
* `run_dadc.py`: run the DADC algorithm only;

### Citation
```
@ARTICLE{bridge_clustering,
  author={Colomba, Luca and Cagliero, Luca and Garza, Paolo},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={Density-Based Clustering by Means of Bridge Point Identification}, 
  year={2023},
  volume={35},
  number={11},
  pages={11274-11287},
  doi={10.1109/TKDE.2022.3232315}
}
```
