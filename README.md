# GrOVe
This paper will appear in IEEE Symposium on Security and Privacy 2024.

This repo contains code that allows you to reproduce experiments for the fingerprinting scheme presented in *GrOVe: Ownership Verification of Graph Neural Networks using Embeddings.*

## Environment Setup
```
conda env create --file environment.yaml &&
conda activate graph_ownership_verification && 
# Install GraphGallery
wget https://github.com/EdisonLeeeee/GraphGallery/archive/refs/tags/1.0.0.tar.gz && 
tar -zxvf 1.0.0.tar.gz && 
cd GraphGallery-1.0.0/ &&
pip3 install -e . --verbose &&
cd .. &&
rm 1.0.0.tar.gz
```

## Run Model Stealing Attack
```
cd model_stealing
python main.py target_model=gat independent_model=gat dataset=dblp surrogate_model=gat attack.recovery_from=embedding attack.robustness=simple_extraction experiment_id=0 surrogate_id=0 attack.structure=original
```

Model Stealing Hyperparameters:
```
dataset:                   ['dblp', 'pubmed', 'citeseer_full', 'coauthor_phy', 'acm', 'amazon_photo']           # Datasets used to train the surrogate model
target_model:              ['gat', 'gin', 'sage']                                                               # Target model's architecuture
independent_model:         ['gat', 'gin', 'sage']                                                               # Benign/Independent model's architecuture
surrogate_model:           ['gat', 'gin']                                                                       # Surrogate model's architecuture
attack.recovery_from:      ['embedding', 'prediction', 'projection']                                            # Target model's response, default is 'embedding'
attack.robustness:         ['simple_extraction', 'double_exctraction', 'pruning', 'fine_tune', 'dist_shift']    # Robustness measures taken by adversary to remove fingerprint
attack.structure:          ['original', 'idgl']                                                                 # Type I/II attacks, 'original' means we use the original graph structure and 'idgl' means we use idgl to reconstruct the graph structure.
experiment_id              [0:10]                                                                               # Random seed for data splitting and target / independent model training initialization
surrogate_id               [0:10]                                                                               # Random seed for model training initialization of surrogate model
attack.overlap:            [True, False]                                                                        # Whether the surrgate training set is overlap of target training set or not
delete_edges:              ["yes", "no"]                                                                        # Whether to deleted edges from surrogate training graph or not
```

All the default parameters can be inspected in the ```conf/``` directory. The file ```conf/config.yaml``` contains the default model architectures, the attack hyperparameters and the dataset.

The wrapper script ```model_stealing/run_attack.py``` can be used to train the target, independent, and surrogate models used in the paper using the same random seeds. 

### Output Structure
Each models' outputs will be saved in a directory with the following naming convention: ```./model_stealing/embeddings_overlap_{attack.overlap}_{attack.recovery_from}_{attack.structure}_{attack.robustness}```

Within each directory for an experimentthe output for each model will be saved in a subdirectory with the following naming convention: ```target_{target_model}_independent_{independent_model}_{dataset}_256_surrogate_{surrogate_model}_{surrogate_id}_{experiment_id}/```

## Train Similarity Model for GrOVe.
```
python train_similarity_model.py <output_dir>
```

Lines 228 - 258 define the embeddings used for training and evaluating the similarity model. By default, surrogate models with surrogate_id=0 are used for training, while the rest are used for testing.

## Analyze results
```
python analyze_results.py <input_dir>
```

The input_dir should be the same as the output_dir used in the training script. This generates the tables and plots used in the paper.
