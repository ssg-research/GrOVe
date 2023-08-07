# Authors: Asim Waheed, Vasisht Duddu
# Copyright 2020 Secure Systems Group, University of Waterloo & Aalto University, https://crysp.uwaterloo.ca/research/SSG/
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import os

from utils import read_data_npy, split_by_dataset

plt.style.use('plot_style.txt')

ATTACK_TYPE = 'embedding'

def feature_engineering(target, suspect):
    return np.linalg.norm(target-suspect, axis=1).reshape(-1, 1)

def process_embds(target, independent, surrogate):
    surrogate_features = feature_engineering(target, surrogate)
    independent_features = feature_engineering(target, independent)

    return surrogate_features, independent_features

def get_features(embeddings_dict):
    all_embeddings = [embds_tupl for embds_tupl in embeddings_dict.values()]

    embds_target = np.concatenate([tupl[0] for tupl in all_embeddings], axis=0)
    embds_independent = np.concatenate([tupl[1] for tupl in all_embeddings], axis=0)
    embds_surrogate = np.concatenate([tupl[2] for tupl in all_embeddings], axis=0)

    print("Target embeddings:", embds_target.shape)

    surrogate_features, independent_features = process_embds(embds_target, embds_independent, embds_surrogate)

    print("Total surrogate features:", surrogate_features.shape)
    print("Total independent features:", independent_features.shape)

    return surrogate_features, independent_features

def visualize_features(surrogate_features, independent_features, dataset):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 6))

    surrogate_features = surrogate_features.reshape(-1)
    independent_features = independent_features.reshape(-1)

    ax.hist(surrogate_features, bins=100, range=(0, 100), label='Target vs. Surrogate', alpha=1)
    ax.hist(independent_features, bins=100, range=(0, 100), label='Target vs. Independent', alpha=0.6)

    titles = {
        'acm': 'ACM',
        'amazon': 'Amazon',
        'citeseer': 'Citeseer',
        'dblp': 'DBLP',
        'pubmed': 'PubMed',
        'coauthor': 'Co-Author',
        'all_datasets': 'All Datasets'
    }

    # ax.set_title(titles[dataset])
    ax.legend(fontsize='large')
    result_dir = 'plots_distance_histograms/'
    os.makedirs(result_dir, exist_ok=True)
    plt.savefig('{}{}.pdf'.format(result_dir, dataset), format='pdf', dpi=400)


if __name__ == '__main__':
    all_embeddings = read_data_npy('./model_stealing/embeddings_overlap_False_embedding_original_simple_extraction'.format(ATTACK_TYPE), 0)

    print("Embeddings read")

    embeddings_per_dataset = split_by_dataset(all_embeddings)

    print(len(embeddings_per_dataset))

    for dataset, embeddings in embeddings_per_dataset.items():
        print("Running for dataset: ", dataset)
        surrogate_features, independent_features = get_features(embeddings)
        visualize_features(surrogate_features, independent_features, dataset)

