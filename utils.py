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
import glob 

def get_model_details(model_name: str):

	model_details = model_name.split('_')

	model_details = [x for x in model_details if x not in ['full', 'photo', 'phy']]
	target_model = model_details[1]
	independent_model = model_details[3]
	dataset = model_details[4]
	embedding_dimension = model_details[5]
	surrogate_model = model_details[7]
	surrogate_id = model_details[8]
	experiment_id = model_details[9]

	return target_model, independent_model, surrogate_model, dataset, embedding_dimension, surrogate_id, experiment_id 

def read_data(directory: str, experiment_id: str) -> dict:
	files = glob.glob('{}/*_{}/*.csv'.format(directory, experiment_id))
	files = [files[i:i+3] for i in range(0, len(files), 3)]
	embds = {}

	for trio in files:
		# Get experiment properties
		start = trio[0].find(directory) + len(directory)
		end = trio[0].rfind('/')
		model_name = trio[0][start:end]

		target_embeddings_file = [file for file in trio if 'target_embeddings.csv' in file][0]
		benign_embeddings_file = [file for file in trio if 'independent_embeddings.csv' in file][0]
		surrogate_embeddings_file = [file for file in trio if 'surrogate_embeddings.csv' in file][0]

		embds_target = np.genfromtxt(target_embeddings_file, delimiter=',')
		embds_benign = np.genfromtxt(benign_embeddings_file, delimiter=',')
		embds_surrogate = np.genfromtxt(surrogate_embeddings_file, delimiter=',')

		embds[model_name] = (embds_target, embds_benign, embds_surrogate)

	return embds

def read_data_npy(directory: str, experiment_id: str) -> dict:
	files = glob.glob('{}/*_{}/*.npy'.format(directory, experiment_id))
	files = [files[i:i+3] for i in range(0, len(files), 3)]
	embds = {}

	for trio in files:
		# Get experiment properties
		start = trio[0].find(directory) + len(directory)
		end = trio[0].rfind('/')
		model_name = trio[0][start:end]

		target_embeddings_file = [file for file in trio if 'target_embeddings.npy' in file][0]
		independent_embeddings_file = [file for file in trio if 'independent_embeddings.npy' in file][0]
		surrogate_embeddings_file = [file for file in trio if 'surrogate_embeddings.npy' in file][0]

		with open(target_embeddings_file, 'rb') as f:
			embds_target = np.load(f)
		with open(independent_embeddings_file, 'rb') as f:
			embds_independent = np.load(f)
		with open(surrogate_embeddings_file, 'rb') as f:
			embds_surrogate = np.load(f)

		embds[model_name] = (embds_target, embds_independent, embds_surrogate)

	return embds

def get_model_details_di(model_name: str):

    model_details = model_name.split('_')
    
    model_details = [x for x in model_details if x not in ['full', 'photo', 'phy']]
    (model_details)
    target_model = model_details[1]
    independent_model = model_details[3]
    dataset = model_details[4]
    embedding_dimension = model_details[5]

    return target_model, independent_model, dataset, embedding_dimension 

def read_data_di(directory: str, experiment_id: str) -> dict:
	files = glob.glob('{}/*_{}/*.csv'.format(directory, experiment_id))
	files = [files[i:i+2] for i in range(0, len(files), 2)]
	embds = {}

	print(len(files))

	for trio in files:
		# Get experiment properties
		start = trio[0].find(directory) + len(directory)
		end = trio[0].rfind('/')
		model_name = trio[0][start:end]

		target_embeddings_file = [file for file in trio if 'target_embeddings.csv' in file][0]
		independent_embeddings_file = [file for file in trio if 'independent_embeddings.csv' in file][0]

		embds_target = np.genfromtxt(target_embeddings_file, delimiter=',')
		embds_independent = np.genfromtxt(independent_embeddings_file, delimiter=',')

		embds[model_name] = (embds_target, embds_independent)

	return embds

def calc_distance(target, test):
	return (target - test)**2

def split_by_dataset(embeddings):
    models = embeddings.keys()

    embeddings_per_dataset = {}
    for model in models:
        _, _, _, dataset, _, _, _ = get_model_details(model)

        if dataset not in embeddings_per_dataset:
            embeddings_per_dataset[dataset] = {}
            
        embeddings_per_dataset[dataset][model] = embeddings[model]

    return embeddings_per_dataset


def split_by_dataset_and_target(embeddings):
	models = embeddings.keys()

	embeddings_per_target = {}
	for model in models:
		target_model, _, _, dataset, _, _, _ = get_model_details(model)
		model_type = dataset + '_' + target_model

		if model_type not in embeddings_per_target:
			embeddings_per_target[model_type] = {}
            
		embeddings_per_target[model_type][model] = embeddings[model]

	return embeddings_per_target