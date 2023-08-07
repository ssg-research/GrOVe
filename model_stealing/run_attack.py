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

import os

target_models = ['gat', 'gin', 'sage']
independent_models = ['gat', 'gin', 'sage']
surrogate_models = ['gat', 'gin']

datasets = ['dblp', 'citeseer_full', 'pubmed', 'coauthor_phy', 'acm', 'amazon_photo']

# [embedding, prediction]
recovery_from = 'embedding'

# [original, idgl]
attack_structure = 'original' 

# [simple_extraction, double_extraction, pruning, fine_tune, dist_shift]
attack_robustness = 'simple_extraction'

experiment_ids = [i for i in range(0, 10)]
surrogate_ids = [i for i in range(0, 10)]

for experiment_id in experiment_ids:
    for dataset in datasets:
        for target_model in target_models:
            for independent_model in independent_models:
                for surrogate_model in surrogate_models:
                    for surrogate_id in surrogate_ids:
                        run_string = "python main.py target_model={} independent_model={} dataset={} surrogate_model={} attack.recovery_from={} attack.robustness={} experiment_id={} surrogate_id={} attack.structure={}".format(
                            target_model,
                            independent_model,
                            dataset,
                            surrogate_model,
                            recovery_from,
                            attack_robustness,
                            experiment_id,
                            surrogate_id,
                            attack_structure
                        )
                        os.system(run_string)

