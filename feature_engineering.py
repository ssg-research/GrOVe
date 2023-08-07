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

def l2_norm(target, suspect):
    distances = np.linalg.norm(target-suspect, axis=1).reshape(-1, 1)
    return distances

def feature_engineering(target, suspect):
    return (target-suspect)**2

def process_embds(target, independent, surrogate):

    surrogate_features = feature_engineering(target, surrogate)
    independent_features = feature_engineering(target, independent)

    return surrogate_features, independent_features

