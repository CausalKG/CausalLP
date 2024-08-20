# Select best parameters for all the different graph splits with different predicates in training data
# such as 1) just causedByType or causesType, 2)   


import os
import numpy as np
import pandas as pd
import numpy as np
import ampligraph #Added
from ampligraph.datasets import load_onet20k, load_ppi5k, load_nl27k, load_cn15k
from ampligraph.latent_features import ComplEx, TransE, DistMult, HolE, ConvE, ConvKB, MODEL_REGISTRY
from ampligraph.utils import save_model, restore_model
from ampligraph.evaluation import evaluate_performance, mrr_score, hits_at_n_score, mr_score
from ampligraph.evaluation import select_best_model_ranking #Added
from ampligraph.datasets import load_from_csv #Added
from sklearn.model_selection import train_test_split #Added
import json

# Causes and causesBy
folderList = [        
        {'causedByType':'CausedByTypeCausesCausedBy'},          
        ]


model_class = [ComplEx, TransE, DistMult, HolE] 

param_grid = {
                     "batches_count": [100],
                   "seed": 0,
                   "epochs": [100,200,300,500],
                    "k": [100, 200],
                  "eta": [5,10,15],
                    "loss": ["nll", "multiclass_nll"],
                    "loss_params": {
                         "margin": [2]
                    },
                    "embedding_model_params": {

                     },
                     "regularizer": ["LP", None],
                    "regularizer_params": {
                         "p": [1, 3],
                         "lambda": [1e-4, 1e-5]
                     },
                     "optimizer": ["adam"],
                     "optimizer_params":{
                         "lr": lambda: np.random.uniform(0.0001, 0.01)
                     },
                     "verbose": False
                 }

for m in model_class:
    for i in folderList:
        print("KG folder: ",list(i.values())[0])
        print("Predicates in training: ",list(i.keys())) 
        X_train = pd.read_csv('../Data/CausalCLEVERERHumanKG_MarkovSplit/Explanation/'+list(i.values())[0]+'/train.txt', sep=',',header=None).to_numpy()
        X_test = pd.read_csv('../Data/CausalCLEVERERHumanKG_MarkovSplit/Explanation/'+list(i.values())[0]+'/test.txt', sep=',',header=None).to_numpy()
        
        best = select_best_model_ranking(m, X_train[:, 0:3], X_valid[:, 0:3], X_test[:, 0:3], param_grid,
                              max_combinations=100, use_filter=True, verbose=True, early_stopping=True)
        print(best)
        print("\n")

    
    
    
    
    
    
    
    
    