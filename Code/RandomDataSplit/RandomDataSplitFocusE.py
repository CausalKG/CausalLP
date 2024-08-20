import os
import numpy as np
import pandas as pd
import numpy as np
import ampligraph #Added
# from ampligraph.datasets import load_onet20k, load_ppi5k, load_nl27k, load_cn15k
from ampligraph.latent_features import ComplEx, TransE, DistMult, HolE, MODEL_REGISTRY
from ampligraph.utils import save_model, restore_model
from ampligraph.evaluation import evaluate_performance, mrr_score, hits_at_n_score, mr_score
from ampligraph.datasets import load_from_csv #Added
from sklearn.model_selection import train_test_split #Added
import json
print("All the package imported \n")

saveResults = pd.DataFrame(columns=['Model','MRR','MR','Hit@1','Hit@3','Hit@10'])

def evaluateModel(Model, ModelFocusE, X_filter, X_test, corruptSide, entitySubset, folderName, model_name):
    global saveResults
    if entitySubset:
        print("Evaluating the regular")
        ranks = evaluate_performance(X_test,
                                     model=Model, 
                                     filter_triples=X_filter,
                                     corrupt_side=str(corruptSide), 
                                     ranking_strategy='worst')


        print("\n Evaluating the FocusE")
        ranksFocusE = evaluate_performance(X_test, 
                                     model=ModelFocusE, 
                                     filter_triples=X_filter,
                                     entities_subset=entitySubset,
                                     corrupt_side=str(corruptSide), 
                                     ranking_strategy='worst')

        print("Regular - MRR: ", mrr_score(ranks), "MR: ", mr_score(ranks), "Hits@1: ",        hits_at_n_score(ranks, 1), "Hits@3: ", hits_at_n_score(ranks, 3), "Hits@10: ", hits_at_n_score(ranks, 10)) 
        
        print("FocusE - MRR: ", mrr_score(ranksFocusE), "MR: ", mr_score(ranksFocusE), "Hits@1: ",        hits_at_n_score(ranksFocusE, 1), "Hits@3: ", hits_at_n_score(ranksFocusE, 3), "Hits@10: ", hits_at_n_score(ranksFocusE, 10)) 

    saveResults = pd.concat([saveResults, pd.DataFrame({'Model':[model_name+'_'+folderName], 'MRR':["{:.4f}".format(mrr_score(ranks))], 'MR':["{:.4f}".format(mr_score(ranks))], 'Hit@1':["{:.4f}".format(hits_at_n_score(ranks, 1))], 'Hit@3':["{:.4f}".format(hits_at_n_score(ranks, 3))], 'Hit@10':["{:.4f}".format(hits_at_n_score(ranks, 10))]})])

    saveResults = pd.concat([saveResults, pd.DataFrame({'Model':[model_name+'_FocusE_'+folderName], 'MRR':["{:.4f}".format(mrr_score(ranks))],'MR':["{:.4f}".format(mr_score(ranksFocusE))], 'Hit@1':["{:.4f}".format(hits_at_n_score(ranksFocusE, 1))], 'Hit@3':["{:.4f}".format(hits_at_n_score(ranksFocusE, 3))], 'Hit@10':["{:.4f}".format(hits_at_n_score(ranksFocusE, 10))]})])
    
    saveResults = pd.concat([saveResults, pd.DataFrame({'Model':[''], 'MR':[''], 'MRR':[''], 'Hit@1':[''], 'Hit@3':[''], 'Hit@5':[''], 'Hit@10':['']})])
    
    return 
    
def generate_focusE_dataset_splits(filterPredicates,folder,split_test_into_top_bottom=True, split_threshold=0.1):
    X_train = pd.read_csv('../Data/CausalCLEVERERHumanKG_RandomSplit/'+folder+'/train.txt', sep=',',header=None).to_numpy()
    X_test = pd.read_csv('../Data/CausalCLEVERERHumanKG_RandomSplit/'+folder+'/test.txt', sep=',',header=None).to_numpy()
    X_valid = pd.read_csv('../Data/CausalCLEVERERHumanKG_RandomSplit'+folder+'/valid.txt', sep=',',header=None).to_numpy()
  
    predicate = ["http://semantic.bosch.com/causal/v00/"+filterPredicates]
    filter_test=np.isin(X_test[:,1], predicate)
    X_test = X_test[filter_test]

    train_numeric_values = X_train[:, 3]
    valid_numeric_values = X_valid[:, 3]
    test_numeric_values = X_test[:, 3]
    
    train = X_train[:, 0:3]
    valid = X_valid[:, 0:3]
    test = X_test[:, 0:3]
    

    sorted_indices = np.argsort(test_numeric_values)
    test = test[sorted_indices]
    test_numeric_values = test_numeric_values[sorted_indices]
    
    if split_test_into_top_bottom:
        split_threshold = int(split_threshold * test.shape[0])
        
        test_bottomk = test[:split_threshold]
        test_bottomk_numeric_values = test_numeric_values[:split_threshold]
        
        test_topk = test[-split_threshold:]
        test_topk_numeric_values = test_numeric_values[-split_threshold:]
        
    dataset = {'train':train, 'valid':valid, 'test':test, 'train_numeric_values':train_numeric_values, 
         'valid_numeric_values':valid_numeric_values,'test_numeric_values':test_numeric_values, 
         'test_bottomk':test_bottomk, 'test_bottomk_numeric_values':test_bottomk_numeric_values, 
         'test_topk':test_topk, 'test_topk_numeric_values':test_topk_numeric_values}
    return dataset


def callTheSplitFunction(filterPredicates,folderName, model_name, vanilla_params, focusE_params):
    X = generate_focusE_dataset_splits(filterPredicates, folderName)
    
    print("X = generate_focusE_dataset_splits \n")
    X_filter = np.concatenate([X['train'], X['valid'], X['test']], 0)
    
    print("X_filter \n")
    X_train = X['train']
    X_train_numeric_values = X['train_numeric_values']
    
    X_valid = X['valid']
    X_valid_numeric_values = X['valid_numeric_values'].astype('float64') #Added dtype
    X_valid = X_valid[(X_valid_numeric_values) >= 0.75] 
    
    X_test = X['test']
    
    modelBuilding(X_train,X_test,X_valid,X_filter,X_train_numeric_values,count, folderName, filterPredicates,model_name, vanilla_params, focusE_params)
    return

def modelBuilding(X_train,X_test,X_valid,X_filter,X_train_numeric_values,count, folderName, filterPredicates,model_name, vanilla_params, focusE_params):
    causalEventEntity_subset=['http://semantic.bosch.com/CausalClevrerHumans/v00/Move',
       'http://semantic.bosch.com/CausalClevrerHumans/v00/Hit',
       'http://semantic.bosch.com/CausalClevrerHumans/v00/Bump',
       'http://semantic.bosch.com/CausalClevrerHumans/v00/Collide',
       'http://semantic.bosch.com/CausalClevrerHumans/v00/ChangeDirection',
       'http://semantic.bosch.com/CausalClevrerHumans/v00/ComeFrom',
       'http://semantic.bosch.com/CausalClevrerHumans/v00/Travel',
       'http://semantic.bosch.com/CausalClevrerHumans/v00/Tag',
       'http://semantic.bosch.com/CausalClevrerHumans/v00/Go',
       'http://semantic.bosch.com/CausalClevrerHumans/v00/Push',
       'http://semantic.bosch.com/CausalClevrerHumans/v00/Bounce',
       'http://semantic.bosch.com/CausalClevrerHumans/v00/Sideswipe',
       'http://semantic.bosch.com/CausalClevrerHumans/v00/Spin',
       'http://semantic.bosch.com/CausalClevrerHumans/v00/Slide',
       'http://semantic.bosch.com/CausalClevrerHumans/v00/Slow',
       'http://semantic.bosch.com/CausalClevrerHumans/v00/Strike',
       'http://semantic.bosch.com/CausalClevrerHumans/v00/Run',
       'http://semantic.bosch.com/CausalClevrerHumans/v00/Roll',
       'http://semantic.bosch.com/CausalClevrerHumans/v00/Stand',
       'http://semantic.bosch.com/CausalClevrerHumans/v00/Halt',
       'http://semantic.bosch.com/CausalClevrerHumans/v00/Touch',
       'http://semantic.bosch.com/CausalClevrerHumans/v00/Hurl',
       'http://semantic.bosch.com/CausalClevrerHumans/v00/Head',
       'http://semantic.bosch.com/CausalClevrerHumans/v00/Start',
       'http://semantic.bosch.com/CausalClevrerHumans/v00/Throw',
       'http://semantic.bosch.com/CausalClevrerHumans/v00/Have']
    
    early_stopping = { 'x_valid': X_valid,
                    'criteria': 'mrr', 
                    'x_filter': X_filter, 
                    'stop_interval': 8, 
                    'burn_in': 200, 
                    'check_interval': 25 }
    
    print("Making the model_focusE \n")
     
    model_transE = MODEL_REGISTRY.get(model_name)(**vanilla_params)
    
    print("Model intialized")
    model_transE.fit(X_train, True, early_stopping)    
    save_model(model_transE, model_name_path = '../Model/CausalCLEVERERHumanKG_RandomSplit/'+folderName+'_'+model_name+'.pkl')
    
    model_focusE = MODEL_REGISTRY.get(model_name)(**focusE_params)
    print("model_focusE done \n")
    model_focusE.fit(X_train, False, X_train_numeric_values)
    save_model(model_transE, model_name_path = '../Model/CausalCLEVERERHumanKG_RandomSplit/'+folderName+'_'+model_name+'_FocusE.pkl')
    

    predicate = ["http://semantic.bosch.com/causal/v00/"+filterPredicates]
    filter_test=np.isin(X_test[:,1], predicate)
    X_test = X_test[filter_test]
    
    evaluateModel(model_transE, model_focusE, X_filter, X_test, "s,o", causalEventEntity_subset, folderName,model_name) 
    return


if __name__ == "__main__":  
    
    json_file_path = './config.json'

    with open(json_file_path, 'r') as j:
         config = json.loads(j.read())
     
    datasets = list(config['regular'].keys())
    for dataset in datasets:
        model_names = list(config['regular'][dataset].keys())
        for model_name in model_names:
            print(dataset, '-', model_name)
            print('-----------------')
            vanilla_params = config['regular'][dataset][model_name]
            focusE_params = config['focusE'][dataset][model_name]
#             compare_vanilla_focusE_models(dataset, model_name, vanilla_params, focusE_params)
            folderList = [
                    {'causedByType':'CausedByTypeCausesCausedBy'},          
                    {'causesType':'CausesTypeCausesCausedBy'},             
                         ]                  
            count = 1
            for i in folderList: 
                if dataset == list(i.values())[0]:
                    print("Predicates in training: ",list(i.keys())[0])
                    callTheSplitFunction(list(i.keys())[0], list(i.values())[0], model_name, vanilla_params, vanilla_params)
                    print("\n")
                
saveResults.to_csv('../Results/CausalCLEVERERHumanKG_RandomSplit/Results_RandomSplit.csv', index=False)
