# Load the packages
import os
import numpy as np
import pandas as pd
import numpy as np
import ampligraph #Added
from ampligraph.evaluation import train_test_split_no_unseen
from ampligraph.datasets import load_onet20k, load_ppi5k, load_nl27k, load_cn15k
from ampligraph.latent_features import ComplEx, TransE, DistMult, MODEL_REGISTRY
from ampligraph.utils import save_model, restore_model
from ampligraph.evaluation import evaluate_performance,mrr_score, hits_at_n_score, mr_score
from ampligraph.datasets import load_from_csv #Added
from sklearn.model_selection import train_test_split #Added
from ampligraph.discovery import discover_facts
from ampligraph.discovery import query_topn
import networkx as nx
import json
import pickle
from collections import ChainMap
import rdflib
from rdflib import Dataset
from rdflib import URIRef
from rdflib.namespace import Namespace, NamespaceManager
from rdflib.namespace import RDF, RDFS
from rdflib import Graph
import uuid

# for the dropdown menu
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
from IPython.display import HTML
from IPython.display import Video
from ipywidgets import Video

import matplotlib.pyplot as plt

from pyvis.network import Network
import networkx as nx
from pyvis import network as net


# from benepar.spacy_plugin import BeneparComponent
import spacy
import benepar
from nltk.tokenize import WhitespaceTokenizer
from pattern.text.en import singularize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet 
print("All the package imported \n")

# import open cv to brake the video into frames
import cv2

import random


# Divide the videos into training and testing based on the CEG graph depth
### Get the maximum depth of the Causal event graph from the root nodes 
### from the graphs with depth 2,3,4,5 (total = 364) randomly choose 177 (885*0.20 =177)) videos for the test set
### and the rest for the depth 0,1,2,3,4,5 goes to training set
### From the test set: divide the CEG nodes into training and testing 
 
 with open('../Data/CegWithNoCycles.txt', 'r') as f:
    cegWithNoCycles = json.load(f)

with open('../Data/VerbObject.json', 'r') as f:
        verbObject = json.load(f)

file = open('../Data/valid_ceg_data_May12.p', 'rb')  
validData = pickle.load(file)
file.close()

file = open('../Data/train_ceg_data_May12.p', 'rb')  
trainData = pickle.load(file)
file.close()

data = ChainMap(trainData, validData)

# Get the longest path in the DAG
depth0List = list()
depth1List = list()
depth2List = list()
depth3List = list()
depth4List = list()
depth5List = list()
depth6List = list()

for ids in cegWithNoCycles:
    depthSet = set()
    ceg_data = data[ids].get('CEG_full')
    edgeList = list()
# From the CEG with no cycles remove the edges with weights equal to 1
    for k in ceg_data.adj.keys():
        for nodes in ceg_data.adj[k]:
             if ceg_data.get_edge_data(k, nodes)['weight']==1:
                edgeList.append((k, nodes))
        ceg_data.remove_edges_from(edgeList)
# Get the root nodes in the CEG and get the depth of the CEG from the root nodes
# Save the max depth of the CEG
    rootList = list()
    [rootList.append(n) for n,d in ceg_data.in_degree() if d==0]
    for root in rootList:
        depthSet.add(len(list(nx.dfs_edges(ceg_data, source=root))))
    if len(list(depthSet))>0:
        if max(list(depthSet))==0:
            depth0List.append(ids)
        if max(list(depthSet))==1:
            depth1List.append(ids)
        if max(list(depthSet))==2:
            depth2List.append(ids)
        if max(list(depthSet))==3:
            depth3List.append(ids)
        if max(list(depthSet))==4:
            depth4List.append(ids)    
        if max(list(depthSet))==5:
            depth5List.append(ids)   
        if max(list(depthSet))==6:
            depth6List.append(ids)   

# Save the list of CEG ids with given Depth
with open('../Data/CausalCLEVERERHumanKG_MarkovSplit/CEGdepth0List.pkl', 'wb') as f:
    pickle.dump(depth0List, f)
with open('../Data/CausalCLEVERERHumanKG_MarkovSplit/CEGdepth1List.pkl', 'wb') as f:
    pickle.dump(depth1List, f)
with open('../Data/CausalCLEVERERHumanKG_MarkovSplit/CEGdepth2List.pkl', 'wb') as f:
    pickle.dump(depth2List, f)
with open('../Data/CausalCLEVERERHumanKG_MarkovSplit/CEGdepth3List.pkl', 'wb') as f:
    pickle.dump(depth3List, f)
with open('../Data/CausalCLEVERERHumanKG_MarkovSplit/CEGdepth4List.pkl', 'wb') as f:
    pickle.dump(depth4List, f)    
with open('../Data/CausalCLEVERERHumanKG_MarkovSplit/CEGdepth5List.pkl', 'wb') as f:
    pickle.dump(depth5List, f)    
with open('../Data/CausalCLEVERERHumanKG_MarkovSplit/CEGdepth6List.pkl', 'wb') as f:
    pickle.dump(depth6List, f) 

print("Total number of CEG videos: ", len(cegWithNoCycles))        
    
print("Number of CEG with depth 0: ",len(depth0List))
print("depth0List: ",depth0List)
print("Number of CEG with depth 1: ",len(depth1List))
print("depth1List: ",depth1List)
print("Number of CEG with depth 2: ",len(depth2List))
print("depth2List: ",depth2List)
print("Number of CEG with depth 3: ",len(depth3List))
print("depth3List: ",depth3List)
print("Number of CEG with depth 4: ",len(depth4List))
print("depth4List: ",depth4List)
print("Number of CEG with depth 5: ",len(depth5List))
print("depth5List: ",depth5List)    
print("Number of CEG with depth 6: ",len(depth6List))
print("depth6List: ",depth6List)


print("Videos with depth 0 and 1: ", len(depth0List)+len(depth1List))
print("Videos with depth 2,3,4,5,6: ", len(depth2List)+len(depth3List)+len(depth4List)+len(depth5List)+len(depth6List))

# # There are 891 CEG with no cycles 
# # len(depth0List) = 23
# # len(depth1List) = 101
# # len(depth2List) = 235
# # len(depth3List) = 295
# # len(depth4List) = 170
# # len(depth5List) = 54
# # len(depth6List) = 10

# # We do not consider the CEG until depth 1 as it includes the root node at level 0, level 1, and level 2 (leaf)
# # Videos with depth 0 and 1:  124
# # For the task of prediction we cut the CEG graph at level leaf - 1, and explanation at 
# # Combining depth2List+depth3List+depth4List+depth5List + depth6List = 764

# # 80, 10, 10 split of 764
# # Train(612), Test(76), Validation(76)  

# # 80, 20 split of 764
# # Train(611), Test(152)  

# depth = list() 
# trainCEG = list() 
# testCEG = list() 
# valCEG = list()


# # Test set contains CEG of depth 5 = 54, depth 6 = 10 => 64 and rest from sample depth 4 = 170-64 = 106

depth56 = depth6List
for i in (depth5List):
    depth56.append(i)

print(len(set(depth56)))
    
testtmpCEG = random.sample((depth56), k=64)
test = random.sample((depth4List), k=88)


for i in test:
    testtmpCEG.append(i)

# # depth = depth2List 
depth = [i for i in depth4List if i not in test]
for i in (depth3List):
    depth.append(i) 
for i in (depth2List):
    depth.append(i) 
for i in (depth5List):
    depth.append(i)     
for i in (depth6List):
    depth.append(i) 

depth = list(set(depth))

testCEG = testCEG(random.sample((depth), k=106))

depth = [i for i in depth if i not in testCEG]

for i in depth:
    trainCEG=trainCEG.append(i)

trainCEG = random.sample((depth), k=612)
depth = [i for i in depth if i not in trainCEG]

valCEG = random.sample((depth), k=76)
depth = [i for i in depth if i not in valCEG]

# Save the list of CEG ids with given Depth
with open('../Data/CausalCLEVERERHumanKG_MarkovSplit/trainVideoIdCEGList.pkl', 'wb') as f:
    pickle.dump(trainCEG, f)
with open('../Data/CausalCLEVERERHumanKG_MarkovSplit/testVideoIdCEGList.pkl', 'wb') as f:
    pickle.dump(testCEG, f)
# with open('../Data/valVideoIdCEGList.pkl', 'wb') as f:
#     pickle.dump(valCEG, f)


# Read the list of CEG ids with given Depth
file = open('../Data/CausalCLEVERERHumanKG_MarkovSplit/trainVideoIdCEGList.pkl', 'rb')  
trainCEG = pickle.load(file)
file.close()

file = open('../Data/CausalCLEVERERHumanKG_MarkovSplit/testVideoIdCEGList.pkl', 'rb')  
testCEG = pickle.load(file)
file.close()

# file = open('../Data/valVideoIdCEGList.pkl', 'rb')  
# valCEG = pickle.load(file)
# file.close()


## Get the KG for the videos in training, testing 
def createKG(depthNodeCEGList,filename):

    file=open(filename+"KG.txt","w")

    clevrerHumans = Namespace("http://semantic.bosch.com/CausalClevrerHumans/v00/")
    clevrerHumansData = Namespace("http://semantic.bosch.com/CausalClevrerHumans/v00/data/")
    clevrer = Namespace("http://semantic.bosch.com/CLEVRER/")
    causal = Namespace("http://semantic.bosch.com/causal/v00/")
    scene = Namespace("http://semantic.bosch.com/scene/v02/")
    ssn = Namespace("http://www.w3.org/ns/ssn/")

#     g = Graph()
    g = Dataset()
    namespace_manager = NamespaceManager(Graph())
    namespace_manager.bind('', clevrerHumansData, override=False)
    namespace_manager.bind("causal", causal, override=True)
    namespace_manager.bind("CCH", clevrerHumans, override=True)
    namespace_manager.bind("scene", scene, override=True)
    namespace_manager.bind("ssn", ssn, override=True)
    g.namespace_manager = namespace_manager

    # # Add for the Shape, color and material property to the graph
    # # Shape: Cube, Sphere, Ball, 
    # # Get unique shape, color, and material
    shapeSet = set()
    colorSet = set()
    materialSet = set()

    # # for ids in cegWithNoCycles:
    # # trainDepthCEGList, testDepthCEGList
    for ids in depthNodeCEGList:
        cegdata = data[ids].get('CEG_full')
        for k in cegdata.nodes(): 
            if 'Color' in verbObject[ids][k]['Object'][0].keys():
                colorSet.add(verbObject[ids][k]['Object'][0]['Color'])
            if 'Material' in verbObject[ids][k]['Object'][0].keys():    
                materialSet.add(verbObject[ids][k]['Object'][0]['Material'])
            if 'Shape' in verbObject[ids][k]['Object'][0].keys():    
                shapeSet.add(verbObject[ids][k]['Object'][0]['Shape'])

    shape = dict()
    color = dict()
    material = dict()
    for c in colorSet:
        colorUUID = URIRef(clevrerHumansData+str(uuid.uuid4()))   
        color[c] = colorUUID
        g.add((colorUUID, RDF.type, URIRef(clevrerHumans+"Color"))) 
        g.add((colorUUID, RDFS.label, rdflib.term.Literal(c))) 
        file.writelines([colorUUID,",", RDF.type,",", URIRef(clevrerHumans+"Color"),"\n"])
        file.writelines([colorUUID,",", RDFS.label,",", rdflib.term.Literal(c),"\n"])

    for s in shapeSet:
        shapeUUID = URIRef(clevrerHumansData+str(uuid.uuid4()))  
        shape[s] = shapeUUID
        g.add((shapeUUID, RDF.type, URIRef(clevrerHumans+"Shape"))) 
        g.add((shapeUUID, RDFS.label, rdflib.term.Literal(s))) 
        file.writelines([shapeUUID,",", RDF.type,",", URIRef(clevrerHumans+"Shape"),"\n"])
        file.writelines([shapeUUID,",", RDFS.label,",", rdflib.term.Literal(s),"\n"])

    for m in materialSet:
        materialUUID = URIRef(clevrerHumansData+str(uuid.uuid4()))   
        material[m] = materialUUID
        g.add((materialUUID, RDF.type, URIRef(clevrerHumans+"Material")))  
        g.add((materialUUID, RDFS.label, rdflib.term.Literal(m))) 
        file.writelines([materialUUID,",", RDF.type,",", URIRef(clevrerHumans+"Material"),"\n"])
        file.writelines([materialUUID,",", RDFS.label,",", rdflib.term.Literal(m),"\n"])

    objectParticipant = dict()
    for c in color:
        for s in shape:
            objectParticipantUUID = URIRef(clevrerHumansData+str(uuid.uuid4()))             
            objectParticipant[c+"_"+s] = objectParticipantUUID
            g.add(((objectParticipantUUID), RDF.type, scene.Object)) 
            g.add(((objectParticipantUUID), RDFS.label, rdflib.term.Literal(c+" "+s))) 
            file.writelines([objectParticipantUUID,",", RDF.type,",", scene.Object,"\n"])
            file.writelines([objectParticipantUUID,",", RDFS.label,",", rdflib.term.Literal(c+" "+s),"\n"])

            g.add(((objectParticipantUUID), ssn.hasProperty, color[c]))
            g.add(((objectParticipantUUID), ssn.hasProperty, shape[s]))

            file.writelines([objectParticipantUUID,",", ssn.hasProperty,",", color[c],"\n"])
            file.writelines([objectParticipantUUID,",", ssn.hasProperty,",", shape[s],"\n"])

    for c in color:
        for s in shape:
            for m in material:
                if c not in ['silver', 'gold']:
                    objectParticipantUUID = URIRef(clevrerHumansData+str(uuid.uuid4()))             
                    objectParticipant[c+"_"+m+"_"+s] = objectParticipantUUID
                    g.add(((objectParticipantUUID), RDF.type, scene.Object)) 
                    g.add(((objectParticipantUUID), RDFS.label, rdflib.term.Literal(c+" "+m+" "+s))) 
                    file.writelines([objectParticipantUUID,",", RDF.type,",", scene.Object,"\n"])
                    file.writelines([objectParticipantUUID,",", RDFS.label,",", rdflib.term.Literal(c+" "+m+" "+s),"\n"])

                    g.add(((objectParticipantUUID), ssn.hasProperty, color[c]))
                    g.add(((objectParticipantUUID), ssn.hasProperty, shape[s]))
                    g.add(((objectParticipantUUID), ssn.hasProperty, material[m]))

                    file.writelines([objectParticipantUUID,",", ssn.hasProperty,",", color[c],"\n"])
                    file.writelines([objectParticipantUUID,",", ssn.hasProperty,",", shape[s],"\n"])
                    file.writelines([objectParticipantUUID,",", ssn.hasProperty,",", material[m],"\n"])



    for ids in depthNodeCEGList:
        g.add((URIRef(clevrerHumansData+(ids)), RDF.type, scene.Scene))
        file.writelines([URIRef(clevrerHumansData+(ids)),",", RDF.type,",", scene.Scene,"\n"])
        
        ceg_data = data[ids]
        ceg_full=ceg_data.get('CEG_full')
        nodeDict = dict()
        for n in ceg_full.nodes():
            nodeDict[n] = str(uuid.uuid4())
        for k in ceg_full.adj.keys(): 
            cegSubjectNode = URIRef(str(k)) 
            subjectUUID = URIRef(str(nodeDict[k]))             
            for nodes in ceg_full.adj[k]:
                if ceg_full.get_edge_data(k, nodes)['weight'] > 1:
                    cegObjectNode =  URIRef(str(nodes)) #URIRef(str(uuid.uuid4()))
                    objectUUID =  URIRef(str(nodeDict[nodes]))

                    g.add((URIRef(clevrerHumansData+(ids)), scene.includes, URIRef(clevrerHumansData+subjectUUID))) 
                    g.add((URIRef(clevrerHumansData+subjectUUID), RDFS.label, rdflib.term.Literal(cegSubjectNode))) 
                    file.writelines([URIRef(clevrerHumansData+(ids)),",", scene.includes,",", URIRef(clevrerHumansData+subjectUUID),"\n"])
                    file.writelines([URIRef(clevrerHumansData+subjectUUID),",", RDFS.label,",", rdflib.term.Literal(cegSubjectNode),"\n"])

                    g.add((URIRef(clevrerHumansData+(ids)), scene.includes, URIRef(clevrerHumansData+objectUUID))) 
                    g.add((URIRef(clevrerHumansData+objectUUID), RDFS.label, rdflib.term.Literal(cegObjectNode))) 
                    file.writelines([URIRef(clevrerHumansData+(ids)),",", scene.includes,",", URIRef(clevrerHumansData+objectUUID),"\n"])
                    file.writelines([URIRef(clevrerHumansData+objectUUID),",", RDFS.label,",", rdflib.term.Literal(cegObjectNode),"\n"])

                    for objects in verbObject[ids][k]['Object']:
                        if objects:
                            if 'Material' in objects.keys():
    #                             print(ids,k,objects, objects.keys(),objects['Material'])
    #                             print(objects['Color'], objects['Material'], objects['Shape'])
                                uid = objectParticipant[objects['Color']+"_"+objects['Material']+"_"+objects['Shape']]

                                g.add((URIRef(clevrerHumansData+(ids)), scene.includes, uid))
                                file.writelines([URIRef(clevrerHumansData+(ids)),",", scene.includes,",", uid,"\n"])

                                g.add((URIRef(clevrerHumansData+subjectUUID), scene.hasParticipant, uid))
                                g.add((uid, scene.isParticipantIn, URIRef(clevrerHumansData+subjectUUID)))
                                file.writelines([URIRef(clevrerHumansData+subjectUUID),",", scene.hasParticipant,",", uid,"\n"])
                                file.writelines([uid,",", scene.isParticipantIn,",", URIRef(clevrerHumansData+subjectUUID),"\n"])

                            else:
                                uid = objectParticipant[objects['Color']+"_"+objects['Shape']]

                                g.add((URIRef(clevrerHumansData+(ids)), scene.includes, uid))
                                file.writelines([URIRef(clevrerHumansData+(ids)),",", scene.includes,",", uid,"\n"])

                                g.add((URIRef(clevrerHumansData+subjectUUID), scene.hasParticipant, uid))
                                g.add((uid, scene.isParticipantIn, URIRef(clevrerHumansData+subjectUUID)))
                                file.writelines([URIRef(clevrerHumansData+subjectUUID),",", scene.hasParticipant,",", uid,"\n"])
                                file.writelines([uid,",", scene.isParticipantIn,",", URIRef(clevrerHumansData+subjectUUID),"\n"])



                    for objects in verbObject[ids][nodes]['Object']:
                        if objects:
                            if 'Material' in objects.keys():
                                uid = objectParticipant[objects['Color']+"_"+objects['Material']+"_"+objects['Shape']]

                                g.add((URIRef(clevrerHumansData+(ids)), scene.includes, uid))
                                file.writelines([URIRef(clevrerHumansData+(ids)),",", scene.includes,",", uid,"\n"])

                                g.add((URIRef(clevrerHumansData+objectUUID), scene.hasParticipant, uid))
                                g.add((uid, scene.isParticipantIn, URIRef(clevrerHumansData+objectUUID)))
                                file.writelines([URIRef(clevrerHumansData+objectUUID),",", scene.hasParticipant,",", uid,"\n"])
                                file.writelines([uid,",", scene.isParticipantIn,",", URIRef(clevrerHumansData+objectUUID),"\n"])

                            else:
                                uid = objectParticipant[objects['Color']+"_"+objects['Shape']]

                                g.add((URIRef(clevrerHumansData+(ids)), scene.includes, uid))
                                file.writelines([URIRef(clevrerHumansData+(ids)),",", scene.includes,",", uid,"\n"])

                                g.add((URIRef(clevrerHumansData+objectUUID), scene.hasParticipant, uid))
                                g.add((uid, scene.isParticipantIn, URIRef(clevrerHumansData+objectUUID)))
                                file.writelines([URIRef(clevrerHumansData+objectUUID),",", scene.hasParticipant,",", uid,"\n"])
                                file.writelines([uid,",", scene.isParticipantIn,",", URIRef(clevrerHumansData+objectUUID),"\n"])


                    if verbObject[ids][nodes]['Verbs'][0] == 'come':
                        verbObject[ids][nodes]['Verbs'][0] = 'ComeFrom'

                    if verbObject[ids][nodes]['Verbs'][0] == 'change':
                        verbObject[ids][nodes]['Verbs'][0] = 'changeDirection'

                    if verbObject[ids][k]['Verbs'][0] == 'come':
                        verbObject[ids][k]['Verbs'][0] = 'ComeFrom'

                    if verbObject[ids][k]['Verbs'][0] == 'change':
                        verbObject[ids][k]['Verbs'][0] = 'changeDirection'
                    
                    # , rdflib.term.Literal(ceg_full.get_edge_data(k, nodes)['weight'])
                    g.add((URIRef(clevrerHumansData+subjectUUID), RDF.type, URIRef(clevrerHumans+verbObject[ids][k]['Verbs'][0].capitalize())))
                    g.add((URIRef(clevrerHumansData+subjectUUID), causal.causesType, URIRef(clevrerHumans+verbObject[ids][nodes]['Verbs'][0].capitalize())))
                    file.writelines([URIRef(clevrerHumansData+subjectUUID),",", RDF.type,",", URIRef(clevrerHumans+verbObject[ids][k]['Verbs'][0].capitalize()),"\n"])
                    file.writelines([URIRef(clevrerHumansData+subjectUUID),",", causal.causesType,",", URIRef(clevrerHumans+verbObject[ids][nodes]['Verbs'][0].capitalize()),",",str(ceg_full.get_edge_data(k, nodes)['weight']),"\n"])
                   
                    g.add((URIRef(clevrerHumansData+objectUUID), RDF.type, URIRef(clevrerHumans+verbObject[ids][nodes]['Verbs'][0].capitalize())))
                    g.add((URIRef(clevrerHumansData+objectUUID), causal.causedByType, URIRef(clevrerHumans+verbObject[ids][k]['Verbs'][0].capitalize())))
                    file.writelines([URIRef(clevrerHumansData+objectUUID),",", RDF.type,",", URIRef(clevrerHumans+verbObject[ids][nodes]['Verbs'][0].capitalize()),"\n"])
                    file.writelines([URIRef(clevrerHumansData+objectUUID),",", causal.causedByType,",", URIRef(clevrerHumans+verbObject[ids][k]['Verbs'][0].capitalize()),",",str(ceg_full.get_edge_data(k, nodes)['weight']),"\n"])

                    g.add((URIRef(clevrerHumansData+subjectUUID), causal.causes, URIRef(clevrerHumansData+objectUUID)))
                    file.writelines([URIRef(clevrerHumansData+subjectUUID),",", causal.causes,",", URIRef(clevrerHumansData+objectUUID),",",str(ceg_full.get_edge_data(k, nodes)['weight']),"\n"])

                    g.add((URIRef(clevrerHumansData+objectUUID), causal.causedBy, URIRef(clevrerHumansData+subjectUUID)))
                    file.writelines([URIRef(clevrerHumansData+objectUUID),",", causal.causedBy,",", URIRef(clevrerHumansData+subjectUUID),",",str(ceg_full.get_edge_data(k, nodes)['weight']),"\n"])


#     g.serialize(filename+"NQuadKG.ttl", format='nquads')
    g.serialize(filename+"KG.ttl", format='n3')                         
    file.close()

# Create a KG for videos in the training and testing
# createKG(depthNodeCEGList,filename)
createKG(trainCEG,"../Data/CausalCLEVERERHumanKG_MarkovSplit/Prediction/train")
createKG(testCEG,"../Data/CausalCLEVERERHumanKG_MarkovSplit/Prediction/test")


# Prediction
### From the test CEG videos:
#### Cut each CEG into further training and testing 
#### Prediction: cut after root level+1, move the causal triples to test set (causes, causesType), ask the questions for nodes
###### at level root+1, use predicate causesType

# Gather the node labels after level root+1 from test set CEGs
# Save it as a dictionary with scene as the key and node labels as value

# Get the root of the network
# The root node will have zero in-degree
testDictNodes = dict()
trainDictNodes = dict()

file = open('../Data/testVideoIdCEGList.pkl', 'rb')  
testDepthCEGList = pickle.load(file)
file.close()


for ids  in testDepthCEGList:
    trainSetNodes = set()
    testSetNodes = set()
    dfs = list()

    ceg_pos = data[ids].get('CEG_full')
    # ceg_pos.adj.keys()

    threshold = 1
    # filter out all edges above threshold and grab id's
    long_edges = list(filter(lambda e: e[2] == threshold, (e for e in ceg_pos.edges.data('weight'))))
    le_ids = list(e[:2] for e in long_edges)

    # remove filtered edges from graph G
    ceg_pos.remove_edges_from(le_ids)

    long_edges = list(filter(lambda e: e[2] == threshold, (e for e in ceg_pos.edges.data('width'))))
    le_ids = list(e[:2] for e in long_edges)
    # remove filtered edges from graph G
    ceg_pos.remove_edges_from(le_ids)

    rootList = list()
    [rootList.append(n) for n,d in ceg_pos.in_degree() if d==0]

    for root in rootList:
        dfs.append(list(nx.dfs_edges(ceg_pos, source=root)))
        trainSetNodes.add(root)

    dfs=sorted(dfs, key=len, reverse=True)
#     print(dfs)
#     print("\n")
    for i in list(dfs):
        if len(i)>1:
            for j in (i[0:1]):
                trainSetNodes.add(j[0])
                trainSetNodes.add(j[1])
            for j in (i[1:2]):
                if (j[0] not in trainSetNodes):
                    trainSetNodes.add(j[0])
                    trainSetNodes.add(j[1])
                if (j[0] in trainSetNodes):
                    testSetNodes.add(j[1])
            for j in (i[3:len(i)]):
                if (j[0] not in trainSetNodes): 
                    testSetNodes.add(j[0])
                    testSetNodes.add(j[1])
                else:
                    testSetNodes.add(j[1])
                    
    testDictNodes[ids]=testSetNodes
    trainDictNodes[ids]=trainSetNodes


def createKGForSpecificNodes(nodeDict,KGFile, filename):

    # Get the UUID from the labels using the 
    # testList = list()
    testList = set()   

    # Read the testGraphFile and use the weight column

    causalTestGraph = Graph()    
    causalTestGraphFile = open(filename+".txt","w")    
    # For a given CEG graph id video   
    # scene UUID is a scene
        # scene UUID includes Object uuid
        # Object uuid label object label

    # Read the testKG
    testKG = Graph()
    testKG.parse("../Data/CausalCLEVERERHumanKG_MarkovSplit/Prediction/testKG.ttl")
    
    causal = Namespace("http://semantic.bosch.com/causal/v00/")
    scene = Namespace("http://semantic.bosch.com/scene/v02/")
    clevrerHumansData = Namespace("http://semantic.bosch.com/CausalClevrerHumans/v00/data/")
    ssn = Namespace("http://www.w3.org/ns/ssn/")
    
    causalPredicate = [causal.causedByType,causal.causes,causal.causedBy,causal.causesType, RDF.type, ssn.hasProperty, scene.hasParticipant]

    for i in nodeDict:
        for j in nodeDict[i]:
    #         print(i,j)
            for sub in testKG.subjects(object=scene.Scene,predicate=RDF.type):
                if sub == URIRef(clevrerHumansData+str(i)):
                    for pred, obj in testKG.predicate_objects(subject=sub):
                        if pred == scene.includes:
                            for pred2, label in testKG.predicate_objects(subject=obj):
                                if (label == rdflib.term.Literal(j)) and (pred2==RDFS.label):
                                    testList.add(tuple([i,j,obj]))
    #                                 print("Label:",label, obj)
                            # Create a new KG with uuid causes, causedByType, causesType
                            # get the predicat
                            # causalTestGraph.add((obj))
                                    for pred3,obj2 in testKG.predicate_objects(subject=obj):
                                        if pred3 in causalPredicate:
                                            causalTestGraph.add((obj,pred3,obj2))
    #                                         print(obj,pred3,obj2)
                                            causalTestGraphFile.writelines([str(obj),",", str(pred3),",", str(obj2),"\n"])
                                        if (pred3==scene.hasParticipant):
                                            for pred4, obj3 in testKG.predicate_objects(subject=obj2):
                                                if (pred4==ssn.hasProperty):
                                                    causalTestGraph.add((obj2,pred4,obj3))
        #                                         print(obj,pred3,obj2)
                                                    causalTestGraphFile.writelines([str(obj2),",", str(pred4),",", str(obj3),"\n"])

                
                                            

    # causalTestGraph.serialize("causalTestGraph.txt", format="n3")
    causalTestGraph.serialize(filename+".ttl", format="n3")
    causalTestGraph.close()
    causalTestGraphFile.close()
    
    with open('../Data/CausalCLEVERERHumanKG_MarkovSplit/Prediction/videoNodeList'+KGFile+'.pkl', 'wb') as f:
        pickle.dump(testList, f)
        
    with open('../Data/CausalCLEVERERHumanKG_MarkovSplit/Prediction/videoNodeList'+KGFile+'.json', 'wb') as f:
        json.dump(testList, f)     

# Create KG for nodes in training part of CEG
# Create KG for nodes in testing part of CEG
createKGForSpecificNodes(testDictNodes,"test","../Data/CausalCLEVERERHumanKG_MarkovSplit/Prediction/causalTestNodeGraphPrediction",)
createKGForSpecificNodes(trainDictNodes,"train","../Data/CausalCLEVERERHumanKG_MarkovSplit/Prediction/causalTrainNodeGraphPrediction",)


# Average the weights for causal predicates in the KG:
def averageCausalWeight(kgFile):
    # Average the causal weights in trainKG as well and
    trainGraphFile = pd.read_csv(kgFile, names=['s','p','o','weight'])
    print(trainGraphFile.shape)
    trainGraphFile=trainGraphFile.groupby(['s','p','o'])['weight'].mean()
    traindf = pd.DataFrame(columns=['s','p','o','weight'])
    for i, v in trainGraphFile.items():
    #     print(i[0],i[1],i[2],v)
        traindf = traindf.append({'s':i[0],'p':i[1],'o':i[2],'weight':(v)}, ignore_index=True)
    traindf.to_csv(kgFile, header=None, index=False)

averageCausalWeight('../Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/testKG.txt')
averageCausalWeight('../Data/CausalCLEVERERHumanKG_MarkovSplit/Prediction/trainKG.txt')

##   Add average causal weight to causalTestGraphFile when subject is same as testKG
def addAveragedWeightKG(filename):
    tl = pd.DataFrame()
    l = list()
    df = pd.read_csv(filename, names=['s','p','o','weight'])
    tmpdf = pd.read_csv("../Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Prediction/testKG.txt", names=['s','p','o','weight'])
    for i in df['s'].unique():
        if i in tmpdf['s'].unique():
            l.append(i)
            tl= tl.append(tmpdf[tmpdf['s']==i])

#     print(len(l))
#     print(l)
    tl.to_csv(filename, header=None, index=False)
    
addAveragedWeightKG('../Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Prediction/causalTestNodeGraphPrediction.txt')
addAveragedWeightKG('../Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Prediction/causalTrainNodeGraphPrediction.txt')

causalTestNodeGraphFile = pd.read_csv('../Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Prediction/causalTestNodeGraphPrediction.txt', names=['s','p','o','weight'])
causalTrainNodeGraphFile = pd.read_csv('../Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Prediction/causalTrainNodeGraphPrediction.txt', names=['s','p','o','weight'])

with open('./Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Prediction/videoNodeListtrain.pkl', 'rb') as f:
    trainList = pickle.load(f)
    
with open('./Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Prediction/videoNodeListtest.pkl', 'rb') as f:
    testList = pickle.load(f)    
    
nodesFromTrainList = list()
for i in trainList:
    nodesFromTrainList.append(str(i[2]))
nodesFromTestList = list()
for i in testList:
    nodesFromTestList.append(str(i[2]))

testSplitCauseType = (causalTestNodeGraphFile[(causalTestNodeGraphFile['p'].isin(causesType))])
trainSplitCauseType = (causalTrainNodeGraphFile[(causalTrainNodeGraphFile['p'].isin(causesType))])

## Remove the causal edges from the training nodes to the testing nodes in the CEG
newTestdfCauses= pd.DataFrame(columns=['s','p','o','weight'])
newTestdfCausesType= pd.DataFrame(columns=['s','p','o','weight'])

newTestdfTestCausedByTrain= pd.DataFrame(columns=['s','p','o','weight'])
newTestdfTrainCausedByTest= pd.DataFrame(columns=['s','p','o','weight'])


causes = ['http://semantic.bosch.com/causal/v00/causes']
causedBy= ['http://semantic.bosch.com/causal/v00/causedBy']
causedByType= ['http://semantic.bosch.com/causal/v00/causedByType']
causesType= ['http://semantic.bosch.com/causal/v00/causesType']

# remove train causes test, train causesType
for i in nodesFromTrainList: 
    testdf = (causalTrainNodeGraphFile[(causalTrainNodeGraphFile['s']==str(i))&(causalTrainNodeGraphFile['p'].isin(causes))&(causalTrainNodeGraphFile['o'].isin(nodesFromTestList))])
    newTestdfCauses = newTestdfCauses.append(testdf)
    testdf = (causalTrainNodeGraphFile[(causalTrainNodeGraphFile['s']==str(i))&(causalTrainNodeGraphFile['p'].isin(causesType))])
    newTestdfCausesType = newTestdfCausesType.append(testdf)
    
# remove test causedBy train if the testList nodes causedBy nodes in the training set
for i in nodesFromTestList:
    testdf = (causalTestNodeGraphFile[(causalTestNodeGraphFile['s']==str(i))&(causalTestNodeGraphFile['p'].isin(causedBy))&(causalTestNodeGraphFile['o'].isin(nodesFromTrainList))])
    newTestdfTestCausedByTrain = newTestdfTestCausedByTrain.append(testdf)
    
# remove train causedBy test     
for i in nodesFromTrainList:
    testdf = (causalTrainNodeGraphFile[(causalTrainNodeGraphFile['s']==str(i))&(causalTrainNodeGraphFile['p'].isin(causedBy))&(causalTrainNodeGraphFile['o'].isin(nodesFromTestList))])
    newTestdfTrainCausedByTest = newTestdfTrainCausedByTest.append(testdf)    

# Triples to add to the test set
addToTest= pd.DataFrame(columns=['s','p','o','weight'])
addToTest = newTestdfCauses
addToTest = addToTest.append(newTestdfCausesType)
    
addToTest = addToTest.drop_duplicates()

dfTrain = pd.merge(causalTrainNodeGraphFile, addToTest,on=['s','p','o','weight'], how='outer', indicator=True).query("_merge == 'left_only'").drop('_merge', axis=1).reset_index(drop=True)
# Remove the new test triples from training CEG
dfTest = pd.merge(causalTestNodeGraphFile, addToTest,on=['s','p','o','weight'], how='outer', indicator=True).query("_merge == 'left_only'").drop('_merge', axis=1).reset_index(drop=True)

test = pd.read_csv("../Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Prediction/testKG.txt",names=['s','p','o','weight'])
test = pd.merge(test, addToTest,on=['s','p','o','weight'], how='outer', indicator=True).query("_merge == 'left_only'").drop('_merge', axis=1).reset_index(drop=True)

addToTest['weight'] = addToTest['weight'].fillna(0.0)
addToTest['weight'] = addToTest['weight'].astype(np.str)
addToTest.to_csv("../Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Prediction/CausesTypeCausesCausedBy/test.txt", index=False, header=None,sep="\t")

test = test.append(dfTrain, ignore_index=True)

test = test.append(dfTest)

train = pd.read_csv('../Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Prediction/trainKG.txt', names=['s','p','o','weight'])

train = train.append(test)
train['weight'] = train['weight'].fillna(0.0)
train.drop_duplicates()
train['weight'] = train['weight'].astype(np.str)

CausesTypeCausesCausedBy =['http://semantic.bosch.com/causal/v00/causesType','http://semantic.bosch.com/causal/v00/causes','http://semantic.bosch.com/causal/v00/causedBy']
train = train[train['p'].isin(CausesType)]

train.to_csv("../Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Prediction/CausesTypeCausesCausedBy/train.txt", index=False, header=None,sep="\t")

t= pd.read_csv("../Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Prediction/CausesTypeCausesCausedBy/train.txt", names=['s','p','o','weight'],sep="\t")

def filterPredicates(folder, filterPredicates):
    train = pd.read_csv("../Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Prediction/predictionTrainFullGraph.txt",names=['h','r','t','tce'],header=None, sep=",")
    test = pd.read_csv("../Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Prediction/predictionTestFullGraph.txt",names=['h','r','t','tce'],header=None, sep=",")
    valid = pd.read_csv("../Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Prediction/validKG.txt",names=['h','r','t','tce'],header=None, sep=",")
    
    # filter the predicates 
    train = train.loc[train['r'].isin(filterPredicates)]
    test = test.loc[test['r'].isin(filterPredicates)]
    valid = valid.loc[valid['r'].isin(filterPredicates)]
    
    train.to_csv('../Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Prediction/'+folder[0]+'/'+'train.txt',columns=None, header=False, index=False) 
    test.to_csv('../Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Prediction/'+folder[0]+'/'+'test.txt',columns=None, header=False, index=False) 
    valid.to_csv('../Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Prediction/'+folder[0]+'/'+'valid.txt',columns=None, header=False, index=False) 
    return

filterTestFlag=[
        {'CausesTypeCausesCausedBy':['http://semantic.bosch.com/causal/v00/causesType','http://semantic.bosch.com/causal/v00/causes','http://semantic.bosch.com/causal/v00/causedBy']},             
]

for i in filterTestFlag:
    print("Test KG contains predicates:",i)
    filterPredicates(list(i.keys()), list(i.values())[0])
    print("\n")



# Explanation
### From the test CEG videos:
#### Cut each CEG into further training and testing 
#### explanation: cut after root level+1, move the causal triples to train set (causes, causesType), ask the questions for nodes
###### at level root+1, use predicate causedByType

# Create a KG for videos in the training and testing
createKG(trainCEG,"../Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Explanation/train")
createKG(testCEG,"../Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Explanation/test")

# Gather the node labels after level root+1 from test set CEGs
# Save it as a dictionary with scene as the key and node labels as value

# Get the root of the network
# The root node will have zero in-degree
testDictNodes = dict()
trainDictNodes = dict()

file = open('../Data/testVideoIdCEGList.pkl', 'rb')  
testDepthCEGList = pickle.load(file)
file.close()


for ids  in testDepthCEGList:
    trainSetNodes = set()
    testSetNodes = set()
    dfs = list()

    ceg_pos = data[ids].get('CEG_full')
    # ceg_pos.adj.keys()

    threshold = 1
    # filter out all edges above threshold and grab id's
    long_edges = list(filter(lambda e: e[2] == threshold, (e for e in ceg_pos.edges.data('weight'))))
    le_ids = list(e[:2] for e in long_edges)

    # remove filtered edges from graph G
    ceg_pos.remove_edges_from(le_ids)

    long_edges = list(filter(lambda e: e[2] == threshold, (e for e in ceg_pos.edges.data('width'))))
    le_ids = list(e[:2] for e in long_edges)
    # remove filtered edges from graph G
    ceg_pos.remove_edges_from(le_ids)

    rootList = list()
    [rootList.append(n) for n,d in ceg_pos.in_degree() if d==0]

    for root in rootList:
        dfs.append(list(nx.dfs_edges(ceg_pos, source=root)))
        testSetNodes.add(root)

    dfs=sorted(dfs, key=len, reverse=True)
#     print(dfs)
#     print("\n")
    for i in list(dfs):
        if len(i)>1:
            for j in (i[0:1]):
                testSetNodes.add(j[0])
                testSetNodes.add(j[1])
            for j in (i[1:2]):
                if (j[0] not in testSetNodes):
                    testSetNodes.add(j[0])
                    testSetNodes.add(j[1])
                if (j[0] in testSetNodes):
                    trainSetNodes.add(j[1])
            for j in (i[3:len(i)]):
                if (j[0] not in testSetNodes): 
                    trainSetNodes.add(j[0])
                    trainSetNodes.add(j[1])
                else:
                    trainSetNodes.add(j[1])
                    
    testDictNodes[ids]=testSetNodes
    trainDictNodes[ids]=trainSetNodes

def createKGForSpecificNodes(nodeDict,KGFile, filename):

    # Get the UUID from the labels using the 
    # testList = list()
    testList = set()   

    causalTestGraph = Graph()    
    causalTestGraphFile = open(filename+".txt","w")    
    # For a given CEG graph id video   
    # scene UUID is a scene
        # scene UUID includes Object uuid
        # Object uuid label object label

    # Read the testKG
    testKG = Graph()
    testKG.parse("../Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Explanation/testKG.ttl")
    
    causal = Namespace("http://semantic.bosch.com/causal/v00/")
    scene = Namespace("http://semantic.bosch.com/scene/v02/")
    clevrerHumansData = Namespace("http://semantic.bosch.com/CausalClevrerHumans/v00/data/")
    ssn = Namespace("http://www.w3.org/ns/ssn/")
    
    causalPredicate = [causal.causedByType,causal.causes,causal.causedBy,causal.causesType, RDF.type, ssn.hasProperty, scene.hasParticipant]

    for i in nodeDict:
        for j in nodeDict[i]:
    #         print(i,j)
            for sub in testKG.subjects(object=scene.Scene,predicate=RDF.type):
                if sub == URIRef(clevrerHumansData+str(i)):
                    for pred, obj in testKG.predicate_objects(subject=sub):
                        if pred == scene.includes:
                            for pred2, label in testKG.predicate_objects(subject=obj):
                                if (label == rdflib.term.Literal(j)) and (pred2==RDFS.label):
                                    testList.add(tuple([i,j,obj]))
    #                                 print("Label:",label, obj)
                            # Create a new KG with uuid causes, causedByType, causesType
                            # get the predicat
                            # causalTestGraph.add((obj))
                                    for pred3,obj2 in testKG.predicate_objects(subject=obj):
                                        if pred3 in causalPredicate:
                                            causalTestGraph.add((obj,pred3,obj2))
    #                                         print(obj,pred3,obj2)
                                            causalTestGraphFile.writelines([str(obj),",", str(pred3),",", str(obj2),".","\n"])
                                        if (pred3==scene.hasParticipant):
                                            for pred4, obj3 in testKG.predicate_objects(subject=obj2):
                                                if (pred4==ssn.hasProperty):
                                                    causalTestGraph.add((obj2,pred4,obj3))
        #                                         print(obj,pred3,obj2)
                                                    causalTestGraphFile.writelines([str(obj2),",", str(pred4),",", str(obj3),".","\n"])

                
                                            

    # causalTestGraph.serialize("causalTestGraph.txt", format="n3")
    causalTestGraph.serialize(filename+".ttl", format="n3")
    causalTestGraph.close()
    causalTestGraphFile.close()
    
    with open('../Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Explanation/videoNodeList'+KGFile+'.pkl', 'wb') as f:
        pickle.dump(testList, f)

# Create KG for nodes in training part of CEG
# Create KG for nodes in testing part of CEG
createKGForSpecificNodes(testDictNodes,"test","../Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Explanation/causalTestNodeGraphPrediction")
createKGForSpecificNodes(trainDictNodes,"train","../Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Explanation/causalTrainNodeGraphPrediction")

# Average the weights for causal predicates in the KG:
def averageCausalWeight(kgFile):
    # Average the causal weights in trainKG as well and
    trainGraphFile = pd.read_csv(kgFile, names=['s','p','o','weight'])
    print(trainGraphFile.shape)
    trainGraphFile=trainGraphFile.groupby(['s','p','o'])['weight'].mean()
    traindf = pd.DataFrame(columns=['s','p','o','weight'])
    for i, v in trainGraphFile.items():
    #     print(i[0],i[1],i[2],v)
        traindf = traindf.append({'s':i[0],'p':i[1],'o':i[2],'weight':v}, ignore_index=True)
    traindf.to_csv(kgFile, header=None, index=False)

averageCausalWeight('../Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Explanation/testKG.txt')
averageCausalWeight('../Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Explanation/trainKG.txt')

##   Add average causal weight to causalTestGraphFile when subject is same as testKG
def addAveragedWeightKG(filename):
    tl = pd.DataFrame()
    l = list()
    df = pd.read_csv(filename, names=['s','p','o','weight'])
    tmpdf = pd.read_csv("./Data/Explanation/testKG.txt", names=['s','p','o','weight'])
    for i in df['s'].unique():
        if i in tmpdf['s'].unique():
            l.append(i)
            tl= tl.append(tmpdf[tmpdf['s']==i])

#     print(len(l))
#     print(l)
    tl.to_csv(filename, header=None, index=False)
    
addAveragedWeightKG('../Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Explanation/causalTestNodeGraphPrediction.txt')
addAveragedWeightKG('../Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Explanation/causalTrainNodeGraphPrediction.txt')

causalTestNodeGraphFile = pd.read_csv('../Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Explanation/causalTestNodeGraphPrediction.txt', names=['s','p','o','weight'])
causalTrainNodeGraphFile = pd.read_csv('../Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Explanation/causalTrainNodeGraphPrediction.txt', names=['s','p','o','weight'])


with open('../Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Explanation/videoNodeListtrain.pkl', 'rb') as f:
    trainList = pickle.load(f)
    
with open('../Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Explanation/videoNodeListtest.pkl', 'rb') as f:
    testList = pickle.load(f)    
    
nodesFromTrainList = list()
for i in trainList:
    nodesFromTrainList.append(str(i[2]))

nodesFromTestList = list()
for i in testList:
    nodesFromTestList.append(str(i[2]))

## Remove the causal edges from the training nodes to the testing nodes in the CEG
newTestdfCauses= pd.DataFrame(columns=['s','p','o','weight'])
newTestdfCausedByType= pd.DataFrame(columns=['s','p','o','weight'])
# newTestdfCausesType= pd.DataFrame(columns=['s','p','o','weight'])


newTestdfTestCausedByTrain= pd.DataFrame(columns=['s','p','o','weight'])
newTestdfTrainCausedByTest= pd.DataFrame(columns=['s','p','o','weight'])


causes = ['http://semantic.bosch.com/causal/v00/causes']
causedBy= ['http://semantic.bosch.com/causal/v00/causedBy']
causedByType= ['http://semantic.bosch.com/causal/v00/causedByType']
# causesType= ['http://semantic.bosch.com/causal/v00/causesType']

# remove train causes test, train causesType
for i in nodesFromTrainList: 
    testdf = (causalTrainNodeGraphFile[(causalTrainNodeGraphFile['s']==str(i))&(causalTrainNodeGraphFile['p'].isin(causes))&(causalTrainNodeGraphFile['o'].isin(nodesFromTestList))])
    newTestdfCauses = newTestdfCauses.append(testdf)
    
    testdf = (causalTrainNodeGraphFile[(causalTrainNodeGraphFile['s']==str(i))&(causalTrainNodeGraphFile['p'].isin(causedByType))])
    newTestdfCausedByType = newTestdfCausedByType.append(testdf)

    
# remove test causedBy train if the testList nodes causedBy nodes in the training set
for i in nodesFromTestList:
#     testdf = (causalTrainNodeGraphFile[(causalTrainNodeGraphFile['s'].isin(nodesFromTestList))&(causalTrainNodeGraphFile['p'].isin(causalPredicate1))&(causalTrainNodeGraphFile['o']==str(i[2]))])
    testdf = (causalTestNodeGraphFile[(causalTestNodeGraphFile['s']==str(i))&(causalTestNodeGraphFile['p'].isin(causedBy))&(causalTestNodeGraphFile['o'].isin(nodesFromTrainList))])
    newTestdfTestCausedByTrain = newTestdfTestCausedByTrain.append(testdf)
    
#     testdf = (causalTestNodeGraphFile[(causalTestNodeGraphFile['s']==str(i))&(causalTrainNodeGraphFile['p'].isin(causedByType))])
#     newTestdfCausedByType = newTestdfCausedByType.append(testdf)
    
# remove train causedBy test     
for i in nodesFromTrainList:
#     testdf = (causalTrainNodeGraphFile[(causalTrainNodeGraphFile['s'].isin(nodesFromTestList))&(causalTrainNodeGraphFile['p'].isin(causalPredicate1))&(causalTrainNodeGraphFile['o']==str(i[2]))])
    testdf = (causalTrainNodeGraphFile[(causalTrainNodeGraphFile['s']==str(i))&(causalTrainNodeGraphFile['p'].isin(causedBy))&(causalTrainNodeGraphFile['o'].isin(nodesFromTestList))])
    newTestdfTrainCausedByTest = newTestdfTrainCausedByTest.append(testdf)    


# Triples to add to the test set
addToTest= pd.DataFrame(columns=['s','p','o','weight'])
addToTest = newTestdfCauses
addToTest = addToTest.append(newTestdfCausedByType)
addToTest = addToTest.append(newTestdfTestCausedByTrain)
addToTest = addToTest.append(newTestdfTrainCausedByTest)

addToTest = addToTest.drop_duplicates()
dfTrain = pd.merge(causalTrainNodeGraphFile, addToTest,on=['s','p','o','weight'], how='outer', indicator=True).query("_merge == 'left_only'").drop('_merge', axis=1).reset_index(drop=True)
# Remove the new test triples from training CEG
dfTest = pd.merge(causalTestNodeGraphFile, addToTest,on=['s','p','o','weight'], how='outer', indicator=True).query("_merge == 'left_only'").drop('_merge', axis=1).reset_index(drop=True)

test = pd.read_csv("../Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Explanation/testKG.txt",names=['s','p','o','weight'])
test = pd.merge(test, addToTest,on=['s','p','o','weight'], how='outer', indicator=True).query("_merge == 'left_only'").drop('_merge', axis=1).reset_index(drop=True)

addToTest['weight'] = addToTest['weight'].fillna(0.0)
addToTest['weight'] = addToTest['weight'].astype(np.str)
addToTest.to_csv("./Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Explanation/CausedByTypeCausesCausedBy/test.txt", index=False, header=None,sep="\t")
test = test.append(dfTrain, ignore_index=True)
test = test.append(dfTest)

train = pd.read_csv('./Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Explanation/trainKG.txt', names=['s','p','o','weight'])
train = train.append(test)
train['weight'] = train['weight'].fillna(0.0)
train.drop_duplicates()
train['weight'] = train['weight'].astype(np.str)

CausedByTypeCausesCausedBy=['http://semantic.bosch.com/causal/v00/causedByType','http://semantic.bosch.com/causal/v00/causes','http://semantic.bosch.com/causal/v00/causedBy']             

train = train[train['p'].isin(CausedByTypeCausesCausedBy)]

train.to_csv("./Data/CausalCLEVERERHumanKG_MarkovSplitPrediction/Explanation/CausedByTypeCausesCausedBy/train.txt", index=False, header=None,sep="\t")

