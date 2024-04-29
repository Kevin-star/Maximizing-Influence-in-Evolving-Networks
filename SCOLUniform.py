# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 09:35:22 2024

@author: hp
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 16:25:50 2023

@author: 28300
"""

# Reconnecting Top-l Relationships 
# Copyright (c) 2022 JadeRay. All Rights Reserved.
# Licensed under the Creative Commons BY-NC-ND 4.0 International License [see LICENSE for details]

"""
Reconnecting Top-l Relationships (RTlR) query
"""

import random
import json
import os.path as osp
from collections import defaultdict
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import sketch_based_greedy_RTlL, order_based_SBG_RTlL, build_upper_bound_label, calculate_candidate_edges, generate_user_groups
from utils import read_temporary_graph_data, read_graph_from_edgefile
from utils import draw_networkx, draw_evaluation
import time
import copy
import random
import heapq
from typing import List
from copy import deepcopy
from functools import reduce
import numpy as np
from tqdm import tqdm

import networkx as nx

def subgraph_update(graphs: List[nx.Graph], users: int):
    """Compute independent cascade in the graph

    Args:
        graph (nx.Graph): graph object
        users (List[int]): a set of user nodes
        mask (List[int], optional): a set of unparticipation cascade mask nodes. default is []. 

    Returns:
        spread, reached_node (int, List[int]): the number of vertexes reached by users in graph and the set of nodes reached by users.
    """
    global n_dic
    for j in range(0,R):
        visited = set()
        if(users in graphs[j]):
            visited.add(users)
            n_dic[j][users] = 0
            Queue1 = []
            Queue1.append(users)
            while(len(Queue1)>0):
                nodecur=Queue1.pop(0)
                neighbors=list(graphs[j].neighbors(nodecur))
                for neighbor in neighbors:
                    if neighbor not in visited:
                       n_dic[j][neighbor] = 0
                       visited.add(neighbor)
                       Queue1.append(neighbor)                       
    return

def Influence_compute_celf(graphs: List[nx.Graph], users: int):
    """The Forward Influence Sketch method

    Args:
        graphs (List[nx.Graph]): a set of snapshot graph
        users (List[int]): a group of user nodes
        reconneted_edge (tuple[int, int]): the reconneted edge

    Returns:
        spread (float): the mean of additional spread of vertexes reached by users in all sketch subgraph.
    """
    R=200
    global n_dic
    Spreadcompu=0
    
    for j in range(0,R):
        visited = set()
        if(users in graphs[j]):
            if(n_dic[j][users]):
                Spreadcompu = Spreadcompu+1
                visited.add(users)
                Queue1 = []
                Queue1.append(users)
                while(len(Queue1)>0):
                    nodecur=Queue1.pop(0)
                    neighbors=list(graphs[j].neighbors(nodecur))
                    for neighbor in neighbors:
                        if(n_dic[j][neighbor]):
                            if neighbor not in visited:
                                Spreadcompu = Spreadcompu+1
                                visited.add(neighbor)
                                Queue1.append(neighbor)                       
    return Spreadcompu/R

def compute_montesimu_spread(graph: nx.Graph, users: List[int], p):
# def compute_montesimu_spread(graph: nx.Graph, users: List[int]):
    """Compute independent cascade in the graph

    Args:
        graph (nx.Graph): graph object
        users (List[int]): a set of user nodes
        mask (List[int], optional): a set of unparticipation cascade mask nodes. default is []. 

    Returns:
        spread, reached_node (int, List[int]): the number of vertexes reached by users in graph and the set of nodes reached by users.
    """
    spread=[]
    popo =[0.1,0.01,0.001]
    for i in range(10000):
        
        new_active, active = users[:], users[:]
            
        # for each newly activated nodes, find its neighbors that becomes activated
        np.random.seed(i)
        while new_active:
            activated_nodes = []
            for node in new_active:               
                for nodenei in list(graph.neighbors(node)):  
                    # if np.random.uniform(0,1) < random.choice(popo):
                    # if np.random.uniform(0,1) < p:
                    if np.random.uniform(0,1) < 1/graph.degree(nodenei):
                       activated_nodes.append(nodenei)                                                 
            new_active = list(set(activated_nodes) - set(active) )
            active += new_active  
        spread.append(len(active))
    return np.mean(spread)

def generate_snapshots(graph: nx.Graph, r: int,  P, seed: int = 42,):
    """Generate r random sketch graph by removing each edges with probability 1-P(u,v), which defined as 1/degree(v). 

    Args:
        graph (nx.Graph): graph object
        r (int): the number of snapshots generated
        seed (int): the random seed of numpy

    Returns:
        snapshots (List[nx.Graph]): r number sketch subgraph
    """

    np.random.seed(seed)
    snapshots = []
    popo =[0.1,0.01,0.001]
    for _ in range(r):
        # select_edges = [edge for edge in graph.edges if np.random.uniform(0, 1) < P]
        # select_edges = [edge for edge in graph.edges if np.random.uniform(0, 1) < random.choice(popo)]
        select_edges = [edge for edge in graph.edges if np.random.uniform(0, 1) < 1/graph.degree(edge[1])]
        snapshots.append(graph.edge_subgraph(select_edges))

    return snapshots

def compute_independent_cascade(graph: nx.Graph, users: List[int], mask: List[int] = []):
    """Compute independent cascade in the graph

    Args:
        graph (nx.Graph): graph object
        users (List[int]): a set of user nodes
        mask (List[int], optional): a set of unparticipation cascade mask nodes. default is []. 

    Returns:
        spread, reached_node (int, List[int]): the number of vertexes reached by users in graph and the set of nodes reached by users.
    """
    
    new_active, active = users[:], users[:]
        
    # for each newly activated nodes, find its neighbors that becomes activated
    while new_active:
        activated_nodes = []
        for node in new_active:
            if graph.has_node(node):
                # determine neighbors that become infected
                neighbors = list(graph.neighbors(node))
                activated_nodes += neighbors
        
        # ensure the newly activated nodes doesn't already exist
        new_active = list(set(activated_nodes) - set(active) - set(mask))
        active += new_active

    return len(active), active

def forward_influence_sketch(graphs: List[nx.Graph], users: List[int]):
    """The Forward Influence Sketch method

    Args:
        graphs (List[nx.Graph]): a set of snapshot graph
        users (List[int]): a group of user nodes
        reconneted_edge (tuple[int, int]): the reconneted edge

    Returns:
        spread (float): the mean of additional spread of vertexes reached by users in all sketch subgraph.
    """
    
    spread = []
    for graph in graphs:
        original_spread = compute_independent_cascade(graph, users)[0]
        spread.append(original_spread)
    
    return np.mean(spread)


T = [20]
R=200
P=0.03
Seedsize = [10] 
# Seedsize = [10, 20, 30, 40, 50]    
# datasets = ['EmailEuCore','AskUbuntu','superuser','StackOverflowsub','Wikitalksub']
datasets = ['Wikitalksub']
for dataset in datasets:    
    predict_graph = read_graph_from_edgefile(f'data/SEALDataset/{dataset}/T20_pred_edge.pt')
    predict_graph_nodes = list(predict_graph.nodes)
    snapshots=generate_snapshots(predict_graph, 200, P)
    timelapse,totalspread=[],[]
    for SZ in Seedsize:
          start_time = time.time()  
          S=[]
# predict_graph = read_graph_from_edgefile(f'data/SEALDataset/EmailEuCore/T20_pred_edge.pt')
          n_dictmp={}
          n_dic=[]
          for j in range(0,R): 
                for node in snapshots[j].nodes:
                    n_dictmp[node]=1
                n_dic.append(deepcopy(n_dictmp))
        # upperbound=[]
        # for users in predict_graph_nodes:
        #     bound=forward_influence_sketch(snapshots, [users]) 
        #     upperbound.append(bound)
        
        
          upperbound=[forward_influence_sketch(snapshots, [users]) for users in predict_graph_nodes]
          Q = sorted(zip(predict_graph_nodes,upperbound), key=lambda x: x[1],reverse=True)
          S = [Q[0][0]]
          subgraph_update(snapshots,Q[0][0])
          Q = Q[1:]
          for _ in range(SZ-1):    
        
                check, node_lookup = False, 0
                
                while not check:
                                      
                    
                    # Recalculate spread of top node
                    current = Q[0][0]
                    
                    # Evaluate the spread function and store the marginal gain in the list
                    Q[0] = (current,Influence_compute_celf(snapshots,current))
        
                    # Re-sort the list
                    Q = sorted(Q, key = lambda x: x[1], reverse = True)
        
                    # Check if previous top node stayed on top after the sort
                    check = (Q[0][0] == current)
        
                # Select the next node
                subgraph_update(snapshots,Q[0][0])
                S.append(Q[0][0])        
        
                # Remove the selected node from the list
                Q = Q[1:]
          timelapse.append(time.time() - start_time)
          eachspread = compute_montesimu_spread(predict_graph,S,P)
          totalspread.append(eachspread)
          
    data={'Seedsize':Seedsize,'Runtime':timelapse,'Totalspread':totalspread}
    df = pd.DataFrame(data)    
    df.to_csv(f'SCELFOP{dataset}{P}.csv', index=False) 
# plt.figure(figsize=(20, 8), dpi=80)
# plt.plot(Seedsize,runtimes,'r-',label='runtime')
# plt.plot(Seedsize,totalspread,'g:',label='total spread')
# plt.savefig("./runtime.png")
# plt.show