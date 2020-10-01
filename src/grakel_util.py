#!/usr/bin/env python3

from __future__ import print_function
print(__doc__)

import numpy as np
from tqdm import tqdm
import os

import igraph as ig

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from grakel.datasets import fetch_dataset
from grakel.kernels import ShortestPath, WeisfeilerLehman, VertexHistogram
import grakel

def get_edge_list(graph, directed=False):
    """Returns the tuple edge dict of graph, where each edge is
    represented as (a,b) and (b,a) when the graph is undirected """
    if directed != False:
        print("Doesn't handle directed graphs yet!")
    
    edges_1 = graph.get_edgelist()
    edges_2 = [(b,a) for (a,b) in edges_1]
    edges = edges_1 + edges_2

    return(set(edges))


def get_node_label_dict(graph, attr_type="label"):
    """Returns a dict with node ids as keys and node attr as values """
    d = {}   
    
    if len(graph.vs.attributes()) > 0:
        if len(attr_type) > 0:
            node_index = graph.vs.indices
            node_labels = [int(i) for i in graph.vs[attr_type]]
            d = dict(zip(node_index, node_labels))
    
    return(d)


def get_edge_label_dict(graph, attr_type="label"):
    """ Return a dict with edge tuple as key and edge attr/label as value"""
    d = {}

    if len(graph.es.attributes()) > 0:
        if len(attr_type) > 0:
            if attr_type == "attribute":
                if "attribute" not in graph.es.attributes():
                    attr_type == "weight"
            for e in graph.es:
                d[(e.source, e.target)] = int(e[attr_type])
                d[(e.target, e.source)] = int(e[attr_type])

    return(d)


def create_grakel_graph(graph, attr):
    """ Create the list of the grakel components"""
    edge_attr = attr["edge"]
    node_attr = attr["vertex"]

    edges = get_edge_list(graph)
    node_labels = get_node_label_dict(graph, attr_type=node_attr)
    edge_labels = get_edge_label_dict(graph, attr_type=edge_attr)
    
    G = grakel.Graph(
            edges, 
            node_labels=node_labels,
            edge_labels=edge_labels
            )
    return(G)


def igraph_to_grakel(graphs, attr):
    """ Creates the grakel graph for all igraphs in the dataset"""
    grakel_graphs = [create_grakel_graph(g, attr) for g in graphs]
    return(grakel_graphs)


if __name__ == "__main__":
    # Loads the MUTAG dataset
    path = "/Users/lobray/Desktop/data/data/BZR/"
    FILE = sorted(os.listdir(path))

    graphs = [
            ig.read(path + filename, format='picklez') for filename in
            tqdm(FILE, desc='File')
        ]
    print(graphs[0])

    graph_attributes = {
            "SP_gkl": {"vertex": "label", "edge": []},
            "EH_gkl": {"vertex": [], "edge": "label"},
            "RW_gkl": {"vertex": "label", "edge": []},
            "WL_gkl": {"vertex": "label", "edge": []},
            "VH_gkl": {"vertex": "label", "edge": []}
            }
    

    # get graphs and y ito the format of grakel
    i#G = igraph_to_grakel(graphs, attr=graph_attributes["WL_gkl"])
    y = [int(g['label']) for g in graphs]

    #MUTAG = fetch_dataset("MUTAG", verbose=False)
    #G, y = MUTAG.data, MUTAG.target

    # Splits the dataset into a training and a test set
    G_train, G_test, y_train, y_test = train_test_split(G, y,
            test_size=0.1, random_state=41)

    # Uses the shortest path kernel to generate the kernel matrices
    #gk = ShortestPath(with_labels=False, normalize=True)
    gk = WeisfeilerLehman(n_iter=0)
    #gk = VertexHistogram() 
    K_train = gk.fit_transform(G_train)
    K_test = gk.transform(G_test)

    # Uses the SVM classifier to perform classification
    clf = SVC(kernel="precomputed")
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)

    # Computes and prints the classification accuracy
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", str(round(acc*100, 2)) + "%")



