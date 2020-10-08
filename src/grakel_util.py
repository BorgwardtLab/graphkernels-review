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
from grakel.kernels import SubgraphMatching 
import grakel


def dirac(a, b):
    return(a == b)


def triangular_kernel(e1, e2, c=0.25):
    e1 = np.linalg.norm(e1)
    e2 = np.linalg.norm(e2)
    return 1.0 / c * max(0, c - abs(e1 - e2))


def brownian_bridge(v1, v2, c=3):
    v1 = np.linalg.norm(v1)
    v2 = np.linalg.norm(v2)

    return(max(0, c - abs(v1 - v2)))


def kv_kernel(v1, v2, c=3):
    """ Updated vertex kernel that multiples the dirac on the node
    labels and a brownian bridge on the node attributes """
    
    if isinstance(v1, int):
        """ If just an integer, then no attr """
        k_dirac = dirac(v1, v2)
        k_brownian_bridge = 1

    elif isinstance(v1, list):
        """ if a list, then [label, attrs] """
        if len(v1) == 1:
            k_dirac = dirac(v1[0], v2[0])
            k_brownian_bridge = 1
        else:
            k_dirac = dirac(v1[0], v2[0])
            k_brownian_bridge = brownian_bridge(v1[1:], v2[1:], c=c)
            print(k_brownian_bridge)

    else:
        print("There is some problem with the data format")
    
    return(k_dirac * k_brownian_bridge)


def ke_kernel(e1, e2, c=0.25):
    """ Updated edge kernel that multiplies the dirac on the edge label
    and the triangulal kernel on the edge weights / attributes"""
    
    if isinstance(e1, int):
        """ If edge is already an int, then no attr """
        k_dirac = dirac(e1, e2)
        k_tp = 1

    elif isinstance(e1, list):
        """ If a list, then label + attr """
        if len(e1) == 1:
            k_dirac = dirac(e1[0], e2[0])
            k_tp = 1

        else: 
            k_dirac = dirac(e1[0], e2[0])
            k_tp = triangular_kernel(e1[1:], e2[1:], c)
    
    return(k_dirac * k_tp)


def set_of_edge_labels(graphs):
    """ Get set of unique edge labels (made from concatenated node
    labels) and st"""
    label = 0
    all_edge_labels = {}

    for graph in graphs:
        for e in graph.es:
            a = graph.vs['label'][e.source]
            b = graph.vs['label'][e.target] 
            k = str(min(a,b)) + "." + str(max(a,b))

            if k not in all_edge_labels:
                all_edge_labels[k] = label
                label += 1

    return(all_edge_labels)


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
    """Returns a dict with node ids as keys and node attr as values. If
    both are required, the label will be the first entry in the list,
    and the remaining attributes will be the others."""
    d = {}   
    node_index = graph.vs.indices

    if len(attr_type) > 0:
        if len(graph.vs.attributes()) > 0:
            if attr_type == "label":
                node_labels = [int(i) for i in graph.vs[attr_type]]

            if attr_type == "both":
                # return list of [label, attrs] 
                node_labels = [[int(i)] for i in graph.vs['label']]
                if "attribute" in graph.vs.attributes():
                    if isinstance(graph.vs['attribute'][0], float): 
                        node_attributes = [[i] for i in graph.vs['attribute']]
                    elif isinstance(graph.vs['attribute'][0], int):
                        print(np.array(graph.vs['attribute']))
                        node_attributes = [[i] for i in graph.vs['attribute']]
                    elif len(graph.vs['attribute'][0]) > 1:
                        node_attributes = [i.tolist() for i in graph.vs['attribute']]

                    label_attr = []
                    for idx in range(len(node_labels)):
                        label_attr.append(node_labels[idx]+list(node_attributes[idx]))
                    node_labels = label_attr
            d = dict(zip(node_index, node_labels))
    return(d)


def get_edge_label_dict(graph, attr_type="label"):
    """ Return a dict with edge tuple as key and edge attr/label as
    value. If both are required, the label will be the first entry in
    the list, and the remaining attributes will come after"""
    d = {}

    if len(graph.es.attributes()) > 0:
        if len(attr_type) > 0:
            if attr_type == "attribute":
                if "attribute" not in graph.es.attributes():
                    attr_type == "weight"
            for e in graph.es:
                if attr_type == "label":
                    d[(e.source, e.target)] = int(e[attr_type])
                    d[(e.target, e.source)] = int(e[attr_type])
                if attr_type == "attribute" or attr_type == "weight":
                    d[(e.source, e.target)] = e[attr_type]
                    d[(e.target, e.source)] = e[attr_type]
                if attr_type == "both":
                    # concatenate label and attributes
                    label_attr = [e['label']]
                    if "attribute" in graph.es.attributes():
                        label_attr.extend(e['attribute'].tolist())
                    elif "weight" in graph.es.attributes():
                        label_attr.extend(e['weight'].tolist())

                    d[(e.source, e.target)] = label_attr
                    d[(e.target, e.source)] = label_attr
    return(d)


def create_grakel_graph(graph, attr):
    """ Create the list of the grakel components"""
    edge_attr = attr["edge"]
    node_attr = attr["vertex"]

    edges = get_edge_list(graph)
    edge_labels = get_edge_label_dict(graph, attr_type=edge_attr)
    node_labels = get_node_label_dict(graph, attr_type=node_attr)

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
            "WL_gkl": {"vertex": "attribute", "edge": []},
            "VH_gkl": {"vertex": "label", "edge": []},
            "CSM_gkl": {"vertex": "both", "edge": "both"}
            }
    
    # get graphs and y ito the format of grakel
    G = igraph_to_grakel(graphs, attr=graph_attributes["CSM_gkl"])
    y = [int(g['label']) for g in graphs]

    #MUTAG = fetch_dataset("BZR", verbose=False,
    #        prefer_attr_nodes=True, prefer_attr_edges=False)
    #G, y = MUTAG.data, MUTAG.target

    #for idx, g in enumerate(G):
    #    G[idx][2] = {}
    #    for edge in G[idx][0]:
    #        G[idx][2][edge] = 0 #[G[idx][0][i]: 0
    # Splits the dataset into a training and a test set
    #print(G[0][2])
    G_train, G_test, y_train, y_test = train_test_split(G, y,
            test_size=0.1, random_state=41)

    # Uses the shortest path kernel to generate the kernel matrices
    #gk = ShortestPath(with_labels=False, normalize=True)
    gk = SubgraphMatching(k=2, ke=ke_kernel)
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



