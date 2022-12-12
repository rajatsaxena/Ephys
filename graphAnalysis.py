# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 08:18:01 2022

@author: Rajat
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import approximation

in1 = np.random.randint(0,300,200)
in2 = np.random.randint(0,300,200)
wt = np.random.uniform(0,1,200)

# make directed graph
g = nx.DiGraph()
g.add_nodes_from(np.arange(300))
for i in range(len(in1)):
    g.add_edge(in1[i], in2[i], weight=wt[i])

# draw the graph
nx.draw(g,with_labels=True)
plt.draw()
plt.show()

# centrality measures
deg_centrality = nx.degree_centrality(g)
in_deg_centrality = nx.in_degree_centrality(g)
out_deg_centrality = nx.out_degree_centrality(g)
eigen_centrality = nx.eigenvector_centrality(g)
closeness_centraliy = nx.closeness_centrality(g)
betweenness_centrality = nx.betweenness_centrality(g)
load_centrality = nx.load_centrality(g)

node_connectivity = nx.node_connectivity(g)
diameter = nx.diameter(g)
pr = nx.pagerank(g)
# reach centrality
#local_reach_centrality = nx.local_reaching_centrality(g,3)
global_reach_centrality = nx.nx.global_reaching_centrality(g)
voterank = nx.voterank(g)
radius = nx.radius(g)
center = nx.center(g)
avg_shortest_path_length = nx.average_shortest_path_length(g)
sigma = nx.sigma(g)
omega = nx.omega(g)
wiener_index = nx.wiener_index(g)