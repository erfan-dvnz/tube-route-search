import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import re

'''--------------------------------------------------------------------
Data Preprocessing
--------------------------------------------------------------------'''

def preprocess(text):
    text = re.sub(' "', '', text)
    text = re.sub('"', '', text)
    return text

    

'''--------------------------------------------------------------------
Heuristic Search Implementation
--------------------------------------------------------------------'''

h_td = pd.read_csv('tubedata.csv', header=None)
ss1 = h_td.iloc[:, 0]
ss = np.copy(ss1)
h_graph = {}

for i in range(len(h_td)):
    h_td[1][i] = preprocess(h_td[1][i])
    h_td[4][i] = preprocess(h_td[4][i])
    h_td[5][i] = preprocess(h_td[5][i])

for i in range(len(h_td)):
    h_td[0][i] = "[" + h_td[4][i] + "," + h_td[5][i] + "]"

for i in range(len(h_td)):
    for j in range(len(h_td)):
        if h_td[1][i] == ss[j]:
            h_td[1][i] = h_td[0][j]

for i in range(len(h_td)):
    if h_td[0][i] in h_graph.keys():
        new_value = { h_td[1][i] : h_td[3][i] }
        h_graph[h_td[0][i]].update(new_value)
    else:
        h_graph[h_td[0][i]] = { h_td[1][i] : h_td[3][i] }

print("Total starting stations: " + str(len(h_graph)))
for start, end_time in h_graph.items():
    print(str(start) + ' --> ' + str(end_time))

nx_h_graph = nx.Graph()
for node, connected_elem in h_graph.items():
    for connected_node, weight in connected_elem.items():
        nx_h_graph.add_edge(node, connected_node, weight = weight)

def heuristic(node):
    node = node.replace('[','')
    node = node.replace(']','')
    x, y = node.split(',', maxsplit=2)
    x = float(x)
    y = float(y)
    return abs(x-9) + abs(y-9)

def Astar(graph, origin, goal):
    admissible_heuristics = {}
    h = heuristic(origin)
    admissible_heuristics[origin] = h
    visited_nodes = {}
    visited_nodes[origin] = (h, [origin])

    paths_to_explore = PriorityQueue()
    paths_to_explore.put((h, [origin], 0))
    while not paths_to_explore.empty():
        _, path, total_cost = paths_to_explore.get()
        current_node = path[-1]
        neighbors = graph.neighbors(current_node)
		
        for neighbor in neighbors:
            edge_data = graph.get_edge_data(path[-1], neighbor)
            if "weight" in edge_data:
                cost_to_neighbor = edge_data["weight"] 
            else:
                cost_to_neighbor = 1 

            if neighbor in admissible_heuristics:
                h = admissible_heuristics[neighbor]
            else:
                h = heuristic(neighbor)
                admissible_heuristics[neighbor] = h

            new_cost = total_cost + cost_to_neighbor
            new_cost_plus_h = new_cost + h
            if (neighbor not in visited_nodes) or (visited_nodes[neighbor][0]>new_cost_plus_h):
                next_node = (new_cost_plus_h, path+[neighbor], new_cost)
                visited_nodes[neighbor] = next_node 
                paths_to_explore.put(next_node) 
                
    return visited_nodes[goal]

solution = Astar(nx_h_graph,"[1,0]","[5,0]")
print(solution)