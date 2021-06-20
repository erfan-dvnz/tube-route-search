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
Imrpoving Cost Function
--------------------------------------------------------------------'''

newtd = pd.read_csv('tubedata.csv', header=None)
newgraph = {}

for i in range(len(newtd)):
    newtd[1][i] = preprocess(newtd[1][i])
    newtd[2][i] = preprocess(newtd[2][i])

for i in range(len(newtd)):
    if newtd[0][i] in newgraph.keys():
        new_value = { newtd[1][i] : { newtd[2][i] : newtd[3][i] } }
        newgraph[newtd[0][i]].update(new_value)
    else:
        newgraph[newtd[0][i]] = { newtd[1][i] : { newtd[2][i] : newtd[3][i] } }

print("Total starting stations: " + str(len(newgraph)))
for start, end_time in newgraph.items():
    print(str(start) + ' --> ' + str(end_time))

new_nx_graph = nx.Graph()

for node, connected_elem in newgraph.items():
    for connected_node, weight in connected_elem.items():
        for uline, time in weight.items():
            new_nx_graph.add_edge(node, connected_node, weight = time)

nx.draw(new_nx_graph, node_size=10)
plt.show()

def new_uniformCostSearch(new_graph, origin, goal):
    visited_nodes = {}
    visited_nodes[origin] = (0, [origin])
    paths_to_explore = PriorityQueue()
    paths_to_explore.put((0, [origin]))
    number_of_explored_nodes = 1
    line_changes = 0

    while not paths_to_explore.empty():
        total_cost, path = paths_to_explore.get()
        current_node = path[-1]
        neighbors = new_graph.neighbors(current_node)

        for neighbor in neighbors:
            edge_data = new_graph.get_edge_data(current_node, neighbor)
            if "weight" in edge_data:
                cost_to_neighbor = edge_data["weight"]
            else:
                cost_to_neighbor = 1
            total_cost_to_neighbor = total_cost + cost_to_neighbor
			
            if (neighbor not in visited_nodes) or (visited_nodes[neighbor][0]>total_cost_to_neighbor):
                next_node = (total_cost_to_neighbor, path+[neighbor])
                visited_nodes[neighbor] = next_node
                paths_to_explore.put(next_node)
                number_of_explored_nodes += 1

    print('number of explorations = {}'.format(number_of_explored_nodes))
    print('number of line changes = {}'.format(line_changes))
    return visited_nodes[goal]

ucs_path = new_uniformCostSearch(new_nx_graph, "Euston", "Victoria")
print(ucs_path)
