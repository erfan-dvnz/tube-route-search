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

td = pd.read_csv('tubedata.csv', header=None)
graph = {}

for i in range(len(td)):
    td[1][i] = preprocess(td[1][i])

for i in range(len(td)):
    if td[0][i] in graph.keys():
        new_value = { td[1][i] : td[3][i] }
        graph[td[0][i]].update(new_value)
    else:
        graph[td[0][i]] = { td[1][i] : td[3][i] }

print("Total starting stations: " + str(len(graph)))
for start, end_time in graph.items():
    print(str(start) + ' --> ' + str(end_time))

nx_graph = nx.Graph()

for node, connected_elem in graph.items():
    for connected_node, weight in connected_elem.items():
        nx_graph.add_edge(node, connected_node, weight = weight)

nx.draw(nx_graph, node_size=10)
plt.show()



'''--------------------------------------------------------------------
DFS Implementation
--------------------------------------------------------------------'''

def construct_path_from_root(node, root):
    path_from_root = [node['label']]
    count = 0
    while node['parent']:
        node = node['parent']
        path_from_root = [node['label']] + path_from_root
        count += 1
    print("Stations to travel: " + str(count))
    return path_from_root

def my_depth_first_graph_search(nxobject, initial, goal):
    frontier = [{'label':initial, 'parent':None}]  
    explored = {initial}
    number_of_explored_nodes = 1 

    while frontier:
        node = frontier.pop()
        number_of_explored_nodes += 1
        
        if node['label']==goal:
            print('number of explorations = {}'.format(number_of_explored_nodes))
            return node

        neighbours = nxobject.neighbors(node['label'])
        for child_label in neighbours:
            child = {'label':child_label, 'parent':node}
            if child_label not in explored:
                frontier.append(child)
                explored.add(child_label)

    return number_of_explored_nodes

dfs_path = my_depth_first_graph_search(nx_graph, 'Euston', 'Victoria')
construct_path_from_root(dfs_path, 'Euston')



'''--------------------------------------------------------------------
BFS Implementation
--------------------------------------------------------------------'''

def my_breadth_first_graph_search(nxobject, initial, goal):
    
    if initial == goal:
        return None
    
    number_of_explored_nodes = 1    
    frontier = [{'label':initial, 'parent':None}]
    explored = {initial}
    
    while frontier:
        node = frontier.pop()
        neighbours = nxobject.neighbors(node['label'])

        for child_label in neighbours:
            child = {'label':child_label, 'parent':node}
            if child_label==goal:
                print('number of explorations = {}'.format(number_of_explored_nodes))
                return child
            if child_label not in explored:
                frontier = [child] + frontier
                number_of_explored_nodes += 1
                explored.add(child_label)
            
    return number_of_explored_nodes

bfs_path = my_breadth_first_graph_search(nx_graph, 'Euston', 'Victoria')
construct_path_from_root(bfs_path, 'Euston')



'''--------------------------------------------------------------------
UCS Implementation
--------------------------------------------------------------------'''

from queue import PriorityQueue

def uniformCostSearch(graph, origin, goal):
	  visited_nodes = {}
	  visited_nodes[origin] = (0, [origin])

	  paths_to_explore = PriorityQueue()
	  paths_to_explore.put((0, [origin]))
	  number_of_explored_nodes = 1

	  while not paths_to_explore.empty():
		    total_cost, path = paths_to_explore.get()
		    current_node = path[-1]
		    neighbors = graph.neighbors(current_node)

		    for neighbor in neighbors:
			      edge_data = graph.get_edge_data(current_node, neighbor)
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
	  return visited_nodes[goal]

ucs_path = uniformCostSearch(nx_graph, "Euston", "Victoria")
print(ucs_path)