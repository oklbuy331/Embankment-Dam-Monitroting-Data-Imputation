from sklearn import
import numpy as np

# Initialize inputs
n_nodes = 7
nodes = np.arange(n_nodes)
edges = np.array([[1,3],[1,2],[1,7],[2,4],[3,5],[3,6],[4,5],[6,7]]) - 1
adj = np.zeros([n_nodes, n_nodes])

for i, j in edges:
    adj[i,j] = 1

t = 3
n_rw = 4

# Generating random walks
def generate_rw()