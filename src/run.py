from networkx import simrank_similarity
from datapreprocessing import load_graph
from  SimRank import SimRank , SimCluster
import networkx as nx
import numpy as np
import random
import time
import networkx as nx
import numpy as np
import random
import time
import pandas as  pd



def Clustering() -> None:
    start=time.perf_counter()

    similarity = simrank_similarity(graph)

    k=10
    print("k value ",15)
    NodeSet = graph.nodes
    center_nodes = random.sample(NodeSet, k)

    for itr in range(15):
        initial_centers = set(center_nodes)
        print(initial_centers)
        cells = SimCluster(similarity,initial_centers)

        data = cells.items()

        r = list(cells.keys())
        centers = np.array(r)

        count=0
        for i in centers: 
            if count==len(center_nodes):
               break
            cluster_nodes = cells[i]

            G1 = nx.Graph()
            flag=False
            if len(cluster_nodes)==1:
                center_nodes[count]=max(cluster_nodes)
                count=count+1
                continue
            for j in cluster_nodes:
                for k in cluster_nodes:
                    if k==j:
                        continue
                    if A[j-1][k-1]==1:
                        G1.add_edge(j,k)
                        flag=True
                if flag==False:
                    center=max(cluster_nodes)
                    center_nodes[count]=center
                    count=count+1
                    continue


            pr= SimRank(G1,similarity,0.85)
            center = max(pr, key=pr.get)


            center_nodes[count]=center
            count=count+1


    finish = time.perf_counter()
    elapsed_time=finish-start

    print("final K centres are")
    print(center_nodes)
    print("time taken")
    print(elapsed_time)


if __name__=="__main__":
    graph = load_graph("C:\Users\mithi\Downloads\mini-project-20231110T142346Z-001\mini-project\minor_project\data\deezer_europe_edges.csv")

    k=15
    print("k value ",k)
    NodeSet = graph.nodes
    center_nodes = random.sample(NodeSet, k)
    print("initial random center_nodes")
    print(center_nodes)

    similarity = simrank_similarity(graph)
    print("similarities found successfully....") 

    adjacency_matrix = nx.to_numpy_array(graph)
    A=adjacency_matrix
    print("adjacency matrix is calculted successfully....")

    Clustering()





