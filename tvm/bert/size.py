import json
import numpy as np
def getGraph():
    f = open('bert/bert_graph.json', 'r')
    graph = json.load(f)
    return graph

sizes = dict()
sizes["int64"] = 8
sizes["float32"] = 4


def computeMemory(ty, graph, i):
    return sizes[ty] * np.prod(np.array(graph['attrs']['shape'][1][i]))


if __name__ == "__main__":
    graph = getGraph()
    storage = set()
    print(graph.keys())
    storage_ids = graph["attrs"]["storage_id"][1]
    nodes = graph["nodes"]
    total_memory = 0
    const_memory = 0
    kaas_memory = 0

    for i in range(len(nodes)):
        if nodes[i]["name"][0] == "p":
            const_memory += computeMemory(graph["attrs"]["dltype"][1][i], graph, i)
        else:
            kaas_memory += computeMemory(graph["attrs"]["dltype"][1][i], graph, i)
            ident = storage_ids[i]
            if not (ident in storage):
                storage.add(ident)
                total_memory += computeMemory(graph["attrs"]["dltype"][1][i], graph, i)
    '''
    for i in range(len(storage_ids)):
        ident = storage_ids[i]
        if not (ident in storage):
            storage.add(ident)
        else:
            total_memory += computeMemory(graph["attrs"]["dltype"][1][i], graph, i)
    '''
    print(storage)
    print("TVM memory: " + str(total_memory))
    print("Const memory: " + str(const_memory))
    print("Kaas memory: " + str(kaas_memory))
