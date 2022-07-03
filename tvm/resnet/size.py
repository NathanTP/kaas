import json
import numpy as np
def getGraph():
    f = open('resInfo/res.txt', 'r')
    graph = json.load(f)
    return graph

sizes = dict()
sizes["int64"] = 8
sizes["float32"] = 4
sizes["int32"] = 4

def compare_size(size1, size2):
    if len(size1) != len(size2):
        return False

    for i in range(len(size1)):
        if size1[i] > size2[i]:
            return False

    return True

def compute_size(array, ty):
    return sizes[ty] * np.prod(array)


def node_analysis(storage_dict, graph, output_list):
   size_map = dict()
   curr_owners = dict()
   storage_dict[0] = 0
   for i in range(1, len(graph["nodes"])):
       found = False
       new_size = np.array(graph['attrs']['shape'][1][i])
       for node in size_map.keys():
           if not node == 0 and not node in output_list and graph['attrs']['dltype'][1][node] == graph['attrs']['dltype'][1][i] and compare_size(size_map[node], new_size) and (not found or compare_size(new_size, size_map[storage_dict[i]])): #check dims:
               bad = False
               owner = curr_owners[node]
               for j in range(i, len(graph["nodes"])):
                   inputs = [x[0] for x in graph["nodes"][j]["inputs"]]
                   if owner in inputs:
                       bad = True
                       break
               #if not bad:
               if not bad:
                   storage_dict[i] = node
                   found = True

       if found:
           curr_owners[storage_dict[i]] = i
       if not found:
           storage_dict[i] = i
           curr_owners[i] = i
           size_map[i] = new_size #sizes[ty] * np.prod(np.array(graph['attrs']['shape'][1][i]))
   return size_map


def node_analysis(storage_map, alloc_map, graph, output_list):
    store_list = graph["attrs"]["storage_id"][1]
    counter = 0
    new_counter = 0
    for i in range(len(store_list)):
        new_size_dims = np.array(graph['attrs']['shape'][1][i])
        ty = graph['attrs']['dltype'][1][i]
        new_size = compute_size(new_size_dims, ty)
        #print(i, store_list[i])
        if store_list[i] in alloc_map.keys():
            if new_size > alloc_map[store_list[i]]:
                alloc_map[store_list[i]] = new_size
            storage_map[i] = store_list[i]
        else:
            alloc_map[counter] = new_size
            storage_map[i] = counter
            counter += 1

def computeMemory(ty, graph, i):
    return sizes[ty] * np.prod(np.array(graph['attrs']['shape'][1][i]))

def computeMemory2(ty, array):
    return sizes[ty] * np.prod(array)


if __name__ == "__main__":
    graph = getGraph()
    storage = set()
    print(graph.keys())
    storage_ids = graph["attrs"]["storage_id"][1]
    nodes = graph["nodes"]
    total_memory = 0
    const_memory = 0
    kaas_memory = 0
    max_dims = 0

    dims = open("dims.txt", 'r').readlines()
    for dim in dims:
        dim_list = eval(dim)
        print(dim_list)
        total = 1
        for i in dim_list:
            for j in i:
                total *= j
        if total > max_dims:
            max_dims = total

    standard_kaas_memory = 0
    for i in range(len(nodes)):
        if nodes[i]["name"][0] == "p":
            const_memory += computeMemory(graph["attrs"]["dltype"][1][i], graph, i)
        else:
            standard_kaas_memory += computeMemory(graph["attrs"]["dltype"][1][i], graph, i)
            ident = storage_ids[i]
            if not (ident in storage):
                storage.add(ident)
                total_memory += computeMemory(graph["attrs"]["dltype"][1][i], graph, i)
    size_map = dict()
    storage_map = dict()
    node_analysis(storage_map, size_map, graph, [167, 171])
    print(size_map)
    for thing in size_map.keys():
        kaas_memory += size_map[thing]
    '''
    for i in range(len(storage_ids)):
        ident = storage_ids[i]
        if not (ident in storage):
            storage.add(ident)
        else:
            total_memory += computeMemory(graph["attrs"]["dltype"][1][i], graph, i)
    '''
    #print(storage)
    print("TVM memory: " + str(total_memory))
    print("Const memory: " + str(const_memory))
    print("Kaas memory: " + str(kaas_memory))
    print("Standard Kaas memory: " + str(standard_kaas_memory))
    print("Max number of threads: " + str(max_dims))
