import json
import numpy as np


sizes = {'float32': 4, 'int32': 4, 'int64': 8}
grap = None

storage = None

def compute_size(array, ty):
    return sizes[ty] * np.prod(array)

def compare_size(size1, size2):
    if len(size1) != len(size2):
        #print("original node", "new node")
        #print(size1, size2)
        raise Exception("invalid assignment")

    for i in range(len(size1)):
        if size1[i] > size2[i]:
            return False

    return True

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

def toString(convert):
    if isinstance(convert, str):
        print(convert)
        return "\"" + convert + "\""
    else:
        return str(convert)

def main(graph, code, metadata):
    text = getStarterCode()
    imm_count = 0

    metadata = json.load(open(metadata, "r"))
    funcs = metadata["func_info"].keys()
    graph = json.load(open(graph))
    kernels = dict()
    for i in range(1, len(graph["nodes"])):
        op = graph["nodes"][i]["op"]
        name = graph["nodes"][i]["name"]
        if op == "null":
            continue
        kernels[name] = []
        name_func = graph["nodes"][i]["attrs"]["func_name"] + "_kernel"
        if "__nop" in name_func:
            kernels[name].append("__nop")
        else:
            counter = 0
            while True:
                temp = name_func + str(counter)
                if temp in funcs:
                    kernels[name].append(temp)
                else:
                    break
                counter += 1

    dims = open("dims.txt", "r").readlines()
    dim_counter = 0

    output_list = []
    for i in graph["heads"]:
        output_list.append(i[0])

    print(len(dims))
    funcDict = getFuncDict()
    loadGraph(graph)
    created = set()
    storage_dict = dict()
    alloc_map = dict()
    node_analysis(storage_dict, alloc_map, graph, output_list)
    print(storage_dict)


    for i in range(1, len(graph["nodes"])):
        node = graph["nodes"][i]
        node_num = storage_dict[i]
        create = False
        if not node_num in created:
            create = True
            created.add(node_num)

        #node_num = getNode(i)
        text += tab() + "# " + str(i) + ". " + node["name"] + "\n"
        if node["op"] == "null":
            text += tab() + "nodes[" + str(node_num) + "] = addToKV(" + str(node_num) + ", params[" + "'" + node['name'] + "'" + "])\n"
            #created.add(node_num)
        elif node["attrs"]["func_name"] == "__nop":
            text += tab() + "nodes[" + str(node_num) + "] = nodes[" + str(getNode(node["inputs"][0][0])) + "]\n"
        else:
            if len(kernels[node["name"]]) > 1:
                text += tab() + "imm = []\n"
            tmp_map = dict()
            tmp_count = 0
            arg_counter = 0
            for j in range(len(kernels[node["name"]]) - 1):
                text += tab() + "# kernel " + str(j) + "\n"
                text += tab() + "output_size = 1806336\n"
                text += tab() + "imm.append(kaas.bufferSpec('a" + str(imm_count) + "', output_size, const=True, ephemeral=True))\n"
                if kernels[node["name"]][j] in funcDict.keys():
                    text += tab() + "arguments = ["
                    argList = funcDict[kernels[node["name"]][j]]
                    for k in range(len(argList)):
                        if argList[k] == "i":
                            input_node = storage_dict[node["inputs"][arg_counter][0]]
                            text += "(nodes[\"" + toString(input_node) + "\"], 'i'), "
                            arg_counter += 1
                        else:
                            name = argList[k][0]
                            inout = argList[k][1]
                            if not name in tmp_map:
                                tmp_map[name] = str(tmp_count)
                                tmp_count += 1
                            if inout == "o":
                                text += "(imm[" + tmp_map[name] + "], 't'), "
                            else:
                                text += "(imm[" + tmp_map[name] + "], 'i'), "
                    text = text[:-2]
                    text += "]\n"
                else:
                    text += tab() + "arguments = [help]\n"

                text += tab() + "shapes = " + dims[dim_counter].replace(",", ", ")
                text += tab() + "kerns.append(makeKern('" + kernels[node["name"]][j] + "', path, shapes, arguments))\n"
                imm_count += 1
                dim_counter += 1

            text += tab() + "# kernel " + str(len(kernels[node["name"]]) - 1) + "\n"
            ty = graph['attrs']['dltype'][1][i]
            if create:
                text += tab() + "output_size = " + str(sizes[ty] * np.prod(alloc_map[node_num])) + "\n"
                text += tab() + "nodes[" + toString(node_num) + "] = kaas.bufferSpec('" + str(node_num) + "', output_size, const=True, ephemeral=True)\n"
                created.add(node_num)
            kernel_name = kernels[node["name"]][len(kernels[node["name"]]) - 1]
            if kernel_name in funcDict.keys():
                text += tab() + "arguments = ["
                argList = funcDict[kernel_name]
                for k in range(len(argList)):
                    if argList[k] == "o":
                        if i in output_list:
                            text += "(nodes[" + toString(node_num) + "], 'o'), "
                        else:
                            text += "(nodes[" + toString(node_num) + "], 't'), "
                    elif argList[k] == "i":
                        text += "(nodes[" + toString(storage_dict[node["inputs"][arg_counter][0]]) + "], 'i'), "
                        arg_counter += 1
                    else:
                        name = argList[k][0]
                        text += "(imm[" + tmp_map[name] + "], 'i'), "

                text = text[:-2]
                text += "]\n"
            else:
                text += tab() + "arguments = [help]\n"

            text += tab() + "shapes = " + dims[dim_counter].replace(",", ", ")
            text += tab() + "kerns.append(makeKern('" + kernel_name + "', path, shapes, arguments))\n"
            dim_counter += 1

        text += "\n"

    text += getEndingCode()

    createFile(text)


def loadGraph(curr):
    global graph
    graph = curr


def getNode(i):
    return storage[i]
    #return i
    #return graph["attrs"]["storage_id"][1][i]


def tab():
    return "    "


def createFile(string):
    f = open("template.py", "w")
    f.write(string)
    f.close()


def getEndingCode():
    code = """
    req = kaas.kaasReq(kerns)
    return req

"""
    return code


def getFuncDict():
    funcDict = dict()
    funcDict["fused_nn_conv2d_add_nn_relu_11_kernel0"] = ["i", "i", "o", "i"]
    funcDict["fused_divide_kernel0"] = ["o", "i", "i"]
    funcDict["fused_nn_dense_add_kernel0"] = ["i", "i", "o", "i"]
    funcDict["fused_nn_conv2d_add_nn_relu_7_kernel0"] = ["i", "i", "o", "i"]
    funcDict["fused_nn_conv2d_add_nn_relu_10_kernel0"] = ["i", "i", "o", "i"]
    funcDict["fused_mean_kernel1"] = ["o", ("k", "i")]
    funcDict["fused_nn_max_pool2d_kernel0"] = ["i", "o"]
    funcDict["fused_nn_conv2d_add_nn_relu_6_kernel0"] = ["i", "i", "o", "i"]
    funcDict["fused_max_kernel0"] = ["i", "o"]
    funcDict["fused_nn_conv2d_add_nn_relu_5_kernel0"] = ["i", "i", "o", "i"]
    funcDict["fused_nn_conv2d_add_nn_relu_9_kernel0"] = ["i", "i", "o", "i"]
    funcDict["fused_mean_kernel0"] = ["i", ("k", "o")]
    funcDict["fused_nn_conv2d_add_nn_relu_kernel0"] = ["i", "i", "o", "i"]
    funcDict["fused_nn_conv2d_add_3_kernel0"] = ["i", "i", "o", "i"]
    funcDict["fused_nn_conv2d_add_add_nn_relu_kernel0"] = ["i", "i", "o", "i", "i"]
    funcDict["fused_nn_conv2d_add_kernel0"] = ["i", "i", "o", "i"]
    funcDict["fused_nn_conv2d_add_nn_relu_2_kernel0"] = ["i", "i", "o", "i"]
    funcDict["fused_nn_conv2d_add_1_kernel0"] = ["i", "i", "o", "i"]
    funcDict["fused_argmax_kernel0"] = ["i", "o"]
    funcDict["fused_nn_conv2d_add_add_nn_relu_2_kernel0"] = ["i", "i", "o", "i", "i"]
    funcDict["fused_nn_conv2d_add_nn_relu_8_kernel0"] = ["i", "i", "o", "i"]
    funcDict["fused_squeeze_nn_batch_flatten_kernel0"] = ["o", "i"]
    funcDict["fused_subtract_exp_kernel0"] = ["o", "i", "i"]
    funcDict["fused_nn_conv2d_add_nn_relu_4_kernel0"] = ["i", "i", "o", "i"]
    funcDict["fused_nn_conv2d_add_2_kernel0"] = ["i", "i", "o", "i"]
    funcDict["fused_nn_conv2d_add_add_nn_relu_1_kernel0"] = ["i", "i", "o", "i", "i"]
    funcDict["fused_nn_conv2d_add_add_nn_relu_3_kernel0"] = ["i", "i", "o", "i", "i"]

    funcDict["fused_sum_kernel0"] = ["i", "o"]

    funcDict["fused_cast_kernel0"] = ["o", "i"]

    funcDict["fused_nn_conv2d_add_nn_relu_3_kernel0"] = ["i", "i", "o", "i"]
    funcDict["fused_nn_conv2d_add_nn_relu_1_kernel0"] = ["i", "i", "o", "i"]


    funcDict["fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2_kernel0"] = ["i", ("k0", "o")]
    funcDict["fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2_kernel1"] = ["i", ("k0", "i"), ("k1", "o")]
    funcDict["fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2_kernel2"] = [("k1", "i"), "o", "i"]


    funcDict["fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel0"] = ["i", ("k0", "o")]
    funcDict["fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel1"] = ["i", ("k0", "i"), ("k1", "o")]
    funcDict["fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel2"] = [("k1", "i"), "o", "i"]

    funcDict["fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_kernel0"] = ["i", ("k0", "o")]
    funcDict["fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_kernel1"] = ["i", ("k0", "i"), ("k1", "o")]
    funcDict["fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_kernel2"] = [("k1", "i"), "o", "i"]

    funcDict["fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3_kernel0"] = ["i", ("k0", "o")]
    funcDict["fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3_kernel1"] = ["i", ("k0", "i"), ("k1", "o")]
    funcDict["fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3_kernel2"] = [("k1", "i"), "o", "i"]

    return funcDict


def getStarterCode():
    code = """import pathlib
import pickle

import kaas
import numpy as np

redisPwd = "Cd+OBWBEAXV0o2fg5yDrMjD9JUkW7J6MATWuGlRtkQXk/CBvf2HYEjKDYw4FC+eWPeVR8cQKWr7IztZy"
testPath = pathlib.Path(__file__).resolve().parent



# Adds the given array to the kv with name node_num.
def addToKV(node_num, arr, const=True, ephemeral=False):
    nByte = arr.nbytes
    buff = kaas.bufferSpec(str(node_num), nByte, const=const, ephemeral=ephemeral)
    return buff


def loadParams(param_path):
    params = pickle.load(open(param_path, 'rb'))
    return params


def makeKern(name_func, path, shapes, arguments):
    return kaas.kernelSpec(path, name_func, shapes[0], shapes[1], arguments=arguments)


def createReq(params, cubinPath, mode='direct'):
    nodes = dict()
    kerns = []
    path = cubinPath
    inp = np.zeros((1, 3, 224, 224))
    nodes.append(addToKV(0, inp, const=False, ephemeral=False))
    # storage = dict()
    # storage['0'] = nodes[0]\n
"""
    return code


if __name__ == "__main__":
    main("resInfo/res.txt", "code.cubin", "tvm_meta.json")
