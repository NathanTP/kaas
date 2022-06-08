import json
import numpy as np


sizes = {'float32': 4, 'int32': 4, 'int64': 8}
grap = None


def main(graph, code, metadata):
    text = getStarterCode()
    imm_count = 0

    metadata = json.load(metadata)
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

    dims = open("newDims.txt", "r").readlines()
    dim_counter = 0

    print(len(dims))
    funcDict = getFuncDict()
    loadGraph(graph)
    created = set()

    for i in range(1, len(graph["nodes"])):
        node = graph["nodes"][i]
        node_num = getNode(i)
        text += tab() + "# " + str(i) + ". " + node["name"] + "\n"
        if node["op"] == "null":
            text += tab() + "nodes[" + str(node_num) + "] = addToKV(" + str(node_num) + ", params[" + "'" + node['name'] + "'" + "])\n"
            created.add(node_num)
        elif node["attrs"]["func_name"] == "__nop":
            text += tab() + "nodes[" + str(node_num) + "] = nodes[" + str(getNode(node["inputs"][0][0])) + "]\n"
        else:
            if len(kernels[node["name"]]) > 1:
                text += tab() + "imm = []\n"
            for j in range(len(kernels[node["name"]]) - 1):
                text += tab() + "# kernel " + str(j) + "\n"
                text += tab() + "output_size = 4096\n"
                text += tab() + "imm.append(kaas.bufferSpec('a" + str(imm_count) + "', output_size, const=False, ephemeral=True))\n"
                if kernels[node["name"]][j] in funcDict.keys():
                    text += tab() + "arguments = ["
                    argList = funcDict[kernels[node["name"]][j]]
                    arg_counter = 0
                    for k in range(len(argList)):
                        if argList[k] == "o":
                            text += "(imm[" + str(j) + "], 'o'), "
                        else:
                            text += "(nodes[" + str(getNode(node["inputs"][arg_counter][0])) + "], 'i'), "
                            arg_counter += 1
                    text += "]\n"
                else:
                    text += tab() + "arguments = [help]\n"
                text += tab() + "shapes = " + dims[dim_counter]
                text += tab() + "kerns.append(makeKern('" + kernels[node["name"]][j] + "', path, shapes, arguments))\n"
                imm_count += 1
                dim_counter += 1

            text += tab() + "# kernel " + str(len(kernels[node["name"]]) - 1) + "\n"
            ty = graph['attrs']['dltype'][1][i]
            if not (node_num in created):
                text += tab() + "output_size = " + str(sizes[ty] * np.prod(np.array(graph['attrs']['shape'][1][i]))) + "\n"
                text += tab() + "nodes[" + str(node_num) + "] = kaas.bufferSpec('" + str(node_num) + "', output_size, const=False, ephemeral=True)\n"
                created.add(node_num)
            kernel_name = kernels[node["name"]][len(kernels[node["name"]]) - 1]
            if kernel_name in funcDict.keys():
                text += tab() + "arguments = ["
                argList = funcDict[kernel_name]
                arg_counter = 0
                for k in range(len(argList)):
                    if argList[k] == "o":
                        text += "(nodes[" + str(node_num) + "], 'o'), "
                    elif argList[k] == 'k':
                        text += "(imm[0], 'i'), "
                    else:
                        text += "(nodes[" + str(getNode(node["inputs"][arg_counter][0])) + "], 'i'), "
                        arg_counter += 1
                text += "]\n"
            else:
                text += tab() + "arguments = [help]\n"
            text += tab() + "shapes = " + dims[dim_counter]
            text += tab() + "kerns.append(makeKern('" + kernel_name + "', path, shapes, arguments))\n"
            dim_counter += 1

        text += "\n"

    text += getEndingCode()

    createFile(text)


def loadGraph(curr):
    global graph
    graph = curr


def getNode(i):
    return graph["attrs"]["storage_id"][1][i]


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

runReq()"""
    return code


def getFuncDict():
    funcDict = dict()
    funcDict["fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0"] = ["o", "i", "i"]
    funcDict["fused_mean_kernel1"] = ["o", "k"]
    funcDict["fused_nn_batch_matmul_4_kernel0"] = ["i", "i", "o"]
    funcDict["fused_nn_batch_matmul_5_kernel0"] = ["i", "i", "o"]
    funcDict["fused_nn_batch_matmul_2_kernel0"] = ["i", "i", "o"]
    funcDict["fused_add_sqrt_divide_multiply_add_reshape_kernel0"] = ["o", "i", "i", "i", "i"]
    funcDict["fused_mean_kernel0"] = ["i", "o"]
    funcDict["fused_add_sqrt_divide_multiply_add_kernel0"] = ["o", "i", "i", "i", "i"]
    funcDict["fused_reshape_add_reshape_transpose_reshape_kernel0"] = ["o", "i", "i"]
    funcDict["fused_subtract_exp_kernel0"] = ["o", "i", "i"]
    funcDict["fused_divide_reshape_kernel0"] = ["o", "i", "i"]
    funcDict["fused_sum_kernel0"] = ["i", "o"]
    funcDict["fused_power_mean_kernel0"] = ["i", "o"]
    funcDict["fused_squeeze_kernel0"] = ["o", "i"]
    funcDict["fused_nn_batch_matmul_1_kernel0"] = ["i", "i", "o"]
    funcDict["fused_nn_batch_matmul_3_kernel0"] = ["i", "i", "o"]
    funcDict["fused_nn_batch_matmul_kernel0"] = ["i", "i", "o"]
    funcDict["fused_reshape_add_add_kernel0"] = ["o", "i", "i", "i"]
    funcDict["fused_reshape_transpose_reshape_kernel0"] = ["o", "i"]
    funcDict["fused_reshape_divide_add_kernel0"] = ["o", "i", "i"]
    funcDict["fused_max_kernel0"] = ["i", "o"]
    funcDict["fused_power_mean_kernel1"] = ["o", "k"]
    funcDict["fused_reshape_add_split_kernel0"] = ["o", "i", "i"]
    funcDict["fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0"] = ["o", "i", "i"]
    funcDict["fused_reshape_add_reshape_transpose_reshape_transpose_kernel0"] = ["o", "i", "i"]
    funcDict["fused_subtract_kernel0"] = ["o", "i", "i"]
    funcDict["fused_reshape_add_split_kernel1"] = ["o", "i", "i"]
    funcDict["fused_squeeze_1_kernel0"] = ["o", "i"]
    return funcDict


def getStarterCode():
    code = """import pathlib
import pickle
import libff as ff
import libff.kv
import libff.invoke

# import kaasServer as kaas
import libff.kaas as kaas

redisPwd = "Cd+OBWBEAXV0o2fg5yDrMjD9JUkW7J6MATWuGlRtkQXk/CBvf2HYEjKDYw4FC+eWPeVR8cQKWr7IztZy"
testPath = pathlib.Path(__file__).resolve().parent


def getCtx(remote=False):
    if remote:
        objStore = ff.kv.Redis(pwd=redisPwd, serialize=True)
    else:
        objStore = ff.kv.Local(copyObjs=False, serialize=False)

    return libff.invoke.RemoteCtx(None, objStore)


''' Adds the given array to the kv with name node_num. '''
def addToKV(node_num, arr, const=True, ephemeral=False):
    kv.put(str(node_num), arr)
    nByte = arr.nbytes
    buff = kaas.bufferSpec(str(node_num), nByte, const=const, ephemeral=ephemeral)
    return buff


def loadParams():
    params = pickle.load(open("params", 'rb'))
    return params


def makeKern(name_func, path, shapes, arguments):
    return kaas.kernelSpec(path, name_func, shapes[0], shapes[1], arguments=arguments)

kv = None
def runReq():
    libffCtx = getCtx(remote=False)
    kv = libffCtx.kv

    req = createReq(inp)

    kaasHandle = kaas.kaasFF.getHandle(mode, libffCtx)
    kaasHandle.Invoke(req.toDict())


def createReq(inp, no_reuse=True, mode='direct'):
    libffCtx = getCtx(remote=(mode == 'process'))
    params = loadParams()
    nodes = dict()
    kerns = []
    path = pathlib.Path(__file__).resolve().parent / 'code.cubin'
    nodes.append(addToKV(libffCtx.kv, 0, inp))
    # storage = dict()
    # storage['0'] = nodes[0]\n
"""
    return code


if __name__ == "__main__":
    main("bertInfo/graph.json", "code.cubin", "bert/bert_meta.tvm_meta.json")
