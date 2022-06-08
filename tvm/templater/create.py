import json
import numpy as np



def main(graph, code):
    #text = "def main():\n\tprint('hello world')\n\nmain()"
    text = getStarterCode() #"def runModel(input):\n"


    metadata = json.load(open("superRes_meta.json")) 
    funcs = metadata["func_info"].keys()
    graph = json.load(open(graph))
    kernels = dict()
    for i in range(1, len(graph["nodes"])):
        op = graph["nodes"][i]["op"]
        name = graph["nodes"][i]["name"]
        if op == "null":
            continue
        kernels[name] = []

    #print(kernels)
    
    for kernel in funcs:
        underscore = kernel.rfind('_')
        function = kernel[:underscore]
        number = int(kernel[underscore+1:][6:])
        func_list = kernels[function]
        if len(func_list) == 0 or number < int(func_list[0][func_list[0].rfind('_') + 1:][6:]):
            func_list.insert(0, kernel)
        
        elif number > int(func_list[len(func_list)-1][func_list[len(func_list)-1].rfind('_') + 1:][6:]):
            func_list.insert(len(func_list), kernel)

        else:
            for i in range(1, len(func_list)):
                prev_number = int(func_list[i - 1][func_list[i - 1].rfind('_') + 1:][6:]) 
                next_number = int(func_list[i][func_list[i].rfind('_') + 1:][6:])
                if prev_number < number and number < next_number:
                    func_list.insert(i, kernel)
                    break

    for i in range(1, len(graph["nodes"])):
        node = graph["nodes"][i]
        text += tab() + "# " + str(i) + ". " + node["name"] + "\n"
        if node["op"] == "null":
            text += tab() + "nodes.append(addToKV(libffCtx.kv, " +  str(i) + ", params[" +"'" +  node['name'] + "'" + "]))\n"                   
        else:
            for i in range(len(kernels[node["name"]]) - 1):
                text += tab() + "#kernel " + str(i) + "\n"                
                text += tab() + "arr = ?\n"
                text += tab() + "nodes.append(addToKV(libffCtx.kv, " + str(i) + ", arr))\n"
                text += tab() + "inputs = ?\n"
                text += tab() + "shapes = ?\n"
                text += tab() + "kerns.append(makeKern('" + kernels[node["name"]][i] + "', nodes[" + str(i) + "], path, inputs, shapes))\n"     
        
                

            text += tab() + "#kernel " + str(len(kernels[node["name"]]) - 1) + "\n"
            text += tab() + "ty = '" + graph['attrs']['dltype'][1][i] + "'\n"
            text += tab() + "output_size = np.array(" + str(graph['attrs']['shape'][1][i]) + ")\n"
            text += tab() + "arr = np.zeros(output_size).astype(ty)\n"
            text += tab() + "nodes.append(addToKV(libffCtx.kv, " + str(i) + ", arr))\n" 
            text += tab() + "inputs = ?\n"
            text += tab() + "shapes = ?\n"        
            text += tab() + "kerns.append(makeKern('" + node["name"] + "_kernel0', nodes[" + str(i) + "], path, inputs, shapes))\n"     
        
        text += "\n\n"        


    text += getEndingCode()

    createFile(text)


def tab():
    return "    "

def createFile(string):
    f = open("template.py", "w")
    f.write(string)
    f.close()



def getEndingCode():
    code = """
    req = kaas.kaasReq(kerns)

    kaasHandle = kaas.getHandle(mode, libffCtx)
    kaasHandle.Invoke(req.toDict())
    """
    return code


def getStarterCode():
    code = """import pathlib
import math
from pprint import pprint
import sys
import subprocess as sp
import pickle
import libff as ff
import libff.kv
import libff.invoke

# import kaasServer as kaas
import libff.kaas as kaas
import numpy as np

redisPwd = "Cd+OBWBEAXV0o2fg5yDrMjD9JUkW7J6MATWuGlRtkQXk/CBvf2HYEjKDYw4FC+eWPeVR8cQKWr7IztZy"
testPath = pathlib.Path(__file__).resolve().parent


def getCtx(remote=False):
    if remote:
        objStore = ff.kv.Redis(pwd=redisPwd, serialize=True)
    else:
        objStore = ff.kv.Local(copyObjs=False, serialize=False)

    return libff.invoke.RemoteCtx(None, objStore)


''' Adds the given array to the kv with name node_num. '''
def addToKV(kv, node_num, arr):
    kv.put(str(node_num), arr)
    nByte = arr.nbytes
    buff = kaas.bufferSpec(str(node_num), nByte)
    return buff 

def loadParams():
    params = pickle.load(open("params", 'rb'))
    return params

def makeKern(name_func, output, path, inputs, shapes):
    #name_func = node['name'] + '_kernel_0'
    return kaas.kernelSpec(path, name_func, shapes[0], shapes[1], inputs=inputs, outputs=[output])

def runModel(inp, mode='direct'):    
    libffCtx = getCtx(remote=(mode == 'process'))
    params = loadParams()
    nodes = []
    kerns = []
    path = pathlib.Path(__file__).resolve().parent / 'code.cubin'
    nodes.append(addToKV(libffCtx.kv, 0, inp))\n
"""
    return code


if __name__ == "__main__":
    main("graph.json", "code.cubin")
