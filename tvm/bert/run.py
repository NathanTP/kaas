import tvm
import infbench
import pathlib
import cv2
import numpy as np
from PIL import Image
from tvm.contrib.debugger import debug_executor as graph_executor
#from tvm.contrib import graph_executor
import pickle
import json
from infbench import bert

import numpy as np

def getInput():
    vocab = open("bert/vocab.txt", 'r').read()
    data = json.load(open("bert/bertInputs.json", 'r'))
    return data, vocab


def loadParams():
    params = pickle.load(open("bertInfo/bert_params.pkl", 'rb'))
    return params

def loadParams():
    path = pathlib.Path.cwd() / "bert" / "bert_params.pkl"
    params = pickle.load(open(path, 'rb'))
    return {'p' + str(i): params[i] for i in range(len(params))}



'''
loader = infbench.dataset.imageNetLoader(pathlib.Path.cwd() / 'data')

loader.preLoad([0])
inp = pre(loader.get(0))[0]

#print(inp)
'''
graphMod = tvm.runtime.load_module("bert/bert.so")

#data, vocab = getInput()

f = open('bert/bert_graph.json', 'r')

model = graph_executor.create(f.read(), graphMod, tvm.cuda())
f.close()


#loader = bert.bertLoader(pathlib.Path.cwd() / "bertData")

#inputs = loader.get(0)

#print(data)
#print(type(data))

constants = bert.bertModel.getConstants(pathlib.Path.cwd())

#pre_input = constants + [inputs[0]]

#pre_output = bert.bertModel.pre(pre_input)

#new_inp = np.frombuffer(pre_output[0], dtype=np.int64)

#print(type(pre_output[1]))

#graph_inputs = [np.frombuffer(array, dtype=np.int64) for array in pre_output]

pre_output = []
pre_output.append(np.random.rand(1, 384, dtype=np.int64))
pre_output.append(np.random.rand(1, 384, dtype=np.int64))
pre_output.append(np.random.rand(1, 384, dtpye=np.int64))



graph_inputs = []
graph_inputs.append(np.frombuffer(pre_output[0], dtype=np.int64))
graph_inputs.append(np.frombuffer(pre_output[1], dtype=np.int64))
graph_inputs.append(np.frombuffer(pre_output[2], dtype=np.int64))

print(type(pre_output[0]))
model.set_input('input_ids', tvm.nd.array(graph_inputs[0]))
model.set_input('input_mask', tvm.nd.array(graph_inputs[1]))
model.set_input('segment_ids', tvm.nd.array(graph_inputs[2]))

params = loadParams()
for key in params.keys():
    model.set_input(key, tvm.nd.array(params[key]))

model.run()

output = tvm.nd.empty((1, 384), dtype="float32")
model.debug_get_output(1122, output)

print(output)

#test = tvm.nd.empty((1, 384, 1), dtype='float32')
#model.debug_get_output(23, test)
#print(test)
#model.run()

'''
inps = bert.bertModel.pre((vocab, data))



'''
#new_inp = np.frombuffer(inp, dtype=np.float32)
'''
model.set_input('input_ids', tvm.nd.array(inps[0]))
model.set_input('input_mask', tvm.nd.array(inps[1]))
model.set_input('segment_ids', tvm.nd.array(inps[2]))


params = loadParams()
for key in params.keys():
    model.set_input(key, tvm.nd.array(params[key]))


a = tvm.nd.empty((1,), dtype='int64')
model.run()
#model.debug_get_output(167, a)
#print(a.asnumpy()[0][1][0][0])
#model.debug_get_output(0, np.zeros((224,)))

#print(a)

'''
