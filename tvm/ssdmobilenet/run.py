import tvm
import infbench
import pathlib
import cv2
import numpy as np
from PIL import Image
#from tvm.contrib.debugger import debug_executor as graph_executor
from tvm.contrib import graph_executor
import pickle
import infbench










def getInput():
    from infbench import ssdmobilenet
    loader = infbench.ssdmobilenet.cocoLoader(pathlib.Path.cwd() / 'data')

    loader.preLoad([0])
    inp = pre(loader.get(0))[0]

    new_inp = np.frombuffer(inp, dtype=np.float32)
    return new_inp


def loadParams():
    params = pickle.load(open("resInfo/resnet50.pickle", 'rb'))
    return params









graphMod = tvm.runtime.load_module("ssdMobilenet.so")


f = open('graph.json', 'r')

model = graph_executor.create(f.read(), graphMod, tvm.cuda())
f.close()
'''
new_inp = np.frombuffer(inp, dtype=np.float32)
'''
new_inp = getInput()
model.set_input('data', tvm.nd.array(new_inp))
params = loadParams()
for key in params.keys():
    model.set_input(key, tvm.nd.array(params[key]))


#a = tvm.nd.empty((1,), dtype='int64') 
model.run()
#model.debug_get_output(167, a)
#print(a.asnumpy()[0][1][0][0])
#model.debug_get_output(0, np.zeros((224,)))

print(a)



