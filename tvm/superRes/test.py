import tvm
import numpy as np
import pathlib
import pickle
import tempfile
import os
import sys
import io

import onnx
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib import graph_executor
from PIL import Image
import matplotlib.pyplot as plt
import json

from tvm.contrib.download import download_testdata


def getSuperRes():
    """Downloads the superres model used in this example"""
    # Pre-Trained ONNX Model
    modelUrl = "".join(
        [
            "https://gist.github.com/zhreshold/",
            "bcda4716699ac97ea44f791c24310193/raw/",
            "93672b029103648953c4e5ad3ac3aadf346a4cdc/",
            "super_resolution_0.2.onnx",
        ]
    )

    modelPath = download_testdata(modelUrl, "super_resolution.onnx", module="onnx")
    return pathlib.Path(modelPath)

def importOnnx(onnxPath, shape):
    """Will load the onnx model (*.onnx) from onnxPath and store it in a
    pre-compiled .so file. It will return a TVM executor capable of running the
    model. Shape is the input shape for the model and must be derived
    manually."""
    #libraryPath = pathlib.Path.cwd() / (onnxPath.stem + ".so")
    libraryPath = pathlib.Path.cwd() / "lib.so"

    if True:
        graphMod = tvm.runtime.load_module(libraryPath)
    #print(type(graphMod))
    #lib = graphMod.get_lib()
    #print(type(graphMod))
    #print(graphMod.get_lib)
    f = open("graph.json", "r")
    js = f.read()#graphMod.get_graph_json()
    f.close()
    #print(graphMod.get_function('fused_nn_conv2d_add_nn_relu_2'))
    #return graph_executor.create(js, graphMod, tvm.cuda())
    #return graph_executor#graph_executor.GraphModule(graphMod['default'](tvm.cuda()))
    return graph_executor.GraphModule(graphMod['default'](tvm.cuda()))


def executeModel(ex, img):
    #params = loadParams()
    arr = np.frombuffer(img, dtype="float32")
    arr.shape = (1, 1, 224, 224)
       
    print(arr.dtype)
    #img = np.array(img)
    #print(type(img))
    img_tvm_arr = tvm.nd.array(arr)
    ex.set_input('1', img)
    #for key in params.keys():
        #ex.set_input(key, params[key])
    #inp = tvm.nd.array(img)
    #a = tvm.nd.empty((1, 9, 224, 224), dtype='float32') 
    ex.run()
    return ex.get_output(0)




from infbench import superres

dataDir = pathlib.Path.cwd() / "data"

loader = superres.superResLoader(dataDir)

loader.preLoad(0)
data = loader.get(0)



thing1, thing2 = superres.superResBase.pre(data)


modelPath = getSuperRes()
ex = importOnnx(1, 1)    
# Execute
print("Running model")
#print(imgNp)
out = executeModel(ex, thing2)

pngBuf = superres.superResBase.post(thing1, runtime)

with open("test.png", "wb") as f:
    f.write(pngBuf[0][0])



