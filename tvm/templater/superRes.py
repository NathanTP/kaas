import pathlib
import math
from pprint import pprint
import sys
import subprocess as sp
import pickle
import libff as ff
import libff.kv
import libff.invoke
from PIL import Image
# import kaasServer as kaas
import libff.kaas as kaas
import numpy as np

from tvm.contrib.download import download_testdata


redisPwd = "Cd+OBWBEAXV0o2fg5yDrMjD9JUkW7J6MATWuGlRtkQXk/CBvf2HYEjKDYw4FC+eWPeVR8cQKWr7IztZy"
testPath = pathlib.Path(__file__).resolve().parent


def getCtx(remote=False):
    if remote:
        objStore = ff.kv.Redis(pwd=redisPwd, serialize=True)
    else:
        objStore = ff.kv.Local(copyObjs=False, serialize=False)

    return libff.invoke.RemoteCtx(None, objStore)


''' Adds the given array to the kv with name node_num. '''
def addToKV(kv, name, arr):
    kv.put(name, arr)
    nByte = arr.nbytes
    buff = kaas.bufferSpec(name, nByte)
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


    #0. load data
    nodes.append(addToKV(libffCtx.kv, '0', inp))

    
    #1. p0
    nodes.append(addToKV(libffCtx.kv, '1', params['p0']))

    #2. p1
    nodes.append(addToKV(libffCtx.kv, '2', params['p1']))

    
    #3.fused_nn_conv2d_expand_dims_add_nn_relu_2"
    nodes.append(kaas.bufferSpec('3', 12845056))
    inputs = [nodes[0], nodes[1], nodes[2]]
    shapes = [(14, 112, 1), (16, 1, 4)]
    kerns.append(makeKern('fused_nn_conv2d_expand_dims_add_nn_relu_2_kernel0', nodes[3], path, inputs, shapes))


    #4. p2
    nodes.append(addToKV(libffCtx.kv, '4', params['p2']))



    #5. p3
    nodes.append(addToKV(libffCtx.kv, '5', params['p3']))

    #6.fused_nn_conv2d_expand_dims_add_nn_relu_1"

    #kernel 0 
    imm = []
    imm.append(kaas.bufferSpec('a0', 589824))
    inputs = [nodes[3]]
    shapes = [(32, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_conv2d_expand_dims_add_nn_relu_1_kernel0', imm[0], path, inputs, shapes))

    #kernel 1
    imm.append(kaas.bufferSpec('a1', 28901376))
    inputs = [nodes[4]]
    shapes = [(1568, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_conv2d_expand_dims_add_nn_relu_1_kernel1', imm[1], path, inputs, shapes))
     
    #kernel 2
    imm.append(kaas.bufferSpec('a2', 28901376))
    inputs = [imm[0], imm[1]]
    shapes = [(49, 1, 36), (16, 4, 1)]
    kerns.append(makeKern('fused_nn_conv2d_expand_dims_add_nn_relu_1_kernel2', imm[2], path, inputs, shapes))

    #kernel 3
    nodes.append(kaas.bufferSpec('6', 12845056))
    inputs = [imm[2], nodes[5]]
    shapes = [(1568, 1, 1), (128, 1, 1)]
    kerns.append(makeKern('fused_nn_conv2d_expand_dims_add_nn_relu_1_kernel3', nodes[6], path, inputs, shapes))

    #7. p4
    nodes.append(addToKV(libffCtx.kv, '7', params['p4']))

    #8. p5
    nodes.append(addToKV(libffCtx.kv, '8', params['p5']))
    
   
     
    #9. 
    nodes.append(kaas.bufferSpec('9', 6422528))
    inputs = [nodes[5], nodes[6], nodes[7]]
    shapes = [(8, 56, 1), (28, 1, 8)]
    kerns.append(makeKern('fused_nn_conv2d_expand_dims_add_nn_relu_kernel0', nodes[9], path, inputs, shapes))

    #10
    nodes.append(addToKV(libffCtx.kv, '10', params['p6']))

    #11
    nodes.append(addToKV(libffCtx.kv, '11', params['p7']))
    
    
    #12
    #arr = np.zeros((1, 9, 224, 224))
    #nodes.append(addToKV(libffCtx.kv, '12', arr))
    nodes.append(kaas.bufferSpec('12', 1806336))
    inputs = [nodes[9], nodes[10], nodes[11]]
    shapes = [(14, 28, 1), (8, 8, 3)]
    kerns.append(makeKern('fused_nn_conv2d_expand_dims_add_kernel0', nodes[12], path, inputs, shapes))

    
    #13
    nodes.append(kaas.bufferSpec('13', 1806336))
    inputs = [nodes[12]]
    shapes = [(256, 1, 1), (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', nodes[13], path, inputs, shapes)) 
    

    ''' 
    #9.fused_nn_conv2d_expand_dims_add_nn_relu"
    inputs = [nodes[5], nodes[6], nodes[7]]
    shapes = [(8, 56, 1), (28, 1, 8)]
    kerns.append(makeKern('fused_nn_conv2d_expand_dims_add_nn_relu_kernel0', nodes[3], path, inputs, shapes))


    #10. p6
    nodes.append(addToKV(libffCtx.kv, '9', params['p6']))
    

    #11. p7
    nodes.append(addToKV(libffCtx.kv, '10', params['p7']))


    #12. fused_nn_conv2d_expand_dims_add" 
    inputs = [nodes[3], nodes[9], nodes[10]]
    shapes = [(14, 28, 1), (8, 8, 3)]
    kerns.append(makeKern('fused_nn_conv2d_expand_dims_add_kernel0', nodes[6], path, inputs, shapes)) 

    #13. fused_reshape_transpose_reshape"
    inputs = [nodes[6]]
    shapes =  [(256, 1, 1), (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', nodes[3], path, inputs, shapes))
    '''
    
    req = kaas.kaasReq(kerns)

    kaasHandle = kaas.getHandle(mode, libffCtx)
    kaasHandle.Invoke(req.toDict())


    #c = np.frombuffer(libffCtx.kv.get('3'), dtype=np.uint32) 
    #print(c)


    c = np.frombuffer(libffCtx.kv.get('0'), dtype=np.float32)
   
    print(c)
 
    outNp = np.uint8(c).clip(0, 255)
     
    print(outNp)

    #print(c)
    #print(c.shape)
     

def getImage():
    """Downloads an example image for the test"""
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_path = download_testdata(img_url, "cat.png", module="data")
    img = Image.open(img_path)

    # We go through bytes to better align with faas requirements
    return img.tobytes()



def imagePreprocess(buf):
    # mode and size were manually read from the png (used Image.open and then
    # inspected the img.mode and img.size attributes). We're gonna just go with
    # this for now.
    img = Image.frombytes("RGB", (256,256), buf)

    img = img.resize((224,224))
    img = img.convert("YCbCr")
    img_y, img_cb, img_cr = img.split()
    imgNp = (np.array(img_y)[np.newaxis, np.newaxis, :, :]).astype("float32")
    return (img, imgNp)

def allocSpace(size):
    length = int(size/4)
    return np.zeros( (length,))

img = getImage()

imgPil, imgNp = imagePreprocess(img)
    
#print(imgNp)

runModel(imgNp)



  
