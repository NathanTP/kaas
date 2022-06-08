import pathlib
import math
from pprint import pprint
import sys
import subprocess as sp
import pickle
import libff as ff
import libff.kv
import libff.invoke
from mnist import MNIST
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
    #print(arr)
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
    nodes.append(addToKV(libffCtx.kv, 0, inp))

    #c = np.frombuffer(libffCtx.kv.get('0'), dtype=np.float32)
    #print(c)

    # 1. fused_nn_batch_flatten
    ty = 'float32'
    output_size = np.array([1, 784])
    print(output_size)
    arr = np.zeros(output_size).astype(ty)
    #print(arr)
    arr[0][783] = 1000
    print(arr)
    nodes.append(addToKV(libffCtx.kv, 1, arr))
    inputs = [nodes[0]]
    shapes = [(1, 1, 1), (784, 1, 1)]
    kerns.append(makeKern('fused_nn_batch_flatten_kernel0', nodes[1], path, inputs, shapes))

    c = np.frombuffer(libffCtx.kv.get('1'), dtype=np.float32)
    print(c)

    # 2. p0
    nodes.append(addToKV(libffCtx.kv, 2, params['p0']))


    # 3. p1
    nodes.append(addToKV(libffCtx.kv, 3, params['p1']))


    # 4. fused_nn_dense_nn_bias_add_nn_relu_1
    ty = 'float32'
    output_size = np.array([1, 128])
    arr = np.zeros(output_size).astype(ty)
    nodes.append(addToKV(libffCtx.kv, 4, arr))
    inputs = [nodes[1], nodes[2], nodes[3]]
    shapes = [(128, 1, 1), (64, 1, 1)]
    kerns.append(makeKern('fused_nn_dense_nn_bias_add_nn_relu_1_kernel0', nodes[4], path, inputs, shapes))

    #c = np.frombuffer(libffCtx.kv.get('4'), dtype=np.float32)
    #print(c)

    # 5. p2
    nodes.append(addToKV(libffCtx.kv, 5, params['p2']))

    #c = np.frombuffer(libffCtx.kv.get('5'), dtype=np.float32)
    #print(c)

    # 6. p3
    nodes.append(addToKV(libffCtx.kv, 6, params['p3']))

    #print(params['p3'])
    #c = np.frombuffer(libffCtx.kv.get('6'), dtype=np.float32)
    #print(c)    

    # 7. fused_nn_dense_nn_bias_add_nn_relu
    ty = 'float32'
    output_size = np.array([1, 64])
    arr = np.zeros(output_size).astype(ty)
    nodes.append(addToKV(libffCtx.kv, 7, arr))
    inputs = [nodes[4], nodes[5], nodes[6]]
    shapes = [(64, 1, 1), (64, 1, 1)]
    kerns.append(makeKern('fused_nn_dense_nn_bias_add_nn_relu_kernel0', nodes[7], path, inputs, shapes))

    c = np.frombuffer(libffCtx.kv.get('7'), dtype=np.float32)
    #print(c)


    # 8. p4
    nodes.append(addToKV(libffCtx.kv, 8, params['p4']))


    # 9. p5
    nodes.append(addToKV(libffCtx.kv, 9, params['p5']))


    # 10. fused_nn_dense_nn_bias_add
    ty = 'float32'
    output_size = np.array([1, 10])
    arr = np.zeros(output_size).astype(ty)
    nodes.append(addToKV(libffCtx.kv, 10, arr))
    inputs = [nodes[7], nodes[8], nodes[9]]
    shapes = [(10, 1, 1), (64, 1, 1)]
    kerns.append(makeKern('fused_nn_dense_nn_bias_add_kernel0', nodes[10], path, inputs, shapes))


    # 11. fused_nn_softmax
    ty = 'float32'
    output_size = np.array([1, 10])
    arr = np.zeros(output_size).astype(ty)
    nodes.append(addToKV(libffCtx.kv, 11, arr))
    inputs = [nodes[10]]
    shapes = [(1, 1, 1), (32, 1, 1)]
    kerns.append(makeKern('fused_nn_softmax_kernel0', nodes[11], path, inputs, shapes))

    req = kaas.kaasReq(kerns)

    kaasHandle = kaas.getHandle(mode, libffCtx)
    kaasHandle.Invoke(req.toDict())

    c = np.frombuffer(libffCtx.kv.get('11'), dtype=np.float32)
    print(c)

def loadMnist(path, dataset='test'):
	mnistData = MNIST(str(path))	
	if dataset == 'train':
		images, labels = mnistData.load_training()
	else:
		images, labels = mnistData.load_testing()

	images = np.asarray(images).astype(np.float32)
	labels = np.asarray(labels).astype(np.uint32)
	return mnistData, images, labels


dataDir = pathlib.Path("fakedata").resolve()
mndata, imgs, lbls = loadMnist(dataDir)

image = imgs[0]
image = (1/255) * image
runModel(image)

