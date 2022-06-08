import pathlib
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
    nodes.append(addToKV(libffCtx.kv, 0, inp))

    # 1. p0
    nodes.append(addToKV(libffCtx.kv, 1, params['p0']))


    # 2. p1
    nodes.append(addToKV(libffCtx.kv, 2, params['p1']))


    # 3. fused_nn_conv2d_expand_dims_add_nn_relu_2
    #kernel 0
    ty = 'float32'
    output_size = np.array([1, 64, 224, 224])
    arr = np.zeros(output_size).astype(ty)
    nodes.append(addToKV(libffCtx.kv, 3, arr))
    inputs = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_expand_dims_add_nn_relu_2_kernel0', nodes[3], path, inputs, shapes))


    # 4. p2
    nodes.append(addToKV(libffCtx.kv, 4, params['p2']))


    # 5. p3
    nodes.append(addToKV(libffCtx.kv, 5, params['p3']))


    # 6. fused_nn_conv2d_expand_dims_add_nn_relu_1
    #kernel 0
    arr = ?
    nodes.append(addToKV(libffCtx.kv, 0, arr))
    inputs = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_expand_dims_add_nn_relu_1_kernel0', nodes[0], path, inputs, shapes))
    #kernel 1
    arr = ?
    nodes.append(addToKV(libffCtx.kv, 1, arr))
    inputs = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_expand_dims_add_nn_relu_1_kernel1', nodes[1], path, inputs, shapes))
    #kernel 2
    arr = ?
    nodes.append(addToKV(libffCtx.kv, 2, arr))
    inputs = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_expand_dims_add_nn_relu_1_kernel2', nodes[2], path, inputs, shapes))
    #kernel 3
    ty = 'float32'
    output_size = np.array([64])
    arr = np.zeros(output_size).astype(ty)
    nodes.append(addToKV(libffCtx.kv, 2, arr))
    inputs = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_expand_dims_add_nn_relu_1_kernel0', nodes[2], path, inputs, shapes))


    # 7. p4
    nodes.append(addToKV(libffCtx.kv, 7, params['p4']))


    # 8. p5
    nodes.append(addToKV(libffCtx.kv, 8, params['p5']))


    # 9. fused_nn_conv2d_expand_dims_add_nn_relu
    #kernel 0
    ty = 'float32'
    output_size = np.array([1, 32, 224, 224])
    arr = np.zeros(output_size).astype(ty)
    nodes.append(addToKV(libffCtx.kv, 9, arr))
    inputs = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_expand_dims_add_nn_relu_kernel0', nodes[9], path, inputs, shapes))


    # 10. p6
    nodes.append(addToKV(libffCtx.kv, 10, params['p6']))


    # 11. p7
    nodes.append(addToKV(libffCtx.kv, 11, params['p7']))


    # 12. fused_nn_conv2d_expand_dims_add
    #kernel 0
    ty = 'float32'
    output_size = np.array([1, 9, 224, 224])
    arr = np.zeros(output_size).astype(ty)
    nodes.append(addToKV(libffCtx.kv, 12, arr))
    inputs = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_expand_dims_add_kernel0', nodes[12], path, inputs, shapes))


    # 13. fused_reshape_transpose_reshape
    #kernel 0
    ty = 'float32'
    output_size = np.array([1, 1, 672, 672])
    arr = np.zeros(output_size).astype(ty)
    nodes.append(addToKV(libffCtx.kv, 13, arr))
    inputs = ?
    shapes = ?
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', nodes[13], path, inputs, shapes))



    req = kaas.kaasReq(kerns)

    kaasHandle = kaas.getHandle(mode, libffCtx)
    kaasHandle.Invoke(req.toDict())
    