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
def addToKV(kv, node_num, arr, const=True, ephemeral=False):
    kv.put(str(node_num), arr)
    nByte = arr.nbytes
    buff = kaas.bufferSpec(str(node_num), nByte, const=const, ephemeral=ephemeral)
    return buff 

def loadParams():
    params = pickle.load(open("params", 'rb'))
    return params

def makeKern(name_func, path, shapes, arguments):
    return kaas.kernelSpec(path, name_func, shapes[0], shapes[1], arguments=arguments)

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


    # 3. fused_nn_conv2d_add_nn_relu_1
    #kernel 0
    output_size = 12845056
    nodes.append(kaas.bufferSpec('3', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_1_kernel0', path, shapes, arguments))


    # 4. p2
    nodes.append(addToKV(libffCtx.kv, 4, params['p2']))


    # 5. p3
    nodes.append(addToKV(libffCtx.kv, 5, params['p3']))


    # 6. fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu
    imm = []
    #kernel 0
    output_size = ?
    imm.append(kaas.bufferSpec('a0', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_kernel0', path, shapes, arguments))
    #kernel 1
    output_size = ?
    imm.append(kaas.bufferSpec('a1', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_kernel1', path, shapes, arguments))
    #kernel 2
    output_size = 12845056
    nodes.append(kaas.bufferSpec('6', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_kernel2', path, shapes, arguments))


    # 7. p4
    nodes.append(addToKV(libffCtx.kv, 7, params['p4']))


    # 8. p5
    nodes.append(addToKV(libffCtx.kv, 8, params['p5']))


    # 9. fused_nn_conv2d_add_nn_relu
    #kernel 0
    output_size = 6422528
    nodes.append(kaas.bufferSpec('9', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_kernel0', path, shapes, arguments))


    # 10. p6
    nodes.append(addToKV(libffCtx.kv, 10, params['p6']))


    # 11. p7
    nodes.append(addToKV(libffCtx.kv, 11, params['p7']))


    # 12. fused_nn_conv2d_add
    #kernel 0
    output_size = 1806336
    nodes.append(kaas.bufferSpec('12', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_kernel0', path, shapes, arguments))


    # 13. fused_reshape_transpose_reshape
    #kernel 0
    output_size = 1806336
    nodes.append(kaas.bufferSpec('13', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))



    req = kaas.kaasReq(kerns)

    kaasHandle = kaas.getHandle(mode, libffCtx)
    kaasHandle.Invoke(req.toDict())
    