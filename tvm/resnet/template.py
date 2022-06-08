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


    # 3. fused_nn_conv2d_add_nn_relu_11
    #kernel 0
    output_size = 3211264
    nodes.append(kaas.bufferSpec('3', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_11_kernel0', path, shapes, arguments))


    # 4. fused_nn_max_pool2d
    #kernel 0
    output_size = 802816
    nodes.append(kaas.bufferSpec('4', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_max_pool2d_kernel0', path, shapes, arguments))


    # 5. p2
    nodes.append(addToKV(libffCtx.kv, 5, params['p2']))


    # 6. p3
    nodes.append(addToKV(libffCtx.kv, 6, params['p3']))


    # 7. fused_nn_conv2d_add_nn_relu_10
    #kernel 0
    output_size = 802816
    nodes.append(kaas.bufferSpec('7', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_10_kernel0', path, shapes, arguments))


    # 8. p4
    nodes.append(addToKV(libffCtx.kv, 8, params['p4']))


    # 9. p5
    nodes.append(addToKV(libffCtx.kv, 9, params['p5']))


    # 10. fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_32
    imm = []
    #kernel 0
    output_size = ?
    imm.append(kaas.bufferSpec('a0', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3_kernel0', path, shapes, arguments))
    #kernel 1
    output_size = ?
    imm.append(kaas.bufferSpec('a1', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3_kernel1', path, shapes, arguments))
    #kernel 2
    output_size = 802816
    nodes.append(kaas.bufferSpec('10', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3_kernel2', path, shapes, arguments))


    # 11. p6
    nodes.append(addToKV(libffCtx.kv, 11, params['p6']))


    # 12. p7
    nodes.append(addToKV(libffCtx.kv, 12, params['p7']))


    # 13. p8
    nodes.append(addToKV(libffCtx.kv, 13, params['p8']))


    # 14. p9
    nodes.append(addToKV(libffCtx.kv, 14, params['p9']))


    # 15. fused_nn_conv2d_add
    #kernel 0
    output_size = 3211264
    nodes.append(kaas.bufferSpec('15', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_kernel0', path, shapes, arguments))


    # 16. fused_nn_conv2d_add_add_nn_relu_32
    #kernel 0
    output_size = 3211264
    nodes.append(kaas.bufferSpec('16', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_add_nn_relu_3_kernel0', path, shapes, arguments))


    # 17. p10
    nodes.append(addToKV(libffCtx.kv, 17, params['p10']))


    # 18. p11
    nodes.append(addToKV(libffCtx.kv, 18, params['p11']))


    # 19. fused_nn_conv2d_add_nn_relu_91
    #kernel 0
    output_size = 802816
    nodes.append(kaas.bufferSpec('19', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_9_kernel0', path, shapes, arguments))


    # 20. p12
    nodes.append(addToKV(libffCtx.kv, 20, params['p12']))


    # 21. p13
    nodes.append(addToKV(libffCtx.kv, 21, params['p13']))


    # 22. fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_31
    imm = []
    #kernel 0
    output_size = ?
    imm.append(kaas.bufferSpec('a2', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3_kernel0', path, shapes, arguments))
    #kernel 1
    output_size = ?
    imm.append(kaas.bufferSpec('a3', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3_kernel1', path, shapes, arguments))
    #kernel 2
    output_size = 802816
    nodes.append(kaas.bufferSpec('22', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3_kernel2', path, shapes, arguments))


    # 23. p14
    nodes.append(addToKV(libffCtx.kv, 23, params['p14']))


    # 24. p15
    nodes.append(addToKV(libffCtx.kv, 24, params['p15']))


    # 25. fused_nn_conv2d_add_add_nn_relu_31
    #kernel 0
    output_size = 3211264
    nodes.append(kaas.bufferSpec('25', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_add_nn_relu_3_kernel0', path, shapes, arguments))


    # 26. p16
    nodes.append(addToKV(libffCtx.kv, 26, params['p16']))


    # 27. p17
    nodes.append(addToKV(libffCtx.kv, 27, params['p17']))


    # 28. fused_nn_conv2d_add_nn_relu_9
    #kernel 0
    output_size = 802816
    nodes.append(kaas.bufferSpec('28', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_9_kernel0', path, shapes, arguments))


    # 29. p18
    nodes.append(addToKV(libffCtx.kv, 29, params['p18']))


    # 30. p19
    nodes.append(addToKV(libffCtx.kv, 30, params['p19']))


    # 31. fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3
    imm = []
    #kernel 0
    output_size = ?
    imm.append(kaas.bufferSpec('a4', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3_kernel0', path, shapes, arguments))
    #kernel 1
    output_size = ?
    imm.append(kaas.bufferSpec('a5', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3_kernel1', path, shapes, arguments))
    #kernel 2
    output_size = 802816
    nodes.append(kaas.bufferSpec('31', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_3_kernel2', path, shapes, arguments))


    # 32. p20
    nodes.append(addToKV(libffCtx.kv, 32, params['p20']))


    # 33. p21
    nodes.append(addToKV(libffCtx.kv, 33, params['p21']))


    # 34. fused_nn_conv2d_add_add_nn_relu_3
    #kernel 0
    output_size = 3211264
    nodes.append(kaas.bufferSpec('34', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_add_nn_relu_3_kernel0', path, shapes, arguments))


    # 35. p22
    nodes.append(addToKV(libffCtx.kv, 35, params['p22']))


    # 36. p23
    nodes.append(addToKV(libffCtx.kv, 36, params['p23']))


    # 37. fused_nn_conv2d_add_nn_relu_8
    #kernel 0
    output_size = 1605632
    nodes.append(kaas.bufferSpec('37', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_8_kernel0', path, shapes, arguments))


    # 38. p24
    nodes.append(addToKV(libffCtx.kv, 38, params['p24']))


    # 39. p25
    nodes.append(addToKV(libffCtx.kv, 39, params['p25']))


    # 40. fused_nn_conv2d_add_nn_relu_7
    #kernel 0
    output_size = 401408
    nodes.append(kaas.bufferSpec('40', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_7_kernel0', path, shapes, arguments))


    # 41. p26
    nodes.append(addToKV(libffCtx.kv, 41, params['p26']))


    # 42. p27
    nodes.append(addToKV(libffCtx.kv, 42, params['p27']))


    # 43. p28
    nodes.append(addToKV(libffCtx.kv, 43, params['p28']))


    # 44. p29
    nodes.append(addToKV(libffCtx.kv, 44, params['p29']))


    # 45. fused_nn_conv2d_add_1
    #kernel 0
    output_size = 1605632
    nodes.append(kaas.bufferSpec('45', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_1_kernel0', path, shapes, arguments))


    # 46. fused_nn_conv2d_add_add_nn_relu_23
    #kernel 0
    output_size = 1605632
    nodes.append(kaas.bufferSpec('46', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_add_nn_relu_2_kernel0', path, shapes, arguments))


    # 47. p30
    nodes.append(addToKV(libffCtx.kv, 47, params['p30']))


    # 48. p31
    nodes.append(addToKV(libffCtx.kv, 48, params['p31']))


    # 49. fused_nn_conv2d_add_nn_relu_62
    #kernel 0
    output_size = 401408
    nodes.append(kaas.bufferSpec('49', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_6_kernel0', path, shapes, arguments))


    # 50. p32
    nodes.append(addToKV(libffCtx.kv, 50, params['p32']))


    # 51. p33
    nodes.append(addToKV(libffCtx.kv, 51, params['p33']))


    # 52. fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_22
    imm = []
    #kernel 0
    output_size = ?
    imm.append(kaas.bufferSpec('a6', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2_kernel0', path, shapes, arguments))
    #kernel 1
    output_size = ?
    imm.append(kaas.bufferSpec('a7', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2_kernel1', path, shapes, arguments))
    #kernel 2
    output_size = 401408
    nodes.append(kaas.bufferSpec('52', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2_kernel2', path, shapes, arguments))


    # 53. p34
    nodes.append(addToKV(libffCtx.kv, 53, params['p34']))


    # 54. p35
    nodes.append(addToKV(libffCtx.kv, 54, params['p35']))


    # 55. fused_nn_conv2d_add_add_nn_relu_22
    #kernel 0
    output_size = 1605632
    nodes.append(kaas.bufferSpec('55', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_add_nn_relu_2_kernel0', path, shapes, arguments))


    # 56. p36
    nodes.append(addToKV(libffCtx.kv, 56, params['p36']))


    # 57. p37
    nodes.append(addToKV(libffCtx.kv, 57, params['p37']))


    # 58. fused_nn_conv2d_add_nn_relu_61
    #kernel 0
    output_size = 401408
    nodes.append(kaas.bufferSpec('58', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_6_kernel0', path, shapes, arguments))


    # 59. p38
    nodes.append(addToKV(libffCtx.kv, 59, params['p38']))


    # 60. p39
    nodes.append(addToKV(libffCtx.kv, 60, params['p39']))


    # 61. fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_21
    imm = []
    #kernel 0
    output_size = ?
    imm.append(kaas.bufferSpec('a8', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2_kernel0', path, shapes, arguments))
    #kernel 1
    output_size = ?
    imm.append(kaas.bufferSpec('a9', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2_kernel1', path, shapes, arguments))
    #kernel 2
    output_size = 401408
    nodes.append(kaas.bufferSpec('61', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2_kernel2', path, shapes, arguments))


    # 62. p40
    nodes.append(addToKV(libffCtx.kv, 62, params['p40']))


    # 63. p41
    nodes.append(addToKV(libffCtx.kv, 63, params['p41']))


    # 64. fused_nn_conv2d_add_add_nn_relu_21
    #kernel 0
    output_size = 1605632
    nodes.append(kaas.bufferSpec('64', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_add_nn_relu_2_kernel0', path, shapes, arguments))


    # 65. p42
    nodes.append(addToKV(libffCtx.kv, 65, params['p42']))


    # 66. p43
    nodes.append(addToKV(libffCtx.kv, 66, params['p43']))


    # 67. fused_nn_conv2d_add_nn_relu_6
    #kernel 0
    output_size = 401408
    nodes.append(kaas.bufferSpec('67', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_6_kernel0', path, shapes, arguments))


    # 68. p44
    nodes.append(addToKV(libffCtx.kv, 68, params['p44']))


    # 69. p45
    nodes.append(addToKV(libffCtx.kv, 69, params['p45']))


    # 70. fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2
    imm = []
    #kernel 0
    output_size = ?
    imm.append(kaas.bufferSpec('a10', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2_kernel0', path, shapes, arguments))
    #kernel 1
    output_size = ?
    imm.append(kaas.bufferSpec('a11', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2_kernel1', path, shapes, arguments))
    #kernel 2
    output_size = 401408
    nodes.append(kaas.bufferSpec('70', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_2_kernel2', path, shapes, arguments))


    # 71. p46
    nodes.append(addToKV(libffCtx.kv, 71, params['p46']))


    # 72. p47
    nodes.append(addToKV(libffCtx.kv, 72, params['p47']))


    # 73. fused_nn_conv2d_add_add_nn_relu_2
    #kernel 0
    output_size = 1605632
    nodes.append(kaas.bufferSpec('73', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_add_nn_relu_2_kernel0', path, shapes, arguments))


    # 74. p48
    nodes.append(addToKV(libffCtx.kv, 74, params['p48']))


    # 75. p49
    nodes.append(addToKV(libffCtx.kv, 75, params['p49']))


    # 76. fused_nn_conv2d_add_nn_relu_5
    #kernel 0
    output_size = 802816
    nodes.append(kaas.bufferSpec('76', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_5_kernel0', path, shapes, arguments))


    # 77. p50
    nodes.append(addToKV(libffCtx.kv, 77, params['p50']))


    # 78. p51
    nodes.append(addToKV(libffCtx.kv, 78, params['p51']))


    # 79. fused_nn_conv2d_add_nn_relu_4
    #kernel 0
    output_size = 200704
    nodes.append(kaas.bufferSpec('79', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_4_kernel0', path, shapes, arguments))


    # 80. p52
    nodes.append(addToKV(libffCtx.kv, 80, params['p52']))


    # 81. p53
    nodes.append(addToKV(libffCtx.kv, 81, params['p53']))


    # 82. p54
    nodes.append(addToKV(libffCtx.kv, 82, params['p54']))


    # 83. p55
    nodes.append(addToKV(libffCtx.kv, 83, params['p55']))


    # 84. fused_nn_conv2d_add_2
    #kernel 0
    output_size = 802816
    nodes.append(kaas.bufferSpec('84', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_2_kernel0', path, shapes, arguments))


    # 85. fused_nn_conv2d_add_add_nn_relu_15
    #kernel 0
    output_size = 802816
    nodes.append(kaas.bufferSpec('85', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_add_nn_relu_1_kernel0', path, shapes, arguments))


    # 86. p56
    nodes.append(addToKV(libffCtx.kv, 86, params['p56']))


    # 87. p57
    nodes.append(addToKV(libffCtx.kv, 87, params['p57']))


    # 88. fused_nn_conv2d_add_nn_relu_34
    #kernel 0
    output_size = 200704
    nodes.append(kaas.bufferSpec('88', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_3_kernel0', path, shapes, arguments))


    # 89. p58
    nodes.append(addToKV(libffCtx.kv, 89, params['p58']))


    # 90. p59
    nodes.append(addToKV(libffCtx.kv, 90, params['p59']))


    # 91. fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_14
    imm = []
    #kernel 0
    output_size = ?
    imm.append(kaas.bufferSpec('a12', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel0', path, shapes, arguments))
    #kernel 1
    output_size = ?
    imm.append(kaas.bufferSpec('a13', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel1', path, shapes, arguments))
    #kernel 2
    output_size = 200704
    nodes.append(kaas.bufferSpec('91', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel2', path, shapes, arguments))


    # 92. p60
    nodes.append(addToKV(libffCtx.kv, 92, params['p60']))


    # 93. p61
    nodes.append(addToKV(libffCtx.kv, 93, params['p61']))


    # 94. fused_nn_conv2d_add_add_nn_relu_14
    #kernel 0
    output_size = 802816
    nodes.append(kaas.bufferSpec('94', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_add_nn_relu_1_kernel0', path, shapes, arguments))


    # 95. p62
    nodes.append(addToKV(libffCtx.kv, 95, params['p62']))


    # 96. p63
    nodes.append(addToKV(libffCtx.kv, 96, params['p63']))


    # 97. fused_nn_conv2d_add_nn_relu_33
    #kernel 0
    output_size = 200704
    nodes.append(kaas.bufferSpec('97', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_3_kernel0', path, shapes, arguments))


    # 98. p64
    nodes.append(addToKV(libffCtx.kv, 98, params['p64']))


    # 99. p65
    nodes.append(addToKV(libffCtx.kv, 99, params['p65']))


    # 100. fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_13
    imm = []
    #kernel 0
    output_size = ?
    imm.append(kaas.bufferSpec('a14', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel0', path, shapes, arguments))
    #kernel 1
    output_size = ?
    imm.append(kaas.bufferSpec('a15', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel1', path, shapes, arguments))
    #kernel 2
    output_size = 200704
    nodes.append(kaas.bufferSpec('100', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel2', path, shapes, arguments))


    # 101. p66
    nodes.append(addToKV(libffCtx.kv, 101, params['p66']))


    # 102. p67
    nodes.append(addToKV(libffCtx.kv, 102, params['p67']))


    # 103. fused_nn_conv2d_add_add_nn_relu_13
    #kernel 0
    output_size = 802816
    nodes.append(kaas.bufferSpec('103', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_add_nn_relu_1_kernel0', path, shapes, arguments))


    # 104. p68
    nodes.append(addToKV(libffCtx.kv, 104, params['p68']))


    # 105. p69
    nodes.append(addToKV(libffCtx.kv, 105, params['p69']))


    # 106. fused_nn_conv2d_add_nn_relu_32
    #kernel 0
    output_size = 200704
    nodes.append(kaas.bufferSpec('106', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_3_kernel0', path, shapes, arguments))


    # 107. p70
    nodes.append(addToKV(libffCtx.kv, 107, params['p70']))


    # 108. p71
    nodes.append(addToKV(libffCtx.kv, 108, params['p71']))


    # 109. fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_12
    imm = []
    #kernel 0
    output_size = ?
    imm.append(kaas.bufferSpec('a16', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel0', path, shapes, arguments))
    #kernel 1
    output_size = ?
    imm.append(kaas.bufferSpec('a17', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel1', path, shapes, arguments))
    #kernel 2
    output_size = 200704
    nodes.append(kaas.bufferSpec('109', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel2', path, shapes, arguments))


    # 110. p72
    nodes.append(addToKV(libffCtx.kv, 110, params['p72']))


    # 111. p73
    nodes.append(addToKV(libffCtx.kv, 111, params['p73']))


    # 112. fused_nn_conv2d_add_add_nn_relu_12
    #kernel 0
    output_size = 802816
    nodes.append(kaas.bufferSpec('112', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_add_nn_relu_1_kernel0', path, shapes, arguments))


    # 113. p74
    nodes.append(addToKV(libffCtx.kv, 113, params['p74']))


    # 114. p75
    nodes.append(addToKV(libffCtx.kv, 114, params['p75']))


    # 115. fused_nn_conv2d_add_nn_relu_31
    #kernel 0
    output_size = 200704
    nodes.append(kaas.bufferSpec('115', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_3_kernel0', path, shapes, arguments))


    # 116. p76
    nodes.append(addToKV(libffCtx.kv, 116, params['p76']))


    # 117. p77
    nodes.append(addToKV(libffCtx.kv, 117, params['p77']))


    # 118. fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_11
    imm = []
    #kernel 0
    output_size = ?
    imm.append(kaas.bufferSpec('a18', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel0', path, shapes, arguments))
    #kernel 1
    output_size = ?
    imm.append(kaas.bufferSpec('a19', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel1', path, shapes, arguments))
    #kernel 2
    output_size = 200704
    nodes.append(kaas.bufferSpec('118', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel2', path, shapes, arguments))


    # 119. p78
    nodes.append(addToKV(libffCtx.kv, 119, params['p78']))


    # 120. p79
    nodes.append(addToKV(libffCtx.kv, 120, params['p79']))


    # 121. fused_nn_conv2d_add_add_nn_relu_11
    #kernel 0
    output_size = 802816
    nodes.append(kaas.bufferSpec('121', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_add_nn_relu_1_kernel0', path, shapes, arguments))


    # 122. p80
    nodes.append(addToKV(libffCtx.kv, 122, params['p80']))


    # 123. p81
    nodes.append(addToKV(libffCtx.kv, 123, params['p81']))


    # 124. fused_nn_conv2d_add_nn_relu_3
    #kernel 0
    output_size = 200704
    nodes.append(kaas.bufferSpec('124', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_3_kernel0', path, shapes, arguments))


    # 125. p82
    nodes.append(addToKV(libffCtx.kv, 125, params['p82']))


    # 126. p83
    nodes.append(addToKV(libffCtx.kv, 126, params['p83']))


    # 127. fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1
    imm = []
    #kernel 0
    output_size = ?
    imm.append(kaas.bufferSpec('a20', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel0', path, shapes, arguments))
    #kernel 1
    output_size = ?
    imm.append(kaas.bufferSpec('a21', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel1', path, shapes, arguments))
    #kernel 2
    output_size = 200704
    nodes.append(kaas.bufferSpec('127', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_1_kernel2', path, shapes, arguments))


    # 128. p84
    nodes.append(addToKV(libffCtx.kv, 128, params['p84']))


    # 129. p85
    nodes.append(addToKV(libffCtx.kv, 129, params['p85']))


    # 130. fused_nn_conv2d_add_add_nn_relu_1
    #kernel 0
    output_size = 802816
    nodes.append(kaas.bufferSpec('130', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_add_nn_relu_1_kernel0', path, shapes, arguments))


    # 131. p86
    nodes.append(addToKV(libffCtx.kv, 131, params['p86']))


    # 132. p87
    nodes.append(addToKV(libffCtx.kv, 132, params['p87']))


    # 133. fused_nn_conv2d_add_nn_relu_2
    #kernel 0
    output_size = 401408
    nodes.append(kaas.bufferSpec('133', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_2_kernel0', path, shapes, arguments))


    # 134. p88
    nodes.append(addToKV(libffCtx.kv, 134, params['p88']))


    # 135. p89
    nodes.append(addToKV(libffCtx.kv, 135, params['p89']))


    # 136. fused_nn_conv2d_add_nn_relu_1
    #kernel 0
    output_size = 100352
    nodes.append(kaas.bufferSpec('136', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_1_kernel0', path, shapes, arguments))


    # 137. p90
    nodes.append(addToKV(libffCtx.kv, 137, params['p90']))


    # 138. p91
    nodes.append(addToKV(libffCtx.kv, 138, params['p91']))


    # 139. p92
    nodes.append(addToKV(libffCtx.kv, 139, params['p92']))


    # 140. p93
    nodes.append(addToKV(libffCtx.kv, 140, params['p93']))


    # 141. fused_nn_conv2d_add_3
    #kernel 0
    output_size = 401408
    nodes.append(kaas.bufferSpec('141', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_3_kernel0', path, shapes, arguments))


    # 142. fused_nn_conv2d_add_add_nn_relu2
    #kernel 0
    output_size = 401408
    nodes.append(kaas.bufferSpec('142', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_add_nn_relu_kernel0', path, shapes, arguments))


    # 143. p94
    nodes.append(addToKV(libffCtx.kv, 143, params['p94']))


    # 144. p95
    nodes.append(addToKV(libffCtx.kv, 144, params['p95']))


    # 145. fused_nn_conv2d_add_nn_relu1
    #kernel 0
    output_size = 100352
    nodes.append(kaas.bufferSpec('145', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_kernel0', path, shapes, arguments))


    # 146. p96
    nodes.append(addToKV(libffCtx.kv, 146, params['p96']))


    # 147. p97
    nodes.append(addToKV(libffCtx.kv, 147, params['p97']))


    # 148. fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu1
    imm = []
    #kernel 0
    output_size = ?
    imm.append(kaas.bufferSpec('a22', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_kernel0', path, shapes, arguments))
    #kernel 1
    output_size = ?
    imm.append(kaas.bufferSpec('a23', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_kernel1', path, shapes, arguments))
    #kernel 2
    output_size = 100352
    nodes.append(kaas.bufferSpec('148', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_kernel2', path, shapes, arguments))


    # 149. p98
    nodes.append(addToKV(libffCtx.kv, 149, params['p98']))


    # 150. p99
    nodes.append(addToKV(libffCtx.kv, 150, params['p99']))


    # 151. fused_nn_conv2d_add_add_nn_relu1
    #kernel 0
    output_size = 401408
    nodes.append(kaas.bufferSpec('151', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_add_nn_relu_kernel0', path, shapes, arguments))


    # 152. p100
    nodes.append(addToKV(libffCtx.kv, 152, params['p100']))


    # 153. p101
    nodes.append(addToKV(libffCtx.kv, 153, params['p101']))


    # 154. fused_nn_conv2d_add_nn_relu
    #kernel 0
    output_size = 100352
    nodes.append(kaas.bufferSpec('154', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_nn_relu_kernel0', path, shapes, arguments))


    # 155. p102
    nodes.append(addToKV(libffCtx.kv, 155, params['p102']))


    # 156. p103
    nodes.append(addToKV(libffCtx.kv, 156, params['p103']))


    # 157. fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu
    imm = []
    #kernel 0
    output_size = ?
    imm.append(kaas.bufferSpec('a24', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_kernel0', path, shapes, arguments))
    #kernel 1
    output_size = ?
    imm.append(kaas.bufferSpec('a25', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_kernel1', path, shapes, arguments))
    #kernel 2
    output_size = 100352
    nodes.append(kaas.bufferSpec('157', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_kernel2', path, shapes, arguments))


    # 158. p104
    nodes.append(addToKV(libffCtx.kv, 158, params['p104']))


    # 159. p105
    nodes.append(addToKV(libffCtx.kv, 159, params['p105']))


    # 160. fused_nn_conv2d_add_add_nn_relu
    #kernel 0
    output_size = 401408
    nodes.append(kaas.bufferSpec('160', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_conv2d_add_add_nn_relu_kernel0', path, shapes, arguments))


    # 161. fused_mean
    imm = []
    #kernel 0
    output_size = ?
    imm.append(kaas.bufferSpec('a26', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    #kernel 1
    output_size = 8192
    nodes.append(kaas.bufferSpec('161', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))


    # 162. fused_squeeze_nn_batch_flatten
    #kernel 0
    output_size = 8192
    nodes.append(kaas.bufferSpec('162', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_squeeze_nn_batch_flatten_kernel0', path, shapes, arguments))


    # 163. p106
    nodes.append(addToKV(libffCtx.kv, 163, params['p106']))


    # 164. p107
    nodes.append(addToKV(libffCtx.kv, 164, params['p107']))


    # 165. fused_nn_dense_add
    #kernel 0
    output_size = 4004
    nodes.append(kaas.bufferSpec('165', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_nn_dense_add_kernel0', path, shapes, arguments))


    # 166. fused_argmax
    #kernel 0
    output_size = 4
    nodes.append(kaas.bufferSpec('166', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_argmax_kernel0', path, shapes, arguments))


    # 167. fused_cast
    #kernel 0
    output_size = 8
    nodes.append(kaas.bufferSpec('167', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_cast_kernel0', path, shapes, arguments))


    # 168. fused_max
    #kernel 0
    output_size = 4
    nodes.append(kaas.bufferSpec('168', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))


    # 169. fused_subtract_exp
    #kernel 0
    output_size = 4004
    nodes.append(kaas.bufferSpec('169', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))


    # 170. fused_sum
    #kernel 0
    output_size = 4
    nodes.append(kaas.bufferSpec('170', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))


    # 171. fused_divide
    #kernel 0
    output_size = 4004
    nodes.append(kaas.bufferSpec('171', output_size, const=True, ephemeral=True))
    arguments = ?
    shapes = ?
    kerns.append(makeKern('fused_divide_kernel0', path, shapes, arguments))



    req = kaas.kaasReq(kerns)

    kaasHandle = kaas.getHandle(mode, libffCtx)
    kaasHandle.Invoke(req.toDict())
    