import pathlib
import pickle
import libff as ff
import libff.kv
import libff.invoke
import libff.kaas.kaasFF

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
    params = pickle.load(open("bert/bert_params.pkl", 'rb'))
    return params


def loadParams():
    path = pathlib.Path.cwd() / "bert" / "bert_params.pkl"
    params = pickle.load(open(path, 'rb'))
    return {'p' + str(i): params[i] for i in range(len(params))}


def makeKern(name_func, path, shapes, arguments):
    return kaas.kernelSpec(path, name_func, shapes[0], shapes[1], arguments=arguments)


kv = None
def runReq():
    libffCtx = getCtx(remote=False)
    global kv
    kv = libffCtx.kv

    from infbench import bert
    import numpy as np
    loader = bert.bertLoader(pathlib.Path.cwd() / "bertData")

    inputs = loader.get(0)


    constants = bert.bertModel.getConstants(pathlib.Path.cwd())

    pre_input = constants + [inputs[0]]

    pre_output = bert.bertModel.pre(pre_input)


    graph_inputs = []
    graph_inputs.append(np.frombuffer(pre_output[0], dtype=np.int64))
    graph_inputs.append(np.frombuffer(pre_output[1], dtype=np.int64))
    graph_inputs.append(np.frombuffer(pre_output[2], dtype=np.int64))

    req = createReq(graph_inputs)

    mode = "direct"
    kaasHandle = kaas.kaasFF.getHandle(mode, libffCtx)
    kaasHandle.Invoke(req.toDict())

    c = np.frombuffer(libffCtx.kv.get('12'), dtype=np.float32)
    print(c)


def createReq(inp, no_reuse=True, mode='direct'):
    params = loadParams()
    nodes = []
    kerns = []
    path = pathlib.Path.cwd() / 'bert' / 'code.cubin'
    nodes.append(addToKV(0, inp[0]))
    # storage = dict()
    # storage['0'] = nodes[0]

    # 1. input_mask
    nodes.append(addToKV(1, inp[1]))

    # 2. segment_ids
    nodes.append(addToKV(2, inp[2]))

    # 3. p0
    nodes.append(addToKV(3, params['p0']))

    # 4. p1
    nodes.append(addToKV(4, params['p1']))

    # 5. p2
    nodes.append(addToKV(5, params['p2']))

    # 6. fused_less_add_where_take_add_less_add_where_take_add
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('6', output_size, const=False, ephemeral=False))
    arguments = [(nodes[6], 'o'), (nodes[3], 'i'), (nodes[0], 'i'), (nodes[4], 'i'), (nodes[5], 'i'), (nodes[2], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_less_add_where_take_add_less_add_where_take_add_kernel0', path, shapes, arguments))

    # 7. fused_mean
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a0', output_size, const=False, ephemeral=True))
    arguments = [(nodes[6], 'i'), (imm[0], 'o')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('7', output_size, const=False, ephemeral=True))
    arguments = [(nodes[7], 'o'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 8. fused_subtract48
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('8', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'o'), (nodes[6], 'i'), (nodes[7], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 9. fused_power_mean48
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a1', output_size, const=False, ephemeral=True))
    arguments = [(nodes[8], 'i'), (imm[0], 'o')]
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('9', output_size, const=False, ephemeral=True))
    arguments = [(nodes[9], 'o'), (imm[0], 'i')]
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 10. p3
    nodes.append(addToKV(10, params['p3']))

    # 11. p4
    nodes.append(addToKV(11, params['p4']))

    # 12. fused_add_sqrt_divide_multiply_add47
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('12', output_size, const=False, ephemeral=False))
    arguments = [(nodes[12], 'o'), (nodes[9], 'i'), (nodes[8], 'i'), (nodes[10], 'i'), (nodes[11], 'i')]
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))
    '''
    # 13. reshape_nop
    nodes.append(addToKV(13, nodes[12]))

    # 14. p5
    nodes.append(addToKV(14, params['p5']))

    # 15. fused_nn_batch_matmul_347
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('15', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 16. p6
    nodes.append(addToKV(16, params['p6']))

    # 17. fused_reshape_add_reshape_transpose_reshape23
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('17', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 18. p7
    nodes.append(addToKV(18, params['p7']))

    # 19. fused_nn_batch_matmul_348
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('19', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 20. p8
    nodes.append(addToKV(20, params['p8']))

    # 21. fused_reshape_add_reshape_transpose_reshape_transpose
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('21', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 22. fused_nn_batch_matmul_523
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('22', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 23. fused_expand_dims_expand_dims_cast_subtract_multiply
    # kernel 0
    output_size = 1536
    nodes.append(kaas.bufferSpec('23', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_expand_dims_expand_dims_cast_subtract_multiply_kernel0', path, shapes, arguments))

    # 24. fused_reshape_divide_add23
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('24', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 25. fused_max
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('25', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 26. fused_subtract_exp23
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('26', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 27. fused_sum
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('27', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 28. fused_divide_reshape23
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('28', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 29. p9
    nodes.append(addToKV(29, params['p9']))

    # 30. fused_nn_batch_matmul_349
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('30', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 31. p10
    nodes.append(addToKV(31, params['p10']))

    # 32. fused_reshape_add_reshape_transpose_reshape_transpose_1
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('32', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 33. fused_nn_batch_matmul_423
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('33', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 34. fused_reshape_transpose_reshape23
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('34', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 35. p11
    nodes.append(addToKV(35, params['p11']))

    # 36. fused_nn_batch_matmul_346
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('36', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 37. p12
    nodes.append(addToKV(37, params['p12']))

    # 38. fused_reshape_add_add47
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('38', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 39. fused_mean1
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a2', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('39', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 40. fused_subtract47
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('40', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 41. fused_power_mean47
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a3', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('41', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 42. p13
    nodes.append(addToKV(42, params['p13']))

    # 43. p14
    nodes.append(addToKV(43, params['p14']))

    # 44. fused_add_sqrt_divide_multiply_add46
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('44', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 45. reshape_nop
    nodes.append(addToKV(`45, nodes[44]))

    # 46. p15
    nodes.append(addToKV(46, params['p15']))

    # 47. fused_nn_batch_matmul_223
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('47', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 48. p16
    nodes.append(addToKV(48, params['p16']))

    # 49. fused_reshape_add_multiply_divide_erf_add_multiply_reshape23
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('49', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 50. p17
    nodes.append(addToKV(50, params['p17']))

    # 51. fused_nn_batch_matmul_123
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('51', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 52. p18
    nodes.append(addToKV(52, params['p18']))

    # 53. fused_reshape_add_add46
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('53', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 54. fused_mean2
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a4', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('54', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 55. fused_subtract46
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('55', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 56. fused_power_mean46
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a5', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('56', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 57. p19
    nodes.append(addToKV(57, params['p19']))

    # 58. p20
    nodes.append(addToKV(58, params['p20']))

    # 59. fused_add_sqrt_divide_multiply_add45
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('59', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 60. reshape_nop
    nodes.append(addToKV(`60, nodes[59]))

    # 61. p21
    nodes.append(addToKV(61, params['p21']))

    # 62. fused_nn_batch_matmul_345
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('62', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 63. p22
    nodes.append(addToKV(63, params['p22']))

    # 64. fused_reshape_add_reshape_transpose_reshape22
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('64', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 65. p23
    nodes.append(addToKV(65, params['p23']))

    # 66. fused_nn_batch_matmul_350
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('66', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 67. p24
    nodes.append(addToKV(67, params['p24']))

    # 68. fused_reshape_add_reshape_transpose_reshape_transpose1
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('68', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 69. fused_nn_batch_matmul_522
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('69', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 70. fused_reshape_divide_add22
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('70', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 71. fused_max1
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('71', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 72. fused_subtract_exp22
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('72', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 73. fused_sum1
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('73', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 74. fused_divide_reshape22
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('74', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 75. p25
    nodes.append(addToKV(75, params['p25']))

    # 76. fused_nn_batch_matmul_351
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('76', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 77. p26
    nodes.append(addToKV(77, params['p26']))

    # 78. fused_reshape_add_reshape_transpose_reshape_transpose_11
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('78', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 79. fused_nn_batch_matmul_422
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('79', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 80. fused_reshape_transpose_reshape22
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('80', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 81. p27
    nodes.append(addToKV(81, params['p27']))

    # 82. fused_nn_batch_matmul_344
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('82', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 83. p28
    nodes.append(addToKV(83, params['p28']))

    # 84. fused_reshape_add_add45
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('84', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 85. fused_mean3
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a6', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('85', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 86. fused_subtract45
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('86', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 87. fused_power_mean45
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a7', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('87', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 88. p29
    nodes.append(addToKV(88, params['p29']))

    # 89. p30
    nodes.append(addToKV(89, params['p30']))

    # 90. fused_add_sqrt_divide_multiply_add44
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('90', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 91. reshape_nop
    nodes.append(addToKV(`91, nodes[90]))

    # 92. p31
    nodes.append(addToKV(92, params['p31']))

    # 93. fused_nn_batch_matmul_222
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('93', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 94. p32
    nodes.append(addToKV(94, params['p32']))

    # 95. fused_reshape_add_multiply_divide_erf_add_multiply_reshape22
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('95', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 96. p33
    nodes.append(addToKV(96, params['p33']))

    # 97. fused_nn_batch_matmul_122
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('97', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 98. p34
    nodes.append(addToKV(98, params['p34']))

    # 99. fused_reshape_add_add44
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('99', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 100. fused_mean4
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a8', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('100', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))
    '''
    '''
    # 101. fused_subtract44
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('101', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 102. fused_power_mean44
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a9', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('102', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 103. p35
    nodes.append(addToKV(103, params['p35']))

    # 104. p36
    nodes.append(addToKV(104, params['p36']))

    # 105. fused_add_sqrt_divide_multiply_add43
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('105', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 106. reshape_nop
    nodes.append(addToKV(`106, nodes[105]))

    # 107. p37
    nodes.append(addToKV(107, params['p37']))

    # 108. fused_nn_batch_matmul_343
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('108', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 109. p38
    nodes.append(addToKV(109, params['p38']))

    # 110. fused_reshape_add_reshape_transpose_reshape21
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('110', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 111. p39
    nodes.append(addToKV(111, params['p39']))

    # 112. fused_nn_batch_matmul_352
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('112', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 113. p40
    nodes.append(addToKV(113, params['p40']))

    # 114. fused_reshape_add_reshape_transpose_reshape_transpose2
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('114', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 115. fused_nn_batch_matmul_521
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('115', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 116. fused_reshape_divide_add21
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('116', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 117. fused_max2
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('117', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 118. fused_subtract_exp21
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('118', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 119. fused_sum2
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('119', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 120. fused_divide_reshape21
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('120', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 121. p41
    nodes.append(addToKV(121, params['p41']))

    # 122. fused_nn_batch_matmul_353
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('122', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 123. p42
    nodes.append(addToKV(123, params['p42']))

    # 124. fused_reshape_add_reshape_transpose_reshape_transpose_12
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('124', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 125. fused_nn_batch_matmul_421
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('125', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 126. fused_reshape_transpose_reshape21
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('126', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 127. p43
    nodes.append(addToKV(127, params['p43']))

    # 128. fused_nn_batch_matmul_342
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('128', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 129. p44
    nodes.append(addToKV(129, params['p44']))

    # 130. fused_reshape_add_add43
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('130', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 131. fused_mean5
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a10', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('131', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 132. fused_subtract43
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('132', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 133. fused_power_mean43
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a11', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('133', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 134. p45
    nodes.append(addToKV(134, params['p45']))

    # 135. p46
    nodes.append(addToKV(135, params['p46']))

    # 136. fused_add_sqrt_divide_multiply_add42
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('136', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 137. reshape_nop
    nodes.append(addToKV(`137, nodes[136]))

    # 138. p47
    nodes.append(addToKV(138, params['p47']))

    # 139. fused_nn_batch_matmul_221
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('139', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 140. p48
    nodes.append(addToKV(140, params['p48']))

    # 141. fused_reshape_add_multiply_divide_erf_add_multiply_reshape21
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('141', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 142. p49
    nodes.append(addToKV(142, params['p49']))

    # 143. fused_nn_batch_matmul_121
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('143', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 144. p50
    nodes.append(addToKV(144, params['p50']))

    # 145. fused_reshape_add_add42
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('145', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 146. fused_mean6
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a12', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('146', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 147. fused_subtract42
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('147', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 148. fused_power_mean42
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a13', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('148', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 149. p51
    nodes.append(addToKV(149, params['p51']))

    # 150. p52
    nodes.append(addToKV(150, params['p52']))

    # 151. fused_add_sqrt_divide_multiply_add41
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('151', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 152. reshape_nop
    nodes.append(addToKV(`152, nodes[151]))

    # 153. p53
    nodes.append(addToKV(153, params['p53']))

    # 154. fused_nn_batch_matmul_341
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('154', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 155. p54
    nodes.append(addToKV(155, params['p54']))

    # 156. fused_reshape_add_reshape_transpose_reshape20
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('156', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 157. p55
    nodes.append(addToKV(157, params['p55']))

    # 158. fused_nn_batch_matmul_354
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('158', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 159. p56
    nodes.append(addToKV(159, params['p56']))

    # 160. fused_reshape_add_reshape_transpose_reshape_transpose3
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('160', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 161. fused_nn_batch_matmul_520
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('161', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 162. fused_reshape_divide_add20
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('162', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 163. fused_max3
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('163', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 164. fused_subtract_exp20
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('164', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 165. fused_sum3
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('165', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 166. fused_divide_reshape20
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('166', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 167. p57
    nodes.append(addToKV(167, params['p57']))

    # 168. fused_nn_batch_matmul_355
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('168', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 169. p58
    nodes.append(addToKV(169, params['p58']))

    # 170. fused_reshape_add_reshape_transpose_reshape_transpose_13
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('170', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 171. fused_nn_batch_matmul_420
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('171', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 172. fused_reshape_transpose_reshape20
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('172', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 173. p59
    nodes.append(addToKV(173, params['p59']))

    # 174. fused_nn_batch_matmul_340
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('174', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 175. p60
    nodes.append(addToKV(175, params['p60']))

    # 176. fused_reshape_add_add41
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('176', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 177. fused_mean7
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a14', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('177', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 178. fused_subtract41
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('178', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 179. fused_power_mean41
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a15', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('179', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 180. p61
    nodes.append(addToKV(180, params['p61']))

    # 181. p62
    nodes.append(addToKV(181, params['p62']))

    # 182. fused_add_sqrt_divide_multiply_add40
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('182', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 183. reshape_nop
    nodes.append(addToKV(`183, nodes[182]))

    # 184. p63
    nodes.append(addToKV(184, params['p63']))

    # 185. fused_nn_batch_matmul_220
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('185', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 186. p64
    nodes.append(addToKV(186, params['p64']))

    # 187. fused_reshape_add_multiply_divide_erf_add_multiply_reshape20
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('187', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 188. p65
    nodes.append(addToKV(188, params['p65']))

    # 189. fused_nn_batch_matmul_120
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('189', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 190. p66
    nodes.append(addToKV(190, params['p66']))

    # 191. fused_reshape_add_add40
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('191', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 192. fused_mean8
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a16', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('192', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 193. fused_subtract40
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('193', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 194. fused_power_mean40
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a17', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('194', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 195. p67
    nodes.append(addToKV(195, params['p67']))

    # 196. p68
    nodes.append(addToKV(196, params['p68']))

    # 197. fused_add_sqrt_divide_multiply_add39
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('197', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 198. reshape_nop
    nodes.append(addToKV(`198, nodes[197]))

    # 199. p69
    nodes.append(addToKV(199, params['p69']))

    # 200. fused_nn_batch_matmul_339
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('200', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 201. p70
    nodes.append(addToKV(201, params['p70']))

    # 202. fused_reshape_add_reshape_transpose_reshape19
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('202', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 203. p71
    nodes.append(addToKV(203, params['p71']))

    # 204. fused_nn_batch_matmul_356
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('204', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 205. p72
    nodes.append(addToKV(205, params['p72']))

    # 206. fused_reshape_add_reshape_transpose_reshape_transpose4
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('206', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 207. fused_nn_batch_matmul_519
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('207', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 208. fused_reshape_divide_add19
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('208', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 209. fused_max4
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('209', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 210. fused_subtract_exp19
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('210', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 211. fused_sum4
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('211', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 212. fused_divide_reshape19
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('212', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 213. p73
    nodes.append(addToKV(213, params['p73']))

    # 214. fused_nn_batch_matmul_357
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('214', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 215. p74
    nodes.append(addToKV(215, params['p74']))

    # 216. fused_reshape_add_reshape_transpose_reshape_transpose_14
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('216', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 217. fused_nn_batch_matmul_419
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('217', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 218. fused_reshape_transpose_reshape19
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('218', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 219. p75
    nodes.append(addToKV(219, params['p75']))

    # 220. fused_nn_batch_matmul_338
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('220', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 221. p76
    nodes.append(addToKV(221, params['p76']))

    # 222. fused_reshape_add_add39
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('222', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 223. fused_mean9
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a18', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('223', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 224. fused_subtract39
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('224', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 225. fused_power_mean39
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a19', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('225', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 226. p77
    nodes.append(addToKV(226, params['p77']))

    # 227. p78
    nodes.append(addToKV(227, params['p78']))

    # 228. fused_add_sqrt_divide_multiply_add38
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('228', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 229. reshape_nop
    nodes.append(addToKV(`229, nodes[228]))

    # 230. p79
    nodes.append(addToKV(230, params['p79']))

    # 231. fused_nn_batch_matmul_219
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('231', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 232. p80
    nodes.append(addToKV(232, params['p80']))

    # 233. fused_reshape_add_multiply_divide_erf_add_multiply_reshape19
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('233', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 234. p81
    nodes.append(addToKV(234, params['p81']))

    # 235. fused_nn_batch_matmul_119
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('235', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 236. p82
    nodes.append(addToKV(236, params['p82']))

    # 237. fused_reshape_add_add38
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('237', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 238. fused_mean10
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a20', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('238', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 239. fused_subtract38
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('239', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 240. fused_power_mean38
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a21', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('240', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 241. p83
    nodes.append(addToKV(241, params['p83']))

    # 242. p84
    nodes.append(addToKV(242, params['p84']))

    # 243. fused_add_sqrt_divide_multiply_add37
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('243', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 244. reshape_nop
    nodes.append(addToKV(`244, nodes[243]))

    # 245. p85
    nodes.append(addToKV(245, params['p85']))

    # 246. fused_nn_batch_matmul_337
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('246', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 247. p86
    nodes.append(addToKV(247, params['p86']))

    # 248. fused_reshape_add_reshape_transpose_reshape18
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('248', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 249. p87
    nodes.append(addToKV(249, params['p87']))

    # 250. fused_nn_batch_matmul_358
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('250', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 251. p88
    nodes.append(addToKV(251, params['p88']))

    # 252. fused_reshape_add_reshape_transpose_reshape_transpose5
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('252', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 253. fused_nn_batch_matmul_518
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('253', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 254. fused_reshape_divide_add18
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('254', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 255. fused_max5
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('255', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 256. fused_subtract_exp18
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('256', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 257. fused_sum5
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('257', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 258. fused_divide_reshape18
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('258', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 259. p89
    nodes.append(addToKV(259, params['p89']))

    # 260. fused_nn_batch_matmul_359
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('260', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 261. p90
    nodes.append(addToKV(261, params['p90']))

    # 262. fused_reshape_add_reshape_transpose_reshape_transpose_15
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('262', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 263. fused_nn_batch_matmul_418
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('263', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 264. fused_reshape_transpose_reshape18
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('264', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 265. p91
    nodes.append(addToKV(265, params['p91']))

    # 266. fused_nn_batch_matmul_336
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('266', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 267. p92
    nodes.append(addToKV(267, params['p92']))

    # 268. fused_reshape_add_add37
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('268', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 269. fused_mean11
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a22', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('269', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 270. fused_subtract37
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('270', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 271. fused_power_mean37
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a23', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('271', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 272. p93
    nodes.append(addToKV(272, params['p93']))

    # 273. p94
    nodes.append(addToKV(273, params['p94']))

    # 274. fused_add_sqrt_divide_multiply_add36
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('274', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 275. reshape_nop
    nodes.append(addToKV(`275, nodes[274]))

    # 276. p95
    nodes.append(addToKV(276, params['p95']))

    # 277. fused_nn_batch_matmul_218
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('277', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 278. p96
    nodes.append(addToKV(278, params['p96']))

    # 279. fused_reshape_add_multiply_divide_erf_add_multiply_reshape18
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('279', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 280. p97
    nodes.append(addToKV(280, params['p97']))

    # 281. fused_nn_batch_matmul_118
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('281', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 282. p98
    nodes.append(addToKV(282, params['p98']))

    # 283. fused_reshape_add_add36
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('283', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 284. fused_mean12
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a24', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('284', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 285. fused_subtract36
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('285', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 286. fused_power_mean36
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a25', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('286', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 287. p99
    nodes.append(addToKV(287, params['p99']))

    # 288. p100
    nodes.append(addToKV(288, params['p100']))

    # 289. fused_add_sqrt_divide_multiply_add35
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('289', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 290. reshape_nop
    nodes.append(addToKV(`290, nodes[289]))

    # 291. p101
    nodes.append(addToKV(291, params['p101']))

    # 292. fused_nn_batch_matmul_335
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('292', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 293. p102
    nodes.append(addToKV(293, params['p102']))

    # 294. fused_reshape_add_reshape_transpose_reshape17
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('294', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 295. p103
    nodes.append(addToKV(295, params['p103']))

    # 296. fused_nn_batch_matmul_360
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('296', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 297. p104
    nodes.append(addToKV(297, params['p104']))

    # 298. fused_reshape_add_reshape_transpose_reshape_transpose6
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('298', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 299. fused_nn_batch_matmul_517
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('299', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 300. fused_reshape_divide_add17
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('300', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 301. fused_max6
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('301', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 302. fused_subtract_exp17
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('302', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 303. fused_sum6
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('303', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 304. fused_divide_reshape17
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('304', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 305. p105
    nodes.append(addToKV(305, params['p105']))

    # 306. fused_nn_batch_matmul_361
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('306', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 307. p106
    nodes.append(addToKV(307, params['p106']))

    # 308. fused_reshape_add_reshape_transpose_reshape_transpose_16
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('308', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 309. fused_nn_batch_matmul_417
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('309', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 310. fused_reshape_transpose_reshape17
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('310', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 311. p107
    nodes.append(addToKV(311, params['p107']))

    # 312. fused_nn_batch_matmul_334
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('312', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 313. p108
    nodes.append(addToKV(313, params['p108']))

    # 314. fused_reshape_add_add35
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('314', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 315. fused_mean13
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a26', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('315', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 316. fused_subtract35
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('316', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 317. fused_power_mean35
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a27', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('317', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 318. p109
    nodes.append(addToKV(318, params['p109']))

    # 319. p110
    nodes.append(addToKV(319, params['p110']))

    # 320. fused_add_sqrt_divide_multiply_add34
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('320', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 321. reshape_nop
    nodes.append(addToKV(`321, nodes[320]))

    # 322. p111
    nodes.append(addToKV(322, params['p111']))

    # 323. fused_nn_batch_matmul_217
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('323', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 324. p112
    nodes.append(addToKV(324, params['p112']))

    # 325. fused_reshape_add_multiply_divide_erf_add_multiply_reshape17
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('325', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 326. p113
    nodes.append(addToKV(326, params['p113']))

    # 327. fused_nn_batch_matmul_117
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('327', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 328. p114
    nodes.append(addToKV(328, params['p114']))

    # 329. fused_reshape_add_add34
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('329', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 330. fused_mean14
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a28', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('330', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 331. fused_subtract34
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('331', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 332. fused_power_mean34
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a29', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('332', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 333. p115
    nodes.append(addToKV(333, params['p115']))

    # 334. p116
    nodes.append(addToKV(334, params['p116']))

    # 335. fused_add_sqrt_divide_multiply_add33
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('335', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 336. reshape_nop
    nodes.append(addToKV(`336, nodes[335]))

    # 337. p117
    nodes.append(addToKV(337, params['p117']))

    # 338. fused_nn_batch_matmul_333
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('338', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 339. p118
    nodes.append(addToKV(339, params['p118']))

    # 340. fused_reshape_add_reshape_transpose_reshape16
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('340', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 341. p119
    nodes.append(addToKV(341, params['p119']))

    # 342. fused_nn_batch_matmul_362
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('342', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 343. p120
    nodes.append(addToKV(343, params['p120']))

    # 344. fused_reshape_add_reshape_transpose_reshape_transpose7
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('344', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 345. fused_nn_batch_matmul_516
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('345', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 346. fused_reshape_divide_add16
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('346', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 347. fused_max7
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('347', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 348. fused_subtract_exp16
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('348', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 349. fused_sum7
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('349', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 350. fused_divide_reshape16
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('350', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 351. p121
    nodes.append(addToKV(351, params['p121']))

    # 352. fused_nn_batch_matmul_363
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('352', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 353. p122
    nodes.append(addToKV(353, params['p122']))

    # 354. fused_reshape_add_reshape_transpose_reshape_transpose_17
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('354', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 355. fused_nn_batch_matmul_416
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('355', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 356. fused_reshape_transpose_reshape16
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('356', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 357. p123
    nodes.append(addToKV(357, params['p123']))

    # 358. fused_nn_batch_matmul_332
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('358', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 359. p124
    nodes.append(addToKV(359, params['p124']))

    # 360. fused_reshape_add_add33
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('360', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 361. fused_mean15
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a30', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('361', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 362. fused_subtract33
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('362', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 363. fused_power_mean33
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a31', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('363', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 364. p125
    nodes.append(addToKV(364, params['p125']))

    # 365. p126
    nodes.append(addToKV(365, params['p126']))

    # 366. fused_add_sqrt_divide_multiply_add32
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('366', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 367. reshape_nop
    nodes.append(addToKV(`367, nodes[366]))

    # 368. p127
    nodes.append(addToKV(368, params['p127']))

    # 369. fused_nn_batch_matmul_216
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('369', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 370. p128
    nodes.append(addToKV(370, params['p128']))

    # 371. fused_reshape_add_multiply_divide_erf_add_multiply_reshape16
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('371', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 372. p129
    nodes.append(addToKV(372, params['p129']))

    # 373. fused_nn_batch_matmul_116
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('373', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 374. p130
    nodes.append(addToKV(374, params['p130']))

    # 375. fused_reshape_add_add32
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('375', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 376. fused_mean16
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a32', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('376', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 377. fused_subtract32
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('377', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 378. fused_power_mean32
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a33', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('378', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 379. p131
    nodes.append(addToKV(379, params['p131']))

    # 380. p132
    nodes.append(addToKV(380, params['p132']))

    # 381. fused_add_sqrt_divide_multiply_add31
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('381', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 382. reshape_nop
    nodes.append(addToKV(`382, nodes[381]))

    # 383. p133
    nodes.append(addToKV(383, params['p133']))

    # 384. fused_nn_batch_matmul_331
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('384', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 385. p134
    nodes.append(addToKV(385, params['p134']))

    # 386. fused_reshape_add_reshape_transpose_reshape15
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('386', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 387. p135
    nodes.append(addToKV(387, params['p135']))

    # 388. fused_nn_batch_matmul_364
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('388', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 389. p136
    nodes.append(addToKV(389, params['p136']))

    # 390. fused_reshape_add_reshape_transpose_reshape_transpose8
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('390', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 391. fused_nn_batch_matmul_515
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('391', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 392. fused_reshape_divide_add15
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('392', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 393. fused_max8
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('393', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 394. fused_subtract_exp15
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('394', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 395. fused_sum8
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('395', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 396. fused_divide_reshape15
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('396', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 397. p137
    nodes.append(addToKV(397, params['p137']))

    # 398. fused_nn_batch_matmul_365
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('398', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 399. p138
    nodes.append(addToKV(399, params['p138']))

    # 400. fused_reshape_add_reshape_transpose_reshape_transpose_18
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('400', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 401. fused_nn_batch_matmul_415
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('401', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 402. fused_reshape_transpose_reshape15
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('402', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 403. p139
    nodes.append(addToKV(403, params['p139']))

    # 404. fused_nn_batch_matmul_330
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('404', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 405. p140
    nodes.append(addToKV(405, params['p140']))

    # 406. fused_reshape_add_add31
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('406', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 407. fused_mean17
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a34', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('407', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 408. fused_subtract31
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('408', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 409. fused_power_mean31
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a35', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('409', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 410. p141
    nodes.append(addToKV(410, params['p141']))

    # 411. p142
    nodes.append(addToKV(411, params['p142']))

    # 412. fused_add_sqrt_divide_multiply_add30
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('412', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 413. reshape_nop
    nodes.append(addToKV(`413, nodes[412]))

    # 414. p143
    nodes.append(addToKV(414, params['p143']))

    # 415. fused_nn_batch_matmul_215
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('415', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 416. p144
    nodes.append(addToKV(416, params['p144']))

    # 417. fused_reshape_add_multiply_divide_erf_add_multiply_reshape15
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('417', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 418. p145
    nodes.append(addToKV(418, params['p145']))

    # 419. fused_nn_batch_matmul_115
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('419', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 420. p146
    nodes.append(addToKV(420, params['p146']))

    # 421. fused_reshape_add_add30
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('421', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 422. fused_mean18
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a36', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('422', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 423. fused_subtract30
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('423', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 424. fused_power_mean30
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a37', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('424', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 425. p147
    nodes.append(addToKV(425, params['p147']))

    # 426. p148
    nodes.append(addToKV(426, params['p148']))

    # 427. fused_add_sqrt_divide_multiply_add29
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('427', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 428. reshape_nop
    nodes.append(addToKV(`428, nodes[427]))

    # 429. p149
    nodes.append(addToKV(429, params['p149']))

    # 430. fused_nn_batch_matmul_329
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('430', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 431. p150
    nodes.append(addToKV(431, params['p150']))

    # 432. fused_reshape_add_reshape_transpose_reshape14
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('432', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 433. p151
    nodes.append(addToKV(433, params['p151']))

    # 434. fused_nn_batch_matmul_366
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('434', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 435. p152
    nodes.append(addToKV(435, params['p152']))

    # 436. fused_reshape_add_reshape_transpose_reshape_transpose9
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('436', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 437. fused_nn_batch_matmul_514
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('437', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 438. fused_reshape_divide_add14
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('438', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 439. fused_max9
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('439', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 440. fused_subtract_exp14
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('440', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 441. fused_sum9
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('441', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 442. fused_divide_reshape14
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('442', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 443. p153
    nodes.append(addToKV(443, params['p153']))

    # 444. fused_nn_batch_matmul_367
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('444', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 445. p154
    nodes.append(addToKV(445, params['p154']))

    # 446. fused_reshape_add_reshape_transpose_reshape_transpose_19
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('446', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 447. fused_nn_batch_matmul_414
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('447', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 448. fused_reshape_transpose_reshape14
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('448', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 449. p155
    nodes.append(addToKV(449, params['p155']))

    # 450. fused_nn_batch_matmul_328
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('450', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 451. p156
    nodes.append(addToKV(451, params['p156']))

    # 452. fused_reshape_add_add29
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('452', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 453. fused_mean19
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a38', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('453', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 454. fused_subtract29
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('454', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 455. fused_power_mean29
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a39', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('455', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 456. p157
    nodes.append(addToKV(456, params['p157']))

    # 457. p158
    nodes.append(addToKV(457, params['p158']))

    # 458. fused_add_sqrt_divide_multiply_add28
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('458', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 459. reshape_nop
    nodes.append(addToKV(`459, nodes[458]))

    # 460. p159
    nodes.append(addToKV(460, params['p159']))

    # 461. fused_nn_batch_matmul_214
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('461', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 462. p160
    nodes.append(addToKV(462, params['p160']))

    # 463. fused_reshape_add_multiply_divide_erf_add_multiply_reshape14
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('463', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 464. p161
    nodes.append(addToKV(464, params['p161']))

    # 465. fused_nn_batch_matmul_114
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('465', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 466. p162
    nodes.append(addToKV(466, params['p162']))

    # 467. fused_reshape_add_add28
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('467', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 468. fused_mean20
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a40', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('468', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 469. fused_subtract28
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('469', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 470. fused_power_mean28
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a41', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('470', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 471. p163
    nodes.append(addToKV(471, params['p163']))

    # 472. p164
    nodes.append(addToKV(472, params['p164']))

    # 473. fused_add_sqrt_divide_multiply_add27
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('473', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 474. reshape_nop
    nodes.append(addToKV(`474, nodes[473]))

    # 475. p165
    nodes.append(addToKV(475, params['p165']))

    # 476. fused_nn_batch_matmul_327
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('476', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 477. p166
    nodes.append(addToKV(477, params['p166']))

    # 478. fused_reshape_add_reshape_transpose_reshape13
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('478', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 479. p167
    nodes.append(addToKV(479, params['p167']))

    # 480. fused_nn_batch_matmul_368
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('480', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 481. p168
    nodes.append(addToKV(481, params['p168']))

    # 482. fused_reshape_add_reshape_transpose_reshape_transpose10
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('482', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 483. fused_nn_batch_matmul_513
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('483', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 484. fused_reshape_divide_add13
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('484', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 485. fused_max10
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('485', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 486. fused_subtract_exp13
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('486', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 487. fused_sum10
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('487', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 488. fused_divide_reshape13
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('488', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 489. p169
    nodes.append(addToKV(489, params['p169']))

    # 490. fused_nn_batch_matmul_369
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('490', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 491. p170
    nodes.append(addToKV(491, params['p170']))

    # 492. fused_reshape_add_reshape_transpose_reshape_transpose_110
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('492', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 493. fused_nn_batch_matmul_413
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('493', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 494. fused_reshape_transpose_reshape13
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('494', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 495. p171
    nodes.append(addToKV(495, params['p171']))

    # 496. fused_nn_batch_matmul_326
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('496', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 497. p172
    nodes.append(addToKV(497, params['p172']))

    # 498. fused_reshape_add_add27
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('498', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 499. fused_mean21
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a42', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('499', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 500. fused_subtract27
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('500', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 501. fused_power_mean27
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a43', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('501', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 502. p173
    nodes.append(addToKV(502, params['p173']))

    # 503. p174
    nodes.append(addToKV(503, params['p174']))

    # 504. fused_add_sqrt_divide_multiply_add26
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('504', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 505. reshape_nop
    nodes.append(addToKV(`505, nodes[504]))

    # 506. p175
    nodes.append(addToKV(506, params['p175']))

    # 507. fused_nn_batch_matmul_213
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('507', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 508. p176
    nodes.append(addToKV(508, params['p176']))

    # 509. fused_reshape_add_multiply_divide_erf_add_multiply_reshape13
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('509', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 510. p177
    nodes.append(addToKV(510, params['p177']))

    # 511. fused_nn_batch_matmul_113
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('511', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 512. p178
    nodes.append(addToKV(512, params['p178']))

    # 513. fused_reshape_add_add26
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('513', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 514. fused_mean22
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a44', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('514', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 515. fused_subtract26
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('515', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 516. fused_power_mean26
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a45', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('516', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 517. p179
    nodes.append(addToKV(517, params['p179']))

    # 518. p180
    nodes.append(addToKV(518, params['p180']))

    # 519. fused_add_sqrt_divide_multiply_add25
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('519', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 520. reshape_nop
    nodes.append(addToKV(`520, nodes[519]))

    # 521. p181
    nodes.append(addToKV(521, params['p181']))

    # 522. fused_nn_batch_matmul_325
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('522', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 523. p182
    nodes.append(addToKV(523, params['p182']))

    # 524. fused_reshape_add_reshape_transpose_reshape12
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('524', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 525. p183
    nodes.append(addToKV(525, params['p183']))

    # 526. fused_nn_batch_matmul_370
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('526', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 527. p184
    nodes.append(addToKV(527, params['p184']))

    # 528. fused_reshape_add_reshape_transpose_reshape_transpose11
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('528', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 529. fused_nn_batch_matmul_512
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('529', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 530. fused_reshape_divide_add12
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('530', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 531. fused_max11
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('531', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 532. fused_subtract_exp12
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('532', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 533. fused_sum11
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('533', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 534. fused_divide_reshape12
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('534', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 535. p185
    nodes.append(addToKV(535, params['p185']))

    # 536. fused_nn_batch_matmul_371
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('536', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 537. p186
    nodes.append(addToKV(537, params['p186']))

    # 538. fused_reshape_add_reshape_transpose_reshape_transpose_111
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('538', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 539. fused_nn_batch_matmul_412
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('539', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 540. fused_reshape_transpose_reshape12
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('540', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 541. p187
    nodes.append(addToKV(541, params['p187']))

    # 542. fused_nn_batch_matmul_324
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('542', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 543. p188
    nodes.append(addToKV(543, params['p188']))

    # 544. fused_reshape_add_add25
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('544', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 545. fused_mean23
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a46', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('545', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 546. fused_subtract25
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('546', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 547. fused_power_mean25
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a47', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('547', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 548. p189
    nodes.append(addToKV(548, params['p189']))

    # 549. p190
    nodes.append(addToKV(549, params['p190']))

    # 550. fused_add_sqrt_divide_multiply_add24
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('550', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 551. reshape_nop
    nodes.append(addToKV(`551, nodes[550]))

    # 552. p191
    nodes.append(addToKV(552, params['p191']))

    # 553. fused_nn_batch_matmul_212
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('553', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 554. p192
    nodes.append(addToKV(554, params['p192']))

    # 555. fused_reshape_add_multiply_divide_erf_add_multiply_reshape12
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('555', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 556. p193
    nodes.append(addToKV(556, params['p193']))

    # 557. fused_nn_batch_matmul_112
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('557', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 558. p194
    nodes.append(addToKV(558, params['p194']))

    # 559. fused_reshape_add_add24
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('559', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 560. fused_mean24
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a48', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('560', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 561. fused_subtract24
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('561', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 562. fused_power_mean24
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a49', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('562', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 563. p195
    nodes.append(addToKV(563, params['p195']))

    # 564. p196
    nodes.append(addToKV(564, params['p196']))

    # 565. fused_add_sqrt_divide_multiply_add23
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('565', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 566. reshape_nop
    nodes.append(addToKV(`566, nodes[565]))

    # 567. p197
    nodes.append(addToKV(567, params['p197']))

    # 568. fused_nn_batch_matmul_323
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('568', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 569. p198
    nodes.append(addToKV(569, params['p198']))

    # 570. fused_reshape_add_reshape_transpose_reshape11
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('570', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 571. p199
    nodes.append(addToKV(571, params['p199']))

    # 572. fused_nn_batch_matmul_372
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('572', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 573. p200
    nodes.append(addToKV(573, params['p200']))

    # 574. fused_reshape_add_reshape_transpose_reshape_transpose12
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('574', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 575. fused_nn_batch_matmul_511
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('575', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 576. fused_reshape_divide_add11
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('576', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 577. fused_max12
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('577', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 578. fused_subtract_exp11
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('578', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 579. fused_sum12
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('579', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 580. fused_divide_reshape11
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('580', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 581. p201
    nodes.append(addToKV(581, params['p201']))

    # 582. fused_nn_batch_matmul_373
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('582', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 583. p202
    nodes.append(addToKV(583, params['p202']))

    # 584. fused_reshape_add_reshape_transpose_reshape_transpose_112
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('584', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 585. fused_nn_batch_matmul_411
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('585', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 586. fused_reshape_transpose_reshape11
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('586', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 587. p203
    nodes.append(addToKV(587, params['p203']))

    # 588. fused_nn_batch_matmul_322
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('588', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 589. p204
    nodes.append(addToKV(589, params['p204']))

    # 590. fused_reshape_add_add23
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('590', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 591. fused_mean25
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a50', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('591', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 592. fused_subtract23
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('592', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 593. fused_power_mean23
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a51', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('593', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 594. p205
    nodes.append(addToKV(594, params['p205']))

    # 595. p206
    nodes.append(addToKV(595, params['p206']))

    # 596. fused_add_sqrt_divide_multiply_add22
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('596', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 597. reshape_nop
    nodes.append(addToKV(`597, nodes[596]))

    # 598. p207
    nodes.append(addToKV(598, params['p207']))

    # 599. fused_nn_batch_matmul_211
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('599', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 600. p208
    nodes.append(addToKV(600, params['p208']))

    # 601. fused_reshape_add_multiply_divide_erf_add_multiply_reshape11
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('601', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 602. p209
    nodes.append(addToKV(602, params['p209']))

    # 603. fused_nn_batch_matmul_111
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('603', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 604. p210
    nodes.append(addToKV(604, params['p210']))

    # 605. fused_reshape_add_add22
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('605', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 606. fused_mean26
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a52', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('606', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 607. fused_subtract22
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('607', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 608. fused_power_mean22
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a53', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('608', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 609. p211
    nodes.append(addToKV(609, params['p211']))

    # 610. p212
    nodes.append(addToKV(610, params['p212']))

    # 611. fused_add_sqrt_divide_multiply_add21
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('611', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 612. reshape_nop
    nodes.append(addToKV(`612, nodes[611]))

    # 613. p213
    nodes.append(addToKV(613, params['p213']))

    # 614. fused_nn_batch_matmul_321
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('614', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 615. p214
    nodes.append(addToKV(615, params['p214']))

    # 616. fused_reshape_add_reshape_transpose_reshape10
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('616', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 617. p215
    nodes.append(addToKV(617, params['p215']))

    # 618. fused_nn_batch_matmul_374
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('618', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 619. p216
    nodes.append(addToKV(619, params['p216']))

    # 620. fused_reshape_add_reshape_transpose_reshape_transpose13
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('620', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 621. fused_nn_batch_matmul_510
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('621', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 622. fused_reshape_divide_add10
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('622', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 623. fused_max13
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('623', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 624. fused_subtract_exp10
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('624', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 625. fused_sum13
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('625', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 626. fused_divide_reshape10
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('626', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 627. p217
    nodes.append(addToKV(627, params['p217']))

    # 628. fused_nn_batch_matmul_375
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('628', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 629. p218
    nodes.append(addToKV(629, params['p218']))

    # 630. fused_reshape_add_reshape_transpose_reshape_transpose_113
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('630', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 631. fused_nn_batch_matmul_410
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('631', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 632. fused_reshape_transpose_reshape10
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('632', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 633. p219
    nodes.append(addToKV(633, params['p219']))

    # 634. fused_nn_batch_matmul_320
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('634', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 635. p220
    nodes.append(addToKV(635, params['p220']))

    # 636. fused_reshape_add_add21
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('636', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 637. fused_mean27
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a54', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('637', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 638. fused_subtract21
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('638', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 639. fused_power_mean21
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a55', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('639', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 640. p221
    nodes.append(addToKV(640, params['p221']))

    # 641. p222
    nodes.append(addToKV(641, params['p222']))

    # 642. fused_add_sqrt_divide_multiply_add20
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('642', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 643. reshape_nop
    nodes.append(addToKV(`643, nodes[642]))

    # 644. p223
    nodes.append(addToKV(644, params['p223']))

    # 645. fused_nn_batch_matmul_210
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('645', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 646. p224
    nodes.append(addToKV(646, params['p224']))

    # 647. fused_reshape_add_multiply_divide_erf_add_multiply_reshape10
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('647', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 648. p225
    nodes.append(addToKV(648, params['p225']))

    # 649. fused_nn_batch_matmul_110
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('649', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 650. p226
    nodes.append(addToKV(650, params['p226']))

    # 651. fused_reshape_add_add20
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('651', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 652. fused_mean28
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a56', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('652', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 653. fused_subtract20
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('653', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 654. fused_power_mean20
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a57', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('654', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 655. p227
    nodes.append(addToKV(655, params['p227']))

    # 656. p228
    nodes.append(addToKV(656, params['p228']))

    # 657. fused_add_sqrt_divide_multiply_add19
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('657', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 658. reshape_nop
    nodes.append(addToKV(`658, nodes[657]))

    # 659. p229
    nodes.append(addToKV(659, params['p229']))

    # 660. fused_nn_batch_matmul_319
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('660', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 661. p230
    nodes.append(addToKV(661, params['p230']))

    # 662. fused_reshape_add_reshape_transpose_reshape9
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('662', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 663. p231
    nodes.append(addToKV(663, params['p231']))

    # 664. fused_nn_batch_matmul_376
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('664', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 665. p232
    nodes.append(addToKV(665, params['p232']))

    # 666. fused_reshape_add_reshape_transpose_reshape_transpose14
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('666', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 667. fused_nn_batch_matmul_59
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('667', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 668. fused_reshape_divide_add9
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('668', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 669. fused_max14
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('669', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 670. fused_subtract_exp9
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('670', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 671. fused_sum14
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('671', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 672. fused_divide_reshape9
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('672', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 673. p233
    nodes.append(addToKV(673, params['p233']))

    # 674. fused_nn_batch_matmul_377
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('674', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 675. p234
    nodes.append(addToKV(675, params['p234']))

    # 676. fused_reshape_add_reshape_transpose_reshape_transpose_114
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('676', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 677. fused_nn_batch_matmul_49
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('677', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 678. fused_reshape_transpose_reshape9
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('678', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 679. p235
    nodes.append(addToKV(679, params['p235']))

    # 680. fused_nn_batch_matmul_318
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('680', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 681. p236
    nodes.append(addToKV(681, params['p236']))

    # 682. fused_reshape_add_add19
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('682', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 683. fused_mean29
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a58', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('683', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 684. fused_subtract19
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('684', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 685. fused_power_mean19
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a59', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('685', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 686. p237
    nodes.append(addToKV(686, params['p237']))

    # 687. p238
    nodes.append(addToKV(687, params['p238']))

    # 688. fused_add_sqrt_divide_multiply_add18
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('688', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 689. reshape_nop
    nodes.append(addToKV(`689, nodes[688]))

    # 690. p239
    nodes.append(addToKV(690, params['p239']))

    # 691. fused_nn_batch_matmul_29
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('691', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 692. p240
    nodes.append(addToKV(692, params['p240']))

    # 693. fused_reshape_add_multiply_divide_erf_add_multiply_reshape9
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('693', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 694. p241
    nodes.append(addToKV(694, params['p241']))

    # 695. fused_nn_batch_matmul_19
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('695', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 696. p242
    nodes.append(addToKV(696, params['p242']))

    # 697. fused_reshape_add_add18
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('697', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 698. fused_mean30
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a60', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('698', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 699. fused_subtract18
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('699', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 700. fused_power_mean18
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a61', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('700', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 701. p243
    nodes.append(addToKV(701, params['p243']))

    # 702. p244
    nodes.append(addToKV(702, params['p244']))

    # 703. fused_add_sqrt_divide_multiply_add17
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('703', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 704. reshape_nop
    nodes.append(addToKV(`704, nodes[703]))

    # 705. p245
    nodes.append(addToKV(705, params['p245']))

    # 706. fused_nn_batch_matmul_317
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('706', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 707. p246
    nodes.append(addToKV(707, params['p246']))

    # 708. fused_reshape_add_reshape_transpose_reshape8
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('708', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 709. p247
    nodes.append(addToKV(709, params['p247']))

    # 710. fused_nn_batch_matmul_378
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('710', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 711. p248
    nodes.append(addToKV(711, params['p248']))

    # 712. fused_reshape_add_reshape_transpose_reshape_transpose15
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('712', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 713. fused_nn_batch_matmul_58
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('713', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 714. fused_reshape_divide_add8
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('714', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 715. fused_max15
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('715', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 716. fused_subtract_exp8
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('716', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 717. fused_sum15
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('717', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 718. fused_divide_reshape8
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('718', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 719. p249
    nodes.append(addToKV(719, params['p249']))

    # 720. fused_nn_batch_matmul_379
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('720', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 721. p250
    nodes.append(addToKV(721, params['p250']))

    # 722. fused_reshape_add_reshape_transpose_reshape_transpose_115
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('722', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 723. fused_nn_batch_matmul_48
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('723', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 724. fused_reshape_transpose_reshape8
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('724', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 725. p251
    nodes.append(addToKV(725, params['p251']))

    # 726. fused_nn_batch_matmul_316
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('726', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 727. p252
    nodes.append(addToKV(727, params['p252']))

    # 728. fused_reshape_add_add17
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('728', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 729. fused_mean31
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a62', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('729', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 730. fused_subtract17
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('730', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 731. fused_power_mean17
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a63', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('731', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 732. p253
    nodes.append(addToKV(732, params['p253']))

    # 733. p254
    nodes.append(addToKV(733, params['p254']))

    # 734. fused_add_sqrt_divide_multiply_add16
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('734', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 735. reshape_nop
    nodes.append(addToKV(`735, nodes[734]))

    # 736. p255
    nodes.append(addToKV(736, params['p255']))

    # 737. fused_nn_batch_matmul_28
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('737', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 738. p256
    nodes.append(addToKV(738, params['p256']))

    # 739. fused_reshape_add_multiply_divide_erf_add_multiply_reshape8
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('739', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 740. p257
    nodes.append(addToKV(740, params['p257']))

    # 741. fused_nn_batch_matmul_18
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('741', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 742. p258
    nodes.append(addToKV(742, params['p258']))

    # 743. fused_reshape_add_add16
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('743', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 744. fused_mean32
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a64', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('744', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 745. fused_subtract16
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('745', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 746. fused_power_mean16
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a65', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('746', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 747. p259
    nodes.append(addToKV(747, params['p259']))

    # 748. p260
    nodes.append(addToKV(748, params['p260']))

    # 749. fused_add_sqrt_divide_multiply_add15
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('749', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 750. reshape_nop
    nodes.append(addToKV(`750, nodes[749]))

    # 751. p261
    nodes.append(addToKV(751, params['p261']))

    # 752. fused_nn_batch_matmul_315
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('752', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 753. p262
    nodes.append(addToKV(753, params['p262']))

    # 754. fused_reshape_add_reshape_transpose_reshape7
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('754', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 755. p263
    nodes.append(addToKV(755, params['p263']))

    # 756. fused_nn_batch_matmul_380
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('756', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 757. p264
    nodes.append(addToKV(757, params['p264']))

    # 758. fused_reshape_add_reshape_transpose_reshape_transpose16
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('758', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 759. fused_nn_batch_matmul_57
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('759', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 760. fused_reshape_divide_add7
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('760', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 761. fused_max16
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('761', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 762. fused_subtract_exp7
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('762', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 763. fused_sum16
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('763', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 764. fused_divide_reshape7
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('764', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 765. p265
    nodes.append(addToKV(765, params['p265']))

    # 766. fused_nn_batch_matmul_381
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('766', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 767. p266
    nodes.append(addToKV(767, params['p266']))

    # 768. fused_reshape_add_reshape_transpose_reshape_transpose_116
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('768', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 769. fused_nn_batch_matmul_47
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('769', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 770. fused_reshape_transpose_reshape7
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('770', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 771. p267
    nodes.append(addToKV(771, params['p267']))

    # 772. fused_nn_batch_matmul_314
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('772', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 773. p268
    nodes.append(addToKV(773, params['p268']))

    # 774. fused_reshape_add_add15
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('774', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 775. fused_mean33
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a66', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('775', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 776. fused_subtract15
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('776', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 777. fused_power_mean15
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a67', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('777', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 778. p269
    nodes.append(addToKV(778, params['p269']))

    # 779. p270
    nodes.append(addToKV(779, params['p270']))

    # 780. fused_add_sqrt_divide_multiply_add14
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('780', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 781. reshape_nop
    nodes.append(addToKV(`781, nodes[780]))

    # 782. p271
    nodes.append(addToKV(782, params['p271']))

    # 783. fused_nn_batch_matmul_27
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('783', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 784. p272
    nodes.append(addToKV(784, params['p272']))

    # 785. fused_reshape_add_multiply_divide_erf_add_multiply_reshape7
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('785', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 786. p273
    nodes.append(addToKV(786, params['p273']))

    # 787. fused_nn_batch_matmul_17
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('787', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 788. p274
    nodes.append(addToKV(788, params['p274']))

    # 789. fused_reshape_add_add14
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('789', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 790. fused_mean34
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a68', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('790', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 791. fused_subtract14
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('791', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 792. fused_power_mean14
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a69', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('792', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 793. p275
    nodes.append(addToKV(793, params['p275']))

    # 794. p276
    nodes.append(addToKV(794, params['p276']))

    # 795. fused_add_sqrt_divide_multiply_add13
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('795', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 796. reshape_nop
    nodes.append(addToKV(`796, nodes[795]))

    # 797. p277
    nodes.append(addToKV(797, params['p277']))

    # 798. fused_nn_batch_matmul_313
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('798', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 799. p278
    nodes.append(addToKV(799, params['p278']))

    # 800. fused_reshape_add_reshape_transpose_reshape6
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('800', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 801. p279
    nodes.append(addToKV(801, params['p279']))

    # 802. fused_nn_batch_matmul_382
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('802', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 803. p280
    nodes.append(addToKV(803, params['p280']))

    # 804. fused_reshape_add_reshape_transpose_reshape_transpose17
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('804', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 805. fused_nn_batch_matmul_56
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('805', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 806. fused_reshape_divide_add6
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('806', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 807. fused_max17
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('807', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 808. fused_subtract_exp6
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('808', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 809. fused_sum17
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('809', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 810. fused_divide_reshape6
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('810', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 811. p281
    nodes.append(addToKV(811, params['p281']))

    # 812. fused_nn_batch_matmul_383
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('812', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 813. p282
    nodes.append(addToKV(813, params['p282']))

    # 814. fused_reshape_add_reshape_transpose_reshape_transpose_117
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('814', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 815. fused_nn_batch_matmul_46
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('815', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 816. fused_reshape_transpose_reshape6
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('816', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 817. p283
    nodes.append(addToKV(817, params['p283']))

    # 818. fused_nn_batch_matmul_312
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('818', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 819. p284
    nodes.append(addToKV(819, params['p284']))

    # 820. fused_reshape_add_add13
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('820', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 821. fused_mean35
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a70', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('821', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 822. fused_subtract13
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('822', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 823. fused_power_mean13
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a71', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('823', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 824. p285
    nodes.append(addToKV(824, params['p285']))

    # 825. p286
    nodes.append(addToKV(825, params['p286']))

    # 826. fused_add_sqrt_divide_multiply_add12
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('826', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 827. reshape_nop
    nodes.append(addToKV(`827, nodes[826]))

    # 828. p287
    nodes.append(addToKV(828, params['p287']))

    # 829. fused_nn_batch_matmul_26
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('829', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 830. p288
    nodes.append(addToKV(830, params['p288']))

    # 831. fused_reshape_add_multiply_divide_erf_add_multiply_reshape6
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('831', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 832. p289
    nodes.append(addToKV(832, params['p289']))

    # 833. fused_nn_batch_matmul_16
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('833', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 834. p290
    nodes.append(addToKV(834, params['p290']))

    # 835. fused_reshape_add_add12
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('835', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 836. fused_mean36
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a72', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('836', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 837. fused_subtract12
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('837', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 838. fused_power_mean12
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a73', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('838', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 839. p291
    nodes.append(addToKV(839, params['p291']))

    # 840. p292
    nodes.append(addToKV(840, params['p292']))

    # 841. fused_add_sqrt_divide_multiply_add11
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('841', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 842. reshape_nop
    nodes.append(addToKV(`842, nodes[841]))

    # 843. p293
    nodes.append(addToKV(843, params['p293']))

    # 844. fused_nn_batch_matmul_311
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('844', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 845. p294
    nodes.append(addToKV(845, params['p294']))

    # 846. fused_reshape_add_reshape_transpose_reshape5
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('846', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 847. p295
    nodes.append(addToKV(847, params['p295']))

    # 848. fused_nn_batch_matmul_384
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('848', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 849. p296
    nodes.append(addToKV(849, params['p296']))

    # 850. fused_reshape_add_reshape_transpose_reshape_transpose18
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('850', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 851. fused_nn_batch_matmul_55
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('851', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 852. fused_reshape_divide_add5
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('852', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 853. fused_max18
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('853', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 854. fused_subtract_exp5
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('854', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 855. fused_sum18
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('855', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 856. fused_divide_reshape5
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('856', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 857. p297
    nodes.append(addToKV(857, params['p297']))

    # 858. fused_nn_batch_matmul_385
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('858', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 859. p298
    nodes.append(addToKV(859, params['p298']))

    # 860. fused_reshape_add_reshape_transpose_reshape_transpose_118
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('860', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 861. fused_nn_batch_matmul_45
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('861', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 862. fused_reshape_transpose_reshape5
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('862', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 863. p299
    nodes.append(addToKV(863, params['p299']))

    # 864. fused_nn_batch_matmul_310
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('864', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 865. p300
    nodes.append(addToKV(865, params['p300']))

    # 866. fused_reshape_add_add11
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('866', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 867. fused_mean37
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a74', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('867', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 868. fused_subtract11
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('868', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 869. fused_power_mean11
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a75', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('869', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 870. p301
    nodes.append(addToKV(870, params['p301']))

    # 871. p302
    nodes.append(addToKV(871, params['p302']))

    # 872. fused_add_sqrt_divide_multiply_add10
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('872', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 873. reshape_nop
    nodes.append(addToKV(`873, nodes[872]))

    # 874. p303
    nodes.append(addToKV(874, params['p303']))

    # 875. fused_nn_batch_matmul_25
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('875', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 876. p304
    nodes.append(addToKV(876, params['p304']))

    # 877. fused_reshape_add_multiply_divide_erf_add_multiply_reshape5
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('877', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 878. p305
    nodes.append(addToKV(878, params['p305']))

    # 879. fused_nn_batch_matmul_15
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('879', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 880. p306
    nodes.append(addToKV(880, params['p306']))

    # 881. fused_reshape_add_add10
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('881', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 882. fused_mean38
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a76', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('882', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 883. fused_subtract10
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('883', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 884. fused_power_mean10
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a77', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('884', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 885. p307
    nodes.append(addToKV(885, params['p307']))

    # 886. p308
    nodes.append(addToKV(886, params['p308']))

    # 887. fused_add_sqrt_divide_multiply_add9
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('887', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 888. reshape_nop
    nodes.append(addToKV(`888, nodes[887]))

    # 889. p309
    nodes.append(addToKV(889, params['p309']))

    # 890. fused_nn_batch_matmul_39
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('890', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 891. p310
    nodes.append(addToKV(891, params['p310']))

    # 892. fused_reshape_add_reshape_transpose_reshape4
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('892', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 893. p311
    nodes.append(addToKV(893, params['p311']))

    # 894. fused_nn_batch_matmul_386
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('894', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 895. p312
    nodes.append(addToKV(895, params['p312']))

    # 896. fused_reshape_add_reshape_transpose_reshape_transpose19
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('896', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 897. fused_nn_batch_matmul_54
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('897', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 898. fused_reshape_divide_add4
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('898', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 899. fused_max19
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('899', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 900. fused_subtract_exp4
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('900', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 901. fused_sum19
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('901', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 902. fused_divide_reshape4
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('902', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 903. p313
    nodes.append(addToKV(903, params['p313']))

    # 904. fused_nn_batch_matmul_387
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('904', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 905. p314
    nodes.append(addToKV(905, params['p314']))

    # 906. fused_reshape_add_reshape_transpose_reshape_transpose_119
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('906', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 907. fused_nn_batch_matmul_44
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('907', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 908. fused_reshape_transpose_reshape4
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('908', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 909. p315
    nodes.append(addToKV(909, params['p315']))

    # 910. fused_nn_batch_matmul_38
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('910', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 911. p316
    nodes.append(addToKV(911, params['p316']))

    # 912. fused_reshape_add_add9
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('912', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 913. fused_mean39
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a78', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('913', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 914. fused_subtract9
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('914', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 915. fused_power_mean9
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a79', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('915', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 916. p317
    nodes.append(addToKV(916, params['p317']))

    # 917. p318
    nodes.append(addToKV(917, params['p318']))

    # 918. fused_add_sqrt_divide_multiply_add8
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('918', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 919. reshape_nop
    nodes.append(addToKV(`919, nodes[918]))

    # 920. p319
    nodes.append(addToKV(920, params['p319']))

    # 921. fused_nn_batch_matmul_24
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('921', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 922. p320
    nodes.append(addToKV(922, params['p320']))

    # 923. fused_reshape_add_multiply_divide_erf_add_multiply_reshape4
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('923', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 924. p321
    nodes.append(addToKV(924, params['p321']))

    # 925. fused_nn_batch_matmul_14
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('925', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 926. p322
    nodes.append(addToKV(926, params['p322']))

    # 927. fused_reshape_add_add8
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('927', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 928. fused_mean40
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a80', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('928', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 929. fused_subtract8
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('929', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 930. fused_power_mean8
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a81', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('930', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 931. p323
    nodes.append(addToKV(931, params['p323']))

    # 932. p324
    nodes.append(addToKV(932, params['p324']))

    # 933. fused_add_sqrt_divide_multiply_add7
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('933', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 934. reshape_nop
    nodes.append(addToKV(`934, nodes[933]))

    # 935. p325
    nodes.append(addToKV(935, params['p325']))

    # 936. fused_nn_batch_matmul_37
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('936', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 937. p326
    nodes.append(addToKV(937, params['p326']))

    # 938. fused_reshape_add_reshape_transpose_reshape3
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('938', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 939. p327
    nodes.append(addToKV(939, params['p327']))

    # 940. fused_nn_batch_matmul_388
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('940', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 941. p328
    nodes.append(addToKV(941, params['p328']))

    # 942. fused_reshape_add_reshape_transpose_reshape_transpose20
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('942', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 943. fused_nn_batch_matmul_53
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('943', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 944. fused_reshape_divide_add3
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('944', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 945. fused_max20
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('945', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 946. fused_subtract_exp3
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('946', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 947. fused_sum20
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('947', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 948. fused_divide_reshape3
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('948', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 949. p329
    nodes.append(addToKV(949, params['p329']))

    # 950. fused_nn_batch_matmul_389
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('950', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 951. p330
    nodes.append(addToKV(951, params['p330']))

    # 952. fused_reshape_add_reshape_transpose_reshape_transpose_120
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('952', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 953. fused_nn_batch_matmul_43
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('953', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 954. fused_reshape_transpose_reshape3
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('954', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 955. p331
    nodes.append(addToKV(955, params['p331']))

    # 956. fused_nn_batch_matmul_36
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('956', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 957. p332
    nodes.append(addToKV(957, params['p332']))

    # 958. fused_reshape_add_add7
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('958', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 959. fused_mean41
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a82', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('959', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 960. fused_subtract7
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('960', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 961. fused_power_mean7
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a83', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('961', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 962. p333
    nodes.append(addToKV(962, params['p333']))

    # 963. p334
    nodes.append(addToKV(963, params['p334']))

    # 964. fused_add_sqrt_divide_multiply_add6
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('964', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 965. reshape_nop
    nodes.append(addToKV(`965, nodes[964]))

    # 966. p335
    nodes.append(addToKV(966, params['p335']))

    # 967. fused_nn_batch_matmul_23
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('967', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 968. p336
    nodes.append(addToKV(968, params['p336']))

    # 969. fused_reshape_add_multiply_divide_erf_add_multiply_reshape3
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('969', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 970. p337
    nodes.append(addToKV(970, params['p337']))

    # 971. fused_nn_batch_matmul_13
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('971', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 972. p338
    nodes.append(addToKV(972, params['p338']))

    # 973. fused_reshape_add_add6
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('973', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 974. fused_mean42
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a84', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('974', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 975. fused_subtract6
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('975', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 976. fused_power_mean6
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a85', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('976', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 977. p339
    nodes.append(addToKV(977, params['p339']))

    # 978. p340
    nodes.append(addToKV(978, params['p340']))

    # 979. fused_add_sqrt_divide_multiply_add5
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('979', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 980. reshape_nop
    nodes.append(addToKV(`980, nodes[979]))

    # 981. p341
    nodes.append(addToKV(981, params['p341']))

    # 982. fused_nn_batch_matmul_35
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('982', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 983. p342
    nodes.append(addToKV(983, params['p342']))

    # 984. fused_reshape_add_reshape_transpose_reshape2
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('984', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 985. p343
    nodes.append(addToKV(985, params['p343']))

    # 986. fused_nn_batch_matmul_390
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('986', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 987. p344
    nodes.append(addToKV(987, params['p344']))

    # 988. fused_reshape_add_reshape_transpose_reshape_transpose21
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('988', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 989. fused_nn_batch_matmul_52
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('989', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 990. fused_reshape_divide_add2
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('990', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 991. fused_max21
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('991', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 992. fused_subtract_exp2
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('992', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 993. fused_sum21
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('993', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 994. fused_divide_reshape2
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('994', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 995. p345
    nodes.append(addToKV(995, params['p345']))

    # 996. fused_nn_batch_matmul_391
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('996', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 997. p346
    nodes.append(addToKV(997, params['p346']))

    # 998. fused_reshape_add_reshape_transpose_reshape_transpose_121
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('998', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 999. fused_nn_batch_matmul_42
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('999', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 1000. fused_reshape_transpose_reshape2
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1000', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 1001. p347
    nodes.append(addToKV(1001, params['p347']))

    # 1002. fused_nn_batch_matmul_34
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1002', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 1003. p348
    nodes.append(addToKV(1003, params['p348']))

    # 1004. fused_reshape_add_add5
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1004', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 1005. fused_mean43
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a86', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('1005', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 1006. fused_subtract5
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1006', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 1007. fused_power_mean5
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a87', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('1007', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 1008. p349
    nodes.append(addToKV(1008, params['p349']))

    # 1009. p350
    nodes.append(addToKV(1009, params['p350']))

    # 1010. fused_add_sqrt_divide_multiply_add4
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1010', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 1011. reshape_nop
    nodes.append(addToKV(`1011, nodes[1010]))

    # 1012. p351
    nodes.append(addToKV(1012, params['p351']))

    # 1013. fused_nn_batch_matmul_22
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('1013', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 1014. p352
    nodes.append(addToKV(1014, params['p352']))

    # 1015. fused_reshape_add_multiply_divide_erf_add_multiply_reshape2
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('1015', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 1016. p353
    nodes.append(addToKV(1016, params['p353']))

    # 1017. fused_nn_batch_matmul_12
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1017', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 1018. p354
    nodes.append(addToKV(1018, params['p354']))

    # 1019. fused_reshape_add_add4
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1019', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 1020. fused_mean44
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a88', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('1020', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 1021. fused_subtract4
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1021', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 1022. fused_power_mean4
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a89', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('1022', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 1023. p355
    nodes.append(addToKV(1023, params['p355']))

    # 1024. p356
    nodes.append(addToKV(1024, params['p356']))

    # 1025. fused_add_sqrt_divide_multiply_add3
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1025', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 1026. reshape_nop
    nodes.append(addToKV(`1026, nodes[1025]))

    # 1027. p357
    nodes.append(addToKV(1027, params['p357']))

    # 1028. fused_nn_batch_matmul_33
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1028', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 1029. p358
    nodes.append(addToKV(1029, params['p358']))

    # 1030. fused_reshape_add_reshape_transpose_reshape1
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1030', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 1031. p359
    nodes.append(addToKV(1031, params['p359']))

    # 1032. fused_nn_batch_matmul_392
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1032', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 1033. p360
    nodes.append(addToKV(1033, params['p360']))

    # 1034. fused_reshape_add_reshape_transpose_reshape_transpose22
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1034', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 1035. fused_nn_batch_matmul_51
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('1035', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 1036. fused_reshape_divide_add1
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('1036', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 1037. fused_max22
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('1037', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 1038. fused_subtract_exp1
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('1038', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 1039. fused_sum22
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('1039', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 1040. fused_divide_reshape1
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('1040', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 1041. p361
    nodes.append(addToKV(1041, params['p361']))

    # 1042. fused_nn_batch_matmul_393
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1042', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 1043. p362
    nodes.append(addToKV(1043, params['p362']))

    # 1044. fused_reshape_add_reshape_transpose_reshape_transpose_122
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1044', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 1045. fused_nn_batch_matmul_41
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1045', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 1046. fused_reshape_transpose_reshape1
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1046', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 1047. p363
    nodes.append(addToKV(1047, params['p363']))

    # 1048. fused_nn_batch_matmul_32
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1048', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 1049. p364
    nodes.append(addToKV(1049, params['p364']))

    # 1050. fused_reshape_add_add3
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1050', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 1051. fused_mean45
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a90', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('1051', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 1052. fused_subtract3
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1052', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 1053. fused_power_mean3
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a91', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('1053', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 1054. p365
    nodes.append(addToKV(1054, params['p365']))

    # 1055. p366
    nodes.append(addToKV(1055, params['p366']))

    # 1056. fused_add_sqrt_divide_multiply_add2
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1056', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 1057. reshape_nop
    nodes.append(addToKV(`1057, nodes[1056]))

    # 1058. p367
    nodes.append(addToKV(1058, params['p367']))

    # 1059. fused_nn_batch_matmul_21
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('1059', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 1060. p368
    nodes.append(addToKV(1060, params['p368']))

    # 1061. fused_reshape_add_multiply_divide_erf_add_multiply_reshape1
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('1061', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 1062. p369
    nodes.append(addToKV(1062, params['p369']))

    # 1063. fused_nn_batch_matmul_11
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1063', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 1064. p370
    nodes.append(addToKV(1064, params['p370']))

    # 1065. fused_reshape_add_add2
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1065', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 1066. fused_mean46
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a92', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('1066', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 1067. fused_subtract2
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1067', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 1068. fused_power_mean2
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a93', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('1068', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 1069. p371
    nodes.append(addToKV(1069, params['p371']))

    # 1070. p372
    nodes.append(addToKV(1070, params['p372']))

    # 1071. fused_add_sqrt_divide_multiply_add1
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1071', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 1072. reshape_nop
    nodes.append(addToKV(`1072, nodes[1071]))

    # 1073. p373
    nodes.append(addToKV(1073, params['p373']))

    # 1074. fused_nn_batch_matmul_31
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1074', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 1075. p374
    nodes.append(addToKV(1075, params['p374']))

    # 1076. fused_reshape_add_reshape_transpose_reshape
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1076', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 1077. p375
    nodes.append(addToKV(1077, params['p375']))

    # 1078. fused_nn_batch_matmul_394
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1078', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 1079. p376
    nodes.append(addToKV(1079, params['p376']))

    # 1080. fused_reshape_add_reshape_transpose_reshape_transpose23
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1080', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_kernel0', path, shapes, arguments))

    # 1081. fused_nn_batch_matmul_5
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('1081', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(6, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_5_kernel0', path, shapes, arguments))

    # 1082. fused_reshape_divide_add
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('1082', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_divide_add_kernel0', path, shapes, arguments))

    # 1083. fused_max23
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('1083', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_max_kernel0', path, shapes, arguments))

    # 1084. fused_subtract_exp
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('1084', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_exp_kernel0', path, shapes, arguments))

    # 1085. fused_sum23
    # kernel 0
    output_size = 24576
    nodes.append(kaas.bufferSpec('1085', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(192, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_sum_kernel0', path, shapes, arguments))

    # 1086. fused_divide_reshape
    # kernel 0
    output_size = 9437184
    nodes.append(kaas.bufferSpec('1086', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_divide_reshape_kernel0', path, shapes, arguments))

    # 1087. p377
    nodes.append(addToKV(1087, params['p377']))

    # 1088. fused_nn_batch_matmul_395
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1088', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 1089. p378
    nodes.append(addToKV(1089, params['p378']))

    # 1090. fused_reshape_add_reshape_transpose_reshape_transpose_123
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1090', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_reshape_transpose_reshape_transpose_1_kernel0', path, shapes, arguments))

    # 1091. fused_nn_batch_matmul_4
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1091', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 6, 16),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_4_kernel0', path, shapes, arguments))

    # 1092. fused_reshape_transpose_reshape
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1092', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_transpose_reshape_kernel0', path, shapes, arguments))

    # 1093. p379
    nodes.append(addToKV(1093, params['p379']))

    # 1094. fused_nn_batch_matmul_3
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1094', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_3_kernel0', path, shapes, arguments))

    # 1095. p380
    nodes.append(addToKV(1095, params['p380']))

    # 1096. fused_reshape_add_add1
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1096', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 1097. fused_mean47
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a94', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('1097', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 1098. fused_subtract1
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1098', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 1099. fused_power_mean1
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a95', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('1099', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 1100. p381
    nodes.append(addToKV(1100, params['p381']))

    # 1101. p382
    nodes.append(addToKV(1101, params['p382']))

    # 1102. fused_add_sqrt_divide_multiply_add
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1102', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_kernel0', path, shapes, arguments))

    # 1103. reshape_nop
    nodes.append(addToKV(`1103, nodes[1102]))

    # 1104. p383
    nodes.append(addToKV(1104, params['p383']))

    # 1105. fused_nn_batch_matmul_2
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('1105', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(64, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_2_kernel0', path, shapes, arguments))

    # 1106. p384
    nodes.append(addToKV(1106, params['p384']))

    # 1107. fused_reshape_add_multiply_divide_erf_add_multiply_reshape
    # kernel 0
    output_size = 6291456
    nodes.append(kaas.bufferSpec('1107', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_multiply_divide_erf_add_multiply_reshape_kernel0', path, shapes, arguments))

    # 1108. p385
    nodes.append(addToKV(1108, params['p385']))

    # 1109. fused_nn_batch_matmul_1
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1109', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(16, 6, 1),  (8, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_1_kernel0', path, shapes, arguments))

    # 1110. p386
    nodes.append(addToKV(1110, params['p386']))

    # 1111. fused_reshape_add_add
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1111', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_add_kernel0', path, shapes, arguments))

    # 1112. fused_mean48
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a96', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('1112', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_mean_kernel1', path, shapes, arguments))

    # 1113. fused_subtract
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1113', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_subtract_kernel0', path, shapes, arguments))

    # 1114. fused_power_mean
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a97', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(12, 1, 1),  (32, 32, 1)]
    kerns.append(makeKern('fused_power_mean_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('1114', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_power_mean_kernel1', path, shapes, arguments))

    # 1115. p387
    nodes.append(addToKV(1115, params['p387']))

    # 1116. p388
    nodes.append(addToKV(1116, params['p388']))

    # 1117. fused_add_sqrt_divide_multiply_add_reshape
    # kernel 0
    output_size = 1572864
    nodes.append(kaas.bufferSpec('1117', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(256, 1, 1),  (1024, 1, 1)]
    kerns.append(makeKern('fused_add_sqrt_divide_multiply_add_reshape_kernel0', path, shapes, arguments))

    # 1118. p389
    nodes.append(addToKV(1118, params['p389']))

    # 1119. fused_nn_batch_matmul
    # kernel 0
    output_size = 3072
    nodes.append(kaas.bufferSpec('1119', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 6, 1),  (2, 8, 1)]
    kerns.append(makeKern('fused_nn_batch_matmul_kernel0', path, shapes, arguments))

    # 1120. p390
    nodes.append(addToKV(1120, params['p390']))

    # 1121. fused_reshape_add_split
    imm = []
    # kernel 0
    output_size = 4096
    imm.append(kaas.bufferSpec('a98', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_split_kernel0', path, shapes, arguments))
    # kernel 1
    output_size = 1536
    nodes.append(kaas.bufferSpec('1121', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_reshape_add_split_kernel1', path, shapes, arguments))

    # 1122. fused_squeeze
    # kernel 0
    output_size = 1536
    nodes.append(kaas.bufferSpec('1122', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_squeeze_kernel0', path, shapes, arguments))

    # 1123. fused_squeeze_1
    # kernel 0
    output_size = 1536
    nodes.append(kaas.bufferSpec('1123', output_size, const=False, ephemeral=True))
    arguments = []
    shapes = [(1, 1, 1),  (384, 1, 1)]
    kerns.append(makeKern('fused_squeeze_1_kernel0', path, shapes, arguments))

    '''
    req = kaas.kaasReq(kerns)
    return req

runReq()
