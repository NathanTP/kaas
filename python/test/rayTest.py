#!/usr/bin/env python
import kaas
import kaas.ray
import kaas.pool

import ray
import pathlib
import numpy as np

testPath = pathlib.Path(__file__).resolve().parent


def getSumReq(arrLen):
    testArray = np.arange(0, arrLen, dtype=np.uint32)
    inpRef = ray.put(testArray)

    args = [(kaas.bufferSpec(kaas.ray.putObj(inpRef), testArray.nbytes), 'i'),
            (kaas.bufferSpec('out', 4, ephemeral=False), 'o')]

    kern = kaas.kernelSpec(testPath / 'kerns' / 'testKerns.cubin',
                           'sumKern',
                           (1, 1), (arrLen, 1, 1),
                           arguments=args)

    reqRef = ray.put(kaas.kaasReq([kern]))

    return reqRef, inpRef, testArray


def checkSum(resArray, testArray):
    expect = testArray.sum()
    got = resArray.sum()
    if got != expect:
        print("Fail: results don't match")
        print("\tExpect: ", expect)
        print("\tGot: ", got)
        return False
    else:
        print("PASS")
        return True


def getDotReq(nElem, offset=False):
    nByte = nElem*4

    aArr = np.arange(0, nElem, dtype=np.uint32)
    bArr = np.arange(nElem, nElem*2, dtype=np.uint32)

    if offset:
        combinedArr = np.concatenate((aArr, bArr))
        combinedRef = ray.put(combinedArr)
        aBuf = kaas.bufferSpec('inpA', nByte, offset=0, key=kaas.ray.putObj(combinedRef))
        bBuf = kaas.bufferSpec('inpB', nByte, offset=nByte, key=kaas.ray.putObj(combinedRef))
        inpRefs = [combinedRef]
    else:
        aRef = ray.put(aArr)
        bRef = ray.put(bArr)
        aBuf = kaas.bufferSpec('inpA', nByte, key=kaas.ray.putObj(aRef))
        bBuf = kaas.bufferSpec('inpB', nByte, key=kaas.ray.putObj(bRef))
        inpRefs = [aRef, bRef]

    prodOutBuf = kaas.bufferSpec('prodOut', nByte, ephemeral=True)
    cBuf = kaas.bufferSpec('output', 8, key='c')

    args_prod = [(aBuf, 'i'), (bBuf, 'i'), (prodOutBuf, 'o')]

    prodKern = kaas.kernelSpec(testPath / 'kerns' / 'testKerns.cubin',
                               'prodKern',
                               (1, 1), (nElem, 1, 1),
                               literals=[kaas.literalSpec('Q', nElem)],
                               arguments=args_prod)

    args_sum = [(prodOutBuf, 'i'), (cBuf, 'o')]

    sumKern = kaas.kernelSpec(testPath / 'kerns' / 'testKerns.cubin',
                              'sumKern',
                              (1, 1), (nElem // 2, 1, 1),
                              arguments=args_sum)

    req = kaas.kaasReq([prodKern, sumKern])
    reqRef = ray.put(req)

    return reqRef, inpRefs, [aArr, bArr]


def checkDot(got, aArr, bArr):
    expect = np.dot(aArr, bArr)
    if got != expect:
        print("Fail: results don't match")
        print("\tExpect: ", expect)
        print("\tGot: ", got)
        return False
    else:
        print("PASS")
        return True


def testMinimal():
    """Test the minimal set of functionality for KaaS"""
    reqRef, _, inputArrs = getDotReq(1024, offset=True)

    taskResRef = kaas.ray.invokerTask.remote((reqRef, {}))

    # taskResRef is the reference returned by the kaas worker task. The task is
    # returning a reference to the output of the kaas req (kaasOutRef). We
    # finally dereference this to get the actual array (resArray).
    kaasOutRef = ray.get(taskResRef)
    resArray = ray.get(kaasOutRef)
    resArray.dtype = np.uint32

    return checkDot(resArray[0], inputArrs[0], inputArrs[1])


def testPool():
    """Test the pool mechanism"""
    clientID = 'testClient'
    pool = kaas.pool.Pool.remote(1, 'balance', kaas.ray.invokerActor)

    # This step is not required, but it ensures that the pool is completely
    # initialized in case we want to avoid cold starts.
    ray.get(pool.ensureReady.remote())

    reqRef, inpRefs, inputArrs = getDotReq(1024, offset=True)

    poolResRef = pool.run.remote('invoke', 1, clientID, inpRefs, [(reqRef, {})], {"clientID": clientID})

    # The pool itself returns a reference to whatever the output of the actor
    # is. The actor returns a reference to whatever the kaas server returns.
    # The kaas server returns a reference to the result array.
    actorResRef = ray.get(poolResRef)
    kaasOutRef = ray.get(actorResRef)
    resArray = ray.get(kaasOutRef)
    resArray.dtype = np.uint32

    return checkDot(resArray[0], inputArrs[0], inputArrs[1])


if __name__ == "__main__":
    ray.init()

    # testMinimal()
    testPool()
