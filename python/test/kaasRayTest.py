#!/usr/bin/env python
import kaas
import kaas.ray
import kaas.pool

import ray
import pathlib
import numpy as np
from pprint import pprint

from kaas import profiling

testPath = pathlib.Path(__file__).resolve().parent


def getSumReq(arrLen):
    testArray = np.arange(0, arrLen, dtype=np.uint32)
    inpRef = ray.put(testArray)

    args = [(kaas.bufferSpec(kaas.ray.putObj(inpRef), testArray.nbytes), 'i'),
            (kaas.bufferSpec('out', 4, ephemeral=True), 'o')]

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
        return True


def getDotReq(nElem, offset=False, nIter=1):
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
    cBuf = kaas.bufferSpec('output', 8, key='c', ephemeral=True)

    args_prod = [(aBuf, 'i'), (bBuf, 'i'), (prodOutBuf, 't')]

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

    req = kaas.kaasReq([prodKern, sumKern], nIter=nIter)
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
        return True


def testMinimal():
    """Test the minimal set of functionality for KaaS"""
    reqRef, _, inputArrs = getDotReq(1024, offset=True)

    taskResRef = kaas.ray.invokerTask.remote((reqRef, {}))

    # taskResRef is the reference returned by the kaas worker task. The task is
    # returning a reference to the output of the kaas req (kaasOutRef). We
    # finally dereference this to get the actual array (resArray).
    kaasOutRef = ray.get(taskResRef)
    resArray = ray.get(kaasOutRef[0])
    resArray.dtype = np.uint32

    return checkDot(resArray[0], inputArrs[0], inputArrs[1])


def stressIterative():
    nIter = 3000
    reqRef, _, inputArrs = getDotReq(1024, offset=True, nIter=nIter)

    profs = profiling.profCollection()
    taskResRef = kaas.ray.invokerTask.remote((reqRef, {}), profs=profs)

    # taskResRef is the reference returned by the kaas worker task. The task is
    # returning a reference to the output of the kaas req (kaasOutRef). We
    # finally dereference this to get the actual array (resArray).
    kaasOutRef, profs = ray.get(taskResRef)
    resArray = ray.get(kaasOutRef[0])
    resArray.dtype = np.uint32

    pprint(profs.report(metrics=['mean']))
    return checkDot(resArray[0], inputArrs[0], inputArrs[1])


def testPool():
    """Test the pool mechanism"""
    groupID = 'testClient'
    pool = kaas.pool.Pool(1, policy=kaas.pool.policies.BALANCE)

    pool.registerGroup(groupID, kaas.ray.invokerActor)

    reqRef, inpRefs, inputArrs = getDotReq(1024, offset=True)

    poolResRef = pool.run(groupID, 'invoke', args=[(reqRef, {})], kwargs={"clientID": groupID})

    # See the pool docs for details, but it returns a ref(actor return
    # references)).
    # The actor in this case is the KaaS executor that returns a reference to
    # the KaaS invocation outputs.
    executorResRef = ray.get(poolResRef)
    kaasResRef = ray.get(executorResRef)
    resArray = ray.get(kaasResRef)
    resArray.dtype = np.uint32

    if not checkDot(resArray[0], inputArrs[0], inputArrs[1]):
        return False

    profs = pool.getProfile().report(metrics=['mean'])
    if 'testClient' not in profs['pool']['groups'] \
       or 'testClient' not in profs['workers']['groups'] \
       or 't_e2e' not in profs['workers']['groups']['testClient']:
        print("Profile missing data:")
        pprint(profs)
        return False

    return True


if __name__ == "__main__":
    ray.init()

    print("Stress Iterative")
    if stressIterative():
        print("PASS")
    else:
        print("FAIL")

    # print("Minimal test")
    # if testMinimal():
    #     print("PASS")
    # else:
    #     print("FAIL")

    # print("Pool Test")
    # if testPool():
    #     print("PASS")
    # else:
    #     print("FAIL")
