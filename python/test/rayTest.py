import kaas
import kaas.ray
import kaas.pool

import ray
import pathlib
import numpy as np

testPath = pathlib.Path(__file__).resolve().parent


def getTestReq(arrLen):
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


def checkRes(resArray, testArray):
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


def testMinimal():
    """Test the minimal set of functionality for KaaS"""
    reqRef, _, testArray = getTestReq(32)

    taskResRef = kaas.ray.invokerTask.remote((reqRef, {}))

    # taskResRef is the reference returned by the kaas worker task. The task is
    # returning a reference to the output of the kaas req (kaasOutRef). We
    # finally dereference this to get the actual array (resArray).
    kaasOutRef = ray.get(taskResRef)
    resArray = ray.get(kaasOutRef)
    resArray.dtype = np.uint32

    return checkRes(resArray, testArray)


def testPool():
    """Test the pool mechanism"""
    clientID = 'testClient'
    pool = kaas.pool.Pool.remote(1, 'balance', kaas.ray.invokerActor)

    # This step is not required, but it ensures that the pool is completely
    # initialized in case we want to avoid cold starts.
    ray.get(pool.ensureReady.remote())

    reqRef, inpRef, testArray = getTestReq(32)

    poolResRef = pool.run.remote('invoke', 1, clientID, [inpRef], [(reqRef, {})], {"clientID": clientID})

    # The pool itself returns a reference to whatever the output of the actor
    # is. The actor returns a reference to whatever the kaas server returns.
    # The kaas server returns a reference to the result array.
    actorResRef = ray.get(poolResRef)
    kaasOutRef = ray.get(actorResRef)
    resArray = ray.get(kaasOutRef)
    resArray.dtype = np.uint32

    return checkRes(resArray, testArray)


if __name__ == "__main__":
    ray.init()

    # testMinimal()
    testPool()
