import kaas
import kaas.ray
import ray
import pathlib
import numpy as np

testPath = pathlib.Path(__file__).resolve().parent


def getTestInput(len):
    return np.arange(0, len, dtype=np.uint32)


def testMinimal():
    """Test the minimal set of functionality for KaaS"""
    testArray = getTestInput(32)
    inpRef = ray.put(testArray)

    args = [(kaas.bufferSpec(kaas.ray.putObj(inpRef), testArray.nbytes), 'i'),
            (kaas.bufferSpec('out', 4, ephemeral=False), 'o')]

    kern = kaas.kernelSpec(testPath / 'kerns' / 'testKerns.cubin',
                           'sumKern',
                           (1, 1), (32, 1, 1),
                           arguments=args)

    reqRef = ray.put(kaas.kaasReq([kern]))

    resRef = kaas.ray.invokerTask.remote((reqRef, {}))

    results = ray.get(resRef)
    resArray = ray.get(results[0])
    resArray.dtype = np.uint32

    expect = testArray.sum()
    got = resArray.sum()
    if got != expect:
        print("Fail: results don't match")
        print("\tExpect: ", expect)
        print("\tGot: ", got)
    else:
        print("PASS")


if __name__ == "__main__":
    ray.init()

    testMinimal()
