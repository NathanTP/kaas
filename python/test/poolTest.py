#!/usr/bin/env python
import ray
import kaas.pool


@ray.remote
class TestWorker(kaas.pool.PoolWorker):
    def returnOne(self, arg):
        return arg

    def returnTwo(self, arg):
        return True, arg


def testMultiRet(policy):
    groupID = "exampleGroup"
    pool = kaas.pool.Pool(3, policy=policy)
    pool.registerGroup(groupID, TestWorker)

    refs = []
    for i in range(3):
        refs.append(pool.run(groupID, "returnTwo", num_returns=2, args=[i]))

    # the pool returns a reference to the TestWorker.returnTwo remote method's
    # return values which should be a tuple of references.
    retRefs = [ray.get(ref) for ref in refs]
    for expect, retRef in enumerate(retRefs):
        success = ray.get(retRef[0])
        if not isinstance(success, bool) or not success:
            print("Test Failed: returned wrong first value expected True, got ", success)
            return False

        val = ray.get(retRef[1])
        if expect != val:
            print(f"Test Failed: returned wrong second value expected '{expect}', got '{val}'")
            return False

    return True


def testOneRet(policy):
    groupID = "exampleGroup"
    pool = kaas.pool.Pool(3, policy=policy)
    pool.registerGroup(groupID, TestWorker)

    refs = []
    for i in range(3):
        refs.append(pool.run(groupID, "returnOne", args=[i]))

    retRefs = [ray.get(ref) for ref in refs]
    for expect, retRef in enumerate(retRefs):
        ret = ray.get(retRef)
        if expect != ret:
            print(f"Test Failed: expected '{expect}', got '{ret}'")
            return False

    return True


if __name__ == "__main__":
    ray.init()
    print("Min test BalancePolicy")
    if not testOneRet(kaas.pool.BalancePolicy):
        print("FAIL")
    else:
        print("SUCCESS")

    print("Min test ExclusivePolicy")
    if not testOneRet(kaas.pool.ExclusivePolicy):
        print("FAIL")
    else:
        print("SUCCESS")

    print("Multiple Returns")
    if not testMultiRet(kaas.pool.BalancePolicy):
        print("FAIL")
    else:
        print("SUCCESS")
