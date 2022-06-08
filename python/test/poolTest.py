#!/usr/bin/env python
import ray
import kaas.pool


@ray.remote
class TestWorker(kaas.pool.PoolWorker):
    def __init__(self):
        # Initialize the PoolWorker parent class to get self.profs
        super().__init__()
        self.profs['n_initialized'].increment(1)

    def returnOne(self, arg):
        self.profs['n_returnOne'].increment(1)
        return arg

    def returnTwo(self, arg):
        self.profs['n_returnTwo'].increment(1)
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


def testOneRet(policy, inputRef=False):
    groupID = "exampleGroup"
    pool = kaas.pool.Pool(3, policy=policy)
    pool.registerGroup(groupID, TestWorker)

    refs = []
    for i in range(3):
        if inputRef:
            arg = ray.put(i)
        else:
            arg = i
        refs.append(pool.run(groupID, "returnOne", args=[arg]))

    retRefs = [ray.get(ref) for ref in refs]
    for expect, retRef in enumerate(retRefs):
        ret = ray.get(retRef)
        if expect != ret:
            print(f"Test Failed: expected '{expect}', got '{ret}'")
            return False

    return True


def testProfs(policy):
    pool = kaas.pool.Pool(3, policy=policy)

    groups = ['group0', 'group1']
    # groups = ['group0']
    for groupID in groups:
        pool.registerGroup(groupID, TestWorker)
        retRefs = []
        for i in range(5):
            retRefs.append(ray.get(pool.run(groupID, 'returnOne', args=['testInp'])))

        ray.get(retRefs)

    profs = pool.getProfile()
    report = profs.report()
    if not set(groups) <= set(report.keys()):
        print("Failure: missing groups")

    return True


if __name__ == "__main__":
    ray.init()

    print("Single Return BalancePolicy")
    if not testOneRet(kaas.pool.policies.BALANCE):
        print("FAIL")
    else:
        print("SUCCESS")

    print("Reference Argument BalancePolicy")
    if not testOneRet(kaas.pool.policies.BALANCE, inputRef=True):
        print("FAIL")
    else:
        print("SUCCESS")

    # print("Min test ExclusivePolicy")
    # if not testOneRet(kaas.pool.policies.EXCLUSIVE):
    #     print("FAIL")
    # else:
    #     print("SUCCESS")
    #
    # print("Multiple Returns")
    # if not testMultiRet(kaas.pool.policies.BALANCE):
    #     print("FAIL")
    # else:
    #     print("SUCCESS")
    #
    # print("Profiling Test")
    # if not testProfs(kaas.pool.policies.EXCLUSIVE):
    #     print("FAIL")
    # else:
    #     print("SUCCESS")
