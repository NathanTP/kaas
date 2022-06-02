#!/usr/bin/env python
import ray
import kaas.pool2


@ray.remote
class TestWorker(kaas.pool2.PoolWorker):
    def exampleMethod(self, arg):
        return f"val is {arg}"


def minTest(policy):
    groupID = "exampleGroup"
    pool = kaas.pool2.Pool.remote(3, policy=policy)
    registerConfirm = pool.registerGroup.remote(groupID, TestWorker)
    ray.get(registerConfirm)

    refs = []
    for i in range(3):
        refs.append(pool.run.remote(groupID, "exampleMethod", 1, args=[i]))

    expects = ["val is 0", "val is 1", "val is 2"]
    rets = [ray.get(ray.get(ref))[0] for ref in refs]
    for expect, ret in zip(expects, rets):
        if expect != ret:
            print(f"Test Failed: expected '{expect}', got '{ret}'")
            return False

    return True


if __name__ == "__main__":
    ray.init()
    print("Min test BalancePolicy")
    if not minTest(kaas.pool2.BalancePolicy):
        print("FAIL")
    else:
        print("SUCCESS")

    print("Min test ExclusivePolicy")
    if not minTest(kaas.pool2.ExclusivePolicy):
        print("FAIL")
    else:
        print("SUCCESS")
