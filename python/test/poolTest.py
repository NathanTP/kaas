#!/usr/bin/env python
import ray
from pprint import pprint  # NOQA
import argparse
import time
import os
import subprocess as sp

import kaas.pool


nResource = 2


def getNGpus():
    """Returns the number of available GPUs on this machine"""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    else:
        proc = sp.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                      stdout=sp.PIPE, text=True)
        if proc.returncode != 0:
            return None
        else:
            return proc.stdout.count('\n')


@ray.remote(num_gpus=1)
class TestWorker(kaas.pool.PoolWorker):
    def __init__(self, **workerKwargs):
        # must be called before accessing inherited methods like getProfs()
        super().__init__(**workerKwargs)
        self.getProfs()['n_initialized'].increment(1)

    def returnOne(self, arg):
        self.getProfs()['n_returnOne'].increment(1)
        return arg

    def returnTwo(self, arg):
        self.getProfs()['n_returnTwo'].increment(1)
        return True, arg

    def delay(self, arg, sleepTime):
        time.sleep(sleepTime)
        return arg


@ray.remote(resources={"testResource": 1})
class TestWorkerNoGPU(kaas.pool.PoolWorker):
    def __init__(self, **workerKwargs):
        # must be called before accessing inherited methods like getProfs()
        super().__init__(**workerKwargs)
        self.getProfs()['n_initialized'].increment(1)

    def returnOne(self, arg):
        self.getProfs()['n_returnOne'].increment(1)
        return arg

    def returnTwo(self, arg):
        self.getProfs()['n_returnTwo'].increment(1)
        return True, arg

    def delay(self, arg, sleepTime):
        time.sleep(sleepTime)
        return arg


def testMultiRet(policy):
    success = True
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
            success = False
            break

        val = ray.get(retRef[1])
        if expect != val:
            print(f"Test Failed: returned wrong second value expected '{expect}', got '{val}'")
            success = False
            break

    pool.shutdown()

    return success


def testOneRet(policy, inputRef=False):
    success = True
    groupID = "exampleGroup"
    pool = kaas.pool.Pool(3, policy=policy)
    pool.registerGroup(groupID, TestWorker)

    refs = []
    for i in range(3):
        if inputRef:
            arg = ray.put(i)
            refDeps = [arg]
        else:
            arg = i
            refDeps = None
        refs.append(pool.run(groupID, "returnOne", args=[arg], refDeps=refDeps))

    retRefs = [ray.get(ref) for ref in refs]
    for expect, retRef in enumerate(retRefs):
        ret = ray.get(retRef)
        if expect != ret:
            print(f"Test Failed: expected '{expect}', got '{ret}'")
            success = False

    pool.shutdown()

    return success


def testStress(policy, gpu=True):
    """Submit many requests from many groups in one batch and wait for returns.
    There are more groups than workers and many requests interleaved between
    groups so this should stress the pool."""
    success = True

    if gpu:
        nGPUs = getNGpus()
        if nGPUs is None:
            raise RuntimeError("gpu==True but there are no GPUs available")
        nWorker = nGPUs
        worker = TestWorker
    else:
        worker = TestWorkerNoGPU
        nWorker = nResource

    # nGroup = 10
    nGroup = 5
    nIter = 2
    sleepTime = 2
    nReqTotal = nGroup * nIter * 2

    pool = kaas.pool.Pool(nWorker, policy=policy)

    groups = ['g' + str(x) for x in range(nGroup)]
    for group in groups:
        if gpu:
            pool.registerGroup(group, worker)
        else:
            pool.registerGroup(group, worker)

    retRefs = []
    args = []
    startTime = time.time()
    for iterIdx in range(nIter):
        for groupIdx, group in enumerate(groups):
            args.append(iterIdx*100 + groupIdx)
            retRefs.append(pool.run(group, 'delay', args=[args[-1], sleepTime]))

            args.append(1000 + iterIdx*100 + groupIdx)
            retRefs.append(pool.run(group, 'delay', args=[args[-1], sleepTime]))

    rets = ray.get(ray.get(retRefs))

    runTime = time.time() - startTime
    expectTime = (nReqTotal * sleepTime) / nWorker
    if runTime < expectTime:
        print("Completed too fast: ")
        print("\tExpected: ", expectTime)
        print("\tGot: ", runTime)
        success = False

    for idx, ret, arg in zip(range(len(rets)), rets, args):
        if ret != arg:
            print(f"Unexpected return from call {idx}: ")
            print("\tExpected: ", arg)
            print("\tGot: ", ret)
            success = False
            break

    pool.shutdown()
    return success


def testProfs(policy):
    success = True
    pool = kaas.pool.Pool(1, policy=policy)

    groups = ['group0', 'group1']
    # groups = ['group0']
    retRefs = []
    for groupID in groups:
        pool.registerGroup(groupID, TestWorker)

    for groupID in groups:
        for i in range(5):
            retRefs.append(pool.run(groupID, 'returnOne', args=['testInp']))

    ray.get(ray.get(retRefs))

    profs = pool.getProfile()

    poolGroups = set([k for k, v in profs.mod('pool').mod('groups').getMods()])
    workerGroups = set([k for k, v in profs.mod('workers').mod('groups').getMods()])
    if not set(groups) <= poolGroups or \
       not set(groups) <= workerGroups:
        print("Failure: missing groups")
        print(profs)
        success = False

    pool.shutdown()
    return success


def simple():
    pool = kaas.pool.Pool(0, policy=kaas.pool.policies.EXCLUSIVE)
    pool.shutdown()


POLICY = kaas.pool.policies.EXCLUSIVE

if __name__ == "__main__":
    availableTests = ['oneRet', 'multiRet', 'profiling', 'stress']

    parser = argparse.ArgumentParser("Non-KaaS unit tests for the pool")
    parser.add_argument('-t', '--test', action='append', choices=availableTests + ['all'])
    parser.add_argument('-p', '--policy', choices=['balance', 'exclusive'])

    args = parser.parse_args()

    ray.init(resources={'testResource': nResource})

    if args.policy == 'balance':
        policy = kaas.pool.policies.BALANCE
    elif args.policy == 'exclusive':
        policy = kaas.pool.policies.EXCLUSIVE
    else:
        raise ValueError("Unrecognized Policy: ", args.policy)

    if args.test == 'all':
        tests = availableTests
    else:
        tests = args.test

    for test in tests:
        print(f"Running with {args.policy}: {test}")
        if test == 'oneRet':
            ret = testOneRet(policy)
        elif test == 'multiRet':
            ret = testMultiRet(policy)
        elif test == 'profiling':
            ret = testProfs(policy)
        elif test == 'stress':
            ret = testStress(policy, gpu=True)
        else:
            raise ValueError("Unrecognized test: ", test)

        if not ret:
            print('FAIL')
        else:
            print('SUCCESS')
