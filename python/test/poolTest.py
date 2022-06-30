#!/usr/bin/env python
import ray
import ray.util.queue
from pprint import pprint  # NOQA
import argparse
import time
import os
import subprocess as sp
import numpy as np
from tornado.ioloop import IOLoop
import asyncio

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


@kaas.pool.remote_with_confirmation()
def testTask(arg):
    return arg


@kaas.pool.remote_with_confirmation(num_returns=2)
def testTaskTwo(arg):
    return arg, arg*10


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

    def delay(self, arg, sleepTime, respQ=None):
        time.sleep(sleepTime)
        if respQ is not None:
            respQ.put(arg)

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


def testConfirm(policy):
    success = True
    groupID = "exampleGroup"
    pool = kaas.pool.Pool(3, policy=policy)
    pool.registerGroup(groupID, TestWorker)

    refs = []
    for i in range(3):
        confRef, retRef = testTask.remote(i)
        refs.append(pool.run(groupID, "returnOne", args=[retRef], refDeps=[confRef]))

    retRefs = [ray.get(ref) for ref in refs]
    for expect, retRef in enumerate(retRefs):
        ret = ray.get(retRef)
        if expect != ret:
            print(f"Test Failed: expected '{expect}', got '{ret}'")
            success = False

    pool.shutdown()

    return success


def testConfirmMultiRet(policy):
    success = True
    groupID = "exampleGroup"
    pool = kaas.pool.Pool(3, policy=policy)
    pool.registerGroup(groupID, TestWorker)

    refs = []
    for i in range(3):
        confRef, retRef0, retRef1 = testTaskTwo.remote(i)
        ret1 = ray.get(retRef1)
        if ret1 != i*10:
            print(f"Test Failed: Wrapped task returned wrong value. Expected {i*10}, Got: {ret1}")

        refs.append(pool.run(groupID, "returnOne", args=[retRef0], refDeps=[confRef]))

    retRefs = [ray.get(ref) for ref in refs]
    for expect, retRef in enumerate(retRefs):
        ret = ray.get(retRef)
        if expect != ret:
            print(f"Test Failed: expected '{expect}', got '{ret}'")
            success = False

    pool.shutdown()

    return success


class TestLooper():
    """ServerTask"""
    def __init__(self, policy, nWorker, nGroup, duration):
        self.nGroup = nGroup

        self.loop = IOLoop.instance()
        self.readyClients = []

        self.pool = kaas.pool.Pool(nWorker, policy=policy)
        groups = ['g' + str(x) for x in range(nGroup)]
        self.rayQ = ray.util.queue.Queue()

        self.rng = np.random.default_rng(0)
        self.nOutstanding = 0
        self.done = 0

        self.startTime = time.time()
        self.stopTime = self.startTime + duration
        self.groupLats = {}
        # Theoretical peak throughput of the pool in requestSeconds/second
        theoreticPeak = nWorker
        # target submission rate per group in requestSeconds/second
        groupBudget = theoreticPeak / nGroup
        for groupIdx, groupID in enumerate(groups):
            self.groupLats[groupID] = []
            self.pool.registerGroup(groupID, TestWorkerNoGPU)
            runtime = ((groupIdx + 1)**2)*0.25
            rate = (groupBudget / runtime) * 0.8
            IOLoop.current().add_callback(self.groupSubmit, groupID, runtime, rate=rate)

        IOLoop.current().add_callback(self.gatherResponses)

    async def groupSubmit(self, groupID, runtime, rate=1):
        while time.time() < self.stopTime:
            self.pool.run(groupID, 'delay', args=[[groupID, time.time()], runtime],
                          kwargs={'respQ': self.rayQ})
            self.nOutstanding += 1
            await asyncio.sleep(self.rng.exponential(rate))

        self.pool.run(groupID, 'delay', args=[[groupID, time.time()], runtime],
                      kwargs={'respQ': self.rayQ})
        self.nOutstanding += 1
        self.done += 1

    async def gatherResponses(self):
        while self.done < self.nGroup:
            groupID, submitTime = await self.rayQ.get_async()
            self.groupLats[groupID].append(time.time() - submitTime)
            self.nOutstanding -= 1

        print("Done with main loop, cleaning up any lingering reqs")
        while self.nOutstanding > 0:
            groupID, submitTime = await self.rayQ.get_async()
            self.groupLats[groupID].append(time.time() - submitTime)
            self.nOutstanding -= 1

        print("Exiting test loop")
        IOLoop.instance().stop()


def testFairness(policy):
    nWorker = nResource
    nGroup = 2
    duration = 60

    testLoop = TestLooper(policy, nWorker, nGroup, duration)
    IOLoop.instance().start()

    gP50s = []
    gP90s = []
    for groupID, lats in testLoop.groupLats.items():
        npLats = np.array(lats)
        gP50s.append(np.quantile(npLats, 0.5))
        gP90s.append(np.quantile(npLats, 0.9))

    print("P50s")
    print(gP50s)
    print("P90s")
    print(gP90s)

    return True


POLICY = kaas.pool.policies.EXCLUSIVE

if __name__ == "__main__":
    availableTests = ['oneRet', 'multiRet', 'profiling', 'stress', 'confirm', 'fairness']

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
        elif test == 'confirm':
            ret = testConfirmMultiRet(policy)
        elif test == 'fairness':
            ret = testFairness(policy)
        else:
            raise ValueError("Unrecognized test: ", test)

        if not ret:
            print('FAIL')
        else:
            print('SUCCESS')
