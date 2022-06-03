#!/usr/bin/env python
import ray
from tornado.ioloop import IOLoop
import asyncio
import collections
import abc


PoolReq = collections.namedtuple('PoolReq', ['groupID', 'fName', 'num_returns', 'args', 'kwargs', 'resFuture'])


class PoolError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return "Pool Error: " + self.msg


class PoolWorker():
    """Inherit from this to make your class compatible with Pools."""
    def _runWithCompletion(self, fName, args, kwargs):
        """Wraps the method 'fname' with a completion signal as required by
        Pools"""
        rets = getattr(self, fName)(*args, **kwargs)
        if not isinstance(rets, tuple):
            rets = (rets,)

        return (True, *rets)


class Policy(abc.ABC):
    """A policy manages workers on behalf of the Pool class. It implements
    the core scheduling algorithm."""

    @abc.abstractmethod
    def __init__(self, numWorker):
        """Args:
               numWorker: Maximum number of concurrent workers to spawn
        """
        pass

    @abc.abstractmethod
    def registerGroup(self, groupID, workerClass: PoolWorker):
        """Register a new group with this policy. Group registration happens
        asynchronously and is not guaranteed to finish immediately. Users
        should ray.get() the return value to ensure that the group is fully
        registered before submitting requests.

        Returns: success boolean
        """
        pass

    @abc.abstractmethod
    def update(self, reqs=None, completedWorkers=None):
        """Update the policy with any new requests and/or newly completed
        workers. If any requests can be scheduled, run them and return their
        result references.
            reqs:
                [PoolReq, ...]

            completedWorkers:
                [workerID, ...]

            Returns:
                {doneFut: wID}
        """
        pass


class BalancePolicy(Policy):
    def __init__(self, maxWorkers):
        """The BalancePolicy balances requests across workers with no affinity
        or isolation between clients. Multiple clients may be registerd, but
        they must all use the same worker class.
            maxWorkers: Maximum number of concurrent workers to spawn
        """
        self.maxWorkers = maxWorkers
        self.numWorkers = 0
        self.workerClass = None
        self.nextWID = 0

        # {workerID: worker actor handle}
        self.freeWorkers = {}  # ready to recieve a request
        self.busyWorkers = {}  # currently running a request
        self.deadWorkers = {}  # finishing up its last request

        self.pendingReqs = collections.deque()

    def scale(self, newMax):
        """Change the target number of workers."""
        delta = newMax - self.maxWorkers
        if delta > 0:
            # scale up
            for i in range(delta):
                self.freeWorkers[self.nextWID] = self.workerClass.remote()
                self.nextWID += 1

        elif delta < 0:
            # Scale down

            # try free workers first
            while len(self.freeWorkers) > 0 and delta > 0:
                _, toKill = self.freeWorkers.popitem()
                toKill.terminate.remote()
                delta -= 1

            # next take from busy pool
            while delta > 0:
                wID, toKill = self.busyWorkers.popitem()
                toKill.terminate.remote()
                self.deadWorkers[wID] = toKill
                delta -= 1
        else:
            # don't actually have to scale
            pass

        self.maxWorkers = newMax
        self.numWorkers = self.maxWorkers

    def registerGroup(self, groupID, workerClass: PoolWorker):
        """The BalancePolicy initializes the maximum number of workers as soon
        as the first worker type is registered and never messes with them
        again. It does not distinguish between groupIDs because there will
        only ever be one worker class."""
        if self.workerClass is not None:
            raise PoolError("The BalancePolicy does not support multiple worker types")
        else:
            self.workerClass = workerClass

        # This should only happen on the first group registration
        if self.numWorkers != self.maxWorkers:
            assert self.numWorkers == 0
            for i in range(self.maxWorkers):
                self.freeWorkers[self.nextWID] = self.workerClass.remote()
                self.nextWID += 1

            self.numWorkers = self.maxWorkers

    def update(self, reqs=(), completedWorkers=()):
        if self.workerClass is None:
            raise PoolError("No groups have been registered!")

        if self.maxWorkers == 0:
            raise PoolError("Requested worker but maxWorkers==0")

        # Update internal state
        self.pendingReqs += reqs
        for wID in completedWorkers:
            doneWorker = self.busyWorkers.pop(wID, None)
            if doneWorker is not None:
                self.freeWorkers[wID] = doneWorker
            else:
                del self.deadWorkers[wID]

        toSchedule = []
        for i in range(min(len(self.freeWorkers), len(self.pendingReqs))):
            toSchedule.append(self.pendingReqs.popleft())

        newRuns = {}
        for req in toSchedule:
            wID, worker = self.freeWorkers.popitem()
            self.busyWorkers[wID] = worker

            #XXX
            nReturns = req.num_returns + 1
            resRefs = worker._runWithCompletion.options(
                num_returns=nReturns).remote(
                req.fName, req.args, req.kwargs)

            req.resFuture.set_result(resRefs[1:])
            doneFut = asyncio.wrap_future(resRefs[1].future())
            newRuns[doneFut] = wID

        return newRuns


class ExclusivePolicy(Policy):
    def __init__(self, maxWorkers):
        self.maxWorkers = maxWorkers
        self.numWorkers = 0

        # {groupID: BalancePolicy}
        self.groups = {}

    def registerGroup(self, groupID, workerClass: PoolWorker):
        group = BalancePolicy(0)
        group.registerGroup(groupID, workerClass)
        self.groups[groupID] = group

    # It's unlikely that we get more than one or two reqs per update so it's
    # just easier to handle each req independently rather than batch
    def handleReq(self, req):
        reqGroup = self.groups[req.groupID]
        if len(reqGroup.freeWorkers) > 0:
            # Group can handle the request right away
            pass
        elif self.numWorkers < self.maxWorkers:
            # free to scale
            reqGroup.scale(reqGroup.numWorkers + 1)
        else:
            # Gonna have to consider shrinking someone
            victimID = None
            maxSize = 0
            for candidateID, candidate in self.groups.items():
                if candidate.numWorkers > maxSize:
                    victimID = candidateID
                    maxSize = candidate.numWorkers

            if reqGroup.numWorkers < maxSize:
                # Scale the victim down and replace it in the dictionary to
                # ensure that the same victim doesn't get repeated scaled
                # down when there are multiple max-sized candidates
                # (dictionaries maintain insertion order)
                victim = self.groups.popitem(victimID)
                victim.scale(victim.numWorkers - 1)
                self.groups[victimID] = victim

                reqGroup.scale(reqGroup.numWorkers + 1)
            # else: nothing we can do, just queue up the request on the
            # reqGroup even though it doesn't have any free workers

        return reqGroup.update(reqs=[req])

    def update(self, reqs=(), completedWorkers=()):
        # Split into per-group updates
        groupCompletions = {}
        for gID, wID in completedWorkers:
            if gID not in groupCompletions:
                groupCompletions[gID] = [wID]
            else:
                groupCompletions[gID].append(wID)

        # Update all the groups with completions so that they have the most
        # up-to-date state in case we have to scale
        for gID, completions in groupCompletions.items():
            self.groups[gID].update(completedWorkers=completions)

        newRuns = {}
        for req in reqs:
            # If there were some unhandled requests piled up, the group might
            # return multiple new runs
            groupRuns = self.handleReq(req)
            for fut, wID in groupRuns.items():
                newRuns[fut] = (req.groupID, wID)

        return newRuns


class Pool():
    """Generic worker pool class. Users must register at least one named group
    of workers. Worker groups are the unit of isolation in the Pool. They are
    defined by a user-defined group name and the worker class to use within the
    group. Once registered, users may request work via run(). The Pool ensures
    that a worker is running for each request and may kill existing workers to
    make room. The exact Pool behavior is defined by the Policy. While users
    can invoke different methods on the workers, it is good practice to limit
    workers to only one predictable behavior so that policies can optimize for
    worker properties.

    How this is different from Ray Actors and Tasks:
        You can think of the Pool as an interface to customize Ray's workers.
        Unlike tasks, workers can opportunistically cache data between requests
        to ammortize initialization costs. Unlike Actors, workers are not
        directly addressable and may be killed at any time. Multiple Policies
        may be implemented to provide more or less isolation between worker
        types and other fairness properties.
    """

    def __init__(self, maxWorkers, policy: Policy = BalancePolicy, nGPUs=0):
        """Arguments:
            maxWorkers: The total number of concurrent workers to allow
            policy: Scheduling policy class to use
            nGPUs: Number of GPUs to assign to each worker. Pool does not
                   currently support per-group nGPUs.
        """
        self.pool = _PoolActor.remote(maxWorkers, policy=policy, nGPUs=nGPUs)

    def registerGroup(self, groupID, workerClass: PoolWorker):
        """Register a new group of workers with this pool. Users must register
        at least one group before calling run().
        """
        registerConfirm = self.pool.registerGroup.remote(groupID, workerClass)
        ray.get(registerConfirm)

    def registerGroupAsync(self, groupID, workerClass: PoolWorker):
        """Like registerGroup() but returns immediately with a reference that
        must be waited on before sending requests for this group."""
        return self.pool.registerGroup.remote(groupID, workerClass)

    def run(self, groupID, methodName, num_returns=1, args=(), kwargs=None):
        """Schedule some work on the pool.

        Arguments:
            groupID: Which worker type to use.
            methodName: Method name within the worker to invoke.
            num_returns: Number of return values for this invocation.
            args, kwargs: Arguments to pass to the method

        Returns:
            Worker return references -
                The pool returns as soon as the request is issued to a worker.
                It returns a reference to whatever the worker invocation would
                return (i.e. a reference or list of references to the remote
                method's returns). In practice, this means that callers will
                need to dereference twice: once to get the worker return
                reference(s), and again to get the value of those return(s).
        """
        return self.pool.run.remote(groupID, methodName, num_returns=num_returns, args=args, kwargs=kwargs)


@ray.remote
class _PoolActor():
    """The Pool is implemented as an asyncio actor that internally runs an event
    loop (handleEvent). Events include new requests via run() or worker
    completions via a "Done" reference. For each event, the Pool updates the
    Policy which handles the actual pool management and worker invocation."""

    def __init__(self, maxWorkers, policy: Policy = BalancePolicy, nGPUs=0):
        """Arguments:
            maxWorkers: The total number of concurrent workers to allow
            policy: Scheduling policy class to use
            nGPUs: Number of GPUs to assign to each worker. Pool does not
                   currently support per-group nGPUs.
        """
        self.loop = IOLoop.instance()
        self.newReqQ = asyncio.Queue()
        self.pendingTasks = set()

        self.policy = policy(maxWorkers)

        # {doneFut: wID}
        self.pendingRuns = {}

        self.getTask = asyncio.create_task(self.newReqQ.get(), name='q')
        self.pendingTasks.add(self.getTask)

        self.loop.add_callback(self.handleEvent)

    async def registerGroup(self, groupID, workerClass: PoolWorker):
        """Register a new group of workers with this pool. Users must register
        at least one group before calling run()."""
        self.policy.registerGroup(groupID, workerClass)

    # run() keeps a Ray remote function alive for the caller so that it can
    # proxy the request to the pool and the response back to the caller. run()
    # will stay alive until the entire request is finished.
    async def run(self, groupID, methodName, num_returns=1, args=(), kwargs=None):
        """Schedule some work on the pool.

        Arguments:
            groupID: Which worker type to use.
            methodName: Method name within the worker to invoke.
            num_returns: Number of return values for this invocation.
            args, kwargs: Arguments to pass to the method

        Returns:
            ref(return value) -
                Since users are calling run() as an actor method, Ray will wrap
                it in another reference. This means that users actually see
                ref(ref(return value)) and must dereference twice to get the
                actual return value
        """
        if kwargs is None:
            kwargs = {}

        resFuture = asyncio.get_running_loop().create_future()
        req = PoolReq(groupID=groupID, fName=methodName,
                      num_returns=num_returns, args=args, kwargs=kwargs,
                      resFuture=resFuture)

        #XXX add code to wait for inputs to be ready

        self.newReqQ.put_nowait(req)
        res = await resFuture

        # Match the behavior of normal Ray tasks or actors. They will return a
        # single value for num_returns=1 or an iterable for num_returns >= 1.
        if num_returns == 1:
            return res[0]
        else:
            return res

    async def handleEvent(self):
        """Event handler for worker completions and new requests (via run())."""
        done, pending = await asyncio.wait(self.pendingTasks, return_when=asyncio.FIRST_COMPLETED)
        self.pendingTasks = pending

        newReq = ()
        doneWorkers = []
        for doneTask in done:
            if doneTask is self.getTask:
                newReq = [doneTask.result()]

                # Start waiting for a new req
                self.getTask = asyncio.create_task(self.newReqQ.get(), name='q')
                self.pendingTasks.add(self.getTask)
            else:
                doneWorkers.append(self.pendingRuns.pop(doneTask))

        newRuns = self.policy.update(reqs=newReq, completedWorkers=doneWorkers)
        self.pendingRuns |= newRuns
        self.pendingTasks |= newRuns.keys()

        self.loop.add_callback(self.handleEvent)
