#!/usr/bin/env python
import ray
from tornado.ioloop import IOLoop
import asyncio
import collections
import abc
import enum
import concurrent
import itertools
import logging
import traceback
from dataclasses import dataclass

from . import profiling

logLevel = logging.DEBUG
# logLevel = logging.WARN
logging.basicConfig(format="pool-%(levelname)s: %(message)s", level=logLevel)


PoolReq = collections.namedtuple('PoolReq', ['groupID', 'fName', 'num_returns', 'args', 'kwargs', 'resFuture'])

metrics = [
    "t_policy_run"  # Time from receiving the request until the worker is invoked
]

# When using the threadpool-based waiter, this is the number of waiter threads
# to use. It's a tradeoff between Python overheads and maximum number of
# outstanding requests.
N_WAIT_THREADS = 32


# Policy name constants
class policies(enum.IntEnum):
    BALANCE = enum.auto()
    EXCLUSIVE = enum.auto()


class PoolError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


class PoolWorker():
    """Inherit from this to make your class compatible with Pools.
    To include profiling data with your worker, you should call
    super().__init__ and use the provided self.profs variable."""
    def __init__(self, profLevel=0, defaultGroup=None):
        self.profLevel = profLevel
        self._defaultGroup = defaultGroup
        self._initProfs()

    def getProfs(self):
        return self._groupProfs

    def _initProfs(self):
        # Top-level profiling object. This is what is reported by _getProfile()
        self._profs = createProfiler(level=self.profLevel)

        # Profiling data for the currently active group
        if self._defaultGroup is None:
            self._groupProfs = self._profs
        else:
            self._groupProfs = self._profs.mod('groups').mod(self._defaultGroup)

    # Note: args and kwargs are passed directly (rather than in a list/dict)
    # because they may contain references that Ray would need to manage.
    def _runWithCompletion(self, fName, *args, groupID=None, **kwargs):
        """Wraps the method 'fname' with a completion signal as required by
        Pools.

        Arguments:
            fName: function to run within the actor.
            groupID: Some policies allow multiple groups to use the same
                worker. In some cases we may still wish to distinguish groups
                within the worker itself. If None, profs will be collected at
                the top-level (global to the worker).
            args,kwargs: These will be passed to the function.
        """
        if groupID is None:
            self._groupProfs = self._profs
        else:
            self._groupProfs = self._profs.mod('groups').mod(groupID)

        rets = getattr(self, fName)(*args, **kwargs)
        if not isinstance(rets, tuple):
            rets = (rets,)

        return (True, *rets)

    def _getProfile(self):
        # Users are not required to initialize profiling
        try:
            if self._profs is None:
                return createProfiler()
            else:
                latestProfs = self._profs
                self._initProfs()
                return latestProfs
        except AttributeError:
            raise PoolError("PoolWorkers must call super().__init__() in order to use profiling")

    def _shutdown(self):
        """Will finish any pending work and then exit gracefully"""
        # Since this calls exit_actor, callers must use ray.wait on it to
        # verify shutdown. Unfortunately
        # https://github.com/ray-project/ray/issues/25280 means that errors are
        # hidden. The try/except makes sure something gets printed.
        try:
            ray.actor.exit_actor()
        except ray.exceptions.AsyncioActorExit:
            # Ray uses an exception for exit_actor() for asyncio actors. I
            # don't want to put it outside the try/except in case there is a
            # different internal error in ray.actor.exit_actor()
            raise
        except Exception as e:
            logging.critical("Shutdown failed: " + traceback.format_exc())
            raise e


class Policy(abc.ABC):
    """A policy manages workers on behalf of the Pool class. It implements
    the core scheduling algorithm."""

    @abc.abstractmethod
    def __init__(self, numWorker, profLevel=0):
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

    @abc.abstractmethod
    def getProfile(self) -> profiling.profCollection:
        """Return any collected profiling data and reset profiling. The exact
        schema is up to each policy, but there is a common structure:

        {
            'pool': {
                global pool profs,
                {'groups':
                    groupID: {per-group pool-specific profs},
                    ...
                }
            }
            'workers': {
                global worker profs,
                {'groups':
                    groupID: {per-group worker profs}
                    ...
                }
            }
            }
        """
        pass


@dataclass
class _WorkerState():
    actor: PoolWorker
    nOutstanding: int


class BalancePolicy(Policy):
    def __init__(self, maxWorkers, profLevel=0, globalGroupID=None):
        """The BalancePolicy balances requests across workers with no affinity
        or isolation between clients. Multiple clients may be registerd, but
        they must all use the same worker class.
            maxWorkers: Maximum number of concurrent workers to spawn
            profLevel: Sets the degree of profiling to be performed (lower is less invasive)
            globalGroupID: The balance policy supports multiple groups
            simultaneously. By default, it uses the provided per-request
            groupID to measure some metrics per-group. If groupID is provided
            here, all metrics (including policy-wide metrics) will be recorded
            under that group.
        """
        self.maxWorkers = maxWorkers
        self.numWorkers = 0
        self.workerClass = None
        self.nextWID = 0

        self.profLevel = profLevel
        self.globalGroupID = globalGroupID
        self.initProfs()

        # References to profiling data from killed workers
        self.pendingProfs = []

        # {workerID: _WorkerState}
        self.freeWorkers = {}  # ready to recieve a request
        self.busyWorkers = {}  # currently running a request

        self.pendingReqs = collections.deque()

    def initProfs(self):
        self.profTop = createProfiler(self.profLevel)
        if self.globalGroupID is None:
            self.profs = self.profTop.mod('pool')
        else:
            self.profs = self.profTop.mod('pool').mod('groups').mod(self.globalGroupID)

    def scale(self, newMax):
        """Change the target number of workers."""
        delta = newMax - self.maxWorkers
        if delta > 0:
            self.profs['n_cold_start'].increment(delta)
            # scale up
            for i in range(delta):
                worker = self.workerClass.remote(profLevel=self.profLevel,
                                                 defaultGroup=self.globalGroupID)

                self.freeWorkers[self.nextWID] = _WorkerState(worker, 0)
                self.nextWID += 1

        elif delta < 0:
            # Scale down
            self.profs['n_killed'].increment(-delta)

            # try free workers first
            while len(self.freeWorkers) > 0 and delta < 0:
                _, toKill = self.freeWorkers.popitem()
                self.pendingProfs.append(toKill.actor._getProfile.remote())
                toKill.actor._shutdown.remote()
                delta += 1

            # next take from busy pool
            while delta < 0:
                wID, toKill = self.busyWorkers.popitem()
                self.pendingProfs.append(toKill.actor._getProfile.remote())
                toKill.actor._shutdown.remote()
                delta += 1
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
        if self.workerClass is None:
            self.workerClass = workerClass
        elif self.workerClass.__class__ != workerClass.__class__:
            raise PoolError("The BalancePolicy does not support multiple worker types")

        # This should only happen on the first group registration. scale()
        # expects the max to have changed so we trick it by seting maxWorkers
        # to 0 before scaling.
        if self.numWorkers != self.maxWorkers:
            assert self.numWorkers == 0
            realMax = self.maxWorkers
            self.maxWorkers = 0
            self.scale(realMax)

    def update(self, reqs=(), completedWorkers=(), force=False):
        """The balance policy will try to run requests if possible, but if
        there are no free workers, it may queue them up pending new
        completions. If force==True, it will submit multiple requests to
        workers right away. This isn't ideal from a load-balance perspective,
        but it means that requests are guaranteed to run even if the pool is
        scaled down."""
        if self.workerClass is None:
            raise PoolError("No groups have been registered!")

        if len(reqs) != 0 and self.maxWorkers == 0:
            raise PoolError("Requested worker but maxWorkers==0")

        for wID in completedWorkers:
            doneWorker = self.busyWorkers.pop(wID, None)

            # We don't need to do anything for completions of dead workers
            if doneWorker is not None:
                doneWorker.nOutstanding -= 1
                if doneWorker.nOutstanding == 0:
                    self.freeWorkers[wID] = doneWorker
                else:
                    self.busyWorkers[wID] = doneWorker

        self.pendingReqs += reqs

        if force:
            toSchedule = self.pendingReqs
            self.pendingReqs = []
        else:
            toSchedule = []
            for i in range(min(len(self.freeWorkers), len(self.pendingReqs))):
                toSchedule.append(self.pendingReqs.popleft())

        newRuns = {}
        for req in toSchedule:
            if len(self.freeWorkers) > 0:
                wID, worker = self.freeWorkers.popitem()
            else:
                assert force
                wID, worker = self.busyWorkers.popitem()

            logging.debug(f"Submitting request for group: {req.groupID}")
            resRefs = worker.actor._runWithCompletion.options(
                num_returns=req.num_returns + 1).remote(
                req.fName, *req.args,
                groupID=req.groupID, **req.kwargs)

            worker.nOutstanding += 1
            self.busyWorkers[wID] = worker

            req.resFuture.set_result(resRefs[1:])
            doneFut = asyncio.wrap_future(resRefs[1].future())
            newRuns[doneFut] = wID

        return newRuns

    async def getProfile(self):
        """Schema: There is only one pool so there are no per-group pool profs
        (top-level pool profs are aggregated across all groups).

        """
        if len(self.busyWorkers) != 0:
            raise PoolError("Cannot call getProfile() when there are pending requests")

        for worker in self.freeWorkers.values():
            self.pendingProfs.append(worker.actor._getProfile.remote())

        workerStats = await asyncio.gather(*self.pendingProfs)
        self.pendingProfs = []

        workerProfs = self.profTop.mod('workers')
        for workerStat in workerStats:
            workerProfs.merge(workerStat)

        latestProfs = self.profTop
        self.initProfs()

        return latestProfs

    def shutdown(self):
        confirmations = []
        for worker in itertools.chain(self.freeWorkers.values(), self.busyWorkers.values()):
            confirmations.append(worker.actor._shutdown.remote())

        ray.wait(confirmations, num_returns=len(confirmations))


class ExclusivePolicy(Policy):
    def __init__(self, maxWorkers, profLevel=0):
        self.maxWorkers = maxWorkers
        self.numWorkers = 0

        self.profLevel = profLevel
        self.profTop = createProfiler(level=self.profLevel)
        self.poolProfs = self.profTop.mod('pool')

        # {groupID: BalancePolicy}
        self.groups = {}

    def registerGroup(self, groupID, workerClass: PoolWorker):
        group = BalancePolicy(0, globalGroupID=groupID, profLevel=self.profLevel)
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
            self.numWorkers += 1
        else:
            # Gonna have to consider shrinking someone
            victimID = None
            maxSize = 0

            # Find the group with the most workers. Break ties by order in
            # dictionary which will be maintained in order of least-recently
            # evicted.
            for candidateID, candidate in self.groups.items():
                if candidate.numWorkers > maxSize:
                    victimID = candidateID
                    maxSize = candidate.numWorkers

            if reqGroup.numWorkers < maxSize:
                # Scale the victim down and replace it in the dictionary to
                # ensure that the same victim doesn't get repeated scaled
                # down when there are multiple max-sized candidates
                # (dictionaries maintain insertion order)
                victim = self.groups.pop(victimID)

                logging.debug(f"Scaling {victim.globalGroupID} down for {reqGroup.globalGroupID}")
                victim.scale(victim.numWorkers - 1)
                self.groups[victimID] = victim

                reqGroup.scale(reqGroup.numWorkers + 1)
            # else: nothing we can do, just queue up the request on the
            # reqGroup even though it doesn't have any free workers

        return reqGroup.update(reqs=[req], force=True)

    def update(self, reqs=(), completedWorkers=()):
        # Split into per-group updates
        groupCompletions = {}
        for gID, wID in completedWorkers:
            if gID not in groupCompletions:
                groupCompletions[gID] = [wID]
            else:
                groupCompletions[gID].append(wID)

        # {gID: {fut: wID}}
        groupRuns = collections.defaultdict(dict)

        # Update all the groups with completions so that they have the most
        # up-to-date state in case we have to scale
        for gID, completions in groupCompletions.items():
            groupRuns[gID] = self.groups[gID].update(completedWorkers=completions)

        for req in reqs:
            groupRuns[req.groupID] |= self.handleReq(req)

        newRuns = {}
        for gID, runs in groupRuns.items():
            for fut, wID in runs.items():
                newRuns[fut] = (gID, wID)

        return newRuns

    async def getProfile(self):
        """Schema: Each group get's it's own private pool so we report
        per-group pool profs. Even though each pool only has one group, we
        still report the per-group worker-specific stats per pool to match the
        behavior of BALANCE.
        """
        workerProfs = self.profTop.mod('workers')
        poolProfs = self.profTop.mod('pool')

        groupPoolProfs = await asyncio.gather(*[group.getProfile() for group in self.groups.values()])

        for groupID, groupProf in zip(self.groups.keys(), groupPoolProfs):
            poolProfs.merge(groupProf.mod('pool'))
            workerProfs.merge(groupProf.mod('workers'))

        latestProfs = self.profTop
        self.profTop = createProfiler(level=self.profLevel)
        self.profs = self.profTop.mod('pool')

        return latestProfs

    def shutdown(self):
        for group in self.groups.values():
            group.shutdown()


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

    def __init__(self, maxWorkers, policy=policies.BALANCE, nGPUs=1, profLevel=0):
        """Arguments:
            maxWorkers: The total number of concurrent workers to allow
            policy: Scheduling policy class to use (from policies enum)
            nGPUs: Number of GPUs to assign to each worker. Pool does not
                   currently support per-group nGPUs.
        """
        self.profile = None
        self.dead = False

        self.pool = _PoolActor.remote(maxWorkers, policy=policy, nGPUs=nGPUs, profLevel=profLevel)

    def registerGroup(self, groupID, workerClass: PoolWorker):
        """Register a new group of workers with this pool. Users must register
        at least one group before calling run().
        """
        if self.dead:
            raise PoolError("Pool has been shutdown")

        registerConfirm = self.pool.registerGroup.remote(groupID, workerClass)
        ray.get(registerConfirm)

    def registerGroupAsync(self, groupID, workerClass: PoolWorker):
        """Like registerGroup() but returns immediately with a reference that
        must be waited on before sending requests for this group."""
        if self.dead:
            raise PoolError("Pool has been shutdown")

        return self.pool.registerGroup.remote(groupID, workerClass)

    def run(self, groupID, methodName, num_returns=1, args=(), kwargs=None, refDeps=None):
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
        if self.dead:
            raise PoolError("Pool has been shutdown")

        return self.pool.run.options(num_returns=num_returns).\
            remote(groupID, methodName, num_returns=num_returns, args=args, kwargs=kwargs, refDeps=refDeps)

    def getProfile(self) -> profiling.profCollection:
        """Returns any profiling data collected so far in this pool and resets
        the profiler. The profile will have one module per registered group as
        well as pool-wide metrics.

        WARNING: The caller must ensure that there is no pending work on the
        pool before calling getProfile(). Calling getProfile() while there are
        outstanding requests will result in undefined behavior."""
        if self.dead:
            return self.profile
        else:
            return ray.get(self.pool.getProfile.remote())

    def shutdown(self):
        """Free any resources associated with this pool. The pool object
        remains and getProfile() will continue to work, but no other methods
        will be valid. A pool cannot be restarted. The user is responsible for
        ensuring that no work is currently pending on the actor, failure to do
        so results in undefined behavior."""
        self.profile = ray.get(self.pool.getProfile.remote())
        self.dead = True
        ray.wait([self.pool.shutdown.remote()], num_returns=1)


@ray.remote
class _PoolActor():
    """The Pool is implemented as an asyncio actor that internally runs an event
    loop (handleEvent). Events include new requests via run() or worker
    completions via a "Done" reference. For each event, the Pool updates the
    Policy which handles the actual pool management and worker invocation."""

    def __init__(self, maxWorkers, policy, nGPUs=0, profLevel=0):
        """Arguments:
            maxWorkers: The total number of concurrent workers to allow
            policy: Scheduling policy class to use
            nGPUs: Number of GPUs to assign to each worker. Pool does not
                   currently support per-group nGPUs.
        """
        self.profLevel = profLevel
        self.loop = IOLoop.instance()
        self.newReqQ = asyncio.Queue()
        self.pendingTasks = set()

        self.nPendingReqs = 0
        self.idle = asyncio.Event()
        self.idle.set()

        self.asyncioLoop = asyncio.get_running_loop()
        self.threadPool = concurrent.futures.ThreadPoolExecutor(max_workers=N_WAIT_THREADS)

        if policy is policies.BALANCE:
            self.policy = BalancePolicy(maxWorkers, profLevel=profLevel)
        elif policy is policies.EXCLUSIVE:
            self.policy = ExclusivePolicy(maxWorkers, profLevel=profLevel)
        else:
            raise PoolError("Unrecognized policy: " + str(policy))

        self.profTop = createProfiler(self.profLevel)
        self.profs = self.profTop.mod('pool')
        self.groupProfs = self.profs.mod('groups')

        # {doneFut: wID}
        self.pendingRuns = {}

        self.getTask = asyncio.create_task(self.newReqQ.get(), name='q')
        self.pendingTasks.add(self.getTask)

        self.loop.add_callback(self.handleEvent)

    # This one uses a thread pool. It avoids materializing the refs, but it can
    # only handle a limited number of outstanding requests. Scaling up the
    # number of threads can have an adverse impact on overall performance.
    async def _waitRefs(self, refs):
        await self.asyncioLoop. \
            run_in_executor(self.threadPool,
                            lambda: ray.wait(refs, fetch_local=False, num_returns=len(refs)))

    # This one uses normal asyncio.wait but it materializes the references
    # which could be slow and wasteful depending on what the references point
    # to. Ideally, we would have a fetch_local=False option for ray futures,
    # but that is WIP: https://github.com/ray-project/ray/issues/25415
    # async def _waitRefs(self, refs):
    #     await asyncio.wait(refs, return_when=asyncio.ALL_COMPLETED)

    async def registerGroup(self, groupID, workerClass: PoolWorker):
        """Register a new group of workers with this pool. Users must register
        at least one group before calling run()."""
        self.policy.registerGroup(groupID, workerClass)

    # run() keeps a Ray remote function alive for the caller so that it can
    # proxy the request to the pool and the response back to the caller. run()
    # will stay alive until the entire request is finished.
    async def run(self, groupID, methodName, num_returns=1, args=(), kwargs=None, refDeps=None):
        """Schedule some work on the pool.

        Arguments:
            groupID: Which worker type to use.
            methodName:
                Method name within the worker to invoke.
            num_returns:
                Number of return values for this invocation.
            args, kwargs:
                Arguments to pass to the method
            refDeps:
                list of references to wait for before submitting the function
                to the pool. This field is not required, but it can result in
                much better utilization by preventing upstream dependencies
                from blocking workers.

        Returns:
            ref(return value) -
                Since users are calling run() as an actor method, Ray will wrap
                it in another reference. This means that users actually see
                ref(ref(return value)) and must dereference twice to get the
                actual return value
        """
        if kwargs is None:
            kwargs = {}

        self.nPendingReqs += 1
        self.idle.clear()

        resFuture = asyncio.get_running_loop().create_future()
        req = PoolReq(groupID=groupID, fName=methodName,
                      num_returns=num_returns, args=args, kwargs=kwargs,
                      resFuture=resFuture)

        if refDeps is not None:
            await self._waitRefs(refDeps)

        with profiling.timer('t_policy_run', self.groupProfs.mod(groupID)):
            self.newReqQ.put_nowait(req)
            res = await resFuture

        # Match the behavior of normal Ray tasks or actors. They will return a
        # single value for num_returns=1 or an iterable for num_returns >= 1.
        if num_returns == 1:
            return res[0]
        else:
            return tuple(res)

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

        self.nPendingReqs -= len(doneWorkers)
        if self.nPendingReqs == 0:
            self.idle.set()

        self.loop.add_callback(self.handleEvent)

    async def getProfile(self) -> profiling.profCollection:
        await self.idle.wait()

        self.profTop.merge(await self.policy.getProfile())

        latestProfs = self.profTop
        self.profTop = createProfiler(self.profLevel)
        self.profs = self.profTop.mod('pool')
        self.groupProfs = self.profs.mod('groups')

        return latestProfs

    async def shutdown(self):
        # Since this calls exit_actor, callers must use ray.wait on it to
        # verify shutdown. Unfortunately
        # https://github.com/ray-project/ray/issues/25280 means that errors are
        # hidden. The try/except makes sure something gets printed.
        try:
            self.policy.shutdown()
            ray.actor.exit_actor()
        except ray.exceptions.AsyncioActorExit:
            # Ray uses an exception for exit_actor() for asyncio actors. I
            # don't want to put it outside the try/except in case there is a
            # different internal error in ray.actor.exit_actor()
            raise
        except Exception as e:
            logging.critical("Shutdown failed: " + traceback.format_exc())
            raise e


def mergePerGroupStats(base, delta):
    for cID, deltaClient in delta.items():
        if cID in base:
            base[cID].merge(deltaClient)
        else:
            base[cID] = deltaClient


def createProfiler(level):
    if level > 0:
        return profiling.profCollection(detail=True)
    else:
        return profiling.profCollection()
