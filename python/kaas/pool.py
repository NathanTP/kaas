#!/usr/bin/env python
import ray
from tornado.ioloop import IOLoop
import asyncio
import collections
import abc
import enum
import itertools
import logging
import traceback
from dataclasses import dataclass

from . import profiling

# logLevel = logging.DEBUG
logLevel = logging.WARN
logging.basicConfig(format="pool-%(levelname)s: %(message)s", level=logLevel)


PoolReq = collections.namedtuple('PoolReq', ['groupID', 'fName', 'num_returns', 'args', 'kwargs', 'resFuture'])

metrics = [
    "t_policy_run"  # Time from receiving the request until the worker is invoked
]


# This is more of a tuning parameter, but the pool will eventually start to
# slow down if you have too many outstanding requests. This is just a heuristic
# but users should try to stay below it if possible for max performance. You
# can always profile your own code and see.
maxOutstanding = 256


# Policy name constants
class policies(enum.IntEnum):
    BALANCE = enum.auto()
    EXCLUSIVE = enum.auto()
    STATIC = enum.auto()


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

    def shutdown(self):
        """User overrideable shutdown function. If there is any client-specific
        code to cleanup, run it here. Note that Ray doesn't delete the object
        before exiting, so __del__ doesn't necessarily run."""
        pass

    def _shutdown(self):
        """Will finish any pending work and then exit gracefully"""
        # Since this calls exit_actor, callers must use ray.wait on it to
        # verify shutdown. Unfortunately
        # https://github.com/ray-project/ray/issues/25280 means that errors are
        # hidden. The try/except makes sure something gets printed.
        self.shutdown()
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


def remote_with_confirmation(**kwargs):
    if 'num_returns' in kwargs:
        kwargs['num_returns'] += 1
    else:
        kwargs['num_returns'] = 2

    def _remote_with_confirmation(func):
        """Decorator to create a ray task that additionally returns a flag used as
        a proxy for task completion. Ray is generally not great about determining
        if references are ready without actually fetching them (ray.wait(...,
        fetch_local=False) only works in certain circumstances). With
        this, you can use the confRef rather than the actual values.

        WARNING: using .options(num_returns=N) will not work as expected. You
        must always use num_returns=N+1 because remote_with_confirmation adds
        an extra argument. If passing num_returns=N to the decorator, you can
        use the expected number of returns, this N+1 is only for .options().
        """
        @ray.remote(**kwargs)
        def with_confirmation_decorated(*args, **kwargs):
            rets = func(*args, **kwargs)
            if not isinstance(rets, tuple):
                rets = (rets,)

            return (True, *rets)

        return with_confirmation_decorated

    return _remote_with_confirmation


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
    def __init__(self, maxWorkers, profLevel=0, globalGroupID=None, strict=False):
        """The BalancePolicy balances requests across workers with no affinity
        or isolation between clients. Multiple clients may be registerd, but
        they must all use the same worker class.
            maxWorkers: Maximum number of concurrent workers to spawn
            profLevel: Sets the degree of profiling to be performed (lower is
                less invasive)
            globalGroupID: The balance policy supports multiple groups
                simultaneously. By default, it uses the provided per-request
                groupID to measure some metrics per-group. If groupID is
                provided here, all metrics (including policy-wide metrics) will
                be recorded under that group.
            strict: If true, the policy will reject any requests that can't be
                scheduled rather than queuing them internally.
        """
        self.strict = strict
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
            # scale up
            self.profs['n_cold_start'].increment(delta)
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

    def update(self, reqs=(), completedWorkers=()):
        """The balance policy will try to run requests if possible, but if
        there are no free workers, it may queue them up pending new
        completions.

        Returns: (newRuns, rejectedReqs)
            newRuns: {doneFuture: wID, ...}
            rejectedReqs: [req]
                Only returned if strict==True, otherwise only the newRuns dict
                is returned.
        """
        if self.workerClass is None:
            raise PoolError("No groups have been registered!")

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

        toSchedule = []
        for i in range(min(len(self.freeWorkers), len(self.pendingReqs))):
            toSchedule.append(self.pendingReqs.popleft())

        newRuns = {}
        for req in toSchedule:
            wID, worker = self.freeWorkers.popitem()

            logging.debug(f"Submitting request for group {req.groupID} to worker {wID}")
            resRefs = worker.actor._runWithCompletion.options(
                num_returns=req.num_returns + 1).remote(
                req.fName, *req.args,
                groupID=req.groupID, **req.kwargs)

            worker.nOutstanding += 1
            self.busyWorkers[wID] = worker

            req.resFuture.set_result(resRefs[1:])
            doneFut = asyncio.wrap_future(resRefs[1].future())
            newRuns[doneFut] = wID

        if self.strict:
            rejectedReqs = self.pendingReqs
            self.pendingReqs = collections.deque()
            return newRuns, rejectedReqs
        else:
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


class StaticPolicy(Policy):
    def __init__(self, maxWorkers, profLevel=0):
        """The static policy assigns workers to groups in a first-come-first
        served basis and returns an error when maxWorkers is exceeded. This
        policy cannot not re-balance clients or create/destroy workers after
        initial creation. It supports non-integer resource requirements for
        workers (must be <= 1)."""
        self.maxResource = maxWorkers

        # Number of resources assigned to workers (doesn't have to be an
        # integer)
        self.resourceUtilization = 0.0
        self.numWorkers = 0

        self.profLevel = profLevel
        self.profTop = createProfiler(level=self.profLevel)
        self.poolProfs = self.profTop.mod('pool')

        # {groupID: BalancePolicy}
        self.groups = {}

    def registerGroup(self, groupID, workerClass: PoolWorker, nWorker=1, workerResources=1):
        """Register a group with this policy.
        Arguments:
            groupID, workerClass: See the Policy ABC
            nWorker: Number of workers assigned to this group, this number will
                never change after registration.
            workerResources: Number of resources consumed by workers for this
                group. This must be <= 1.
        """
        requestedResources = workerResources * nWorker
        if self.resourceUtilization + requestedResources > self.maxResource:
            raise ValueError("Resources exhuasted, cannot register group")

        self.resourceUtilization += requestedResources

        group = BalancePolicy(nWorker, globalGroupID=groupID, profLevel=self.profLevel)
        group.registerGroup(groupID, workerClass)
        self.groups[groupID] = group

    def update(self, reqs=(), completedWorkers=()):
        # Split into per-group updates
        groupCompletions = collections.defaultdict(list)
        for gID, wID in completedWorkers:
            groupCompletions[gID].append(wID)

        groupReqs = collections.defaultdict(list)
        for req in reqs:
            groupReqs[req.groupID].append(req)

        # {gID: {fut: wID}}
        groupRuns = collections.defaultdict(dict)

        # Update all the groups
        for gID, group in self.groups.items():
            groupRuns[gID] = group.update(completedWorkers=groupCompletions[gID],
                                          reqs=groupReqs[gID])

        # Merge into a single list of new runs
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


class ExclusivePolicy(Policy):
    def __init__(self, maxWorkers, profLevel=0):
        self.maxWorkers = maxWorkers
        self.numWorkers = 0

        self.profLevel = profLevel
        self.profTop = createProfiler(level=self.profLevel)
        self.poolProfs = self.profTop.mod('pool')

        # {groupID: BalancePolicy}
        self.groups = {}

        # List of [req] that haven't been scheduled yet
        self.pendingReqs = []

    def registerGroup(self, groupID, workerClass: PoolWorker):
        group = BalancePolicy(0, globalGroupID=groupID, profLevel=self.profLevel, strict=True)
        group.registerGroup(groupID, workerClass)
        self.groups[groupID] = group

    # It's unlikely that we get more than one or two reqs per update so it's
    # just easier to handle each req independently rather than batch
    def handleReq(self, req):
        """Attempt to schedule req on its group. This may involve rebalancing
        groups. If the req can't be scheduled (group is busy and we decide not
        to rebalance), we will return it.

        returns:
            None: The req could not be scheduled
            newRuns: newly scheduled run
        """
        reqGroup = self.groups[req.groupID]
        newRuns, rejectedReqs = reqGroup.update([req])

        # If we succeed the first time, no need to try and scale
        if len(rejectedReqs) == 0:
            return newRuns

        # The group couldn't handle the request immediately, consider scaling
        if self.numWorkers < self.maxWorkers:
            # free to scale
            logging.debug(f"Scaling up {req.groupID} to {reqGroup.numWorkers + 1} from unused workers")
            reqGroup.scale(reqGroup.numWorkers + 1)
            self.numWorkers += 1
            newRuns, rejected = reqGroup.update(reqs=[req])
            assert len(rejected) == 0
            return newRuns
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

            if reqGroup.numWorkers == 0 or reqGroup.numWorkers + 1 < maxSize:
                # Scale the victim down and replace it in the dictionary to
                # ensure that the same victim doesn't get repeated scaled
                # down when there are multiple max-sized candidates
                # (dictionaries maintain insertion order). If the reqGroup
                # doesn't have any workers, we have to scale someone down to
                # avoid starvation. If it's within one of the max, scaling
                # would just swap their places and maintain imballance, leading
                # to unnecessary thrashing.
                victim = self.groups.pop(victimID)

                logging.debug(f"Scaling down {victim.globalGroupID} to {victim.numWorkers - 1} for {reqGroup.globalGroupID}")
                victim.scale(victim.numWorkers - 1)
                self.groups[victimID] = victim

                logging.debug(f"Scaling up {reqGroup.globalGroupID} to {reqGroup.numWorkers + 1} after rebalance")
                reqGroup.scale(reqGroup.numWorkers + 1)
                newRuns, rejected = reqGroup.update(reqs=[req])
                assert len(rejected) == 0
                return newRuns
            else:
                # Nothing we can do, just reject the request and let the caller
                # deal with it
                logging.debug(f"Rejected request for {req.groupID}")
                return None

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
            logging.debug(f"Received completion for {gID}")
            newRuns, rejectedRuns = self.groups[gID].update(completedWorkers=completions)
            assert len(newRuns) == 0 and len(rejectedRuns) == 0

        self.pendingReqs += reqs

        # Try to schedule any pending requests. If they can't be scheduled, put
        # them back on the list of pending requests. This preserves FCFS order.
        # This is kind of naive because it will always iterate all pending
        # requests, but the busyGroups optimization should make this not too
        # bad. Hopefully the number of pendingReqs won't get too big (it's on
        # the user of the pool to keep a reasonable number of outstanding
        # requests)
        remainingReqs = []
        busyGroups = set()
        for req in self.pendingReqs:
            # Optimization to avoid attempting to schedule a req that will be
            # rejected
            if req.groupID in busyGroups:
                remainingReqs.append(req)
            else:
                newRun = self.handleReq(req)
                if newRun is None:
                    remainingReqs.append(req)
                    busyGroups.add(req.groupID)
                else:
                    groupRuns[req.groupID] |= newRun
        self.pendingReqs = remainingReqs

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

    WARNING: There is currently a limitation that workers must use a resource
    that has the same availability as maxWorkers. In other words, if they
    require one GPU then maxWorkers MUST be the number of GPUs in the system.
    If they don't require a real resource, you MUST create a custom resource
    and set maxWorkers to whatever the resource limit is. I don't have a good
    way of enforcing this yet, so be warned. The symptom will likely be a hang.

    How this is different from Ray Actors and Tasks:
        You can think of the Pool as an interface to customize Ray's workers.
        Unlike tasks, workers can opportunistically cache data between requests
        to ammortize initialization costs. Unlike Actors, workers are not
        directly addressable and may be killed at any time. Multiple Policies
        may be implemented to provide more or less isolation between worker
        types and other fairness properties.
    """

    def __init__(self, maxWorkers, policy=policies.BALANCE, profLevel=0, policyArgs: dict = {}):
        """Arguments:
            maxWorkers: The total number of concurrent workers to allow. If
            there are multiple nodes in the system, each node will get an even
            share of maxWorkers.
            policy: Scheduling policy class to use (from policies enum)
        """
        self.profile = None
        self.dead = False

        self.pool = _PoolActor.remote(maxWorkers, policy=policy, profLevel=profLevel, policyArgs=policyArgs)

    def registerGroup(self, groupID, workerClass: PoolWorker, **policyArgs):
        """Register a new group of workers with this pool. Users must register
        at least one group before calling run().
        Arguments:
            groupID: Unique identifer for this tenant/worker/group
            workerClass: This will be used by the pool to run requests for this group
            policyArgs: Some scheduling policies can take additional argumentskwargs for the policy
        """
        if self.dead:
            raise PoolError("Pool has been shutdown")

        registerConfirm = self.pool.registerGroup.remote(groupID, workerClass, **policyArgs)
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

        if refDeps is None:
            refDeps = []

        return self.pool.run.options(num_returns=num_returns).\
            remote(groupID, methodName, *refDeps, num_returns=num_returns, args=args, kwargs=kwargs)

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

    def __init__(self, maxWorkers, policy, profLevel=0, policyArgs: dict = {}):
        """Arguments:
            maxWorkers: The total number of concurrent workers to allow
            policy: Scheduling policy class to use
        """
        self.profLevel = profLevel
        self.loop = IOLoop.instance()
        self.newReqQ = asyncio.Queue()
        self.pendingTasks = set()

        self.nPendingReqs = 0
        self.idle = asyncio.Event()
        self.idle.set()

        if policy is policies.BALANCE:
            self.policy = BalancePolicy(maxWorkers, profLevel=profLevel, **policyArgs)
        elif policy is policies.EXCLUSIVE:
            self.policy = ExclusivePolicy(maxWorkers, profLevel=profLevel, **policyArgs)
        elif policy is policies.STATIC:
            self.policy = StaticPolicy(maxWorkers, profLevel=profLevel, **policyArgs)
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

    async def registerGroup(self, groupID, workerClass: PoolWorker, **policyArgs):
        """Register a new group of workers with this pool. Users must register
        at least one group before calling run()."""
        self.policy.registerGroup(groupID, workerClass, **policyArgs)

    # run() keeps a Ray remote function alive for the caller so that it can
    # proxy the request to the pool and the response back to the caller. run()
    # will stay alive until the entire request is finished.
    async def run(self, groupID, methodName, *refDeps, num_returns=1, args=(), kwargs=None):
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
