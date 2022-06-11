# from . import _server_light as _server
from . import _server_prof as _server
from . import profiling
from . import pool

import ray


class _rayKV():
    """A libff.kv compatible(ish) KV store for Ray plasma store. Unlike normal
    KV stores, plasma does not allow arbitrary key names which is incompatible
    with libff.kv. Instead, we add a new 'flush' method that returns handles
    for any newly added objects that can be passed back to the caller."""

    def __init__(self):
        self.newRefs = {}

    def put(self, k, v, profile=None, profFinal=True):
        self.newRefs[k] = ray.put(v)

    def get(self, k, profile=None, profFinal=True):
        # Ray returns immutable objects so we have to make a copy
        ref = ray.cloudpickle.loads(k)
        return ray.get(ref)

    def delete(self, *keys, profile=None, profFinal=True):
        # Ray is immutable
        pass

    def destroy(self):
        # Ray plasma store associated with the ray session, can't clean up
        # independently
        pass


def init():
    _server.initServer()


def putObj(obj):
    """KaaS uses the ray object store behind the scenes for bufferSpec inputs,
    but it has to handle it carefully. You should use this function to write
    objects intended for kaas bufferSpecs. You can also pass a ray reference to
    be wrapped appropriately."""
    if isinstance(obj, ray._raylet.ObjectRef):
        ref = obj
    else:
        ref = ray.put(obj)

    return ray.cloudpickle.dumps(ref)


# Request serialization/deserialization is a pretty significant chunk of time,
# but they only change in very minor ways each time so we cache the
# deserialization here.
reqCache = {}


def invoke(rawReq, stats=None, clientID=None):
    """Handle a single KaaS request in the current thread/actor/task. GPU state
    is cached and no attempt is made to be polite in sharing the GPU. The user
    should ensure that the only GPU-enabled functions running are
    kaasServeRay(). Returns a list of handles of outputs (in the same order as
    the request).

    rawReq: (kaasReqDense reference, rename map)
        The first argument is a reference to a kaasReqDense, this may be cached
        by the kaas server. The second argument is a rename map
        {bufferName -> newRef} that reassigns keys (ray references) to names in
        the requests buffer list.  This helps avoid extra
        serialization/deserialization (which is expensive for kaasReq in some
        cases).

    Returns: tuple (i.e. multiple returns) of references to output objects
    """
    kv = _rayKV()
    with profiling.timer('t_e2e', stats):
        reqRef = rawReq[0]
        renameMap = rawReq[1]
        with profiling.timer("t_parse_request", stats):
            if reqRef in reqCache:
                req = reqCache[reqRef]
            else:
                req = ray.get(reqRef)
                reqCache[reqRef] = req

            req.reKey(renameMap)

        visibleOutputs = _server.kaasServeInternal(req, kv, stats, clientID=clientID)

    returns = []
    for outKey in visibleOutputs:
        returns.append(kv.newRefs[outKey])

    return tuple(returns)


@ray.remote(num_gpus=1)
class invokerActor(pool.PoolWorker):
    def __init__(self):
        """invokerActor is the ray version of a kaas worker, it is assigned a
        single GPU and supports requests from multiple clients."""
        super().__init__()
        init()

        # {clientID -> profiling.profCollection}
        self.stats = {}

    def ensureReady(self):
        """A dummy call that will only return once ray has fully initialized
        this actor. Used to ensure warm starts for subsequent calls."""
        pass

    def invoke(self, req, clientID=None):
        """Invoke the kaasReq req on this actor. You may optionally pass a
        clientID. clientIDs are used for per-client profiling and may affect
        scheduling/caching policies."""
        # if clientID not in self.stats:
        #     self.stats[clientID] = profiling.profCollection()

        clientProfs = self.profs.mod(clientID)
        with profiling.timer('t_e2e', clientProfs):
            res = invoke(req, clientProfs)

        return res


@ray.remote(num_gpus=1)
def invokerTask(req):
    """Handle a single KaaS request as a ray task. This isn't the recommended
    way to use kaas (it is intended for persistent allocations like actors),
    but it can be useful from time to time."""
    init()

    return invoke(req)
