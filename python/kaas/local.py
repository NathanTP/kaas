from . import profiling
import cloudpickle as pickle
import copy
from . import _server_prof as _server
# from ._server_prof import kaasServeInternal


class KVKeyError(Exception):
    def __init__(self, key):
        self.key = key

    def __str__(self):
        return "Key " + str(self.key) + " does not exist"


class LocalKV():
    """A baseline "local" kv store. Really just a dictionary. Note: no copy is
    made, be careful not to re-use the reference."""

    def __init__(self, copyObjs=False, serialize=True):
        """If copyObjs is set, all puts and gets will make deep copies of the
        object, otherwise the existing objects will be stored. If
        serialize=True, objects will be serialized in the store. This isn't
        needed for the local kv store, but it mimics the behavior of a real KV
        store better."""
        self.store = {}
        self.copy = copyObjs
        self.serialize = serialize

    def put(self, k, v, profile=None, profFinal=True):
        with profiling.timer("t_serialize", profile, final=profFinal):
            if self.serialize:
                v = pickle.dumps(v)
            elif self.copy:
                v = copy.deepcopy(v)

        with profiling.timer("t_write", profile, final=profFinal):
            self.store[k] = v

    def get(self, k, profile=None, profFinal=True):
        if isinstance(k, list):
            keys = k
        else:
            keys = [k]

        values = []
        for key in keys:
            with profiling.timer("t_read", profile, final=False):
                try:
                    raw = self.store[key]
                except KeyError:
                    raise KVKeyError(key)

            with profiling.timer("t_deserialize", profile, final=False):
                if self.serialize:
                    values.append(pickle.loads(raw))
                elif self.copy:
                    values.append(copy.deepcopy(raw))
                else:
                    values.append(raw)

        if profFinal and profile is not None:
            profile['t_read'].increment()
            profile['t_deserialize'].increment()

        if isinstance(k, list):
            return values
        else:
            return values[0]

    def delete(self, keys, profile=None, profFinal=True):
        with profiling.timer("t_delete", profile, final=profFinal):
            for k in keys:
                try:
                    del self.store[k]
                except KeyError:
                    pass

    def destroy(self):
        pass


def init():
    _server.initServer()


def invoke(rawReq, kv, stats=None):
    with profiling.timer("t_e2e", stats):
        req = rawReq[0]
        renameMap = rawReq[1]
        req.reKey(renameMap)
        _server.kaasServeInternal(req, kv, stats)
