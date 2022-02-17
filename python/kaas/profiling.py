import collections.abc
import time
from contextlib import contextmanager
import jsonpickle as json


class prof():
    def __init__(self, fromDict=None):
        """A profiler object for a metric or event type. The counter can be
        updated multiple times per event, while calling increment() moves on to
        a new event."""
        if fromDict is not None:
            self.total = fromDict['total']
            self.nevent = fromDict['nevent']
        else:
            self.total = 0.0
            self.nevent = 0

    def update(self, n):
        """Update increases the value of this entry for the current event."""
        self.total += n

    def increment(self, n=0):
        """Finalize the current event (increment the event counter). If n is
        provided, the current event will be updated by n before finalizing."""
        self.update(n)
        self.nevent += 1

    def total(self):
        """Report the total value of the counter for all events"""
        return self.total

    def mean(self):
        """Report the average value per event"""
        return self.total / self.nevent


class profCollection(collections.abc.MutableMapping):
    """This is basically a dictionary. It has a few nice additional features
    though. In particular, it will generate an empty prof whenever a
    non-existant key is accessed."""

    def __init__(self):
        # a map of modules included in these stats. Each module is a
        # profCollection. Submodules can nest indefinitely.
        self.mods = {}

        self.profs = dict()

    def __getitem__(self, key):
        if key not in self.profs:
            self.profs[key] = prof()
        return self.profs[key]

    def __contains__(self, key):
        return key in self.profs

    def __setitem__(self, key, value):
        self.profs[key] = value

    def __delitem__(self, key):
        del self.profs[key]

    def __iter__(self):
        return iter(self.profs)

    def __len__(self):
        return len(self.profs)

    def __str__(self):
        return json.dumps(self.report(), indent=4)

    def mod(self, name):
        if name not in self.mods:
            self.mods[name] = profCollection()

        return self.mods[name]

    def merge(self, new, prefix=''):
        # Start by merging the direct stats
        for k, v in new.items():
            newKey = prefix+k
            if newKey in self.profs:
                self.profs[newKey].increment(v.total)
            else:
                self.profs[newKey] = v

        # Now recursively handle modules
        for name, mod in new.mods.items():
            # Merging into an empty profCollection causes a deep copy
            if name not in self.mods:
                self.mods[name] = profCollection()
            self.mods[name].merge(mod)

    def report(self):
        flattened = {name: v.mean() for name, v in self.profs.items()}

        for name, mod in self.mods.items():
            flattened = {**flattened, **{name+":"+itemName: v for itemName, v in mod.report().items()}}

        return flattened

    def reset(self):
        """Clears all existing metrics. Any instantiated modules will continue
        to exist, but will be empty (it is safe to keep references to modules
        after reset()).
        """
        self.profs = {}
        for mod in self.mods.values():
            mod.reset()


# ms
timeScale = 1000


@contextmanager
def timer(name, timers, final=True):
    if timers is None:
        yield
    else:
        start = time.time() * timeScale
        try:
            yield
        finally:
            if final:
                timers[name].increment((time.time()*timeScale) - start)
            else:
                timers[name].update((time.time()*timeScale) - start)


def reportTimers(times):
    if times is None:
        return {}
    else:
        return {name: v.mean() for name, v in times.items()}


def printTimers(times):
    print(json.dumps(reportTimers(times), indent=4))
