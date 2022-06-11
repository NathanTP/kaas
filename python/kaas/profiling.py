import collections.abc
import time
from contextlib import contextmanager
# import jsonpickle as json
import json
import numpy as np
import contextlib
import ctypes


cudaRT = None
profilerStarted = False


class prof():
    def __init__(self, fromDict=None, detail=False):
        """A profiler object for a metric or event type. The counter can be
        updated multiple times per event, while calling increment() moves on to
        a new event. If detail==true, all events are logged allowing more
        complex statistics. This may affect performance if there are many
        events."""
        self.detail = detail
        if fromDict is not None:
            if self.detail:
                self.events = fromDict['events']
            self.total = fromDict['total']
            self.nevent = fromDict['nevent']
        else:
            self.total = 0.0
            self.nevent = 0

            if self.detail:
                self.currentEvent = 0.0
                self.events = []

    def update(self, n):
        """Update increases the value of this entry for the current event."""
        self.total += n
        if self.detail:
            self.currentEvent += n

    def increment(self, n=0):
        """Finalize the current event (increment the event counter). If n is
        provided, the current event will be updated by n before finalizing."""
        self.update(n)
        self.nevent += 1
        if self.detail:
            self.events.append(self.currentEvent)
            self.currentEvent = 0.0

    def report(self, includeEvents=False, metrics=None):
        """Create a report dictionary from this profile. If the profile was
        created with detail=True, full summary statistics are included.
        Otherwise only the mean is reported.

        Args:
            includeEvents: Include the full trace of all events.
            metrics: List of metrics to report. If "None", all available
                metrics are reported. If the profile was created with
                detail=False, only 'mean' and 'total' are available.
            """
        if self.nevent == 0:
            raise RuntimeError("Metric has no events")

        rep = {}
        rep['total'] = self.total
        rep['mean'] = self.total / self.nevent

        if self.detail:
            events = np.array(self.events)
            rep['min'] = events.min()
            rep['max'] = events.max()
            rep['std'] = np.std(events)
            rep['p50'] = np.quantile(events, 0.50)
            rep['p90'] = np.quantile(events, 0.90)
            rep['p99'] = np.quantile(events, 0.99)
            if includeEvents:
                rep['events'] = self.events

        if metrics is None:
            metrics = ['total', 'mean', 'min', 'max', 'std', 'p50', 'p90', 'p99', 'events']

        return {metric: rep[metric] for metric in metrics if metric in rep}


class profCollection(collections.abc.MutableMapping):
    """This is basically a dictionary and can be used anywhere a dictionary of
    profs was previously used. It has a few nice additional features though. In
    particular, it will generate an empty prof whenever a non-existant key is
    accessed."""

    def __init__(self, detail=False):
        # a map of modules included in these stats. Each module is a
        # profCollection. Submodules can nest indefinitely.
        self.detail = detail

        self.mods = {}

        self.profs = dict()

    def __contains__(self, key):
        return key in self.profs

    def __getitem__(self, key):
        if key not in self.profs:
            self.profs[key] = prof(detail=self.detail)
        return self.profs[key]

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
            self.mods[name] = profCollection(detail=self.detail)

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
                self.mods[name] = profCollection(detail=self.detail)
            self.mods[name].merge(mod)

    def report(self, includeEvents=False, metrics=None):
        """Create a report dctionary from this collection of the form
        {metricName: metricReport} where metricReport is returned by
        prof.report(). If there are modules, they will be sub-dictionaries in
        the report. See prof.report() for details of the contents of each
        metricReport, including the behavior of 'includeEvents' and 'metrics'."""
        report = {}
        for name, v in self.profs.items():
            try:
                report[name] = v.report(includeEvents=includeEvents, metrics=metrics)
            except Exception as e:
                raise RuntimeError(f"Failed to report metric '{name}'") from e

        for name, mod in self.mods.items():
            report[name] = mod.report(includeEvents=includeEvents, metrics=metrics)

        return report

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


profilerStarted = False


def cudaProfilerResetCtx():
    # nvprof can't handle changes in context once profiling has been enabled.
    # If you change contexts (i.e. creating one during initialization), you
    # should call this.
    if profilerStarted:
        cudaRT.cudaProfilerStart()


def cudaProfilerStart():
    global cudaRT
    global profilerStarted

    if cudaRT is None:
        cudaRT = ctypes.CDLL('libcudart.so')
    profilerStarted = True
    cudaRT.cudaProfilerStart()


def cudaProfilerStop():
    global profilerStarted
    cudaRT.cudaProfilerStop()
    profilerStarted = False


@contextlib.contextmanager
def cudaProfile(enable=True):
    if enable:
        cudaProfilerStart()
    yield
    if enable:
        cudaProfilerStop()
