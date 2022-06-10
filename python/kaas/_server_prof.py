import pycuda.driver as cuda
import pycuda.tools
import numpy as np
import collections
import logging
import atexit
import os
import time

from . import cutlass

from . import complexCutlass

logLevel = logging.WARN
logging.basicConfig(format="%(levelname)s: %(message)s", level=logLevel)

# Profiling level sets how aggressive we are in profiling.
#   - 0 no profiling
#   - 1 metrics that will have little effect on performance
#   - 2 synchronize cuda aggresively to get accurate measurements (hurts e2e
#       perf)
profLevel = 2

# This is a global reference to the current ff.profCollection (reset for each call)
profs = None

kCache = None
bCache = None

dir_path = os.path.dirname(os.path.realpath(__file__))


def profSync():
    """If required by the profiling level, synchronize the cuda context"""
    if profLevel > 1:
        cuda.Context.synchronize()


def startTimer(level=1):
    if level <= profLevel:
        return time.time()
    else:
        return None


def updateTimer(name, startTime, level=1, final=True):
    if level <= profLevel:
        if final:
            profs[name].increment((time.time() - startTime)*1000)
        else:
            profs[name].update((time.time() - startTime)*1000)


def updateProf(name, val, level=1, final=True):
    if level <= profLevel:
        profs[name].update(val)


def getProf(level=1, mod=None):
    """Returns the profs object (ff.profCollection) for this level. If level is
    above the current profLevel, None is returned. This is suitable for use in a
    ff.timer context."""
    if level <= profLevel:
        if mod is None:
            return profs
        else:
            return profs.mod(mod)
    else:
        return None


# A list of all metrics that must be finalized after each invocation
# n_* is an event count, s_* is a size metric in bytes, t_* is a time measurement in ms
eventMetrics = [
    'n_hostDMiss',
    'n_hostDHit',
    's_hostDLoad',
    't_hostDLoad',
    'n_hostDWriteBack',
    't_hostDWriteBack',
    'n_devDHit',
    'n_devDMiss',
    'n_devDEvict',
    'n_invoke',
    't_devDEvict',
    's_devDWriteBack',
    's_htod',
    's_dtoh',
    't_htod',
    't_dtoh',
    't_zero',
    'n_KMiss',
    'n_KHit',
    't_kernelLoad',
    't_cudaMM',
    't_hostMM']

eventMetrics += ['t_makeRoom', 't_invokeExternal', 't_setupArgs', 't_bCacheLoad']

# These metrics need to be handled with special cases
metricSpecial = ['t_invoke']


class kaasBuf():
    @classmethod
    def fromSpec(cls, bID, spec, src=None):
        return cls(bID, spec.name, spec.key, src,
                   size=spec.size, offset=spec.offset, ephemeral=spec.ephemeral)

    def __init__(self, bID, name, key, src, size=None, offset=0, ephemeral=False):
        """A kaasBuffer represnts a binary data buffer managed by the kaas
        system, it can be either on the device or on the host. If src is
        provided, it will be used as the buffer, otherwise size will be used to
        represent a zeroed buffer. kaasBuffers do not automatically manage
        consistency, it is up to the user to decide when and how to synchronize
        host and device memory.
        """
        self.bID = bID
        self.name = name
        self.key = key
        self.dirty = False
        self.ephemeral = ephemeral
        self.useCount = 0

        self.dbuf = None
        self.onDevice = False

        if src is None:
            self.hbuf = None
        else:
            self.hbuf = memoryview(src).cast('B')

        self.offset = offset
        if size is None:
            self.size = self.hbuf.nbytes
        else:
            self.size = size

    def __str__(self):
        return self.name

    def __repr__(self):
        return "KaaS Buffer (name={}, key={}, dirty={}, ephemeral={}, onDevice={}, size={})".format(
                self.name, self.key, self.dirty, self.ephemeral, self.onDevice, self.size)

    def updateValue(self, newBuf):
        """Replace the data in this buffer. If there is already a device
        buffer, it will be updated as well. If there is no existing device
        buffer, we will not allocate one. If newBuf==None, we will free host
        memory, but the device buffer will not be updated (use clear() for
        that)."""
        if newBuf is None:
            self.hbuf = None
        else:
            newBuf = memoryview(newBuf).cast('B')
            if newBuf.nbytes != self.size:
                # Maybe this should be an error. If self.size < newBuf.nbytes
                # we'll get a CUDA error, otherwise it's a warning
                logging.warn(f"(buffer {self.bID}) Unexpected size for new data: buffer is {self.size} but new value is {newBuf.nbytes}")

            self.hbuf = newBuf
            if self.onDevice:
                logging.debug(f"Moving {self.size}B from host buffer '{self.name}' to device address 0x{hex(int(self.dbuf))}")
                pstart = startTimer()
                cuda.memcpy_htod(self.dbuf, self.hbuf[self.offset:self.size])
                profSync()
                updateTimer('t_htod', pstart, final=False)

    def toDevice(self):
        """Place the buffer onto the device if it is not already there.  If no
        host buffer is set, zeroed device memory will be allocated.

        Returns: amount of additional memory used on the device"""
        if self.onDevice:
            return 0
        else:
            updateProf('s_htod', self.size)

            pstart = startTimer()
            self.dbuf = cuda.mem_alloc(self.size)
            profSync()
            updateTimer('t_cudaMM', pstart, final=False)

            if self.hbuf is not None:
                logging.debug(f"Moving {self.size}B from host buffer '{self.name}' to device address 0x{hex(int(self.dbuf))}")
                pstart = startTimer()
                cuda.memcpy_htod(self.dbuf, self.hbuf[self.offset:self.offset+self.size])
                profSync()
                updateTimer('t_htod', pstart, final=False)

            self.onDevice = True
            return self.size

    def toHost(self):
        """Copy data from the device (if present). If the kaasBuf does not have
        a host buffer set, one will be allocated, otherwise the currently
        configured host buffer will be overwritten with the device buffer
        data."""
        if not self.onDevice:
            return
        else:
            logging.debug("Moving {}B from device (0x{}) {} to host".format(
                self.size, hex(int(self.dbuf)), self.name))

            if self.hbuf is None:
                pstart = startTimer()
                self.hbuf = np.empty(self.size, dtype=np.int8)
                updateTimer('t_hostMM', pstart, final=False)

            updateProf('s_dtoh', self.size)
            pstart = startTimer()
            cuda.memcpy_dtoh(self.hbuf[self.offset:], self.dbuf)
            profSync()
            updateTimer('t_dtoh', pstart, final=False)

    def freeDevice(self):
        """Free any memory that is allocated on the device."""
        if not self.onDevice:
            return
        else:
            pstart = startTimer()
            self.dbuf.free()
            profSync()
            updateTimer('t_cudaMM', pstart, final=False)

            self.dbuf = None
            self.onDevice = False

    def clear(self):
        logging.debug("Clearing existing buffer " + self.name)
        pstart = startTimer()
        self.hbuf = None
        if self.onDevice:
            cuda.memset_d8(self.dbuf, 0, self.size)
            profSync()
        updateTimer('t_zero', pstart, final=False)


class kaasFunc():
    def __init__(self, library, fName, literalTypes, nBuf):
        self.func = library.get_function(fName)
        self.func.prepare(literalTypes + ["P"]*nBuf)

        self.fName = fName
        self.lib = library

    def __str__(self):
        return self.func

    def Invoke(self, literals, bufs, gridDim, blockDim, sharedSize):
        """Invoke the func with the provided diminsions:
            - bufs is a list of kaasBuf objects
            - outs is a list of indexes of bufs to copy back to host memory
              after invocation
        """

        literalVals = [lit.val for lit in literals]

        dAddrs = []
        for b in bufs:
            if not b.onDevice:
                raise RuntimeError(f"Provided buffer was not resident on the device (insufficient memory?): {b.name} ({b.key})")

            dAddrs.append(b.dbuf)

        args = literalVals + dAddrs

        if self.fName == "_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock12MmaPipelinedINS1_9GemmShapeILi128ELi128ELi8EEENS_9transform11threadblock22PredicatedTileIteratorINS_11MatrixShapeILi128ELi8EEEfNS_6layout8RowMajorELi1ENS8_30PitchLinearStripminedThreadMapINSD_16PitchLinearShapeILi8ELi128EEELi256ELi1EEELi1EEENS9_19RegularTileIteratorISC_fNSD_11ColumnMajorELi1ENS8_33TransposePitchLinearThreadMapSimtISI_EELi4EEENSA_INSB_ILi8ELi128EEEfSE_Li0ENSF_INSG_ILi128ELi8EEELi256ELi1EEELi1EEENSK_ISP_fSE_Li0ESR_Li4EEEfSE_NS4_9MmaPolicyINS1_4warp7MmaSimtINS6_ILi32ELi64ELi8EEEfSL_fSE_fSE_NSV_13MmaSimtPolicyINSB_ILi4ELi8EEENSD_19RowMajorInterleavedILi2EEENS6_ILi4ELi4ELi1EEEEELi1ELNS_16ComplexTransformE0ELS14_0EbEENSB_ILi4ELi0EEENSB_ILi0ELi0EEELi1EEENS_21NumericArrayConverterIffLi4ELNS_15FloatRoundStyleE2EEES1B_bEENS_8epilogue11threadblock8EpilogueIS7_S15_Li1ENS1E_22PredicatedTileIteratorINS1E_26OutputTileOptimalThreadMapINS1E_15OutputTileShapeILi128ELi1ELi4ELi4ELi1EEENS1I_ILi1ELi4ELi2ELi1ELi8EEELi256ELi1ELi32EEEfEENS1D_4warp20FragmentIteratorSimtISX_NS1_6thread3MmaINS6_ILi8ELi8ELi1EEEfSL_fSE_fSE_NS_4arch13OpMultiplyAddEbEESE_S13_EENS1N_16TileIteratorSimtISX_S1U_fSE_S13_EENS1E_18SharedLoadIteratorINS1L_18CompactedThreadMapEfLi4EEENS1D_6thread17LinearCombinationIfLi1EffLNS21_9ScaleType4KindE0ELS1A_2EEENSB_ILi0ELi17EEELi1EEENS4_30GemmIdentityThreadblockSwizzleILi1EEELb0EEEEEvNT_6ParamsE":
            logging.debug("Invoking <<<{}, {}, {}>>>cutlassSgemm({})".format(gridDim, blockDim, sharedSize,
                          ", ".join([str(lit) for lit in literalVals] + [str(b.name) for b in bufs])))

            cuda.memset_d8(bufs[2].dbuf, 0, bufs[2].size)
            params = cutlass.parseSgemmArgs(literalVals, dAddrs, kCache.cutlassAdapter)
            self.func.prepare("320s")
            return self.func.prepared_timed_call(gridDim, blockDim, params.contents, shared_size=sharedSize)
        elif self.fName == "_ZN7cutlass6KernelINS_4gemm6kernel4GemmINS1_11threadblock12MmaPipelinedINS1_9GemmShapeILi128ELi128ELi8EEENS_9transform11threadblock22PredicatedTileIteratorINS_11MatrixShapeILi128ELi8EEENS_7complexIfEENS_6layout8RowMajorELi1ENS8_30PitchLinearStripminedThreadMapINSF_16PitchLinearShapeILi8ELi128EEELi256ELi1EEELi1EEENS9_19RegularTileIteratorISC_SE_NSF_11ColumnMajorELi1ENS8_33TransposePitchLinearThreadMapSimtISK_EELi8EEENSA_INSB_ILi8ELi128EEESE_SG_Li0ENSH_INSI_ILi128ELi8EEELi256ELi1EEELi1EEENSM_ISR_SE_SG_Li0EST_Li8EEESE_SG_NS4_9MmaPolicyINS1_4warp7MmaSimtINS6_ILi32ELi64ELi8EEESE_SN_SE_SG_SE_SG_NSX_13MmaSimtPolicyINSB_ILi4ELi8EEENSF_19RowMajorInterleavedILi2EEENS6_ILi2ELi2ELi1EEEEELi1ELNS_16ComplexTransformE0ELS16_0EbEENSB_ILi2ELi0EEENSB_ILi0ELi0EEELi1EEENS_21NumericArrayConverterISE_SE_Li4ELNS_15FloatRoundStyleE2EEES1D_bEENS_8epilogue11threadblock8EpilogueIS7_S17_Li1ENS1G_22PredicatedTileIteratorINS1G_26OutputTileOptimalThreadMapINS1G_15OutputTileShapeILi128ELi1ELi4ELi4ELi1EEENS1K_ILi1ELi2ELi4ELi1ELi8EEELi256ELi1ELi64EEESE_EENS1F_4warp20FragmentIteratorSimtISZ_NS1_6thread3MmaINS6_ILi8ELi8ELi1EEESE_SN_SE_SG_SE_SG_NS_4arch13OpMultiplyAddEbEESG_S15_EENS1P_16TileIteratorSimtISZ_S1W_SE_SG_S15_EENS1G_18SharedLoadIteratorINS1N_18CompactedThreadMapESE_Li8EEENS1F_6thread17LinearCombinationISE_Li1ESE_SE_LNS23_9ScaleType4KindE0ELS1C_2EEENSB_ILi0ELi9EEELi1EEENS4_30GemmIdentityThreadblockSwizzleILi1EEELb0EEEEEvNT_6ParamsE":
            logging.debug("Invoking <<<{}, {}, {}>>>cutlassComplexGemm({})".format(gridDim, blockDim, sharedSize,
                          ", ".join([str(lit) for lit in literalVals] + [str(b.name) for b in bufs])))
            cuda.memset_d8(bufs[2].dbuf, 0, bufs[2].size)
            params = complexCutlass.parseSgemmArgs(literalVals, dAddrs, kCache.complexAdapter)
            self.func.prepare("328s")
            return self.func.prepared_timed_call(gridDim, blockDim, params.contents, shared_size=sharedSize)
        else:
            logging.debug("Invoking <<<{}, {}, {}>>>{}({})".format(gridDim, blockDim, sharedSize, self.fName,
                          ", ".join([str(lit) for lit in literalVals] + [str(b.name) for b in bufs])))
            return self.func.prepared_timed_call(gridDim, blockDim, *args, shared_size=sharedSize)


class kernelCache():
    def __init__(self):
        self.libs = {}
        self.kerns = {}

        pycuda.driver.init()
        self.cudaCtx = pycuda.tools.make_default_context()

        self.cutlassAdapter = cutlass.loadSgemmAdapter()
        self.complexAdapter = complexCutlass.loadAdapter()

    def get(self, spec):
        if spec.name not in self.kerns:
            updateProf('n_KMiss', 1)
            pstart = startTimer()
            try:
                if spec.library not in self.libs:
                    self.libs[spec.library] = cuda.module_from_file(spec.library)
            except pycuda._driver.RuntimeError as e:
                # raise RuntimeError(f"Could not load kaas library at {libPath}: {e.args}") from e
                raise RuntimeError(f"Could not load kaas library at {spec.library}: {e.args}") from e

            litTypes = [lit.dtype for lit in spec.literals]
            self.kerns[spec.name] = kaasFunc(self.libs[spec.library], spec.kernel,
                                             litTypes, len(spec.arguments))
            updateTimer('t_kernelLoad', pstart, final=False)
        else:
            updateProf('n_KHit', 1)

        return self.kerns[spec.name]


class lruPolicy():
    def __init__(self):
        # Only contains buffers that are currently on the device. Constants are
        # given higher priority than other types of buffers.
        self.lruConst = collections.deque()
        self.lruOther = collections.deque()

    def remove(self, buf):
        """Remove buffer from consideration"""
        if buf.useCount > 2:
            self.lruConst.remove(buf)
        else:
            self.lruOther.remove(buf)

    def push(self, buf):
        """Place buffer in the most-recently-used slot"""
        if buf.useCount > 2:
            self.lruConst.appendleft(buf)
        else:
            self.lruOther.appendleft(buf)

    def pop(self):
        """Remove and return the least recently used buffer"""
        # pull from lruOther if we can, otherwise start evicting consts
        if self.lruOther:
            b = self.lruOther.pop()
        else:
            b = self.lruConst.pop()

        return b


class bufferCache():
    """The buffer cache tracks buffers both on and off the device. Any sort of
    distributed storage properties are provided by an external KV store.
    Buffers are explicitly loaded onto the device by clients and automatically
    migrated off the device as needed. The cache is inclusive (anything on the
    device also exists on the host). Buffers are only committed back to the KV
    store explicitly by the client (dirty() and flush()). Eventually, this
    should really be integrated with the normal cache but for now it's a
    separate thing.

    WARNING: This cache doesn't really do everything it should correctly, I'm
    just trying to get something to work. It works with the server as written
    but it's really bad at keeping things consistent. Writing a real cache is a
    task for another day."""

    def __init__(self):
        """Initialize a device buffer cache with byte capacity cap. If this cap
        is exceeded, things will be evicted"""
        # Contains all bufs, on device or not
        self.bufs = {}
        self.dirtyBufs = {}
        self.ephemerals = {}

        self.policy = lruPolicy()

        memFree, memAvail = cuda.mem_get_info()

        # Maximum number of bytes on the device
        # We build in a bit of a margin because we can't track size very
        # accurately. We assume we won't be off by more than this margin within
        # a single request (this is just a guess).
        self.cap = memAvail - 10*1024*1024

        # Size represents the amount of memory used on the device, it's more
        # complicated than just the sum of buffer sizes because of device
        # memory overheads (e.g. cudaMalloc() uses a lot of device space)
        self.size = memAvail - memFree

    def setKV(self, kv):
        self.kv = kv

    def updateSize(self):
        memFree, memAvail = cuda.mem_get_info()
        self.size = memAvail - memFree

    def makeRoom(self, sz):
        while self.cap - self.size < sz:
            updateProf('n_devDEvict', 1)
            b = self.policy.pop()
            logging.debug(f"Evicting {b.name} ({b.key})")

            if b.dirty:
                updateProf('s_devDWriteBack', b.size())
                b.toHost()

            assert b.onDevice
            b.freeDevice()
            self.size -= b.size

    def makeRoomForBufs(self, bSpecs):
        """Ensure that we have enough room to load all the buffers in bSpecs.
        This accounts for if any of the buffers are already on the device. This
        function helps performance and possibly memory fragmentation by
        batching frees()."""
        total = 0
        for bSpec in bSpecs:
            cacheBuf = self.bufs.get(bSpec.key, None)
            if cacheBuf is None:
                updateProf('n_devDMiss', 1)
                total += bSpec.size
            else:
                if cacheBuf.onDevice:
                    updateProf('n_devDHit', 1)
                else:
                    updateProf('n_devDMiss', 1)
                    total += bSpec.size

        pstart = startTimer()
        self.makeRoom(total)
        updateTimer('t_devDEvict', pstart, final=False)

    def load(self, bSpec, overwrite=False, clientID=None):
        """Load a buffer onto the device, fetching from the KV store if
        necessary. If overwrite=True, a new buffer will be created. If the
        buffer is already in the cache and dirty, it will be written back and
        re-read (you should probably make sure this doesn't happen by flushing
        when needed). Buffers are pinned in device memory when loaded, you must
        explicitly release() them when you are done."""

        # key = bSpec.key
        bID = f"{clientID}:{bSpec.name}"

        buf = self.bufs.get(bID, None)
        if buf is None:
            if bSpec.ephemeral or overwrite:
                logging.debug(f"Creating new buffer: bID:{bID}, name:{bSpec.name}, size:{bSpec.size}")
                hbuf = None
            else:
                #XXX need to track repeated keys so we don't store a million copies of the same buffer (this could be serious for things like BERT where the aggregate constant size is huge and it has many constants).
                hbuf = self.kv.get(bSpec.key)
                logging.debug(f"Loading buffer from KV: bID:{bID}, name:{bSpec.name}, size:{len(hbuf)}, key:{bSpec.key}")

            buf = kaasBuf.fromSpec(bID, bSpec, src=hbuf)

            if bSpec.ephemeral:
                self.ephemerals[bID] = buf

            self.bufs[bID] = buf
        else:
            if buf.key != bSpec.key and not (bSpec.ephemeral or overwrite):
                logging.debug(f"Loading new value into buffer {bID} from key {bSpec.key}")
                hbuf = self.kv.get(bSpec.key)
                buf.updateValue(hbuf)
            else:
                logging.debug(f"Re-using cached buffer {bID}")

            # Pin the buffer in memory by removing it from the eviction policy.
            # release() will place it back in the policy.
            if buf.onDevice:
                self.policy.remove(buf)

        # buf = self.bufs.get(key, None)
        # if buf is not None:
        #     logging.debug("Loading from Cache: {}".format(bSpec.name))
        #     updateProf('n_hostDHit', 1)
        #
        #     # Reset LRU
        #     if buf.onDevice:
        #         self.policy.remove(buf)
        # else:
        #     updateProf('n_hostDMiss', 1)
        #     if bSpec.ephemeral or overwrite:
        #         logging.debug("Loading (new buffer): {}".format(bSpec.name))
        #         buf = kaasBuf.fromSpec(bSpec)
        #
        #         if buf.ephemeral:
        #             self.ephemerals[key] = buf
        #     else:
        #         pstart = startTimer()
        #         raw = self.kv.get(key, profile=getProf(mod='kv'), profFinal=False)
        #         updateTimer('t_hostDLoad', pstart, final=False)
        #
        #         if raw is None:
        #             logging.debug("Loading (new buffer): {}".format(bSpec.name))
        #             buf = kaasBuf.fromSpec(bSpec)
        #         else:
        #             logging.debug("Loading from KV: {} (key: {})".format(bSpec.name, key))
        #             updateProf('s_hostDLoad', bSpec.size)
        #             buf = kaasBuf.fromSpec(bSpec, src=raw)

        self.makeRoom(buf.size)

        self.size += buf.toDevice()

        return buf

    def release(self, buf):
        """When buffers are loaded, they are automatically pinned in memory.
        You must explicitly release them to make them eligible for eviction.
        The primary reason for this is so that we can guarantee a single
        kernel's memory will be available."""
        buf.useCount += 1
        self.policy.push(buf)

    def dirty(self, bID):
        buf = self.bufs[bID]
        buf.dirty = True
        self.dirtyBufs[bID] = buf

    def _flushOne(self, buf):
        if buf.dirty:
            if buf.onDevice:
                buf.toHost()
            if not buf.ephemeral:
                # Data are stored as numpy arrays because memoryviews can't be
                # pickled. This should still be zero copy.
                logging.debug("Writing back to kv: {} (key: {})".format(buf.name, buf.key))
                updateProf('n_hostDWriteBack', 1)
                pstart = startTimer()
                self.kv.put(buf.key, buf.hbuf, profile=getProf(mod='kv'), profFinal=False)
                updateTimer('t_hostDWriteBack', pstart, final=False)
            buf.dirty = False

    def flush(self):
        for b in self.dirtyBufs.values():
            self._flushOne(b)

        self.dirtyBufs = {}

    def drop(self, bID):
        """Remove a buffer from the cache (writing back if dirty). This frees
        any device memory and drops references to the host buffer (Python's GC
        will pick it up eventually)."""
        logging.debug("Dropping " + str(bID))
        buf = self.bufs[bID]
        self._flushOne(buf)

        if buf.onDevice:
            self.policy.remove(buf)
            buf.freeDevice()
            self.size -= buf.size

        if buf.dirty:
            self.dirtyBufs.pop(buf.key, None)
        if buf.ephemeral:
            self.ephemerals.pop(buf.key, None)

        del self.bufs[buf.key]


def initServer():
    # These should only get initialized upon invocation, not at import. This is
    # most important for the kCache which has to manage a cuda device context
    global kCache, bCache

    if kCache is None:
        kCache = kernelCache()

    if bCache is None:
        bCache = bufferCache()


def kaasServeInternal(req, kv, newProfs=None, clientID=None):
    """Internal implementation of kaas execution. Req is a kaas.kaasReq, not a
    dictionary"""

    global profs
    profs = newProfs

    # If the user doesn't pass us anything for profiling, we should just avoid
    # it completely. This is actually needed for correctness (can't collect
    # stats into None).
    if newProfs is None:
        global profLevel
        profLevel = 0

    # This gets reset every call because libff only gives us the kv handle per
    # call and it could (in theory) change in some way between calls.
    bCache.setKV(kv)

    # We can't estimate memory utilization perfectly so we update our estimate
    # on every request
    bCache.updateSize()

    # We try to help the cuda memory allocator out by freeing all the buffers
    # at once instead of mixing frees and mallocs at a fine-grain. This loop
    # finds all the unique buffers in the request
    pstart = startTimer()
    bCache.makeRoomForBufs(req.bufferMap.values())
    updateTimer('t_makeRoom', pstart, final=False)

    invokeTimes = []
    visibleOutputs = []

    # when nIter > 1, we may see the same output multiple times, but we only
    # want to report it once. OrderedDict keys are unique and ordered, we don't
    # use the values.
    visibleOutputs = collections.OrderedDict()

    for i in range(req.nIter):
        for kSpec in req.kernels:
            updateProf('n_invoke', 1)
            kern = kCache.get(kSpec)

            arguments = []
            for argName, ioType in zip(kSpec.arguments, kSpec.ioTypes):
                arg = req.bufferMap[argName]
                pstart = startTimer()
                if ioType == 'o':
                    argBuf = bCache.load(arg, overwrite=True, clientID=clientID)
                else:
                    argBuf = bCache.load(arg, clientID=clientID)
                updateTimer('t_setupArgs', pstart, final=False)

                if (ioType == 'o' or ioType == 'io') and not argBuf.ephemeral:
                    bCache.dirty(argBuf.bID)
                    visibleOutputs[argBuf.key] = None

                arguments.append(argBuf)

            pstart = startTimer()
            timer = kern.Invoke(kSpec.literals, arguments,
                                kSpec.gridDim, kSpec.blockDim, kSpec.sharedSize)
            profSync()
            updateTimer('t_invokeExternal', pstart, final=False)
            invokeTimes.append(timer)

            for buf in arguments:
                bCache.release(buf)

    # Make sure all outputs are visible externally (basically this merges our
    # private state into whatever consistency properties the KV gives us.
    bCache.flush()

    if profLevel > 0:
        for metric in eventMetrics:
            profs[metric].increment()
        for p in profs.mod('kv').values():
            p.increment()

        t_invoke = 0
        for t in invokeTimes:
            t_invoke += t()
        profs['t_invoke'].increment(t_invoke*1000)

    return list(visibleOutputs.keys())


@atexit.register
def kaasCleanup():
    global kCache
    global bCache

    if bCache is not None:
        bCache.flush()
        bCache = None

    if kCache is not None:
        kCache.cudaCtx.detach()
        kCache = None
