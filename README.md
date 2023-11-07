This repository, along with
[kaasBenchmarks](https://github.com/NathanTP/kaasBenchmarks), contains the code
used to generate results in the following publications. However, no attempt has
been made to make the code easily used by others.

Nathan Pemberton. "The Serverless Datacenter: Hardware and Software Techniques
for Resource Disaggregation". 2022. University of Californiaat Berkeley, PhD
Dissertation. [Link](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2022/EECS-2022-86.html).

Nathan Pemberton, Anton Zabreyko, Zhoujie Ding, Randy Katz, and Joseph
Gonzalez. 2022. Kernel-as-a-Service: A Serverless Interface to GPUs.
[arXiv:2212.08146](https://arxiv.org/abs/2212.08146).

# Overview
Kernel-as-a-Service is a serverless function type that is tailored specifically
for GPUs. Rather than running arbitrary host code as is done in most serverless
functions, KaaS runs graphs of CUDA kernels directly. This allows KaaS to
tailor its runtime and avoid many of the challenges that typical FaaS systems
have to deal with (especially cold starts).

For now, this package works with the Ray distributed computing framework. In
effect, KaaS functions represent an alternative task type for Ray. They are
described by a request object rather than a python function, but otherwise
behave similarly to Ray tasks.

# Request Format

## Top Level Request
This is the request format expected by the kaasServer.

    { "kernels" : [ # List of kernelSpecs to invoke (in order) ] }

## KernelSpec
Kernel specs represent a single invocation of some kernel. Inputs are read from
a KV store, while outputs are temps are created as needed. Only outputs are
saved back to the KV store.

    {
        "library" : # Absolute path to the CUDA wrapper shared library,
        "kernel"  : # name of the wrapper func in library to call,
        "arguments": [ # list of tuples, first value is bufferSpec to use, second value is string denoting type (i -> input, o -> output, t -> temporary ]
        "nGrid"   : # Number of CUDA grids to use when invoking,
        "nBlock"  : # Number of CUDA blocks to use when invoking 
    }

## BufferSpec
Describes a single buffer that will be used by kaas.

    {
        "name"      : # string identifier for this buffer, will be the key in the KV store (if used)
        "size"      : # Number of bytes in the buffer,
        "ephemeral" : # boolean indicating whether the buffer should be
                        persisted to the KV store or not. Ephemeral buffers have a lifetime equal to a
                        single kaas request.,
        "const"     : # A guarantee that the buffer will never change
                        externally. This is a bit of a hack until we get a proper cache figured out.
    }
