#!/bin/bash
# Profiling code in the kaas executor is really slow, even when disabled in the code.
# Python doesn't seem to have an easy-to-use syntactic macro system, so we do
# it with sed which is good enough. Running this will remove any line ending
# with # PROFDISABLE in _server.py. You should run it for final performance
# runs and use _server_light.py.
sed '/.*# PROFDISABLE/d;s/profLevel = .*/profLevel = 0/;s/logLevel = .*/logLevel = logging.ERROR/' _server.py > _server_light.py
