import argparse
import os
import pathlib

print('TEST.PY RUNNING...')
print('environ:', os.environ)

import horovod.torch as hvd
import numpy as np


parser = argparse.ArgumentParser(description="Horovod MPI connection test harness")
args = parser.parse_args()

print('TEST.PY MODULES IMPORTED...')
print('args:', args)


# initialize horovod
hvd.init()

print('hvd.inited - local rank:', hvd.local_rank(), ' global rank:', hvd.rank(), ' ranks:', hvd.size())

