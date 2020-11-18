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

print('hvd.inited - initialized:', hvd.is_initialized(), 
      ' local rank:', hvd.local_rank(), ' global rank:', hvd.rank(), 
      ' local ranks:', hvd.local_size(), ' global ranks:', hvd.size())
print('hvd built - mpi_built:', hvd.mpi_built(), 
      ' nccl_built:', hvd.nccl_built(), ' cuda_built:', cuda_built())
print('hvd MPI - mpi_enabled:', hvd.mpi_enabled(), 
      ' mpi_threads_supported:', hvd.mpi_threads_supported())

# shudown horovod
hvd.shutdown()

print('hvd.shutdown')


