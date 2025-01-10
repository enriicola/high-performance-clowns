#/usr/bin/env sh

set -xe

NVARCH=`uname -s`_`uname -m`; export NVARCH
NVCOMPILERS=/opt/nvidia/hpc_sdk; export NVCOMPILERS
MANPATH=$MANPATH:$NVCOMPILERS/$NVARCH/23.7/compilers/man; export MANPATH
PATH=$NVCOMPILERS/$NVARCH/23.7/compilers/bin:$PATH; export PATH

nvc++ heat.cu -o heat_cuda
