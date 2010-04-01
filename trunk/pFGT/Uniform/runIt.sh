#!/bin/sh

echo $PETSC_DIR
echo $PETSC_ARCH

#Replace this  
export MPIEXEC=/cygdrive/c/petsc-3.0.0-p12/optCxx/bin/mpiexec

export writeOut=0
export fMag=10.0 
export epsilon=1.0e-3

export numPtsPerProc=1000000
export numPtsStr=1M

export delta=0.01

export numProcStr=1

${MPIEXEC} -n ${numProcStr} ./runUniform ${numPtsPerProc} ${fMag} ${epsilon} ${delta} ${writeOut} -log_summary >& upfgt.${numPtsStr}.${numProcStr}.txt

