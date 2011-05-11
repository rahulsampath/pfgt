#!/bin/sh

export writeOut=1
export fMag=10.0 
export epsilon=1.0e-6

export forceType=2

export numPtsPerProc=10000
export numPtsStr=10K

#export delta=0.0625
export delta=1.0

export numProcStr=1

./runUniform ${numPtsPerProc} ${fMag} ${epsilon} ${delta} ${writeOut} ${forceType} 
# ./runUniform ${numPtsPerProc} ${fMag} ${epsilon} ${delta} ${writeOut} ${forceType} >& upfgt.${numPtsStr}.${numProcStr}.${forceType}.txt

./directSum ${delta}

