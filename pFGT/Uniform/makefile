
SHELL = /bin/sh

include ${PETSC_DIR}/${PETSC_ARCH}/conf/petscvariables
include ${PETSC_DIR}/conf/variables

#include ${DENDRO_DIR}/makeVariables

CEXT = C

CFLAGS = -DPETSC_USE_LOG  -D__USE_K_FOR_TYPE2__  -D__USE_TENSOR_PRODUCT_GRID__  -O3 
# -O2 -O1

# -D__DEBUG__

#-D__USE_MG_INIT_TYPE2__
#-D__USE_64_BIT_INT__

#-g -O0

include ./makefileCore

