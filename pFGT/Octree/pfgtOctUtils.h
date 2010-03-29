
#ifndef __PFGT_OCT_UTILS__
#define __PFGT_OCT_UTILS__

#include "petsc.h"
#include "TreeNode.h"
#include <vector>

PetscErrorCode pfgt(std::vector<ot::TreeNode> & linOct, unsigned int maxDepth,
    double delta, double fMag, unsigned int numFgtPtsPerDimPerProc,
    int P, int L, int K, int writeOut);


#endif

