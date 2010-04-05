
#ifndef __PFGT_OCT_UTILS__
#define __PFGT_OCT_UTILS__

#include "petsc.h"
#include "TreeNode.h"
#include <vector>

PetscErrorCode pfgt(std::vector<ot::TreeNode> & linOct, unsigned int maxDepth,
    double delta, double fMag, unsigned int numFgtPtsPerDimPerProc,
    int P, int L, int K, int DirectHfactor, int writeOut);

void directW2L(PetscScalar**** WlArr, PetscScalar**** WgArr, int xs, int ys, int zs,
    int nx, int ny, int nz, int Ne, double h, const int StencilWidth,
    const int PforType2, const double lambda);

void sweepW2L(PetscScalar**** WlArr, PetscScalar**** WgArr, int xs, int ys, int zs,
    int nx, int ny, int nz, int Ne, double h, const int StencilWidth,
    const int PforType2, const double lambda);

void directLayer(PetscScalar**** WlArr, PetscScalar**** WgArr, int xs, int ys, int zs, 
    int nx, int ny, int nz, int Ne, double h, const int StencilWidth, 
    const int PforType2, const double lambda);

#endif

