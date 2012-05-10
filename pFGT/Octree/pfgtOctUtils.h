
#ifndef __PFGT_OCT_UTILS__
#define __PFGT_OCT_UTILS__

#include "petsc.h"
#include "oct/TreeNode.h"
#include <vector>

#define __PI__ 3.1415926535897932

PetscErrorCode pfgt(std::vector<ot::TreeNode> & linOct, const unsigned int maxDepth,
    const double delta, const double fMag, const unsigned int numFgtPtsPerDimPerProc,
    const int P, const int L, const int K, const int DirectHfactor, const int writeOut);

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

