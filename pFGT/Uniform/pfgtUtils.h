
#ifndef PFGT_UTILS
#define PFGT_UTILS

#include "petsc.h"

PetscErrorCode pfgtType1(double delta, int K, double fMag, unsigned int numPtsPerProc, int writeOut);

PetscErrorCode pfgtType2(double delta, double fMag, 
    unsigned int numPtsPerProc, int P, int L, int K, int writeOut);

void directW2L(PetscScalar**** WlArr, PetscScalar**** WgArr, int xs, int ys, int zs, int nx, int ny, int nz, int Ne, double h, const int StencilWidth, const int PforType2, const double lambda);
void sweepW2L(PetscScalar**** WlArr, PetscScalar**** WgArr, int xs, int ys, int zs, int nx, int ny, int nz, int Ne, double h, const int StencilWidth, const int PforType2, const double lambda);

#endif

