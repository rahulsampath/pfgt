
#include "mpi.h"
#include "petscda.h"
#include "pfgtOctUtils.h"
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <iostream>

extern PetscLogEvent fgtEvent;
extern PetscLogEvent s2wEvent;
extern PetscLogEvent s2wCommEvent;
extern PetscLogEvent w2dEvent;
extern PetscLogEvent d2dEvent;
extern PetscLogEvent w2lEvent;
extern PetscLogEvent d2lEvent;
extern PetscLogEvent l2tCommEvent;
extern PetscLogEvent l2tEvent;

#define __PI__ 3.14159265

PetscErrorCode pfgt(std::vector<ot::TreeNode> & linOct, unsigned int maxDepth,
    double delta, double fMag, unsigned int numPtsPerProc, 
    int P, int L, int K, int writeOut)
{
  PetscFunctionBegin;

  PetscLogEventBegin(fgtEvent, 0, 0, 0, 0);

  MPI_Comm comm = MPI_COMM_WORLD;

  int npes, rank;
  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &rank);

  const double hRg = sqrt(delta);

  //Create DA for FGT boxes

  const unsigned int Ne = static_cast<unsigned int>(1.0/hRg);

  if(!rank) {
    std::cout<<"Ne = "<< Ne <<std::endl;
  }

  //2P complex coefficients for each dimension.  
  const unsigned int Ndofs = 16*P*P*P;

  if(!rank) {
    std::cout<<"Ndofs = "<<Ndofs<<std::endl;
  }

  if(!rank) {
    std::cout<<"StencilWidth = "<< K <<std::endl;
  }

  DA da;
  DACreate3d(comm, DA_NONPERIODIC, DA_STENCIL_BOX, Ne, Ne, Ne,
      PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, Ndofs, K,
      PETSC_NULL, PETSC_NULL, PETSC_NULL, &da);

  PetscInt xs, ys, zs, nx, ny, nz;
  DAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);

  //Split octree into 2 sets
  std::vector<ot::TreeNode> expandTree;
  std::vector<ot::TreeNode> directTree;

  const unsigned int numLocalOcts = linOct.size();

  const double hOctFac = 1.0/static_cast<double>(1u << maxDepth);

  for(unsigned int i = 0; i < numLocalOcts; i++) {
    unsigned int lev = linOct[i].getLevel();
    double hCurrOct = hOctFac*static_cast<double>(1u << (maxDepth - lev));

    if(hCurrOct <= hRg) {
      expandTree.push_back(linOct[i]);
    } else {
      directTree.push_back(linOct[i]);
    }
  }//end for i
  linOct.clear();

  const unsigned int numLocalExpandOcts = expandTree.size();
  const unsigned int numLocalDirectOcts = directTree.size();

  //Tensor-Product grid on each octant

  //Maximum value
  unsigned int numPtsPerBox = numPtsPerProc/numLocalOcts;

  //Tensor-Product Grid
  const unsigned int ptGridSizeWithinBox = static_cast<unsigned int>(floor(pow( numPtsPerBox, (1.0/3.0) )));
  long long trueLocalNumPts = ptGridSizeWithinBox*ptGridSizeWithinBox*ptGridSizeWithinBox*numLocalOcts;

  //S2W
  PetscLogEventBegin(s2wEvent, 0, 0, 0, 0);

  //Loop over local expand octants and execute S2W in each octant

  const double lambda = static_cast<double>(L)/(static_cast<double>(P)*sqrt(delta));

  std::vector<std::vector<std::vector<double> > > tmp1R;
  std::vector<std::vector<std::vector<double> > > tmp1C;

  std::vector<std::vector<std::vector<double> > > tmp2R;
  std::vector<std::vector<std::vector<double> > > tmp2C;

  std::vector<std::vector<double> > Woct(numLocalExpandOcts);

  std::vector<unsigned int> oct2fgtPtmap(3*numLocalExpandOcts);
  std::vector<unsigned int> oct2fgtIdmap(numLocalExpandOcts);

  for(unsigned int i = 0; i < numLocalExpandOcts; i++) {
    unsigned int lev = expandTree[i].getLevel();
    double hCurrOct = hOctFac*static_cast<double>(1u << (maxDepth - lev));

    double ptGridOff = 0.1*hCurrOct;
    double ptGridH = 0.8*hCurrOct/(static_cast<double>(ptGridSizeWithinBox) - 1.0);

    //Anchor of the octant
    unsigned int anchX = expandTree[i].getX();
    unsigned int anchY = expandTree[i].getY();
    unsigned int anchZ = expandTree[i].getZ();

    double aOx =  hOctFac*(static_cast<double>(anchX));
    double aOy =  hOctFac*(static_cast<double>(anchY));
    double aOz =  hOctFac*(static_cast<double>(anchZ));

    //Anchor of the FGT box
    unsigned int fgtxid = static_cast<unsigned int>(floor(aOx/hRg));
    unsigned int fgtyid = static_cast<unsigned int>(floor(aOy/hRg));
    unsigned int fgtzid = static_cast<unsigned int>(floor(aOz/hRg));

    oct2fgtIdmap.push_back( ( (fgtzid*Ne*Ne) + (fgtyid*Ne) + fgtxid ) );

    oct2fgtPtmap.push_back(fgtxid);
    oct2fgtPtmap.push_back(fgtyid);
    oct2fgtPtmap.push_back(fgtzid);

    double aFx = hRg*static_cast<double>(fgtxid);
    double aFy = hRg*static_cast<double>(fgtyid);
    double aFz = hRg*static_cast<double>(fgtzid);

    //Center of the FGT box
    double halfH = (0.5*hRg);
    double cx =  aFx + halfH;
    double cy =  aFy + halfH;
    double cz =  aFz + halfH;

    //Tensor-Product Acceleration 

    //Stage-1

    tmp1R.resize(2*P);
    tmp1C.resize(2*P);
    for(int k1 = -P; k1 < P; k1++) {
      int shiftK1 = (k1 + P);

      tmp1R[shiftK1].resize(ptGridSizeWithinBox);
      tmp1C[shiftK1].resize(ptGridSizeWithinBox);

      for(int j3 = 0; j3 < ptGridSizeWithinBox; j3++) {
        tmp1R[shiftK1][j3].resize(ptGridSizeWithinBox);
        tmp1C[shiftK1][j3].resize(ptGridSizeWithinBox);

        for(int j2 = 0; j2 < ptGridSizeWithinBox; j2++) {
          tmp1R[shiftK1][j3][j2] = 0.0;
          tmp1C[shiftK1][j3][j2] = 0.0;

          for(int j1 = 0; j1 < ptGridSizeWithinBox; j1++) {
            double px = aOx + ptGridOff + (ptGridH*(static_cast<double>(j1)));

            double theta = lambda*(static_cast<double>(k1)*(cx - px));

            //Replace fMag by drand48() if you want
            tmp1R[shiftK1][j3][j2] += (fMag*cos(theta));
            tmp1C[shiftK1][j3][j2] += (fMag*sin(theta));

          }//end for j1
        }//end for j2
      }//end for j3
    }//end for k1

    //Stage-2

    tmp2R.resize(2*P);
    tmp2C.resize(2*P);
    for(int k2 = -P; k2 < P; k2++) {
      int shiftK2 = (k2 + P);

      tmp2R[shiftK2].resize(2*P);
      tmp2C[shiftK2].resize(2*P);

      for(int k1 = -P; k1 < P; k1++) {
        int shiftK1 = (k1 + P);

        tmp2R[shiftK2][shiftK1].resize(ptGridSizeWithinBox);
        tmp2C[shiftK2][shiftK1].resize(ptGridSizeWithinBox);

        for(int j3 = 0; j3 < ptGridSizeWithinBox; j3++) {
          tmp2R[shiftK2][shiftK1][j3] = 0.0;
          tmp2C[shiftK2][shiftK1][j3] = 0.0;

          for(int j2 = 0; j2 < ptGridSizeWithinBox; j2++) {
            double py = aOy + ptGridOff + (ptGridH*(static_cast<double>(j2)));

            double theta = lambda*(static_cast<double>(k2)*(cy - py));

            double rVal = tmp1R[shiftK1][j3][j2];
            double cVal = tmp1C[shiftK1][j3][j2];

            tmp2R[shiftK2][shiftK1][j3] += ( (rVal*cos(theta)) - (cVal*sin(theta)) );
            tmp2C[shiftK2][shiftK1][j3] += ( (rVal*sin(theta)) + (cVal*cos(theta)) );

          }//end for j2
        }//end for j3
      }//end for k1
    }//end for k2

    //Stage-3

    Woct[i].resize(Ndofs);

    for(int k3 = -P, di = 0; k3 < P; k3++) {
      for(int k2 = -P; k2 < P; k2++) {
        int shiftK2 = (k2 + P);

        for(int k1 = -P; k1 < P; k1++, di++) {
          int shiftK1 = (k1 + P);

          Woct[i][2*di] = 0.0;
          Woct[i][(2*di) + 1] = 0.0;

          for(int j3 = 0; j3 < ptGridSizeWithinBox; j3++) {
            double pz = aOz + ptGridOff + (ptGridH*(static_cast<double>(j3)));

            double theta = lambda*(static_cast<double>(k3)*(cz - pz));

            double rVal = tmp2R[shiftK2][shiftK1][j3];
            double cVal = tmp2C[shiftK2][shiftK1][j3];

            Woct[i][2*di] += ( (rVal*cos(theta)) - (cVal*sin(theta)) );
            Woct[i][(2*di) + 1] += ( (rVal*sin(theta)) + (cVal*cos(theta)) );

          }//end for j3
        }//end for k1
      }//end for k2
    }//end for k3

  }//end for i

  PetscLogEventEnd(s2wEvent, 0, 0, 0, 0);

  //S2W-Comm
  PetscLogEventBegin(s2wCommEvent, 0, 0, 0, 0);

  PetscLogEventEnd(s2wCommEvent, 0, 0, 0, 0);

  //W2D
  PetscLogEventBegin(w2dEvent, 0, 0, 0, 0);

  std::vector<std::vector<double> > directResults(numLocalDirectOcts);

  PetscLogEventEnd(w2dEvent, 0, 0, 0, 0);

  //D2D
  PetscLogEventBegin(d2dEvent, 0, 0, 0, 0);

  PetscLogEventEnd(d2dEvent, 0, 0, 0, 0);

  //W2L
  PetscLogEventBegin(w2lEvent, 0, 0, 0, 0);

  PetscLogEventEnd(w2lEvent, 0, 0, 0, 0);

  //D2L
  PetscLogEventBegin(d2lEvent, 0, 0, 0, 0);

  PetscLogEventEnd(d2lEvent, 0, 0, 0, 0);

  //L2T-Comm
  PetscLogEventBegin(l2tCommEvent, 0, 0, 0, 0);

  PetscLogEventEnd(l2tCommEvent, 0, 0, 0, 0);

  //L2T
  PetscLogEventBegin(l2tEvent, 0, 0, 0, 0);

  const double C0 = ( pow((0.5/sqrt(__PI__)), 3.0)*
      pow((static_cast<double>(L)/static_cast<double>(P)), 3.0) );

  std::vector<std::vector<double> > expandResults(numLocalExpandOcts);

  PetscLogEventEnd(l2tEvent, 0, 0, 0, 0);

  if(writeOut) {
    char fname[256];
    sprintf(fname, "OctExpandOutType2_%d_%d.txt", rank, npes);
    FILE* fp = fopen(fname, "w");
    fprintf(fp, "%d\n", (expandResults.size()));
    for(unsigned int i = 0; i < expandResults.size(); i++) {
      fprintf(fp, "%d\n", expandResults[i].size());
      for(unsigned int j = 0; j < expandResults[i].size(); j++) {
        fprintf(fp, "%lf \n", expandResults[i][j]);
      }//end for j
    }//end for i
    fclose(fp);
  }

  if(writeOut) {
    char fname[256];
    sprintf(fname, "OctDirectOutType2_%d_%d.txt", rank, npes);
    FILE* fp = fopen(fname, "w");
    fprintf(fp, "%d\n", (directResults.size()));
    for(unsigned int i = 0; i < directResults.size(); i++) {
      fprintf(fp, "%d\n", directResults[i].size());
      for(unsigned int j = 0; j < directResults[i].size(); j++) {
        fprintf(fp, "%lf \n", directResults[i][j]);
      }//end for j
    }//end for i
    fclose(fp);
  }

  long long trueTotalPts;
  MPI_Reduce(&trueLocalNumPts, &trueTotalPts, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, comm);

  if(!rank) {
    std::cout<<"True Total NumPts: "<<trueTotalPts<<std::endl; 
  }

  PetscLogEventEnd(fgtEvent, 0, 0, 0, 0);

  PetscFunctionReturn(0);

}

