
#include "mpi.h"
#include "petscda.h"
#include "pfgtOctUtils.h"
#include "seqUtils.h"
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
    double delta, double fMag, unsigned int ptGridSizeWithinBox, 
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
    std::cout<<"Ndofs = "<< Ndofs <<std::endl;
  }

  if(!rank) {
    std::cout<<"StencilWidth = "<< K <<std::endl;
  }

  DA da;
  DACreate3d(comm, DA_NONPERIODIC, DA_STENCIL_BOX, Ne, Ne, Ne,
      PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, Ndofs, K,
      PETSC_NULL, PETSC_NULL, PETSC_NULL, &da);

  if(!rank) {
    std::cout<<"Created DA"<<std::endl;
  }

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

  if(!rank) {
    std::cout<<"Marked Octants"<<std::endl;
  }

  const unsigned int numLocalExpandOcts = expandTree.size();
  const unsigned int numLocalDirectOcts = directTree.size();

  //Tensor-Product grid on each octant

  //Tensor-Product Grid
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

  if(!rank) {
    std::cout<<"Finished S2W"<<std::endl;
  }

  //S2W-Comm
  PetscLogEventBegin(s2wCommEvent, 0, 0, 0, 0);

  std::vector<unsigned int> uniqueOct2fgtIdmap = oct2fgtIdmap;
  seq::makeVectorUnique<unsigned int>(uniqueOct2fgtIdmap, false);

  std::vector<std::vector<double> > Wfgt(uniqueOct2fgtIdmap.size());
  for(unsigned int i = 0; i < Wfgt.size(); i++) {
    Wfgt[i].resize(Ndofs);
    for(unsigned int j = 0; j < Ndofs; j++) {
      Wfgt[i][j] = 0.0;
    }//end for j
  }//end for i

  for(unsigned int i = 0; i < numLocalExpandOcts; i++) {
    unsigned int fgtId = oct2fgtIdmap[i];
    unsigned int fgtIndex;
    bool idxFound = seq::BinarySearch<unsigned int>( (&(*(uniqueOct2fgtIdmap.begin()))), 
        uniqueOct2fgtIdmap.size(), fgtId, &fgtIndex);
    assert(idxFound);
    assert(uniqueOct2fgtIdmap[fgtIndex] == fgtId);

    //Reset map value 
    oct2fgtIdmap[i] = fgtIndex;

    for(unsigned int j = 0; j < Ndofs; j++) {
      Wfgt[fgtIndex][j] += Woct[i][j];
    }//end for j
  }//end for i

  //Do Not Free lx, ly, lz. They are managed by DA
  const PetscInt* lx = NULL;
  const PetscInt* ly = NULL;
  const PetscInt* lz = NULL;
  PetscInt npx, npy, npz;

  //Information about DA partition
  DAGetOwnershipRanges(da, &lx, &ly, &lz);
  DAGetInfo(da, PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL, &npx, &npy, &npz,	
      PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL);

  std::vector<unsigned int> scanLx(npx);
  std::vector<unsigned int> scanLy(npy);
  std::vector<unsigned int> scanLz(npz);

  scanLx[0] = 0.0;
  scanLy[0] = 0.0;
  scanLz[0] = 0.0;
  for(int i = 1; i < npx; i++) {
    scanLx[i] = scanLx[i - 1] + lx[i - 1];
  }
  for(int i = 1; i < npy; i++) {
    scanLy[i] = scanLy[i - 1] + ly[i - 1];
  }
  for(int i = 1; i < npz; i++) {
    scanLz[i] = scanLz[i - 1] + lz[i - 1];
  }

  Vec Wglobal;
  DACreateGlobalVector(da, &Wglobal);

  VecZeroEntries(Wglobal);

  PetscScalar**** WgArr;
  DAVecGetArrayDOF(da, Wglobal, &WgArr);

  PetscInt xs, ys, zs, nx, ny, nz;
  DAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);

  std::vector<bool> isFGTboxEmpty(nx*ny*nz);
  for(unsigned int i = 0; i < isFGTboxEmpty.size(); i++) {
    isFGTboxEmpty[i] = true;
  }//end for i

  std::vector<int> sendCnts(npes); 
  for(int i = 0; i < npes; i++) {
    sendCnts[i] = 0;
  }//end for i

  std::vector<int> part(Wfgt.size());
  for(unsigned int i = 0; i < Wfgt.size(); i++) {
    unsigned int fgtId = uniqueOct2fgtIdmap[i];
    unsigned int fgtzid = (fgtId/(Ne*Ne));
    unsigned int fgtyid = ((fgtId%(Ne*Ne))/Ne);
    unsigned int fgtxid = ((fgtId%(Ne*Ne))%Ne);

    unsigned int xRes, yRes, zRes;
    seq::maxLowerBound<unsigned int>(scanLx, fgtxid, xRes, 0, 0);
    seq::maxLowerBound<unsigned int>(scanLy, fgtyid, yRes, 0, 0);
    seq::maxLowerBound<unsigned int>(scanLz, fgtzid, zRes, 0, 0);

    //Processor that owns the FGT box
    part[i] = (((zRes*npy) + yRes)*npx) + xRes;

    if(part[i] == rank) {
      unsigned int boxId = ( ((fgtzid - zs)*nx*ny) + ((fgtyid - ys)*nx) + (fgtxid - xs) );
      isFGTboxEmpty[boxId] = false;
      for(unsigned int j = 0; j < Ndofs; j++) {
        WgArr[fgtzid][fgtyid][fgtxid][j] += Wfgt[i][j];
      }//end for j
    } else {
      sendCnts[part[i]]++;
    }
  }//end for i

  std::vector<int> recvCnts(npes); 

  MPI_Alltoall( (&(*(sendCnts.begin()))), 1, MPI_INT,
      (&(*(recvCnts.begin()))), 1, MPI_INT, comm );

  std::vector<int> sendDisps(npes);
  std::vector<int> recvDisps(npes);
  sendDisps[0] = 0;
  recvDisps[0] = 0;
  for(int i = 1; i < npes; i++) {
    sendDisps[i] = sendDisps[i - 1] + sendCnts[i - 1];
    recvDisps[i] = recvDisps[i - 1] + recvCnts[i - 1];
  }//end for i

  std::vector<unsigned int> sendFgtIds(sendDisps[npes - 1] + sendCnts[npes - 1]);

  for(int i = 0; i < npes; i++) {
    sendCnts[i] = 0;
  }//end for i

  for(unsigned int i = 0; i < Wfgt.size(); i++) {
    if(part[i] != rank) {
      sendFgtIds[ sendDisps[part[i]] + sendCnts[part[i]] ] = uniqueOct2fgtIdmap[i];
      sendCnts[part[i]]++;
    }
  }//end for i

  std::vector<unsigned int> recvFgtIds(recvDisps[npes - 1] + recvCnts[npes - 1]);

  MPI_Alltoallv( (&(*(sendFgtIds.begin()))), (&(*(sendCnts.begin()))), (&(*(sendDisps.begin()))), MPI_UNSIGNED, 
      (&(*(recvFgtIds.begin()))), (&(*(recvCnts.begin()))), (&(*(recvDisps.begin()))), MPI_UNSIGNED, comm );

  for(unsigned int i = 0; i < npes; i++) {
    sendDisps[i] *= Ndofs;
    recvCnts[i] *= Ndofs;
    recvDisps[i] *= Ndofs;
  }//end for i

  std::vector<double> sendFgtVals((Ndofs*(sendFgtIds.size())));

  for(int i = 0; i < npes; i++) {
    sendCnts[i] = 0;
  }//end for i

  for(unsigned int i = 0; i < Wfgt.size(); i++) {
    if(part[i] != rank) {
      for(unsigned int j = 0; j < Ndofs; j++) {
        sendFgtVals[ sendDisps[part[i]] + sendCnts[part[i]] + j ] = Wfgt[i][j];
      }//end for j
      sendCnts[part[i]] += Ndofs;
    }
  }//end for i

  std::vector<double> recvFgtVals(recvDisps[npes - 1] + recvCnts[npes - 1]);

  MPI_Alltoallv( (&(*(sendFgtVals.begin()))), (&(*(sendCnts.begin()))), (&(*(sendDisps.begin()))), MPI_DOUBLE, 
      (&(*(recvFgtVals.begin()))), (&(*(recvCnts.begin()))), (&(*(recvDisps.begin()))), MPI_DOUBLE, comm );

  for(unsigned int i = 0; i < recvFgtIds.size(); i++) {
    unsigned int fgtId = recvFgtIds[i];
    unsigned int fgtzid = (fgtId/(Ne*Ne));
    unsigned int fgtyid = ((fgtId%(Ne*Ne))/Ne);
    unsigned int fgtxid = ((fgtId%(Ne*Ne))%Ne);

    unsigned int boxId = ( ((fgtzid - zs)*nx*ny) + ((fgtyid - ys)*nx) + (fgtxid - xs) );
    isFGTboxEmpty[boxId] = false;
    for(unsigned int j = 0; j < Ndofs; j++) {
      WgArr[fgtzid][fgtyid][fgtxid][j] += recvFgtVals[(i*Ndofs) + j];
    }//end for j
  }//end for i

  sendFgtVals.clear();
  recvFgtVals.clear();

  sendFgtIds.clear();
  recvFgtIds.clear();

  DAVecRestoreArrayDOF(da, Wglobal, &WgArr);

  PetscLogEventEnd(s2wCommEvent, 0, 0, 0, 0);

  if(!rank) {
    std::cout<<"Finished S2Wcomm"<<std::endl;
  }

  const double C0 = ( pow((0.5/sqrt(__PI__)), 3.0)*
      pow((static_cast<double>(L)/static_cast<double>(P)), 3.0) );

  //W2D
  PetscLogEventBegin(w2dEvent, 0, 0, 0, 0);

  std::vector<std::vector<double> > directResults(numLocalDirectOcts);

  PetscLogEventEnd(w2dEvent, 0, 0, 0, 0);

  if(!rank) {
    std::cout<<"Finished W2D"<<std::endl;
  }

  //D2D
  PetscLogEventBegin(d2dEvent, 0, 0, 0, 0);

  PetscLogEventEnd(d2dEvent, 0, 0, 0, 0);

  if(!rank) {
    std::cout<<"Finished D2D"<<std::endl;
  }

  //W2L
  PetscLogEventBegin(w2lEvent, 0, 0, 0, 0);

  Vec Wlocal;
  DACreateLocalVector(da, &Wlocal);

  DAGlobalToLocalBegin(da, Wglobal, INSERT_VALUES, Wlocal);
  DAGlobalToLocalEnd(da, Wglobal, INSERT_VALUES, Wlocal);

  //Sequential W2L

  PetscLogEventEnd(w2lEvent, 0, 0, 0, 0);

  if(!rank) {
    std::cout<<"Finished W2L"<<std::endl;
  }

  //D2L
  PetscLogEventBegin(d2lEvent, 0, 0, 0, 0);

  PetscLogEventEnd(d2lEvent, 0, 0, 0, 0);

  if(!rank) {
    std::cout<<"Finished D2L"<<std::endl;
  }

  //L2T-Comm
  PetscLogEventBegin(l2tCommEvent, 0, 0, 0, 0);

  PetscLogEventEnd(l2tCommEvent, 0, 0, 0, 0);

  if(!rank) {
    std::cout<<"Finished L2Tcomm"<<std::endl;
  }

  //L2T
  PetscLogEventBegin(l2tEvent, 0, 0, 0, 0);

  std::vector<std::vector<double> > expandResults(numLocalExpandOcts);

  PetscLogEventEnd(l2tEvent, 0, 0, 0, 0);

  if(!rank) {
    std::cout<<"Finished L2T"<<std::endl;
  }

  DADestroy(da);

  VecDestroy(Wlocal);
  VecDestroy(Wglobal);

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

