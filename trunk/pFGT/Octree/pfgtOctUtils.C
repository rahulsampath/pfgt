
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

#define __COMP_MUL_RE(a, ai, b, bi) ( a*b - ai*bi )
#define __COMP_MUL_IM(a, ai, b, bi) ( a*bi + ai*b )

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

  long long numLocalExpandOcts = expandTree.size();
  long long numLocalDirectOcts = directTree.size();

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
  PetscScalar**** WlArr;
  DAVecGetArrayDOF(da, Wlocal, &WlArr);

  VecZeroEntries(Wglobal);
  DAVecGetArrayDOF(da, Wglobal, &WgArr);

  // directW2L(WlArr, WgArr, xs, ys, zs, nx, ny, nz, Ne, h, K, P, lambda);
  sweepW2L(WlArr, WgArr, xs, ys, zs, nx, ny, nz, Ne, hRg, K, P, lambda);

  DAVecRestoreArrayDOF(da, Wlocal, &WlArr);

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

  DAVecRestoreArrayDOF(da, Wglobal, &WgArr);

  sendFgtVals.clear();
  recvFgtVals.clear();

  sendFgtIds.clear();
  recvFgtIds.clear();

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

  VecDestroy(Wlocal);
  VecDestroy(Wglobal);

  DADestroy(da);

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

  long long totalNumExpandOcts;
  long long totalNumDirectOcts;

  MPI_Reduce(&numLocalExpandOcts, &totalNumExpandOcts, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&numLocalDirectOcts, &totalNumDirectOcts, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if(!rank) {
    std::cout<<"Total Num Expand Octs: "<< totalNumExpandOcts << std::endl;
    std::cout<<"Total Num Direct Octs: "<< totalNumDirectOcts << std::endl;
  }

  PetscLogEventEnd(fgtEvent, 0, 0, 0, 0);

  PetscFunctionReturn(0);

}

void directW2L(PetscScalar**** WlArr, PetscScalar**** WgArr, int xs, int ys, int zs, int nx, int ny, int nz, int Ne, double h, const int StencilWidth, const int P, const double lambda) {
  //Loop over local boxes and their Interaction lists and do a direct translation

  for(PetscInt zi = 0; zi < nz; zi++) {
    for(PetscInt yi = 0; yi < ny; yi++) {
      for(PetscInt xi = 0; xi < nx; xi++) {
        int xx = xi +xs;
        int yy = yi +ys;
        int zz = zi +zs;

        //Center of the box B
        /*
           double cBx =  h*(0.5 + static_cast<double>(xi + xs));
           double cBy =  h*(0.5 + static_cast<double>(yi + ys));
           double cBz =  h*(0.5 + static_cast<double>(zi + zs));
           */
        //Bounds for Ilist of box B
        int Ixs = xi + xs - StencilWidth;
        int Ixe = xi + xs + StencilWidth;

        int Iys = yi + ys - StencilWidth;
        int Iye = yi + ys + StencilWidth;

        int Izs = zi + zs - StencilWidth;
        int Ize = zi + zs + StencilWidth;

        if(Ixs < 0) {
          Ixs = 0;
        }
        if(Ixe >= Ne) {
          Ixe = (Ne - 1);
        }

        if(Iys < 0) {
          Iys = 0;
        }
        if(Iye >= Ne) {
          Iye = (Ne - 1);
        }

        if(Izs < 0) {
          Izs = 0;
        }
        if(Ize >= Ne) {
          Ize = (Ne - 1);
        }

#ifdef __DEBUG__
        assert(Ixs >= gxs);
        assert(Iys >= gys);
        assert(Izs >= gzs);

        assert(Ixe < (gxs + gnx));
        assert(Iye < (gys + gny));
        assert(Ize < (gzs + gnz));
#endif

        //Loop over Ilist of box B
        for(int zj = Izs; zj <= Ize; zj++) {
          for(int yj = Iys; yj <= Iye; yj++) {
            for(int xj = Ixs; xj <= Ixe; xj++) {

              //Center of the box C
              /*
                 double cCx =  h*(0.5 + static_cast<double>(xj));
                 double cCy =  h*(0.5 + static_cast<double>(yj));
                 double cCz =  h*(0.5 + static_cast<double>(zj));
                 */

              for(int k3 = -P, di = 0; k3 < P; k3++) {
                for(int k2 = -P; k2 < P; k2++) {
                  for(int k1 = -P; k1 < P; k1++, di++) {

                    double theta = lambda*h*( static_cast<double>(k1*(xj - xx) + k2*(yj - yy) + k3*(zj - zz) ) );

                    WgArr[zi + zs][yi + ys][xi + xs][2*di] += (WlArr[zj][yj][xj][2*di]*cos(theta));
                    WgArr[zi + zs][yi + ys][xi + xs][(2*di) + 1] += (WlArr[zj][yj][xj][(2*di) + 1]*sin(theta));

                  }//end for k1
                }//end for k2
              }//end for k3

            }//end for xj
          }//end for yj
        }//end for zj

      }//end for xi
    }//end for yi
  }//end for zi
}

void sweepW2L(PetscScalar**** WlArr, PetscScalar**** WgArr, 
    int xs, int ys, int zs, 
    int nx, int ny, int nz, 
    int Ne, double h, const int K, 
    const int P, const double lambda) {

  // compute the first layer directly ...  
  /*
     directW2L(WlArr, WgArr, xs, ys, zs, nx, ny, 1, Ne, h, K, P, lambda); // XY Plane
     directW2L(WlArr, WgArr, xs, ys+1, zs, 1, ny-1, nz, Ne, h, K, P, lambda); // YZ Plane
     directW2L(WlArr, WgArr, xs+1, ys, zs+1, nx-1, 1, nz-1, Ne, h, K, P, lambda); // ZX Plane 
     */
  directLayer(WlArr, WgArr, xs+1, ys, zs+1, nx-1, 1, nz-1, Ne, h, K, P, lambda); // ZX Plane 

  // return;

  int num_layers = std::min(std::min(nx, ny), nz);

  double *fac = new double [7*2*8*P*P*P]; // 7* (2P)^3 complex terms ...
  double theta, ct, st;

  for(int k3 = -P, di = 0; k3 < P; k3++) {
    for(int k2 = -P; k2 < P; k2++) {
      for(int k1 = -P; k1 < P; k1++, di++) {
        // i,j,k
        theta = lambda*h* ( (static_cast<double>(k1 + k2 + k3) ) );
        fac[14*di]     = cos(theta);
        fac[14*di + 1] = sin(theta);

        theta = lambda*h* ( (static_cast<double>(k1) ) );
        fac[14*di + 2] = cos(theta);
        fac[14*di + 3] = sin(theta);

        theta = lambda*h* ( (static_cast<double>(k2) ) );
        fac[14*di + 4] = cos(theta);
        fac[14*di + 5] = sin(theta);

        theta = lambda*h* ( (static_cast<double>(k3) ) );
        fac[14*di + 6] = cos(theta);
        fac[14*di + 7] = sin(theta);

        theta = lambda*h* ( (static_cast<double>(k2 + k3) ) );
        fac[14*di + 8] = cos(theta);
        fac[14*di + 9] = sin(theta);

        theta = lambda*h* ( (static_cast<double>(k1 + k3) ) );
        fac[14*di + 10] = cos(theta);
        fac[14*di + 11] = sin(theta);

        theta = lambda*h* ( (static_cast<double>(k1 + k2) ) );
        fac[14*di + 12] = cos(theta);
        fac[14*di + 13] = sin(theta);
      }
    }
  }

  int i,j,k;
  // have the first layer, now propagate ...
  for (int layer = 1; layer < num_layers; layer++) {
    int lx = xs + layer;
    int ly = ys + layer;
    int lz = zs + layer;
    // do XY Plane ... z = lz;
    k=lz;
    for (i=lx; i<xs+nx; i++) {
      for (j=ly; j<ys+ny; j++) {
        // At this stage ...  (i,j,k) needs to be computed and (i-1,j-1,k-1),
        // and the other 6 boxes between them should have already been
        // computed.

        for(int k3 = -P, di = 0; k3 < P; k3++) {
          for(int k2 = -P; k2 < P; k2++) {
            for(int k1 = -P; k1 < P; k1++, di++) {
              WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WgArr[k-1][j-1][i-1][2*di], WgArr[k-1][j-1][i-1][2*di+1],  fac[14*di], fac[14*di+1] );
              WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WgArr[k-1][j-1][i-1][2*di], WgArr[k-1][j-1][i-1][2*di+1],  fac[14*di], fac[14*di+1] );

              WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WgArr[k][j][i-1][2*di], WgArr[k][j][i-1][2*di+1],  fac[14*di+2], fac[14*di+3] );
              WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WgArr[k][j][i-1][2*di], WgArr[k][j][i-1][2*di+1],  fac[14*di+2], fac[14*di+3] );
              WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WgArr[k][j-1][i][2*di], WgArr[k][j-1][i][2*di+1],  fac[14*di+4], fac[14*di+5] );
              WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WgArr[k][j-1][i][2*di], WgArr[k][j-1][i][2*di+1],  fac[14*di+4], fac[14*di+5] );
              WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WgArr[k-1][j][i][2*di], WgArr[k-1][j][i][2*di+1],  fac[14*di+6], fac[14*di+7] );
              WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WgArr[k-1][j][i][2*di], WgArr[k-1][j][i][2*di+1],  fac[14*di+6], fac[14*di+7] );

              WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WgArr[k-1][j-1][i][2*di], WgArr[k-1][j-1][i][2*di+1],  fac[14*di+8], fac[14*di+9] );
              WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WgArr[k-1][j-1][i][2*di], WgArr[k-1][j-1][i][2*di+1],  fac[14*di+8], fac[14*di+9] );
              WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WgArr[k-1][j][i-1][2*di], WgArr[k-1][j][i-1][2*di+1],  fac[14*di+10], fac[14*di+11] );
              WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WgArr[k-1][j][i-1][2*di], WgArr[k-1][j][i-1][2*di+1],  fac[14*di+10], fac[14*di+11] );
              WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WgArr[k][j-1][i-1][2*di], WgArr[k][j-1][i-1][2*di+1],  fac[14*di+12], fac[14*di+13] );
              WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WgArr[k][j-1][i-1][2*di], WgArr[k][j-1][i-1][2*di+1],  fac[14*di+12], fac[14*di+13] );
            } // k3
          } // k2 
        } // k1 

        // now the corner corrections ...
        for(int k3 = -P, di = 0; k3 < P; k3++) {
          for(int k2 = -P; k2 < P; k2++) {
            for(int k1 = -P; k1 < P; k1++, di++) {
              // corner 000
              if ( ( (k-K-1) >=0) && ((j-K-1) >=0) && ((i-K-1) >=0) ) {  
                theta = lambda*h* ( (static_cast<double>((K+1)*(k1 + k2 + k3)) ) );
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WlArr[k-K-1][j-K-1][i-K-1][2*di], WlArr[k-K-1][j-K-1][i-K-1][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WlArr[k-K-1][j-K-1][i-K-1][2*di], WlArr[k-K-1][j-K-1][i-K-1][2*di+1],  ct, st );
              }
              // corner 110
              if ( ( (k-K-1) >=0) && ((j+K) < Ne) && ((i+K) < Ne) ) {  
                theta = lambda*h* ( (static_cast<double>((K+1)* k3 -K*(k1 + k2))) ) ;
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WlArr[k-K-1][j+K][i+K][2*di], WlArr[k-K-1][j+K][i+K][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WlArr[k-K-1][j+K][i+K][2*di], WlArr[k-K-1][j+K][i+K][2*di+1],  ct, st );
              }
              // corner 011
              if ( ( (i-K-1) >=0) && ((j+K) < Ne) && ((k+K) < Ne) ) {  
                theta = lambda*h* ( (static_cast<double>((K+1)* k1 -K*(k2 + k3))) ) ;
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WlArr[k+K][j+K][i-K-1][2*di], WlArr[k+K][j+K][i-K-1][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WlArr[k+K][j+K][i-K-1][2*di], WlArr[k+K][j+K][i-K-1][2*di+1],  ct, st );
              }
              // corner 101
              if ( ( (j-K-1) >=0) && ((k+K) < Ne) && ((i+K) < Ne) ) {  
                theta = lambda*h* ( (static_cast<double>((K+1)* k2 -K*(k1 + k3))) ) ;
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WlArr[k+K][j-K-1][i+K][2*di], WlArr[k+K][j-K-1][i+K][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WlArr[k+K][j-K-1][i+K][2*di], WlArr[k+K][j-K-1][i+K][2*di+1],  ct, st );
              }
              // corner 100
              if ( ( (k-K-1) >=0) && ((j-K-1) >= 0) && ((i+K) < Ne) ) {  
                theta = lambda*h* ( (static_cast<double>((K+1)*(k2 + k3) -K*k1) ) );
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   -= __COMP_MUL_RE( WlArr[k-K-1][j-K-1][i+K][2*di], WlArr[k-K-1][j-K-1][i+K][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] -= __COMP_MUL_IM( WlArr[k-K-1][j-K-1][i+K][2*di], WlArr[k-K-1][j-K-1][i+K][2*di+1],  ct, st );
              }
              // corner 010
              if ( ( (k-K-1) >=0) && ((i-K-1) >= 0) && ((j+K) < Ne) ) {  
                theta = lambda*h* ( (static_cast<double>((K+1)*(k1 + k3) - K*k2) ) );
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   -= __COMP_MUL_RE( WlArr[k-K-1][j+K][i-K-1][2*di], WlArr[k-K-1][j+K][i-K-1][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] -= __COMP_MUL_IM( WlArr[k-K-1][j+K][i-K-1][2*di], WlArr[k-K-1][j+K][i-K-1][2*di+1],  ct, st );
              }
              // corner 001
              if ( ( (i-K-1) >=0) && ((j-K-1) >= 0) && ((k+K) < Ne) ) {  
                theta = lambda*h* ( (static_cast<double>((K+1)*(k1 + k2) - K*k3) ) );
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   -= __COMP_MUL_RE( WlArr[k+K][j-K-1][i-K-1][2*di], WlArr[k+K][j-K-1][i-K-1][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] -= __COMP_MUL_IM( WlArr[k+K][j-K-1][i-K-1][2*di], WlArr[k+K][j-K-1][i-K-1][2*di+1],  ct, st );
              }
              // corner 111
              if ( ( (j+K) < Ne) && ((k+K) < Ne) && ((i+K) < Ne) ) {  
                theta = lambda*h* ( (static_cast<double>((-K)*(k1 + k2 + k3)) ) );
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   -= __COMP_MUL_RE( WlArr[k+K][j+K][i+K][2*di], WlArr[k+K][j+K][i+K][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] -= __COMP_MUL_IM( WlArr[k+K][j+K][i+K][2*di], WlArr[k+K][j+K][i+K][2*di+1],  ct, st );
              }
            }//end for k1
          }//end for k2
        }//end for k3

      }
    } // XY Plane
    // do YZ Plane ... x = lx;
    i=lx;
    for (j=ly+1; j<ys+ny; j++) { // 1st y layer has already been computed ...
      for (k=lz; k<zs+nz; k++) { 
        for(int k3 = -P, di = 0; k3 < P; k3++) {
          for(int k2 = -P; k2 < P; k2++) {
            for(int k1 = -P; k1 < P; k1++, di++) {
              WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WgArr[k-1][j-1][i-1][2*di], WgArr[k-1][j-1][i-1][2*di+1],  fac[14*di], fac[14*di+1] );
              WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WgArr[k-1][j-1][i-1][2*di], WgArr[k-1][j-1][i-1][2*di+1],  fac[14*di], fac[14*di+1] );

              WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WgArr[k][j][i-1][2*di], WgArr[k][j][i-1][2*di+1],  fac[14*di+2], fac[14*di+3] );
              WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WgArr[k][j][i-1][2*di], WgArr[k][j][i-1][2*di+1],  fac[14*di+2], fac[14*di+3] );
              WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WgArr[k][j-1][i][2*di], WgArr[k][j-1][i][2*di+1],  fac[14*di+4], fac[14*di+5] );
              WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WgArr[k][j-1][i][2*di], WgArr[k][j-1][i][2*di+1],  fac[14*di+4], fac[14*di+5] );
              WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WgArr[k-1][j][i][2*di], WgArr[k-1][j][i][2*di+1],  fac[14*di+6], fac[14*di+7] );
              WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WgArr[k-1][j][i][2*di], WgArr[k-1][j][i][2*di+1],  fac[14*di+6], fac[14*di+7] );

              WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WgArr[k-1][j-1][i][2*di], WgArr[k-1][j-1][i][2*di+1],  fac[14*di+8], fac[14*di+9] );
              WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WgArr[k-1][j-1][i][2*di], WgArr[k-1][j-1][i][2*di+1],  fac[14*di+8], fac[14*di+9] );
              WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WgArr[k-1][j][i-1][2*di], WgArr[k-1][j][i-1][2*di+1],  fac[14*di+10], fac[14*di+11] );
              WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WgArr[k-1][j][i-1][2*di], WgArr[k-1][j][i-1][2*di+1],  fac[14*di+10], fac[14*di+11] );
              WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WgArr[k][j-1][i-1][2*di], WgArr[k][j-1][i-1][2*di+1],  fac[14*di+12], fac[14*di+13] );
              WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WgArr[k][j-1][i-1][2*di], WgArr[k][j-1][i-1][2*di+1],  fac[14*di+12], fac[14*di+13] );
            } // k3
          } // k2 
        } // k1 

        // now the corner corrections ...
        for(int k3 = -P, di = 0; k3 < P; k3++) {
          for(int k2 = -P; k2 < P; k2++) {
            for(int k1 = -P; k1 < P; k1++, di++) {
              // corner 000
              if ( ( (k-K-1) >=0) && ((j-K-1) >=0) && ((i-K-1) >=0) ) {  
                theta = lambda*h* ( (static_cast<double>((K+1)*(k1 + k2 + k3)) ) );
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WlArr[k-K-1][j-K-1][i-K-1][2*di], WlArr[k-K-1][j-K-1][i-K-1][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WlArr[k-K-1][j-K-1][i-K-1][2*di], WlArr[k-K-1][j-K-1][i-K-1][2*di+1],  ct, st );
              }
              // corner 110
              if ( ( (k-K-1) >=0) && ((j+K) < Ne) && ((i+K) < Ne) ) {  
                theta = lambda*h* ( (static_cast<double>((K+1)* k3 -K*(k1 + k2))) ) ;
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WlArr[k-K-1][j+K][i+K][2*di], WlArr[k-K-1][j+K][i+K][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WlArr[k-K-1][j+K][i+K][2*di], WlArr[k-K-1][j+K][i+K][2*di+1],  ct, st );
              }
              // corner 011
              if ( ( (i-K-1) >=0) && ((j+K) < Ne) && ((k+K) < Ne) ) {  
                theta = lambda*h* ( (static_cast<double>((K+1)* k1 -K*(k2 + k3))) ) ;
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WlArr[k+K][j+K][i-K-1][2*di], WlArr[k+K][j+K][i-K-1][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WlArr[k+K][j+K][i-K-1][2*di], WlArr[k+K][j+K][i-K-1][2*di+1],  ct, st );
              }
              // corner 101
              if ( ( (j-K-1) >=0) && ((k+K) < Ne) && ((i+K) < Ne) ) {  
                theta = lambda*h* ( (static_cast<double>((K+1)* k2 -K*(k1 + k3))) ) ;
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WlArr[k+K][j-K-1][i+K][2*di], WlArr[k+K][j-K-1][i+K][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WlArr[k+K][j-K-1][i+K][2*di], WlArr[k+K][j-K-1][i+K][2*di+1],  ct, st );
              }
              // corner 100
              if ( ( (k-K-1) >=0) && ((j-K-1) >= 0) && ((i+K) < Ne) ) {  
                theta = lambda*h* ( (static_cast<double>((K+1)*(k2 + k3) -K*k1) ) );
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   -= __COMP_MUL_RE( WlArr[k-K-1][j-K-1][i+K][2*di], WlArr[k-K-1][j-K-1][i+K][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] -= __COMP_MUL_IM( WlArr[k-K-1][j-K-1][i+K][2*di], WlArr[k-K-1][j-K-1][i+K][2*di+1],  ct, st );
              }
              // corner 010
              if ( ( (k-K-1) >=0) && ((i-K-1) >= 0) && ((j+K) < Ne) ) {  
                theta = lambda*h* ( (static_cast<double>((K+1)*(k1 + k3) - K*k2) ) );
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   -= __COMP_MUL_RE( WlArr[k-K-1][j+K][i-K-1][2*di], WlArr[k-K-1][j+K][i-K-1][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] -= __COMP_MUL_IM( WlArr[k-K-1][j+K][i-K-1][2*di], WlArr[k-K-1][j+K][i-K-1][2*di+1],  ct, st );
              }
              // corner 001
              if ( ( (i-K-1) >=0) && ((j-K-1) >= 0) && ((k+K) < Ne) ) {  
                theta = lambda*h* ( (static_cast<double>((K+1)*(k1 + k2) - K*k3) ) );
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   -= __COMP_MUL_RE( WlArr[k+K][j-K-1][i-K-1][2*di], WlArr[k+K][j-K-1][i-K-1][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] -= __COMP_MUL_IM( WlArr[k+K][j-K-1][i-K-1][2*di], WlArr[k+K][j-K-1][i-K-1][2*di+1],  ct, st );
              }
              // corner 111
              if ( ( (j+K) < Ne) && ((k+K) < Ne) && ((i+K) < Ne) ) {  
                theta = lambda*h* ( (static_cast<double>((-K)*(k1 + k2 + k3)) ) );
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   -= __COMP_MUL_RE( WlArr[k+K][j+K][i+K][2*di], WlArr[k+K][j+K][i+K][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] -= __COMP_MUL_IM( WlArr[k+K][j+K][i+K][2*di], WlArr[k+K][j+K][i+K][2*di+1],  ct, st );
              }
            }//end for k1
          }//end for k2
        }//end for k3

      }
    } // YZ Plane
    // do ZX Plane ... y = ly;
    j=ly;
    for (k=lz+1; k<zs+nz; k++) {
      for (i=lx+1; i<xs+nx; i++) {
        for(int k3 = -P, di = 0; k3 < P; k3++) {
          for(int k2 = -P; k2 < P; k2++) {
            for(int k1 = -P; k1 < P; k1++, di++) {
              WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WgArr[k-1][j-1][i-1][2*di], WgArr[k-1][j-1][i-1][2*di+1],  fac[14*di], fac[14*di+1] );
              WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WgArr[k-1][j-1][i-1][2*di], WgArr[k-1][j-1][i-1][2*di+1],  fac[14*di], fac[14*di+1] );

              WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WgArr[k][j][i-1][2*di], WgArr[k][j][i-1][2*di+1],  fac[14*di+2], fac[14*di+3] );
              WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WgArr[k][j][i-1][2*di], WgArr[k][j][i-1][2*di+1],  fac[14*di+2], fac[14*di+3] );
              WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WgArr[k][j-1][i][2*di], WgArr[k][j-1][i][2*di+1],  fac[14*di+4], fac[14*di+5] );
              WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WgArr[k][j-1][i][2*di], WgArr[k][j-1][i][2*di+1],  fac[14*di+4], fac[14*di+5] );
              WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WgArr[k-1][j][i][2*di], WgArr[k-1][j][i][2*di+1],  fac[14*di+6], fac[14*di+7] );
              WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WgArr[k-1][j][i][2*di], WgArr[k-1][j][i][2*di+1],  fac[14*di+6], fac[14*di+7] );

              WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WgArr[k-1][j-1][i][2*di], WgArr[k-1][j-1][i][2*di+1],  fac[14*di+8], fac[14*di+9] );
              WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WgArr[k-1][j-1][i][2*di], WgArr[k-1][j-1][i][2*di+1],  fac[14*di+8], fac[14*di+9] );
              WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WgArr[k-1][j][i-1][2*di], WgArr[k-1][j][i-1][2*di+1],  fac[14*di+10], fac[14*di+11] );
              WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WgArr[k-1][j][i-1][2*di], WgArr[k-1][j][i-1][2*di+1],  fac[14*di+10], fac[14*di+11] );
              WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WgArr[k][j-1][i-1][2*di], WgArr[k][j-1][i-1][2*di+1],  fac[14*di+12], fac[14*di+13] );
              WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WgArr[k][j-1][i-1][2*di], WgArr[k][j-1][i-1][2*di+1],  fac[14*di+12], fac[14*di+13] );
            } // k3
          } // k2 
        } // k1 

        // now the corner corrections ...
        for(int k3 = -P, di = 0; k3 < P; k3++) {
          for(int k2 = -P; k2 < P; k2++) {
            for(int k1 = -P; k1 < P; k1++, di++) {
              // corner 000
              if ( ( (k-K-1) >=0) && ((j-K-1) >=0) && ((i-K-1) >=0) ) {  
                theta = lambda*h* ( (static_cast<double>((K+1)*(k1 + k2 + k3)) ) );
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WlArr[k-K-1][j-K-1][i-K-1][2*di], WlArr[k-K-1][j-K-1][i-K-1][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WlArr[k-K-1][j-K-1][i-K-1][2*di], WlArr[k-K-1][j-K-1][i-K-1][2*di+1],  ct, st );
              }
              // corner 110
              if ( ( (k-K-1) >=0) && ((j+K) < Ne) && ((i+K) < Ne) ) {  
                theta = lambda*h* ( (static_cast<double>((K+1)* k3 -K*(k1 + k2))) ) ;
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WlArr[k-K-1][j+K][i+K][2*di], WlArr[k-K-1][j+K][i+K][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WlArr[k-K-1][j+K][i+K][2*di], WlArr[k-K-1][j+K][i+K][2*di+1],  ct, st );
              }
              // corner 011
              if ( ( (i-K-1) >=0) && ((j+K) < Ne) && ((k+K) < Ne) ) {  
                theta = lambda*h* ( (static_cast<double>((K+1)* k1 -K*(k2 + k3))) ) ;
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WlArr[k+K][j+K][i-K-1][2*di], WlArr[k+K][j+K][i-K-1][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WlArr[k+K][j+K][i-K-1][2*di], WlArr[k+K][j+K][i-K-1][2*di+1],  ct, st );
              }
              // corner 101
              if ( ( (j-K-1) >=0) && ((k+K) < Ne) && ((i+K) < Ne) ) {  
                theta = lambda*h* ( (static_cast<double>((K+1)* k2 -K*(k1 + k3))) ) ;
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WlArr[k+K][j-K-1][i+K][2*di], WlArr[k+K][j-K-1][i+K][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WlArr[k+K][j-K-1][i+K][2*di], WlArr[k+K][j-K-1][i+K][2*di+1],  ct, st );
              }
              // corner 100
              if ( ( (k-K-1) >=0) && ((j-K-1) >= 0) && ((i+K) < Ne) ) {  
                theta = lambda*h* ( (static_cast<double>((K+1)*(k2 + k3) -K*k1) ) );
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   -= __COMP_MUL_RE( WlArr[k-K-1][j-K-1][i+K][2*di], WlArr[k-K-1][j-K-1][i+K][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] -= __COMP_MUL_IM( WlArr[k-K-1][j-K-1][i+K][2*di], WlArr[k-K-1][j-K-1][i+K][2*di+1],  ct, st );
              }
              // corner 010
              if ( ( (k-K-1) >=0) && ((i-K-1) >= 0) && ((j+K) < Ne) ) {  
                theta = lambda*h* ( (static_cast<double>((K+1)*(k1 + k3) - K*k2) ) );
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   -= __COMP_MUL_RE( WlArr[k-K-1][j+K][i-K-1][2*di], WlArr[k-K-1][j+K][i-K-1][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] -= __COMP_MUL_IM( WlArr[k-K-1][j+K][i-K-1][2*di], WlArr[k-K-1][j+K][i-K-1][2*di+1],  ct, st );
              }
              // corner 001
              if ( ( (i-K-1) >=0) && ((j-K-1) >= 0) && ((k+K) < Ne) ) {  
                theta = lambda*h* ( (static_cast<double>((K+1)*(k1 + k2) - K*k3) ) );
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   -= __COMP_MUL_RE( WlArr[k+K][j-K-1][i-K-1][2*di], WlArr[k+K][j-K-1][i-K-1][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] -= __COMP_MUL_IM( WlArr[k+K][j-K-1][i-K-1][2*di], WlArr[k+K][j-K-1][i-K-1][2*di+1],  ct, st );
              }
              // corner 111
              if ( ( (j+K) < Ne) && ((k+K) < Ne) && ((i+K) < Ne) ) {  
                theta = lambda*h* ( (static_cast<double>((-K)*(k1 + k2 + k3)) ) );
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   -= __COMP_MUL_RE( WlArr[k+K][j+K][i+K][2*di], WlArr[k+K][j+K][i+K][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] -= __COMP_MUL_IM( WlArr[k+K][j+K][i+K][2*di], WlArr[k+K][j+K][i+K][2*di+1],  ct, st );
              }
            }//end for k1
          }//end for k2
        }//end for k3

      }
    } // ZX Plane

  } // layer ...

  delete [] fac;
}

void directLayer(PetscScalar**** WlArr, PetscScalar**** WgArr, 
    int xs, int ys, int zs, 
    int nx, int ny, int nz, 
    int Ne, double h, 
    const int K, const int P, const double lambda) {

  int i,j,k;
  int p,q,r;
  double theta, ct, st;

  // 0. compute directly for first box ... xs, ys, zs
  directW2L(WlArr, WgArr, xs, ys, zs, 1, 1, 1, Ne, h, K, P, lambda); 

  // 1. Precompute the factors ...
  double *fac = new double [3*2*8*P*P*P]; // 3* (2P)^3 complex terms ... one for X,Y and Z shifts, 
  for(int k3 = -P, di = 0; k3 < P; k3++) {
    for(int k2 = -P; k2 < P; k2++) {
      for(int k1 = -P; k1 < P; k1++, di++) {
        theta = lambda*h* ( (static_cast<double>(k1) ) );
        fac[6*di]     = cos(theta);
        fac[6*di + 1] = sin(theta);

        theta = lambda*h* ( (static_cast<double>(k2) ) );
        fac[6*di + 2] = cos(theta);
        fac[6*di + 3] = sin(theta);

        theta = lambda*h* ( (static_cast<double>(k3) ) );
        fac[6*di + 4] = cos(theta);
        fac[6*di + 5] = sin(theta);
      }
    }
  }

  // printf("Precomputed facs\n");
  // MPI_Barrier(MPI_COMM_WORLD);

  // 2. Now incrementaly for the XY plane ...
  k = zs;
  for (j=ys; j<ys+ny; j++) {
    for (i=xs; i<xs+nx; i++) {
      if (i == xs) { // special treatment ...
        if (j == ys)
          continue;
        // Propagate from -Y neighbour instead ...
        for(int k3 = -P, di = 0; k3 < P; k3++) {
          for(int k2 = -P; k2 < P; k2++) {
            for(int k1 = -P; k1 < P; k1++, di++) {
              WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WgArr[k][j-1][i][2*di], WgArr[k][j-1][i-1][2*di+1],  fac[6*di+2], fac[6*di+3] );
              WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WgArr[k][j-1][i][2*di], WgArr[k][j-1][i-1][2*di+1],  fac[6*di+2], fac[6*di+3] );
            }
          }
        }
        // add/sub layer ... will add/sub the XZ plane ...
        for (r=-K; r<=K; r++) {
          for (p=-K; p<=K; p++) {
            for(int k3 = -P, di = 0; k3 < P; k3++) {
              for(int k2 = -P; k2 < P; k2++) {
                for(int k1 = -P; k1 < P; k1++, di++) {
                  // add the layer ...
                  q = j+K;
                  if ( ( (j+K) < Ne) && ((i+p) >= 0) && ((i+p) < Ne) && ((k+r) >= 0) && ((k+r) < Ne) ) {  
                    theta = lambda*h* ( (static_cast<double>( -K*k2 - p*k1 - r*k3 ) ) );
                    ct = cos(theta);
                    st = sin(theta);
                    WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WlArr[k+r][j+K][i+p][2*di], WlArr[k+r][j+K][i+p][2*di+1],  ct, st );
                    WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WlArr[k+r][j+K][i+p][2*di], WlArr[k+r][j+K][i+p][2*di+1],  ct, st );
                  }
                  // remove the layer 
                  q = j-K-1;
                  if ( ( (j-K-1) >= 0) && ((i+p) >= 0) && ((i+p) < Ne) && ((k+r) >= 0) && ((k+r) < Ne) ) {  
                    theta = lambda*h* ( (static_cast<double>( (K+1)*k2 - p*k1 - r*k3 ) ) );
                    ct = cos(theta);
                    st = sin(theta);
                    WgArr[k][j][i][2*di]   -= __COMP_MUL_RE( WlArr[k+r][j-K-1][i+p][2*di], WlArr[k+r][j-K-1][i+p][2*di+1],  ct, st );
                    WgArr[k][j][i][2*di+1] -= __COMP_MUL_IM( WlArr[k+r][j-K-1][i+p][2*di], WlArr[k+r][j-K-1][i+p][2*di+1],  ct, st );
                  }
                } // k1
              } // k2
            } //k3
          } //q
        } // r
      } else {
        // Get from -X neighbour ... 
        for(int k3 = -P, di = 0; k3 < P; k3++) {
          for(int k2 = -P; k2 < P; k2++) {
            for(int k1 = -P; k1 < P; k1++, di++) {
              WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WgArr[k][j][i-1][2*di], WgArr[k][j][i-1][2*di+1],  fac[6*di], fac[6*di+1] );
              WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WgArr[k][j][i-1][2*di], WgArr[k][j][i-1][2*di+1],  fac[6*di], fac[6*di+1] );
            }
          }
        }
        // add/sub layer ... will add/sub the YZ plane ...
        for (r=-K; r<=K; r++) {
          for (q=-K; q<=K; q++) {
            for(int k3 = -P, di = 0; k3 < P; k3++) {
              for(int k2 = -P; k2 < P; k2++) {
                for(int k1 = -P; k1 < P; k1++, di++) {
                  // add the layer ...
                  p = i+K;
                  if ( ( (i+K) < Ne) && ((j+q) >= 0) && ((j+q) < Ne) && ((k+r) >= 0) && ((k+r) < Ne) ) {  
                    theta = lambda*h* ( (static_cast<double>( -K*k1 - q*k2 - r*k3 ) ) );
                    ct = cos(theta);
                    st = sin(theta);
                    WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WlArr[k+r][j+q][i+K][2*di], WlArr[k+r][j+q][i+K][2*di+1],  ct, st );
                    WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WlArr[k+r][j+q][i+K][2*di], WlArr[k+r][j+q][i+K][2*di+1],  ct, st );
                  }
                  // remove the layer 
                  p = i-K-1;
                  if ( ( (i-K-1) >= 0) && ((j+q) >= 0) && ((j+q) < Ne) && ((k+r) >= 0) && ((k+r) < Ne) ) {  
                    theta = lambda*h* ( (static_cast<double>( (K+1)*k1 - q*k2 - r*k3 ) ) );
                    ct = cos(theta);
                    st = sin(theta);
                    WgArr[k][j][i][2*di]   -= __COMP_MUL_RE( WlArr[k+r][j+q][i-K-1][2*di], WlArr[k+r][j+q][i-K-1][2*di+1],  ct, st );
                    WgArr[k][j][i][2*di+1] -= __COMP_MUL_IM( WlArr[k+r][j+q][i-K-1][2*di], WlArr[k+r][j+q][i-K-1][2*di+1],  ct, st );
                  }
                } // k1
              } // k2
            } //k3
          } //q
        } // r
      } // special 
    } // i
  } // j

  // MPI_Barrier(MPI_COMM_WORLD);
  // printf("Done plane XY\n");
  // 3. Now for YZ
  i = xs;
  for (k=zs+1; k<zs+nz; k++) {
    for (j=ys; j<ys+ny; j++) {
      // Get from -Z neighbour ... 
      for(int k3 = -P, di = 0; k3 < P; k3++) {
        for(int k2 = -P; k2 < P; k2++) {
          for(int k1 = -P; k1 < P; k1++, di++) {
            WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WgArr[k-1][j][i][2*di], WgArr[k-1][j][i][2*di+1],  fac[6*di+4], fac[6*di+5] );
            WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WgArr[k-1][j][i][2*di], WgArr[k-1][j][i][2*di+1],  fac[6*di+4], fac[6*di+5] );
          }
        }
      }

      // add/sub layer ... will add/sub the XY plane ...
      for (q=-K; q<=K; q++) {
        for (p=-K; p<=K; p++) {
          for(int k3 = -P, di = 0; k3 < P; k3++) {
            for(int k2 = -P; k2 < P; k2++) {
              for(int k1 = -P; k1 < P; k1++, di++) {
                // add the layer ...
                r = k+K;
                if ( ( (k+K) < Ne) && ((j+q) >= 0) && ((j+q) < Ne) && ((i+p) >= 0) && ((i+r) < Ne) ) {  
                  theta = lambda*h* ( (static_cast<double>( -p*k1 - q*k2 - K*k3 ) ) );
                  ct = cos(theta);
                  st = sin(theta);
                  WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WlArr[k+K][j+q][i+p][2*di], WlArr[k+K][j+q][i+p][2*di+1],  ct, st );
                  WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WlArr[k+K][j+q][i+p][2*di], WlArr[k+K][j+q][i+p][2*di+1],  ct, st );
                }
                // remove the layer 
                r = k-K-1;
                if ( ( (k-K-1) >= 0) && ((j+q) >= 0) && ((j+q) < Ne) && ((i+p) >= 0) && ((i+p) < Ne) ) {  
                  theta = lambda*h* ( (static_cast<double>( -p*k1 - q*k2 + (K+1)*k3 ) ) );
                  ct = cos(theta);
                  st = sin(theta);
                  WgArr[k][j][i][2*di]   -= __COMP_MUL_RE( WlArr[k-K-1][j+q][i+p][2*di], WlArr[k-K-1][j+q][i+p][2*di+1],  ct, st );
                  WgArr[k][j][i][2*di+1] -= __COMP_MUL_IM( WlArr[k-K-1][j+q][i+p][2*di], WlArr[k-K-1][j+q][i+p][2*di+1],  ct, st );
                }
              } // k1
            } // k2
          } //k3
        } //q
      } // r

    } // j
  } // k

  // MPI_Barrier(MPI_COMM_WORLD);
  // printf("Done plane YZ\n");
  // MPI_Barrier(MPI_COMM_WORLD);

  // 4. And finally the ZX plane
  j = ys;
  for (k=zs+1; k<zs+nz; k++) {
    for (i=xs+1; i<xs+nx; i++) {
      // Get from -X neighbour ... 
      for(int k3 = -P, di = 0; k3 < P; k3++) {
        for(int k2 = -P; k2 < P; k2++) {
          for(int k1 = -P; k1 < P; k1++, di++) {
            WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WgArr[k][j][i-1][2*di], WgArr[k][j][i-1][2*di+1],  fac[6*di], fac[6*di+1] );
            WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WgArr[k][j][i-1][2*di], WgArr[k][j][i-1][2*di+1],  fac[6*di], fac[6*di+1] );
          }
        }
      }
      // add/sub layer ... will add/sub the XY plane ...
      for (q=-K; q<=K; q++) {
        for (p=-K; p<=K; p++) {
          for(int k3 = -P, di = 0; k3 < P; k3++) {
            for(int k2 = -P; k2 < P; k2++) {
              for(int k1 = -P; k1 < P; k1++, di++) {
                // add the layer ...
                r = k+K;
                if ( ( (k+K) < Ne) && ((j+q) >= 0) && ((j+q) < Ne) && ((i+p) >= 0) && ((i+r) < Ne) ) {  
                  theta = lambda*h* ( (static_cast<double>( -p*k1 - q*k2 - K*k3 ) ) );
                  ct = cos(theta);
                  st = sin(theta);
                  WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WlArr[k+K][j+q][i+p][2*di], WlArr[k+K][j+q][i+p][2*di+1],  ct, st );
                  WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WlArr[k+K][j+q][i+p][2*di], WlArr[k+K][j+q][i+p][2*di+1],  ct, st );
                }
                // remove the layer 
                r = k-K-1;
                if ( ( (k-K-1) >= 0) && ((j+q) >= 0) && ((j+q) < Ne) && ((i+p) >= 0) && ((i+p) < Ne) ) {  
                  theta = lambda*h* ( (static_cast<double>( -p*k1 - q*k2 + (K+1)*k3 ) ) );
                  ct = cos(theta);
                  st = sin(theta);
                  WgArr[k][j][i][2*di]   -= __COMP_MUL_RE( WlArr[k-K-1][j+q][i+p][2*di], WlArr[k-K-1][j+q][i+p][2*di+1],  ct, st );
                  WgArr[k][j][i][2*di+1] -= __COMP_MUL_IM( WlArr[k-K-1][j+q][i+p][2*di], WlArr[k-K-1][j+q][i+p][2*di+1],  ct, st );
                }
              } // k1
            } // k2
          } //k3
        } //q
      } // r
    } // i
  } // j

  // MPI_Barrier(MPI_COMM_WORLD);
  // printf("Done plane ZX\n");
  // clean up 
  delete [] fac;
}


