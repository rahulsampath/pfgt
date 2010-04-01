
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

#define __COMP_MUL_RE(a, ai, b, bi) ( ((a)*(b)) - ((ai)*(bi)) )
#define __COMP_MUL_IM(a, ai, b, bi) ( ((a)*(bi)) + ((ai)*(b)) )

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

  const double LbyP = static_cast<double>(L)/static_cast<double>(P);
  const double ReExpZfactor = -0.25*LbyP*LbyP;
  const double ImExpZfactor = LbyP/sqrt(delta); 

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

            double theta = ImExpZfactor*(static_cast<double>(k1)*(cx - px));

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

            double theta = ImExpZfactor*(static_cast<double>(k2)*(cy - py));

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

            double theta = ImExpZfactor*(static_cast<double>(k3)*(cz - pz));

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

  Woct.clear();

  DA da;
  DACreate3d(comm, DA_NONPERIODIC, DA_STENCIL_BOX, Ne, Ne, Ne,
      PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, Ndofs, K,
      PETSC_NULL, PETSC_NULL, PETSC_NULL, &da);

  if(!rank) {
    std::cout<<"Created DA"<<std::endl;
  }

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

  std::vector<int> s2wSendCnts(npes); 
  for(int i = 0; i < npes; i++) {
    s2wSendCnts[i] = 0;
  }//end for i

  std::vector<int> s2wPart(Wfgt.size());
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
    s2wPart[i] = (((zRes*npy) + yRes)*npx) + xRes;

    if(s2wPart[i] == rank) {
      unsigned int boxId = ( ((fgtzid - zs)*nx*ny) + ((fgtyid - ys)*nx) + (fgtxid - xs) );
      isFGTboxEmpty[boxId] = false;
      for(unsigned int j = 0; j < Ndofs; j++) {
        WgArr[fgtzid][fgtyid][fgtxid][j] = Wfgt[i][j];
      }//end for j
    } else {
      s2wSendCnts[s2wPart[i]]++;
    }
  }//end for i

  std::vector<int> s2wRecvCnts(npes); 

  MPI_Alltoall( (&(*(s2wSendCnts.begin()))), 1, MPI_INT,
      (&(*(s2wRecvCnts.begin()))), 1, MPI_INT, comm );

  std::vector<int> s2wSendDisps(npes);
  std::vector<int> s2wRecvDisps(npes);
  s2wSendDisps[0] = 0;
  s2wRecvDisps[0] = 0;
  for(int i = 1; i < npes; i++) {
    s2wSendDisps[i] = s2wSendDisps[i - 1] + s2wSendCnts[i - 1];
    s2wRecvDisps[i] = s2wRecvDisps[i - 1] + s2wRecvCnts[i - 1];
  }//end for i

  std::vector<unsigned int> s2wSendFgtIds(s2wSendDisps[npes - 1] + s2wSendCnts[npes - 1]);

  for(int i = 0; i < npes; i++) {
    s2wSendCnts[i] = 0;
  }//end for i

  for(unsigned int i = 0; i < Wfgt.size(); i++) {
    if(s2wPart[i] != rank) {
      s2wSendFgtIds[ s2wSendDisps[s2wPart[i]] + s2wSendCnts[s2wPart[i]] ] = uniqueOct2fgtIdmap[i];
      s2wSendCnts[s2wPart[i]]++;
    }
  }//end for i

  std::vector<unsigned int> s2wRecvFgtIds(s2wRecvDisps[npes - 1] + s2wRecvCnts[npes - 1]);

  MPI_Alltoallv( (&(*(s2wSendFgtIds.begin()))), (&(*(s2wSendCnts.begin()))), (&(*(s2wSendDisps.begin()))), MPI_UNSIGNED, 
      (&(*(s2wRecvFgtIds.begin()))), (&(*(s2wRecvCnts.begin()))), (&(*(s2wRecvDisps.begin()))), MPI_UNSIGNED, comm );

  for(unsigned int i = 0; i < npes; i++) {
    s2wSendDisps[i] *= Ndofs;
    s2wRecvCnts[i] *= Ndofs;
    s2wRecvDisps[i] *= Ndofs;
  }//end for i

  std::vector<double> s2wSendFgtVals((Ndofs*(s2wSendFgtIds.size())));

  for(int i = 0; i < npes; i++) {
    s2wSendCnts[i] = 0;
  }//end for i

  for(unsigned int i = 0; i < Wfgt.size(); i++) {
    if(s2wPart[i] != rank) {
      for(unsigned int j = 0; j < Ndofs; j++) {
        s2wSendFgtVals[ s2wSendDisps[s2wPart[i]] + s2wSendCnts[s2wPart[i]] + j ] = Wfgt[i][j];
      }//end for j
      s2wSendCnts[s2wPart[i]] += Ndofs;
    }
  }//end for i

  std::vector<double> s2wRecvFgtVals(s2wRecvDisps[npes - 1] + s2wRecvCnts[npes - 1]);

  MPI_Alltoallv( (&(*(s2wSendFgtVals.begin()))), (&(*(s2wSendCnts.begin()))), (&(*(s2wSendDisps.begin()))), MPI_DOUBLE, 
      (&(*(s2wRecvFgtVals.begin()))), (&(*(s2wRecvCnts.begin()))), (&(*(s2wRecvDisps.begin()))), MPI_DOUBLE, comm );

  for(unsigned int i = 0; i < s2wRecvFgtIds.size(); i++) {
    unsigned int fgtId = s2wRecvFgtIds[i];
    unsigned int fgtzid = (fgtId/(Ne*Ne));
    unsigned int fgtyid = ((fgtId%(Ne*Ne))/Ne);
    unsigned int fgtxid = ((fgtId%(Ne*Ne))%Ne);

    unsigned int boxId = ( ((fgtzid - zs)*nx*ny) + ((fgtyid - ys)*nx) + (fgtxid - xs) );
    isFGTboxEmpty[boxId] = false;
    for(unsigned int j = 0; j < Ndofs; j++) {
      WgArr[fgtzid][fgtyid][fgtxid][j] += s2wRecvFgtVals[(i*Ndofs) + j];
    }//end for j
  }//end for i

  PetscLogEventEnd(s2wCommEvent, 0, 0, 0, 0);

  if(!rank) {
    std::cout<<"Finished S2Wcomm"<<std::endl;
  }

  //W2D
  PetscLogEventBegin(w2dEvent, 0, 0, 0, 0);

  std::vector<unsigned int> requiredFgtIds; 

  for(unsigned int i = 0; i < numLocalDirectOcts; i++) {
    unsigned int lev = directTree[i].getLevel();
    double hCurrOct = hOctFac*static_cast<double>(1u << (maxDepth - lev));

    double ptGridOff = 0.1*hCurrOct;
    double ptGridH = 0.8*hCurrOct/(static_cast<double>(ptGridSizeWithinBox) - 1.0);

    //Anchor of the octant
    unsigned int anchX = directTree[i].getX();
    unsigned int anchY = directTree[i].getY();
    unsigned int anchZ = directTree[i].getZ();

    double aOx =  hOctFac*(static_cast<double>(anchX));
    double aOy =  hOctFac*(static_cast<double>(anchY));
    double aOz =  hOctFac*(static_cast<double>(anchZ));

    for(int j3 = 0; j3 < ptGridSizeWithinBox; j3++) {
      double pz = aOz + ptGridOff + (ptGridH*(static_cast<double>(j3)));

      int minZid = static_cast<int>(ceil(pz/hRg)) - K - 1;
      int maxZid = static_cast<int>(floor(pz/hRg)) + K + 1;

      if(minZid < 0) {
        minZid = 0;
      }

      if(maxZid > Ne) {
        maxZid = Ne;
      }

      for(int j2 = 0; j2 < ptGridSizeWithinBox; j2++) {
        double py = aOy + ptGridOff + (ptGridH*(static_cast<double>(j2)));

        int minYid = static_cast<int>(ceil(py/hRg)) - K - 1;
        int maxYid = static_cast<int>(floor(py/hRg)) + K + 1;

        if(minYid < 0) {
          minYid = 0;
        }

        if(maxYid > Ne) {
          maxYid = Ne;
        }

        for(int j1 = 0; j1 < ptGridSizeWithinBox; j1++) {
          double px = aOx + ptGridOff + (ptGridH*(static_cast<double>(j1)));

          int minXid = static_cast<int>(ceil(px/hRg)) - K - 1;
          int maxXid = static_cast<int>(floor(px/hRg)) + K + 1;

          if(minXid < 0) {
            minXid = 0;
          }

          if(maxXid > Ne) {
            maxXid = Ne;
          }

          for(int zid = minZid; zid < maxZid; zid++) {
            for(int yid = minYid; yid < maxYid; yid++) {
              for(int xid = minXid; xid < maxXid; xid++) {
                unsigned int fgtId = ( (zid*Ne*Ne) + (yid*Ne) + xid );

                unsigned int foundIdx;
                bool foundIt = seq::maxLowerBound<unsigned int>(requiredFgtIds, fgtId, foundIdx, 0, 0);

                if(foundIt) {
                  assert( requiredFgtIds[foundIdx] <= fgtId );
                  if( (foundIdx + 1) < (requiredFgtIds.size()) ) {
                    assert( requiredFgtIds[foundIdx + 1] > fgtId );
                  }

                  if( requiredFgtIds[foundIdx] != fgtId ) {
                    requiredFgtIds.insert( (requiredFgtIds.begin() + foundIdx + 1), fgtId );
                  }

                  assert( (foundIdx + 1) < (requiredFgtIds.size()) );
                  assert( requiredFgtIds[foundIdx + 1] == fgtId );
                } else {
                  if( !(requiredFgtIds.empty()) ) {
                    assert( requiredFgtIds[0] > fgtId );
                  }

                  requiredFgtIds.insert( requiredFgtIds.begin(), fgtId );

                  assert( requiredFgtIds[0] == fgtId );
                }

              }//end for xid
            }//end for yid
          }//end for zid

        }//end for j1
      }//end for j2
    }//end for j3

  }//end for i

  std::vector<int> w2dSendCnts(npes); 
  for(int i = 0; i < npes; i++) {
    w2dSendCnts[i] = 0;
  }//end for i

  std::vector<int> w2dPart(requiredFgtIds.size());
  for(unsigned int i = 0; i < requiredFgtIds.size(); i++) {
    unsigned int fgtId = requiredFgtIds[i];
    unsigned int fgtzid = (fgtId/(Ne*Ne));
    unsigned int fgtyid = ((fgtId%(Ne*Ne))/Ne);
    unsigned int fgtxid = ((fgtId%(Ne*Ne))%Ne);

    unsigned int xRes, yRes, zRes;
    seq::maxLowerBound<unsigned int>(scanLx, fgtxid, xRes, 0, 0);
    seq::maxLowerBound<unsigned int>(scanLy, fgtyid, yRes, 0, 0);
    seq::maxLowerBound<unsigned int>(scanLz, fgtzid, zRes, 0, 0);

    //Processor that owns the FGT box
    w2dPart[i] = (((zRes*npy) + yRes)*npx) + xRes;

    if(w2dPart[i] != rank) {
      w2dSendCnts[w2dPart[i]]++;
    }
  }//end for i

  std::vector<int> w2dRecvCnts(npes); 

  MPI_Alltoall( (&(*(w2dSendCnts.begin()))), 1, MPI_INT,
      (&(*(w2dRecvCnts.begin()))), 1, MPI_INT, comm );

  std::vector<int> w2dSendDisps(npes);
  std::vector<int> w2dRecvDisps(npes);
  w2dSendDisps[0] = 0;
  w2dRecvDisps[0] = 0;
  for(int i = 1; i < npes; i++) {
    w2dSendDisps[i] = w2dSendDisps[i - 1] + w2dSendCnts[i - 1];
    w2dRecvDisps[i] = w2dRecvDisps[i - 1] + w2dRecvCnts[i - 1];
  }//end for i

  std::vector<unsigned int> w2dSendFgtIds(w2dSendDisps[npes - 1] + w2dSendCnts[npes - 1]);

  for(int i = 0; i < npes; i++) {
    w2dSendCnts[i] = 0;
  }//end for i

  std::vector<unsigned int> w2dCommMap(requiredFgtIds.size());
  for(unsigned int i = 0; i < requiredFgtIds.size(); i++) {
    if(w2dPart[i] != rank) {
      w2dSendFgtIds[ w2dSendDisps[w2dPart[i]] + w2dSendCnts[w2dPart[i]] ] = requiredFgtIds[i];
      w2dCommMap[i] = ( w2dSendDisps[w2dPart[i]] + w2dSendCnts[w2dPart[i]] );
      w2dSendCnts[w2dPart[i]]++;
    }
  }//end for i

  std::vector<unsigned int> w2dRecvFgtIds(w2dRecvDisps[npes - 1] + w2dRecvCnts[npes - 1]);

  MPI_Alltoallv( (&(*(w2dSendFgtIds.begin()))), (&(*(w2dSendCnts.begin()))), (&(*(w2dSendDisps.begin()))), MPI_UNSIGNED, 
      (&(*(w2dRecvFgtIds.begin()))), (&(*(w2dRecvCnts.begin()))), (&(*(w2dRecvDisps.begin()))), MPI_UNSIGNED, comm );

  for(unsigned int i = 0; i < npes; i++) {
    w2dSendCnts[i] *= Ndofs;
    w2dSendDisps[i] *= Ndofs;
    w2dRecvCnts[i] *= Ndofs;
    w2dRecvDisps[i] *= Ndofs;
  }//end for i

  std::vector<double> w2dSendFgtVals((Ndofs*(w2dRecvFgtIds.size())));

  for(unsigned int i = 0; i < w2dRecvFgtIds.size(); i++) {
    unsigned int fgtId = w2dRecvFgtIds[i];
    unsigned int fgtzid = (fgtId/(Ne*Ne));
    unsigned int fgtyid = ((fgtId%(Ne*Ne))/Ne);
    unsigned int fgtxid = ((fgtId%(Ne*Ne))%Ne);

    for(unsigned int j = 0; j < Ndofs; j++) {
      w2dSendFgtVals[ (Ndofs*i) + j ] = WgArr[fgtzid][fgtyid][fgtxid][j];
    }//end for j
  }//end for i

  std::vector<double> w2dRecvFgtVals((Ndofs*(w2dSendFgtIds.size())));

  //Reverse communication
  MPI_Alltoallv( (&(*(w2dSendFgtVals.begin()))), (&(*(w2dRecvCnts.begin()))), (&(*(w2dRecvDisps.begin()))), MPI_DOUBLE,
      (&(*(w2dRecvFgtVals.begin()))), (&(*(w2dSendCnts.begin()))), (&(*(w2dSendDisps.begin()))), MPI_DOUBLE, comm );

  const double C0 = ( pow((0.5/sqrt(__PI__)), 3.0)*
      pow((static_cast<double>(L)/static_cast<double>(P)), 3.0) );

  std::vector<std::vector<double> > directResults(numLocalDirectOcts);

  for(unsigned int i = 0; i < numLocalDirectOcts; i++) {
    unsigned int lev = directTree[i].getLevel();
    double hCurrOct = hOctFac*static_cast<double>(1u << (maxDepth - lev));

    double ptGridOff = 0.1*hCurrOct;
    double ptGridH = 0.8*hCurrOct/(static_cast<double>(ptGridSizeWithinBox) - 1.0);

    //Anchor of the octant
    unsigned int anchX = directTree[i].getX();
    unsigned int anchY = directTree[i].getY();
    unsigned int anchZ = directTree[i].getZ();

    double aOx =  hOctFac*(static_cast<double>(anchX));
    double aOy =  hOctFac*(static_cast<double>(anchY));
    double aOz =  hOctFac*(static_cast<double>(anchZ));

    directResults[i].resize(ptGridSizeWithinBox*ptGridSizeWithinBox*ptGridSizeWithinBox);

    for(int j3 = 0, pt = 0; j3 < ptGridSizeWithinBox; j3++) {
      double pz = aOz + ptGridOff + (ptGridH*(static_cast<double>(j3)));

      int minZid = static_cast<int>(ceil(pz/hRg)) - K - 1;
      int maxZid = static_cast<int>(floor(pz/hRg)) + K + 1;

      if(minZid < 0) {
        minZid = 0;
      }

      if(maxZid > Ne) {
        maxZid = Ne;
      }

      for(int j2 = 0; j2 < ptGridSizeWithinBox; j2++) {
        double py = aOy + ptGridOff + (ptGridH*(static_cast<double>(j2)));

        int minYid = static_cast<int>(ceil(py/hRg)) - K - 1;
        int maxYid = static_cast<int>(floor(py/hRg)) + K + 1;

        if(minYid < 0) {
          minYid = 0;
        }

        if(maxYid > Ne) {
          maxYid = Ne;
        }

        for(int j1 = 0; j1 < ptGridSizeWithinBox; j1++, pt++) {
          double px = aOx + ptGridOff + (ptGridH*(static_cast<double>(j1)));

          int minXid = static_cast<int>(ceil(px/hRg)) - K - 1;
          int maxXid = static_cast<int>(floor(px/hRg)) + K + 1;

          if(minXid < 0) {
            minXid = 0;
          }

          if(maxXid > Ne) {
            maxXid = Ne;
          }

          directResults[i][pt] = 0.0;

          for(int zid = minZid; zid < maxZid; zid++) {
            for(int yid = minYid; yid < maxYid; yid++) {
              for(int xid = minXid; xid < maxXid; xid++) {
                unsigned int fgtId = ( (zid*Ne*Ne) + (yid*Ne) + xid );

                //Center of the FGT box
                double cx = hRg*(0.5 + static_cast<double>(xid));
                double cy = hRg*(0.5 + static_cast<double>(yid));
                double cz = hRg*(0.5 + static_cast<double>(zid));

                unsigned int foundIdx;
                bool foundIt = seq::BinarySearch<unsigned int>( (&(*(requiredFgtIds.begin()))),
                    requiredFgtIds.size(), fgtId, &foundIdx);
                assert(foundIt);
                assert(requiredFgtIds[foundIdx] == fgtId);

                double sum = 0.0;
                if(w2dPart[foundIdx] == rank) {
                  for(int k3 = -P, di = 0; k3 < P; k3++) {
                    for(int k2 = -P; k2 < P; k2++) {
                      for(int k1 = -P; k1 < P; k1++, di++) {
                        double factor = exp(ReExpZfactor*static_cast<double>( (k1*k1) + (k2*k2) + (k3*k3) ));

                        double theta = ImExpZfactor*( (static_cast<double>(k1)*(px - cx)) +
                            (static_cast<double>(k2)*(py - cy)) + (static_cast<double>(k3)*(pz - cz)) );

                        double a = WgArr[zid][yid][xid][2*di];
                        double b = WgArr[zid][yid][xid][(2*di) + 1];
                        double c = cos(theta);
                        double d = sin(theta);

                        sum += (factor*( (a*c) - (b*d) ));
                      }//end for k1
                    }//end for k2
                  }//end for k3
                } else {
                  for(int k3 = -P, di = 0; k3 < P; k3++) {
                    for(int k2 = -P; k2 < P; k2++) {
                      for(int k1 = -P; k1 < P; k1++, di++) {
                        double factor = exp(ReExpZfactor*static_cast<double>( (k1*k1) + (k2*k2) + (k3*k3) ));

                        double theta = ImExpZfactor*( (static_cast<double>(k1)*(px - cx)) +
                            (static_cast<double>(k2)*(py - cy)) + (static_cast<double>(k3)*(pz - cz)) );

                        double a = w2dRecvFgtVals[ (Ndofs*w2dCommMap[foundIdx]) + (2*di) ];
                        double b = w2dRecvFgtVals[ (Ndofs*w2dCommMap[foundIdx]) + (2*di) + 1 ];
                        double c = cos(theta);
                        double d = sin(theta);

                        sum += (factor*( (a*c) - (b*d) ));
                      }//end for k1
                    }//end for k2
                  }//end for k3
                }

                directResults[i][pt] += (C0*sum);

              }//end for xid
            }//end for yid
          }//end for zid

        }//end for j1
      }//end for j2
    }//end for j3

  }//end for i

  DAVecRestoreArrayDOF(da, Wglobal, &WgArr);

  PetscLogEventEnd(w2dEvent, 0, 0, 0, 0);

  if(!rank) {
    std::cout<<"Finished W2D"<<std::endl;
  }

  //D2D
  PetscLogEventBegin(d2dEvent, 0, 0, 0, 0);

  ot::TreeNode rootOct(3, maxDepth);
  ot::TreeNode minDirectOct = rootOct;
  if( !(directTree.empty()) ) {
    minDirectOct = directTree[0];
  }

  std::vector<ot::TreeNode> directTreePartInfo(npes);

  MPI_Allgather( &minDirectOct, 1, par::Mpi_datatype<ot::TreeNode>::value(),
      (&(*(directTreePartInfo.begin()))), 1, par::Mpi_datatype<ot::TreeNode>::value(), comm );

  std::vector<ot::TreeNode> directTreeMins;
  for(unsigned int i = 0; i < npes; i++) {
    if(directTreePartInfo[i] != rootOct) {
      directTreePartInfo[i].setWeight(i);
      directTreeMins.push_back(directTreePartInfo[i]);
    }
  }//end for i
  directTreePartInfo.clear();

  std::vector<std::vector<double> > d2dSendPts(npes);

  for(unsigned int i = 0; i < numLocalDirectOcts; i++) {
    unsigned int lev = directTree[i].getLevel();
    double hCurrOct = hOctFac*static_cast<double>(1u << (maxDepth - lev));

    double ptGridOff = 0.1*hCurrOct;
    double ptGridH = 0.8*hCurrOct/(static_cast<double>(ptGridSizeWithinBox) - 1.0);

    //Anchor of the octant
    unsigned int anchX = directTree[i].getX();
    unsigned int anchY = directTree[i].getY();
    unsigned int anchZ = directTree[i].getZ();

    double aOx =  hOctFac*(static_cast<double>(anchX));
    double aOy =  hOctFac*(static_cast<double>(anchY));
    double aOz =  hOctFac*(static_cast<double>(anchZ));

    for(int j3 = 0; j3 < ptGridSizeWithinBox; j3++) {
      double pz = aOz + ptGridOff + (ptGridH*(static_cast<double>(j3)));

      double minZ = pz - (static_cast<double>(K)*hRg);
      double maxZ = pz + (static_cast<double>(K)*hRg);

      if(minZ < 0.0) {
        minZ = 0.0;
      }

      if(maxZ > 1.0) {
        maxZ = 1.0;
      }

      unsigned int uiMinZ = static_cast<unsigned int>(floor(minZ*static_cast<double>(1u << maxDepth)));
      unsigned int uiMaxZ = static_cast<unsigned int>(ceil(maxZ*static_cast<double>(1u << maxDepth)));

      for(int j2 = 0; j2 < ptGridSizeWithinBox; j2++) {
        double py = aOy + ptGridOff + (ptGridH*(static_cast<double>(j2)));

        double minY = py - (static_cast<double>(K)*hRg);
        double maxY = py + (static_cast<double>(K)*hRg);

        if(minY < 0.0) {
          minY = 0.0;
        }

        if(maxY > 1.0) {
          maxY = 1.0;
        }

        unsigned int uiMinY = static_cast<unsigned int>(floor(minY*static_cast<double>(1u << maxDepth)));
        unsigned int uiMaxY = static_cast<unsigned int>(ceil(maxY*static_cast<double>(1u << maxDepth)));

        for(int j1 = 0; j1 < ptGridSizeWithinBox; j1++) {
          double px = aOx + ptGridOff + (ptGridH*(static_cast<double>(j1)));

          double minX = px - (static_cast<double>(K)*hRg);
          double maxX = px + (static_cast<double>(K)*hRg);

          if(minX < 0.0) {
            minX = 0.0;
          }

          if(maxX > 1.0) {
            maxX = 1.0;
          }

          unsigned int uiMinX = static_cast<unsigned int>(floor(minX*static_cast<double>(1u << maxDepth)));
          unsigned int uiMaxX = static_cast<unsigned int>(ceil(maxX*static_cast<double>(1u << maxDepth)));

          ot::TreeNode minPt(uiMinX, uiMinY, uiMinZ, maxDepth, 3, maxDepth);
          ot::TreeNode maxPt( (uiMaxX - 1), (uiMaxY - 1), (uiMaxZ - 1), maxDepth, 3, maxDepth);

          unsigned int minPtIdx;
          bool foundMin = seq::maxLowerBound<ot::TreeNode>(directTreeMins, minPt, minPtIdx, 0, 0);

          if(!foundMin) {
            minPtIdx = 0;
          }

          unsigned int maxPtIdx;
          bool foundMax = seq::maxLowerBound<ot::TreeNode>(directTreeMins, maxPt, maxPtIdx, 0, 0);

          //maxPt > currPt and currPt > currOct and currOct is a direct octant
          assert(foundMax);

          for(unsigned int idx = minPtIdx; idx <= maxPtIdx; idx++) {
            unsigned procId = directTreeMins[idx].getWeight();

            d2dSendPts[procId].push_back(px);
            d2dSendPts[procId].push_back(py);
            d2dSendPts[procId].push_back(pz);

            //Use drand48() instead if you want
            d2dSendPts[procId].push_back(fMag);
          }//end idx

        }//end for j1
      }//end for j2
    }//end for j3

  }//end for i

  std::vector<int> d2dSendCnts(npes);
  for(int i = 0; i < npes; i++) {
    d2dSendCnts[i] = d2dSendPts[i].size();
  }//end for i

  std::vector<int> d2dRecvCnts(npes);

  MPI_Alltoall( (&(*(d2dSendCnts.begin()))), 1, MPI_INT,
      (&(*(d2dRecvCnts.begin()))), 1, MPI_INT, comm );

  std::vector<int> d2dSendDisps(npes);
  std::vector<int> d2dRecvDisps(npes);
  d2dSendDisps[0] = 0;
  d2dRecvDisps[0] = 0;
  for(int i = 1; i < npes; i++) {
    d2dSendDisps[i] = d2dSendDisps[i - 1] + d2dSendCnts[i - 1];
    d2dRecvDisps[i] = d2dRecvDisps[i - 1] + d2dRecvCnts[i - 1];
  }//end for i

  std::vector<double> d2dSendVals(d2dSendDisps[npes - 1] + d2dSendCnts[npes - 1]);
  for(int i = 0; i < npes; i++) {
    for(int j = 0; j < d2dSendCnts[i]; j++) {
      d2dSendVals[d2dSendDisps[i] + j] = d2dSendPts[i][j];
    }//end for j
  }//end for i
  d2dSendPts.clear();

  std::vector<double> d2dRecvVals(d2dRecvDisps[npes - 1] + d2dRecvCnts[npes - 1]);

  MPI_Alltoallv( (&(*(d2dSendVals.begin()))), (&(*(d2dSendCnts.begin()))), (&(*(d2dSendDisps.begin()))), MPI_DOUBLE, 
      (&(*(d2dRecvVals.begin()))), (&(*(d2dRecvCnts.begin()))), (&(*(d2dRecvDisps.begin()))), MPI_DOUBLE, comm );

  d2dSendVals.clear();
  d2dSendCnts.clear();
  d2dSendDisps.clear();

  d2dRecvCnts.clear();
  d2dRecvDisps.clear();

  const unsigned int numRecvSourcePts = d2dRecvVals.size()/4;

  const double IlistWidthSquare = (hRg*hRg*static_cast<double>(K*K));

  for(unsigned int i = 0; i < numRecvSourcePts; i++) {
    //Source point
    double sx = d2dRecvVals[4*i];
    double sy = d2dRecvVals[(4*i) + 1];
    double sz = d2dRecvVals[(4*i) + 2];
    double sf = d2dRecvVals[(4*i) + 3];

    //Ilist of source point
    double IminX = sx - (static_cast<double>(K)*hRg);
    double ImaxX = sx + (static_cast<double>(K)*hRg);

    double IminY = sy - (static_cast<double>(K)*hRg);
    double ImaxY = sy + (static_cast<double>(K)*hRg);

    double IminZ = sz - (static_cast<double>(K)*hRg);
    double ImaxZ = sz + (static_cast<double>(K)*hRg);

    if(IminX < 0.0) {
      IminX = 0.0;
    }

    if(ImaxX > 1.0) {
      ImaxX = 1.0;
    }

    if(IminY < 0.0) {
      IminY = 0.0;
    }

    if(ImaxY > 1.0) {
      ImaxY = 1.0;
    }

    if(IminZ < 0.0) {
      IminZ = 0.0;
    }

    if(ImaxZ > 1.0) {
      ImaxZ = 1.0;
    }

    unsigned int uiMinX = static_cast<unsigned int>(floor(IminX*static_cast<double>(1u << maxDepth)));
    unsigned int uiMaxX = static_cast<unsigned int>(ceil(ImaxX*static_cast<double>(1u << maxDepth)));

    unsigned int uiMinY = static_cast<unsigned int>(floor(IminY*static_cast<double>(1u << maxDepth)));
    unsigned int uiMaxY = static_cast<unsigned int>(ceil(ImaxY*static_cast<double>(1u << maxDepth)));

    unsigned int uiMinZ = static_cast<unsigned int>(floor(IminZ*static_cast<double>(1u << maxDepth)));
    unsigned int uiMaxZ = static_cast<unsigned int>(ceil(ImaxZ*static_cast<double>(1u << maxDepth)));

    ot::TreeNode minPt(uiMinX, uiMinY, uiMinZ, maxDepth, 3, maxDepth);
    ot::TreeNode maxPt( (uiMaxX - 1), (uiMaxY - 1), (uiMaxZ - 1), maxDepth, 3, maxDepth);

    unsigned int minPtIdx;
    bool foundMin = seq::maxLowerBound<ot::TreeNode>(directTree, minPt, minPtIdx, 0, 0);

    if(!foundMin) {
      minPtIdx = 0;
    }

    unsigned int maxPtIdx;
    bool foundMax = seq::maxLowerBound<ot::TreeNode>(directTree, maxPt, maxPtIdx, 0, 0);

    //This source point was sent only to those procs
    //whose directTreeMin <= maxPt
    assert(foundMax);

    for(unsigned int idx = minPtIdx; idx <= maxPtIdx; idx++) {
      unsigned int lev = directTree[idx].getLevel();
      double hCurrOct = hOctFac*static_cast<double>(1u << (maxDepth - lev));

      double ptGridOff = 0.1*hCurrOct;
      double ptGridH = 0.8*hCurrOct/(static_cast<double>(ptGridSizeWithinBox) - 1.0);

      //Anchor of the octant
      unsigned int anchX = directTree[idx].getX();
      unsigned int anchY = directTree[idx].getY();
      unsigned int anchZ = directTree[idx].getZ();

      double aOx =  hOctFac*(static_cast<double>(anchX));
      double aOy =  hOctFac*(static_cast<double>(anchY));
      double aOz =  hOctFac*(static_cast<double>(anchZ));

      unsigned int stXid, stYid, stZid;

      if( IminX >= (aOx + ptGridOff) ) {
        stXid = static_cast<unsigned int>(ceil((IminX - (aOx + ptGridOff))/ptGridH));
      } else {
        stXid = 0; 
      }

      if( IminY >= (aOy + ptGridOff) ) {
        stYid = static_cast<unsigned int>(ceil((IminY - (aOy + ptGridOff))/ptGridH));
      } else {
        stYid = 0; 
      }

      if( IminZ >= (aOz + ptGridOff) ) {
        stZid = static_cast<unsigned int>(ceil((IminZ - (aOz + ptGridOff))/ptGridH));
      } else {
        stZid = 0; 
      }

      unsigned int endXid, endYid, endZid;

      if( ImaxX >= (aOx + ptGridOff) ) {
        endXid = static_cast<unsigned int>(ceil((ImaxX - (aOx + ptGridOff))/ptGridH));
      } else {
        endXid = 0;
      }

      if( ImaxY >= (aOy + ptGridOff) ) {
        endYid = static_cast<unsigned int>(ceil((ImaxY - (aOy + ptGridOff))/ptGridH));
      } else {
        endYid = 0;
      }

      if( ImaxZ >= (aOz + ptGridOff) ) {
        endZid = static_cast<unsigned int>(ceil((ImaxZ - (aOz + ptGridOff))/ptGridH));
      } else {
        endZid = 0;
      }

      if(endXid > ptGridSizeWithinBox) {
        endXid = ptGridSizeWithinBox;
      }

      if(endYid > ptGridSizeWithinBox) {
        endYid = ptGridSizeWithinBox;
      }

      if(endZid > ptGridSizeWithinBox) {
        endZid = ptGridSizeWithinBox;
      }

      for(unsigned int zid = stZid; zid < endZid; zid++) {
        double tz = aOz + ptGridOff + (ptGridH*(static_cast<double>(zid)));

        double distZsqr = (tz - sz)*(tz - sz);

        for(unsigned int yid = stYid; yid < endYid; yid++) {
          double ty = aOy + ptGridOff + (ptGridH*(static_cast<double>(yid)));

          double distYsqr = (ty - sy)*(ty - sy);

          for(unsigned int xid = stXid; xid < endXid; xid++) {
            double tx = aOx + ptGridOff + (ptGridH*(static_cast<double>(xid)));

            double distXsqr = (tx - sx)*(tx - sx);

            unsigned int ptId = ( (zid*ptGridSizeWithinBox*ptGridSizeWithinBox) + (yid*ptGridSizeWithinBox) + xid );

            double distSqr = (distXsqr + distYsqr + distZsqr);

            if( distSqr < IlistWidthSquare ) {
              directResults[idx][ptId] += (sf*exp(-distSqr/delta));
            }
          }//end for xid
        }//end for yid
      }//end for zid

    }//end for idx

  }//end for i

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

  // directW2L(WlArr, WgArr, xs, ys, zs, nx, ny, nz, Ne, h, K, P, ImExpZfactor);
  sweepW2L(WlArr, WgArr, xs, ys, zs, nx, ny, nz, Ne, hRg, K, P, ImExpZfactor);

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

  for(unsigned int i = 0; i < s2wRecvFgtIds.size(); i++) {
    unsigned int fgtId = s2wRecvFgtIds[i];
    unsigned int fgtzid = (fgtId/(Ne*Ne));
    unsigned int fgtyid = ((fgtId%(Ne*Ne))/Ne);
    unsigned int fgtxid = ((fgtId%(Ne*Ne))%Ne);

    for(unsigned int j = 0; j < Ndofs; j++) {
      s2wRecvFgtVals[(i*Ndofs) + j] = WgArr[fgtzid][fgtyid][fgtxid][j];
    }//end for j
  }//end for i

  //Reverse of S2W comm
  MPI_Alltoallv( (&(*(s2wRecvFgtVals.begin()))), (&(*(s2wRecvCnts.begin()))), (&(*(s2wRecvDisps.begin()))), MPI_DOUBLE,
      (&(*(s2wSendFgtVals.begin()))), (&(*(s2wSendCnts.begin()))), (&(*(s2wSendDisps.begin()))), MPI_DOUBLE, comm );

  for(int i = 0; i < npes; i++) {
    s2wSendCnts[i] = 0;
  }//end for i

  for(unsigned int i = 0; i < Wfgt.size(); i++) {
    if(s2wPart[i] == rank) {
      unsigned int fgtId = uniqueOct2fgtIdmap[i];
      unsigned int fgtzid = (fgtId/(Ne*Ne));
      unsigned int fgtyid = ((fgtId%(Ne*Ne))/Ne);
      unsigned int fgtxid = ((fgtId%(Ne*Ne))%Ne);

      for(unsigned int j = 0; j < Ndofs; j++) {
        Wfgt[i][j] = WgArr[fgtzid][fgtyid][fgtxid][j];
      }//end for j
    } else {
      for(unsigned int j = 0; j < Ndofs; j++) {
        Wfgt[i][j] = s2wSendFgtVals[ s2wSendDisps[s2wPart[i]] + s2wSendCnts[s2wPart[i]] + j ];
      }//end for j
      s2wSendCnts[s2wPart[i]] += Ndofs;
    }
  }//end for i

  DAVecRestoreArrayDOF(da, Wglobal, &WgArr);

  PetscLogEventEnd(l2tCommEvent, 0, 0, 0, 0);

  if(!rank) {
    std::cout<<"Finished L2Tcomm"<<std::endl;
  }

  //L2T
  PetscLogEventBegin(l2tEvent, 0, 0, 0, 0);

  std::vector<std::vector<double> > expandResults(numLocalExpandOcts);

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

    unsigned int fgtIndex = oct2fgtIdmap[i];
    unsigned int fgtId = uniqueOct2fgtIdmap[fgtIndex];
    unsigned int fgtzid = (fgtId/(Ne*Ne));
    unsigned int fgtyid = ((fgtId%(Ne*Ne))/Ne);
    unsigned int fgtxid = ((fgtId%(Ne*Ne))%Ne);

    //Anchor of the FGT box
    double aFx = hRg*static_cast<double>(fgtxid);
    double aFy = hRg*static_cast<double>(fgtyid);
    double aFz = hRg*static_cast<double>(fgtzid);

    //Center of the FGT box
    double halfH = (0.5*hRg);
    double cx =  aFx + halfH;
    double cy =  aFy + halfH;
    double cz =  aFz + halfH;

    //Tensor Product Acceleration

    //Stage - 1

    tmp1R.resize(ptGridSizeWithinBox);
    tmp1C.resize(ptGridSizeWithinBox);
    for(unsigned int k1 = 0; k1 < ptGridSizeWithinBox; k1++) {
      tmp1R[k1].resize(2*P);
      tmp1C[k1].resize(2*P);

      double px = aOx + ptGridOff + (ptGridH*(static_cast<double>(k1)));

      for(int j3 = -P, di = 0; j3 < P; j3++) {
        int shiftJ3 = (j3 + P);

        tmp1R[k1][shiftJ3].resize(2*P);
        tmp1C[k1][shiftJ3].resize(2*P);

        for(int j2 = -P; j2 < P; j2++) {
          int shiftJ2 = (j2 + P);

          double rSum = 0.0;
          double cSum = 0.0;

          for(int j1 = -P; j1 < P; j1++, di++) {
            double theta = ImExpZfactor*(static_cast<double>(j1)*(px - cx)) ;

            double a = Wfgt[fgtIndex][2*di];
            double b = Wfgt[fgtIndex][(2*di) + 1];
            double c = cos(theta);
            double d = sin(theta);
            double factor = exp(ReExpZfactor*static_cast<double>( (j1*j1) + (j2*j2) + (j3*j3) ));

            rSum += (factor*( (a*c) - (b*d) ));
            cSum += (factor*( (a*d) + (b*c) ));
          }//end for j1

          tmp1R[k1][shiftJ3][shiftJ2] = (C0*rSum);
          tmp1C[k1][shiftJ3][shiftJ2] = (C0*cSum);
        }//end for j2
      }//end for j3
    }//end for k1

    //Stage - 2

    tmp2R.resize(ptGridSizeWithinBox);
    tmp2C.resize(ptGridSizeWithinBox);
    for(unsigned int k2 = 0; k2 < ptGridSizeWithinBox; k2++) {
      tmp2R[k2].resize(ptGridSizeWithinBox);
      tmp2C[k2].resize(ptGridSizeWithinBox);

      double py = aOy + ptGridOff + (ptGridH*(static_cast<double>(k2)));

      for(unsigned int k1 = 0; k1 < ptGridSizeWithinBox; k1++) {
        tmp2R[k2][k1].resize(2*P);
        tmp2C[k2][k1].resize(2*P);

        for(int j3 = -P; j3 < P; j3++) {
          int shiftJ3 = (j3 + P);

          tmp2R[k2][k1][shiftJ3] = 0.0;
          tmp2C[k2][k1][shiftJ3] = 0.0;

          for(int j2 = -P; j2 < P; j2++) {
            int shiftJ2 = (j2 + P);

            double theta = ImExpZfactor*(static_cast<double>(j2)*(py - cy)) ;

            double a = tmp1R[k1][shiftJ3][shiftJ2];
            double b = tmp1C[k1][shiftJ3][shiftJ2];
            double c = cos(theta);
            double d = sin(theta);

            tmp2R[k2][k1][shiftJ3] += ( (a*c) - (b*d) );
            tmp2C[k2][k1][shiftJ3] += ( (a*d) + (b*c) );
          }//end for j2
        }//end for j3
      }//end for k1
    }//end for k2

    //Stage - 3

    expandResults[i].resize(ptGridSizeWithinBox*ptGridSizeWithinBox*ptGridSizeWithinBox);

    for(unsigned int k3 = 0, pt = 0; k3 < ptGridSizeWithinBox; k3++) {
      double pz = aOz + ptGridOff + (ptGridH*(static_cast<double>(k3)));

      for(unsigned int k2 = 0; k2 < ptGridSizeWithinBox; k2++) {
        for(unsigned int k1 = 0; k1 < ptGridSizeWithinBox; k1++, pt++) {

          expandResults[i][pt] = 0.0;

          for(int j3 = -P; j3 < P; j3++) {
            int shiftJ3 = (j3 + P);

            double theta = ImExpZfactor*(static_cast<double>(j3)*(pz - cz)) ;

            double a = tmp2R[k2][k1][shiftJ3];
            double b = tmp2C[k2][k1][shiftJ3];
            double c = cos(theta);
            double d = sin(theta);

            expandResults[i][pt] += ( (a*c) - (b*d) );
          }//end for j3
        }//end for k1
      }//end for k2
    }//end for k3

  }//end for i

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

void directW2L(PetscScalar**** WlArr, PetscScalar**** WgArr, 
    int xs, int ys, int zs, int nx, int ny, int nz, 
    int Ne, double h, const int StencilWidth, const int P, const double ImExpZfactor) {
  //Loop over local boxes and their Interaction lists and do a direct translation

  for(PetscInt zi = 0; zi < nz; zi++) {
    for(PetscInt yi = 0; yi < ny; yi++) {
      for(PetscInt xi = 0; xi < nx; xi++) {
        int xx = xi + xs;
        int yy = yi + ys;
        int zz = zi + zs;

        //Bounds for Ilist of box B
        int Ixs = xx - StencilWidth;
        int Ixe = xx + StencilWidth;

        int Iys = yy - StencilWidth;
        int Iye = yy + StencilWidth;

        int Izs = zz - StencilWidth;
        int Ize = zz + StencilWidth;

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

              for(int k3 = -P, di = 0; k3 < P; k3++) {
                for(int k2 = -P; k2 < P; k2++) {
                  for(int k1 = -P; k1 < P; k1++, di++) {

                    double theta = ImExpZfactor*h*( static_cast<double>(k1*(xx - xj) + k2*(yy - yj) + k3*(zz - zj) ) );
                    double ct = cos(theta);
                    double st = sin(theta);

                    WgArr[zz][yy][xx][2*di] += __COMP_MUL_RE(WlArr[zj][yj][xj][2*di], WlArr[zj][yj][xj][(2*di) + 1], ct, st);
                    WgArr[zz][yy][xx][(2*di) + 1] += __COMP_MUL_IM(WlArr[zj][yj][xj][2*di], WlArr[zj][yj][xj][(2*di) + 1], ct, st);

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
    const int P, const double ImExpZfactor) {

  // compute the first layer directly ...  
  /*
     directW2L(WlArr, WgArr, xs, ys, zs, nx, ny, 1, Ne, h, K, P, ImExpZfactor); // XY Plane
     directW2L(WlArr, WgArr, xs, ys+1, zs, 1, ny-1, nz, Ne, h, K, P, ImExpZfactor); // YZ Plane
     directW2L(WlArr, WgArr, xs+1, ys, zs+1, nx-1, 1, nz-1, Ne, h, K, P, ImExpZfactor); // ZX Plane 
     */
  directLayer(WlArr, WgArr, xs+1, ys, zs+1, nx-1, 1, nz-1, Ne, h, K, P, ImExpZfactor); // ZX Plane 

  // return;

  int num_layers = std::min(std::min(nx, ny), nz);

  double *fac = new double [7*2*8*P*P*P]; // 7* (2P)^3 complex terms ...
  double theta, ct, st;

  for(int k3 = -P, di = 0; k3 < P; k3++) {
    for(int k2 = -P; k2 < P; k2++) {
      for(int k1 = -P; k1 < P; k1++, di++) {
        // i,j,k
        theta = ImExpZfactor*h* ( (static_cast<double>(k1 + k2 + k3) ) );
        fac[14*di]     = cos(theta);
        fac[14*di + 1] = sin(theta);

        theta = ImExpZfactor*h* ( (static_cast<double>(k1) ) );
        fac[14*di + 2] = cos(theta);
        fac[14*di + 3] = sin(theta);

        theta = ImExpZfactor*h* ( (static_cast<double>(k2) ) );
        fac[14*di + 4] = cos(theta);
        fac[14*di + 5] = sin(theta);

        theta = ImExpZfactor*h* ( (static_cast<double>(k3) ) );
        fac[14*di + 6] = cos(theta);
        fac[14*di + 7] = sin(theta);

        theta = ImExpZfactor*h* ( (static_cast<double>(k2 + k3) ) );
        fac[14*di + 8] = cos(theta);
        fac[14*di + 9] = sin(theta);

        theta = ImExpZfactor*h* ( (static_cast<double>(k1 + k3) ) );
        fac[14*di + 10] = cos(theta);
        fac[14*di + 11] = sin(theta);

        theta = ImExpZfactor*h* ( (static_cast<double>(k1 + k2) ) );
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
                theta = ImExpZfactor*h* ( (static_cast<double>((K+1)*(k1 + k2 + k3)) ) );
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WlArr[k-K-1][j-K-1][i-K-1][2*di], WlArr[k-K-1][j-K-1][i-K-1][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WlArr[k-K-1][j-K-1][i-K-1][2*di], WlArr[k-K-1][j-K-1][i-K-1][2*di+1],  ct, st );
              }
              // corner 110
              if ( ( (k-K-1) >=0) && ((j+K) < Ne) && ((i+K) < Ne) ) {  
                theta = ImExpZfactor*h* ( (static_cast<double>((K+1)* k3 -K*(k1 + k2))) ) ;
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WlArr[k-K-1][j+K][i+K][2*di], WlArr[k-K-1][j+K][i+K][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WlArr[k-K-1][j+K][i+K][2*di], WlArr[k-K-1][j+K][i+K][2*di+1],  ct, st );
              }
              // corner 011
              if ( ( (i-K-1) >=0) && ((j+K) < Ne) && ((k+K) < Ne) ) {  
                theta = ImExpZfactor*h* ( (static_cast<double>((K+1)* k1 -K*(k2 + k3))) ) ;
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WlArr[k+K][j+K][i-K-1][2*di], WlArr[k+K][j+K][i-K-1][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WlArr[k+K][j+K][i-K-1][2*di], WlArr[k+K][j+K][i-K-1][2*di+1],  ct, st );
              }
              // corner 101
              if ( ( (j-K-1) >=0) && ((k+K) < Ne) && ((i+K) < Ne) ) {  
                theta = ImExpZfactor*h* ( (static_cast<double>((K+1)* k2 -K*(k1 + k3))) ) ;
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WlArr[k+K][j-K-1][i+K][2*di], WlArr[k+K][j-K-1][i+K][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WlArr[k+K][j-K-1][i+K][2*di], WlArr[k+K][j-K-1][i+K][2*di+1],  ct, st );
              }
              // corner 100
              if ( ( (k-K-1) >=0) && ((j-K-1) >= 0) && ((i+K) < Ne) ) {  
                theta = ImExpZfactor*h* ( (static_cast<double>((K+1)*(k2 + k3) -K*k1) ) );
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   -= __COMP_MUL_RE( WlArr[k-K-1][j-K-1][i+K][2*di], WlArr[k-K-1][j-K-1][i+K][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] -= __COMP_MUL_IM( WlArr[k-K-1][j-K-1][i+K][2*di], WlArr[k-K-1][j-K-1][i+K][2*di+1],  ct, st );
              }
              // corner 010
              if ( ( (k-K-1) >=0) && ((i-K-1) >= 0) && ((j+K) < Ne) ) {  
                theta = ImExpZfactor*h* ( (static_cast<double>((K+1)*(k1 + k3) - K*k2) ) );
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   -= __COMP_MUL_RE( WlArr[k-K-1][j+K][i-K-1][2*di], WlArr[k-K-1][j+K][i-K-1][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] -= __COMP_MUL_IM( WlArr[k-K-1][j+K][i-K-1][2*di], WlArr[k-K-1][j+K][i-K-1][2*di+1],  ct, st );
              }
              // corner 001
              if ( ( (i-K-1) >=0) && ((j-K-1) >= 0) && ((k+K) < Ne) ) {  
                theta = ImExpZfactor*h* ( (static_cast<double>((K+1)*(k1 + k2) - K*k3) ) );
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   -= __COMP_MUL_RE( WlArr[k+K][j-K-1][i-K-1][2*di], WlArr[k+K][j-K-1][i-K-1][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] -= __COMP_MUL_IM( WlArr[k+K][j-K-1][i-K-1][2*di], WlArr[k+K][j-K-1][i-K-1][2*di+1],  ct, st );
              }
              // corner 111
              if ( ( (j+K) < Ne) && ((k+K) < Ne) && ((i+K) < Ne) ) {  
                theta = ImExpZfactor*h* ( (static_cast<double>((-K)*(k1 + k2 + k3)) ) );
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
                theta = ImExpZfactor*h* ( (static_cast<double>((K+1)*(k1 + k2 + k3)) ) );
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WlArr[k-K-1][j-K-1][i-K-1][2*di], WlArr[k-K-1][j-K-1][i-K-1][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WlArr[k-K-1][j-K-1][i-K-1][2*di], WlArr[k-K-1][j-K-1][i-K-1][2*di+1],  ct, st );
              }
              // corner 110
              if ( ( (k-K-1) >=0) && ((j+K) < Ne) && ((i+K) < Ne) ) {  
                theta = ImExpZfactor*h* ( (static_cast<double>((K+1)* k3 -K*(k1 + k2))) ) ;
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WlArr[k-K-1][j+K][i+K][2*di], WlArr[k-K-1][j+K][i+K][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WlArr[k-K-1][j+K][i+K][2*di], WlArr[k-K-1][j+K][i+K][2*di+1],  ct, st );
              }
              // corner 011
              if ( ( (i-K-1) >=0) && ((j+K) < Ne) && ((k+K) < Ne) ) {  
                theta = ImExpZfactor*h* ( (static_cast<double>((K+1)* k1 -K*(k2 + k3))) ) ;
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WlArr[k+K][j+K][i-K-1][2*di], WlArr[k+K][j+K][i-K-1][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WlArr[k+K][j+K][i-K-1][2*di], WlArr[k+K][j+K][i-K-1][2*di+1],  ct, st );
              }
              // corner 101
              if ( ( (j-K-1) >=0) && ((k+K) < Ne) && ((i+K) < Ne) ) {  
                theta = ImExpZfactor*h* ( (static_cast<double>((K+1)* k2 -K*(k1 + k3))) ) ;
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WlArr[k+K][j-K-1][i+K][2*di], WlArr[k+K][j-K-1][i+K][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WlArr[k+K][j-K-1][i+K][2*di], WlArr[k+K][j-K-1][i+K][2*di+1],  ct, st );
              }
              // corner 100
              if ( ( (k-K-1) >=0) && ((j-K-1) >= 0) && ((i+K) < Ne) ) {  
                theta = ImExpZfactor*h* ( (static_cast<double>((K+1)*(k2 + k3) -K*k1) ) );
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   -= __COMP_MUL_RE( WlArr[k-K-1][j-K-1][i+K][2*di], WlArr[k-K-1][j-K-1][i+K][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] -= __COMP_MUL_IM( WlArr[k-K-1][j-K-1][i+K][2*di], WlArr[k-K-1][j-K-1][i+K][2*di+1],  ct, st );
              }
              // corner 010
              if ( ( (k-K-1) >=0) && ((i-K-1) >= 0) && ((j+K) < Ne) ) {  
                theta = ImExpZfactor*h* ( (static_cast<double>((K+1)*(k1 + k3) - K*k2) ) );
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   -= __COMP_MUL_RE( WlArr[k-K-1][j+K][i-K-1][2*di], WlArr[k-K-1][j+K][i-K-1][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] -= __COMP_MUL_IM( WlArr[k-K-1][j+K][i-K-1][2*di], WlArr[k-K-1][j+K][i-K-1][2*di+1],  ct, st );
              }
              // corner 001
              if ( ( (i-K-1) >=0) && ((j-K-1) >= 0) && ((k+K) < Ne) ) {  
                theta = ImExpZfactor*h* ( (static_cast<double>((K+1)*(k1 + k2) - K*k3) ) );
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   -= __COMP_MUL_RE( WlArr[k+K][j-K-1][i-K-1][2*di], WlArr[k+K][j-K-1][i-K-1][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] -= __COMP_MUL_IM( WlArr[k+K][j-K-1][i-K-1][2*di], WlArr[k+K][j-K-1][i-K-1][2*di+1],  ct, st );
              }
              // corner 111
              if ( ( (j+K) < Ne) && ((k+K) < Ne) && ((i+K) < Ne) ) {  
                theta = ImExpZfactor*h* ( (static_cast<double>((-K)*(k1 + k2 + k3)) ) );
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
                theta = ImExpZfactor*h* ( (static_cast<double>((K+1)*(k1 + k2 + k3)) ) );
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WlArr[k-K-1][j-K-1][i-K-1][2*di], WlArr[k-K-1][j-K-1][i-K-1][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WlArr[k-K-1][j-K-1][i-K-1][2*di], WlArr[k-K-1][j-K-1][i-K-1][2*di+1],  ct, st );
              }
              // corner 110
              if ( ( (k-K-1) >=0) && ((j+K) < Ne) && ((i+K) < Ne) ) {  
                theta = ImExpZfactor*h* ( (static_cast<double>((K+1)* k3 -K*(k1 + k2))) ) ;
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WlArr[k-K-1][j+K][i+K][2*di], WlArr[k-K-1][j+K][i+K][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WlArr[k-K-1][j+K][i+K][2*di], WlArr[k-K-1][j+K][i+K][2*di+1],  ct, st );
              }
              // corner 011
              if ( ( (i-K-1) >=0) && ((j+K) < Ne) && ((k+K) < Ne) ) {  
                theta = ImExpZfactor*h* ( (static_cast<double>((K+1)* k1 -K*(k2 + k3))) ) ;
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WlArr[k+K][j+K][i-K-1][2*di], WlArr[k+K][j+K][i-K-1][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WlArr[k+K][j+K][i-K-1][2*di], WlArr[k+K][j+K][i-K-1][2*di+1],  ct, st );
              }
              // corner 101
              if ( ( (j-K-1) >=0) && ((k+K) < Ne) && ((i+K) < Ne) ) {  
                theta = ImExpZfactor*h* ( (static_cast<double>((K+1)* k2 -K*(k1 + k3))) ) ;
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WlArr[k+K][j-K-1][i+K][2*di], WlArr[k+K][j-K-1][i+K][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WlArr[k+K][j-K-1][i+K][2*di], WlArr[k+K][j-K-1][i+K][2*di+1],  ct, st );
              }
              // corner 100
              if ( ( (k-K-1) >=0) && ((j-K-1) >= 0) && ((i+K) < Ne) ) {  
                theta = ImExpZfactor*h* ( (static_cast<double>((K+1)*(k2 + k3) -K*k1) ) );
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   -= __COMP_MUL_RE( WlArr[k-K-1][j-K-1][i+K][2*di], WlArr[k-K-1][j-K-1][i+K][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] -= __COMP_MUL_IM( WlArr[k-K-1][j-K-1][i+K][2*di], WlArr[k-K-1][j-K-1][i+K][2*di+1],  ct, st );
              }
              // corner 010
              if ( ( (k-K-1) >=0) && ((i-K-1) >= 0) && ((j+K) < Ne) ) {  
                theta = ImExpZfactor*h* ( (static_cast<double>((K+1)*(k1 + k3) - K*k2) ) );
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   -= __COMP_MUL_RE( WlArr[k-K-1][j+K][i-K-1][2*di], WlArr[k-K-1][j+K][i-K-1][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] -= __COMP_MUL_IM( WlArr[k-K-1][j+K][i-K-1][2*di], WlArr[k-K-1][j+K][i-K-1][2*di+1],  ct, st );
              }
              // corner 001
              if ( ( (i-K-1) >=0) && ((j-K-1) >= 0) && ((k+K) < Ne) ) {  
                theta = ImExpZfactor*h* ( (static_cast<double>((K+1)*(k1 + k2) - K*k3) ) );
                ct = cos(theta);
                st = sin(theta);
                WgArr[k][j][i][2*di]   -= __COMP_MUL_RE( WlArr[k+K][j-K-1][i-K-1][2*di], WlArr[k+K][j-K-1][i-K-1][2*di+1],  ct, st );
                WgArr[k][j][i][2*di+1] -= __COMP_MUL_IM( WlArr[k+K][j-K-1][i-K-1][2*di], WlArr[k+K][j-K-1][i-K-1][2*di+1],  ct, st );
              }
              // corner 111
              if ( ( (j+K) < Ne) && ((k+K) < Ne) && ((i+K) < Ne) ) {  
                theta = ImExpZfactor*h* ( (static_cast<double>((-K)*(k1 + k2 + k3)) ) );
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
    const int K, const int P, const double ImExpZfactor) {

  int i,j,k;
  int p,q,r;
  double theta, ct, st;

  // 0. compute directly for first box ... xs, ys, zs
  directW2L(WlArr, WgArr, xs, ys, zs, 1, 1, 1, Ne, h, K, P, ImExpZfactor); 

  // 1. Precompute the factors ...
  double *fac = new double [3*2*8*P*P*P]; // 3* (2P)^3 complex terms ... one for X,Y and Z shifts, 
  for(int k3 = -P, di = 0; k3 < P; k3++) {
    for(int k2 = -P; k2 < P; k2++) {
      for(int k1 = -P; k1 < P; k1++, di++) {
        theta = ImExpZfactor*h* ( (static_cast<double>(k1) ) );
        fac[6*di]     = cos(theta);
        fac[6*di + 1] = sin(theta);

        theta = ImExpZfactor*h* ( (static_cast<double>(k2) ) );
        fac[6*di + 2] = cos(theta);
        fac[6*di + 3] = sin(theta);

        theta = ImExpZfactor*h* ( (static_cast<double>(k3) ) );
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
                    theta = ImExpZfactor*h* ( (static_cast<double>( -K*k2 - p*k1 - r*k3 ) ) );
                    ct = cos(theta);
                    st = sin(theta);
                    WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WlArr[k+r][j+K][i+p][2*di], WlArr[k+r][j+K][i+p][2*di+1],  ct, st );
                    WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WlArr[k+r][j+K][i+p][2*di], WlArr[k+r][j+K][i+p][2*di+1],  ct, st );
                  }
                  // remove the layer 
                  q = j-K-1;
                  if ( ( (j-K-1) >= 0) && ((i+p) >= 0) && ((i+p) < Ne) && ((k+r) >= 0) && ((k+r) < Ne) ) {  
                    theta = ImExpZfactor*h* ( (static_cast<double>( (K+1)*k2 - p*k1 - r*k3 ) ) );
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
                    theta = ImExpZfactor*h* ( (static_cast<double>( -K*k1 - q*k2 - r*k3 ) ) );
                    ct = cos(theta);
                    st = sin(theta);
                    WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WlArr[k+r][j+q][i+K][2*di], WlArr[k+r][j+q][i+K][2*di+1],  ct, st );
                    WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WlArr[k+r][j+q][i+K][2*di], WlArr[k+r][j+q][i+K][2*di+1],  ct, st );
                  }
                  // remove the layer 
                  p = i-K-1;
                  if ( ( (i-K-1) >= 0) && ((j+q) >= 0) && ((j+q) < Ne) && ((k+r) >= 0) && ((k+r) < Ne) ) {  
                    theta = ImExpZfactor*h* ( (static_cast<double>( (K+1)*k1 - q*k2 - r*k3 ) ) );
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
                  theta = ImExpZfactor*h* ( (static_cast<double>( -p*k1 - q*k2 - K*k3 ) ) );
                  ct = cos(theta);
                  st = sin(theta);
                  WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WlArr[k+K][j+q][i+p][2*di], WlArr[k+K][j+q][i+p][2*di+1],  ct, st );
                  WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WlArr[k+K][j+q][i+p][2*di], WlArr[k+K][j+q][i+p][2*di+1],  ct, st );
                }
                // remove the layer 
                r = k-K-1;
                if ( ( (k-K-1) >= 0) && ((j+q) >= 0) && ((j+q) < Ne) && ((i+p) >= 0) && ((i+p) < Ne) ) {  
                  theta = ImExpZfactor*h* ( (static_cast<double>( -p*k1 - q*k2 + (K+1)*k3 ) ) );
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
                  theta = ImExpZfactor*h* ( (static_cast<double>( -p*k1 - q*k2 - K*k3 ) ) );
                  ct = cos(theta);
                  st = sin(theta);
                  WgArr[k][j][i][2*di]   += __COMP_MUL_RE( WlArr[k+K][j+q][i+p][2*di], WlArr[k+K][j+q][i+p][2*di+1],  ct, st );
                  WgArr[k][j][i][2*di+1] += __COMP_MUL_IM( WlArr[k+K][j+q][i+p][2*di], WlArr[k+K][j+q][i+p][2*di+1],  ct, st );
                }
                // remove the layer 
                r = k-K-1;
                if ( ( (k-K-1) >= 0) && ((j+q) >= 0) && ((j+q) < Ne) && ((i+p) >= 0) && ((i+p) < Ne) ) {  
                  theta = ImExpZfactor*h* ( (static_cast<double>( -p*k1 - q*k2 + (K+1)*k3 ) ) );
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


