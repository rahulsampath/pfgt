
#include "mpi.h"
#include "petscda.h"
#include "pfgtUtils.h"
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <iostream>

extern PetscLogEvent type1Event;
extern PetscLogEvent type2Event;
extern PetscLogEvent s2wEvent;
extern PetscLogEvent w2lEvent;
extern PetscLogEvent l2tEvent;

#define __PI__ 3.14159265

#define __GRID_ID__(xi, yi, zi) ( ((zi)*NgridX*NgridY) + ((yi)*NgridX) + (xi) )

#define __COMP_MUL_RE(a, ai, b, bi) ( a*b - ai*bi )
#define __COMP_MUL_IM(a, ai, b, bi) ( a*bi + ai*b )

#define __LOCAL_COMPUTATION_BLOCK__ { \
  for(unsigned int k = 0; k < localGridLists[otherGridId].size(); k++) { \
    unsigned int otherListId = localGridLists[otherGridId][k]; \
    double otherX = ptsAndSources[4*otherListId]; \
    double otherY = ptsAndSources[(4*otherListId) + 1]; \
    double otherZ = ptsAndSources[(4*otherListId) + 2]; \
    double distSqr = ( ((myX - otherX)*(myX - otherX)) + \
        ((myY - otherY)*(myY - otherY)) + \
        ((myZ - otherZ)*(myZ - otherZ)) ); \
    if( distSqr < IlistWidthSquare ) { \
      localResults[otherListId] += (myF*exp(-distSqr/delta)); \
    } \
  } \
  for(unsigned int k = 0; k < alienGridLists[otherGridId].size(); k++) { \
    unsigned int otherListId = alienGridLists[otherGridId][k]; \
    double otherX = recvData[3*otherListId]; \
    double otherY = recvData[(3*otherListId) + 1]; \
    double otherZ = recvData[(3*otherListId) + 2]; \
    double distSqr = ( ((myX - otherX)*(myX - otherX)) + \
        ((myY - otherY)*(myY - otherY)) + \
        ((myZ - otherZ)*(myZ - otherZ)) ); \
    if( distSqr < IlistWidthSquare ) { \
      sendResults[otherListId] += (myF*exp(-distSqr/delta)); \
    } \
  } \
}

PetscErrorCode pfgtType1(double delta, int K, double fMag, unsigned int numPtsPerProc, int writeOut)
{
  PetscFunctionBegin;

  PetscLogEventBegin(type1Event, 0, 0, 0, 0);

  MPI_Comm comm = MPI_COMM_WORLD;

  int npes, rank;
  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &rank);

  //One box per processor
  const int Nxe = static_cast<int>(pow(npes, (1.0/3.0)));
  assert(Nxe > 0);

  const int Nye = static_cast<int>(sqrt(static_cast<double>(npes/Nxe)));
  assert(Nye > 0);

  const int Nze = npes/(Nxe*Nye);

  assert( (Nxe*Nye*Nze) == npes );

  const double hx = 1.0/(static_cast<double>(Nxe));
  const double hy = 1.0/(static_cast<double>(Nye));
  const double hz = 1.0/(static_cast<double>(Nze));

  const unsigned int myZid = rank/(Nxe*Nye);
  const unsigned int myYid = (rank%(Nxe*Nye))/Nxe;
  const unsigned int myXid = ((rank%(Nxe*Nye))%Nxe);

  assert( rank == ( (myZid*Nxe*Nye) + (myYid*Nxe) + myXid ) );

  //Create local points
  std::vector<double> ptsAndSources;

  double xOff = (static_cast<double>(myXid)*hx);
  double yOff = (static_cast<double>(myYid)*hy);
  double zOff = (static_cast<double>(myZid)*hz);

  for(unsigned int i = 0; i < numPtsPerProc; i++) {
    double x = drand48();
    double y = drand48();
    double z = drand48();
    double f = fMag*(drand48());

    if( (x > 0.0) && (y > 0.0) && (z > 0.0) &&
        (x < 1.0) && (y < 1.0) && (z < 1.0) ) {
      ptsAndSources.push_back(xOff + (x*hx));
      ptsAndSources.push_back(yOff + (y*hy));
      ptsAndSources.push_back(zOff + (z*hz));
      ptsAndSources.push_back(f);
    }
  }//end for i

  //Exact value
  long long numLocalPts = (static_cast<long long>(ptsAndSources.size())/static_cast<long long>(4));

  double IlistWidth = static_cast<double>(K)*sqrt(delta);

  //Send points to other processors
  std::vector<std::vector<unsigned int> > ranksToSend(numLocalPts);
  for(unsigned int i = 0; i < numLocalPts; i++) {
    double myX = ptsAndSources[4*i];
    double myY = ptsAndSources[(4*i) + 1];
    double myZ = ptsAndSources[(4*i) + 2];

    double boxXmin = myX - IlistWidth;
    double boxXmax = myX + IlistWidth;

    double boxYmin = myY - IlistWidth;
    double boxYmax = myY + IlistWidth;

    double boxZmin = myZ - IlistWidth;
    double boxZmax = myZ + IlistWidth;

    if(boxXmin < 0.0) {
      boxXmin = 0.0;
    }
    if(boxXmax > 1.0) {
      boxXmax = 1.0;
    }

    if(boxYmin < 0.0) {
      boxYmin = 0.0;
    }
    if(boxYmax > 1.0) {
      boxYmax = 1.0;
    }

    if(boxZmin < 0.0) {
      boxZmin = 0.0;
    }
    if(boxZmax > 1.0) {
      boxZmax = 1.0;
    }

    unsigned int minXid = static_cast<unsigned int>(floor(boxXmin/hx));
    unsigned int maxXid = static_cast<unsigned int>(ceil(boxXmax/hx));

    unsigned int minYid = static_cast<unsigned int>(floor(boxYmin/hy));
    unsigned int maxYid = static_cast<unsigned int>(ceil(boxYmax/hy));

    unsigned int minZid = static_cast<unsigned int>(floor(boxZmin/hz));
    unsigned int maxZid = static_cast<unsigned int>(ceil(boxZmax/hz));

    for(unsigned int zi = minZid; zi < maxZid; zi++) {
      for(unsigned int yi = minYid; yi < maxYid; yi++) {
        for(unsigned int xi = minXid; xi < maxXid; xi++) {
          unsigned int tmpRank = ( (zi*Nxe*Nye) + (yi*Nxe) + xi );
          if(tmpRank != rank) {
            ranksToSend[i].push_back(tmpRank);
          }
        }//end for xi
      }//end for yi
    }//end for zi
  }//end for i

  std::vector<int> sendCnts(npes);
  std::vector<int> recvCnts(npes);

  for(int i = 0; i < npes; i++) {
    sendCnts[i] = 0;
  }//end for i

  for(unsigned int i = 0; i < ranksToSend.size(); i++) {
    for(unsigned int j = 0; j < ranksToSend[i].size(); j++) {
      sendCnts[ranksToSend[i][j]] += 3;
    }//end for j
  }//end for i

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

  std::vector<double> sendData(sendDisps[npes - 1] + sendCnts[npes - 1]);

  unsigned int numSendPts = ((sendData.size())/3);

  std::vector<unsigned int> commMap(numSendPts);

  for(int i = 0; i < npes; i++) {
    sendCnts[i] = 0;
  }//end for i

  for(unsigned int i = 0; i < ranksToSend.size(); i++) {
    for(unsigned int j = 0; j < ranksToSend[i].size(); j++) {
      sendData[sendDisps[ranksToSend[i][j]] + sendCnts[ranksToSend[i][j]]] = ptsAndSources[4*i];
      sendData[sendDisps[ranksToSend[i][j]] + sendCnts[ranksToSend[i][j]] + 1] = ptsAndSources[(4*i) + 1];
      sendData[sendDisps[ranksToSend[i][j]] + sendCnts[ranksToSend[i][j]] + 2] = ptsAndSources[(4*i) + 2];
      commMap[(sendDisps[ranksToSend[i][j]] + sendCnts[ranksToSend[i][j]])/3] = i;
      sendCnts[ranksToSend[i][j]] += 3;
    }//end for j
  }//end for i

  ranksToSend.clear();

  std::vector<double> recvData(recvDisps[npes - 1] + recvCnts[npes - 1]);

  unsigned int numRecvPts = ((recvData.size())/3);

  MPI_Alltoallv( (&(*(sendData.begin()))), (&(*(sendCnts.begin()))),
      (&(*(sendDisps.begin()))), MPI_DOUBLE, (&(*(recvData.begin()))),
      (&(*(recvCnts.begin()))), (&(*(recvDisps.begin()))), MPI_DOUBLE, comm );

  sendData.clear();

  //Create grid:
  //1) Each box in the grid is of size IlistWidth 
  //2) The grid encloses the region [(h-IlistWidth), (h + IlistWidth)) (Left
  //inclusive), (Right exclusive)
  //3) The grid is aligned with the region on the -ve XYZ directions, but may
  //include some extra space on the +ve XYZ directions if h + 2*IlistWidth is
  //not perfectly divisible by IlistWidth.
  double gridMinX = (static_cast<double>(myXid)*hx) - IlistWidth;
  double gridMaxX = (static_cast<double>(myXid + 1)*hx) + IlistWidth;

  double gridMinY = (static_cast<double>(myYid)*hy) - IlistWidth;
  double gridMaxY = (static_cast<double>(myYid + 1)*hy) + IlistWidth;

  double gridMinZ = (static_cast<double>(myZid)*hz) - IlistWidth;
  double gridMaxZ = (static_cast<double>(myZid + 1)*hz) + IlistWidth;

  if(gridMinX < 0.0) {
    gridMinX = 0.0;
  }
  if(gridMaxX > 1.0) {
    gridMaxX = 1.0;
  }

  if(gridMinY < 0.0) {
    gridMinY = 0.0;
  }
  if(gridMaxY > 1.0) {
    gridMaxY = 1.0;
  }

  if(gridMinZ < 0.0) {
    gridMinZ = 0.0;
  }
  if(gridMaxZ > 1.0) {
    gridMaxZ = 1.0;
  }

  unsigned int NgridX = static_cast<unsigned int>(ceil((gridMaxX - gridMinX)/IlistWidth));
  unsigned int NgridY = static_cast<unsigned int>(ceil((gridMaxY - gridMinY)/IlistWidth));
  unsigned int NgridZ = static_cast<unsigned int>(ceil((gridMaxZ - gridMinZ)/IlistWidth));

  //Can be made more memory efficient with some more book-keeping
  std::vector<std::vector<unsigned int> > localGridLists(NgridX*NgridY*NgridZ);
  std::vector<std::vector<unsigned int> > alienGridLists(NgridX*NgridY*NgridZ);

  //Bin local points
  for(unsigned int i = 0; i < numLocalPts; i++) {
    double myX = ptsAndSources[4*i];
    double myY = ptsAndSources[(4*i) + 1];
    double myZ = ptsAndSources[(4*i) + 2];

    assert(myX >= gridMinX);
    assert(myX < gridMaxX);

    assert(myY >= gridMinY);
    assert(myY < gridMaxY);

    assert(myZ >= gridMinZ);
    assert(myZ < gridMaxZ);

    unsigned int xid = static_cast<unsigned int>(floor((myX - gridMinX)/IlistWidth));
    unsigned int yid = static_cast<unsigned int>(floor((myY - gridMinY)/IlistWidth));
    unsigned int zid = static_cast<unsigned int>(floor((myZ - gridMinZ)/IlistWidth));

    localGridLists[(zid*NgridX*NgridY) + (yid*NgridX) + xid].push_back(i);
  }//end for i

  //Bin alien points
  for(unsigned int i = 0; i < numRecvPts; i++) {
    double myX = recvData[3*i];
    double myY = recvData[(3*i) + 1];
    double myZ = recvData[(3*i) + 2];

    assert(myX >= gridMinX);
    assert(myX < gridMaxX);

    assert(myY >= gridMinY);
    assert(myY < gridMaxY);

    assert(myZ >= gridMinZ);
    assert(myZ < gridMaxZ);

    unsigned int xid = static_cast<unsigned int>(floor((myX - gridMinX)/IlistWidth));
    unsigned int yid = static_cast<unsigned int>(floor((myY - gridMinY)/IlistWidth));
    unsigned int zid = static_cast<unsigned int>(floor((myZ - gridMinZ)/IlistWidth));

    alienGridLists[(zid*NgridX*NgridY) + (yid*NgridX) + xid].push_back(i);
  }//end for i

  double IlistWidthSquare = (IlistWidth*IlistWidth);

  std::vector<double> sendResults(numRecvPts);
  std::vector<double> localResults(numLocalPts);

  for(unsigned int i = 0; i < numRecvPts; i++) {
    sendResults[i] = 0.0;
  }//end for i

  for(unsigned int i = 0; i < numLocalPts; i++) {
    localResults[i] = 0.0;
  }//end for i

  //my = source; other = target
  for(unsigned int zi = 0; zi < NgridZ; zi++) {
    for(unsigned int yi = 0; yi < NgridY; yi++) {
      for(unsigned int xi = 0; xi < NgridX; xi++) {
        unsigned int myGridId = __GRID_ID__(xi, yi, zi) ; 

        for(unsigned int j = 0; j < localGridLists[myGridId].size(); j++) {
          unsigned int myListId = localGridLists[myGridId][j];

          double myX = ptsAndSources[4*myListId];
          double myY = ptsAndSources[(4*myListId) + 1];
          double myZ = ptsAndSources[(4*myListId) + 2];
          double myF = ptsAndSources[(4*myListId) + 3];

          //Self
          {
            unsigned int otherGridId = myGridId;
            __LOCAL_COMPUTATION_BLOCK__
          }

          //-ve X
          if(xi > 0) {
            unsigned int otherGridId = __GRID_ID__((xi - 1), yi, zi) ;
            __LOCAL_COMPUTATION_BLOCK__
          }

          //+ve X
          if(xi < (NgridX - 1)) {
            unsigned int otherGridId = __GRID_ID__((xi + 1), yi, zi) ;
            __LOCAL_COMPUTATION_BLOCK__
          }

          //-ve Y
          if(yi > 0) {
            unsigned int otherGridId = __GRID_ID__(xi, (yi - 1), zi) ;
            __LOCAL_COMPUTATION_BLOCK__
          }

          //+ve Y
          if(yi < (NgridY - 1)) {
            unsigned int otherGridId = __GRID_ID__(xi, (yi + 1), zi) ;
            __LOCAL_COMPUTATION_BLOCK__
          }

          //-ve Z
          if(zi > 0) {
            unsigned int otherGridId = __GRID_ID__(xi, yi, (zi - 1)) ;
            __LOCAL_COMPUTATION_BLOCK__
          }

          //+ve Z
          if(zi < (NgridZ - 1)) {
            unsigned int otherGridId = __GRID_ID__(xi, yi, (zi + 1)) ;
            __LOCAL_COMPUTATION_BLOCK__
          }

        }//end for j
      }//end for xi
    }//end for yi
  }//end for zi

  recvData.clear();

  //Reverse communication for results
  for(int i = 0; i < npes; i++) {
    sendCnts[i] = (sendCnts[i]/3);
    sendDisps[i] = (sendDisps[i]/3);
    recvCnts[i] = (recvCnts[i]/3);
    recvDisps[i] = (recvDisps[i]/3);
  }//end for i

  std::vector<double> recvResults(numSendPts);

  MPI_Alltoallv( (&(*(sendResults.begin()))), (&(*(recvCnts.begin()))),
      (&(*(recvDisps.begin()))), MPI_DOUBLE,
      (&(*(recvResults.begin()))), (&(*(sendCnts.begin()))), 
      (&(*(sendDisps.begin()))), MPI_DOUBLE, comm );

  sendResults.clear();
  recvCnts.clear();
  recvDisps.clear();
  sendCnts.clear();
  sendDisps.clear();

  //Global sum for results
  for(unsigned int i = 0; i < numSendPts; i++) {
    localResults[commMap[i]] += recvResults[i];
  }//end for i

  if(writeOut) {
    char fname[256];
    sprintf(fname, "inpType1_%d_%d.txt", rank, npes);
    FILE* fp = fopen(fname, "w");
    fprintf(fp, "%lld\n", numLocalPts);
    for(unsigned int i = 0; i < numLocalPts; i++) {
      fprintf(fp, "%lf %lf %lf %lf\n", ptsAndSources[4*i], ptsAndSources[(4*i) + 1], 
          ptsAndSources[(4*i) + 2], ptsAndSources[(4*i) + 3]);
    }//end for i
    fclose(fp);
  }

  if(writeOut) {
    char fname[256];
    sprintf(fname, "outType1_%d_%d.txt", rank, npes);
    FILE* fp = fopen(fname, "w");
    fprintf(fp, "%lld\n", numLocalPts);
    for(unsigned int i = 0; i < numLocalPts; i++) {
      fprintf(fp, "%lf \n", localResults[i]);
    }//end for i
    fclose(fp);
  }

  long long trueTotalPts;
  MPI_Reduce(&numLocalPts, &trueTotalPts, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, comm);

  if(!rank) {
    std::cout<<"True Total NumPts: "<<trueTotalPts<<std::endl; 
  }

  PetscLogEventEnd(type1Event, 0, 0, 0, 0);

  PetscFunctionReturn(0);
}

PetscErrorCode pfgtType2(double delta, double fMag, unsigned int numPtsPerProc,
    int P, int L, int K, int writeOut)
{
  PetscFunctionBegin;

  PetscLogEventBegin(type2Event, 0, 0, 0, 0);

  MPI_Comm comm = MPI_COMM_WORLD;

  int npes, rank;
  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &rank);

  const double h = sqrt(delta);

  const unsigned int Ne = static_cast<unsigned int>(1.0/h);

  if(!rank) {
    std::cout<<"Ne = "<< Ne <<std::endl;
  }

  //2p complex coefficients for each dimension.  
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

  //Maximum value
  unsigned int numPtsPerBox = numPtsPerProc/(nx*ny*nz);

  //Tensor-Product Grid
  const unsigned int ptGridSizeWithinBox = static_cast<unsigned int>(floor(pow( numPtsPerBox, (1.0/3.0) )));
  const double ptGridOff = 0.1*h;
  const double ptGridH = 0.8*h/(static_cast<double>(ptGridSizeWithinBox) - 1.0);
  long long trueLocalNumPts = ptGridSizeWithinBox*ptGridSizeWithinBox*ptGridSizeWithinBox*nx*ny*nz;

  PetscLogEventBegin(s2wEvent, 0, 0, 0, 0);

  //Loop over local boxes and execute S2W in each box
  Vec Wglobal;
  PetscScalar**** WgArr;
  DACreateGlobalVector(da, &Wglobal);
  DAVecGetArrayDOF(da, Wglobal, &WgArr);

  const double lambda = static_cast<double>(L)/(static_cast<double>(P)*sqrt(delta));

  std::vector<std::vector<std::vector<double> > > tmp1R;
  std::vector<std::vector<std::vector<double> > > tmp1C;

  std::vector<std::vector<std::vector<double> > > tmp2R;
  std::vector<std::vector<std::vector<double> > > tmp2C;

  for(PetscInt zi = 0, boxId = 0; zi < nz; zi++) {
    for(PetscInt yi = 0; yi < ny; yi++) {
      for(PetscInt xi = 0; xi < nx; xi++, boxId++) {

        //Anchor of the box
        double ax =  h*(static_cast<double>(xi + xs));
        double ay =  h*(static_cast<double>(yi + ys));
        double az =  h*(static_cast<double>(zi + zs));

        //Center of the box
        double halfH = (0.5*h);
        double cx =  ax + halfH;
        double cy =  ay + halfH;
        double cz =  az + halfH;

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
                double px = ax + ptGridOff + (ptGridH*(static_cast<double>(j1)));

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
                double py = ay + ptGridOff + (ptGridH*(static_cast<double>(j2)));

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

        for(int k3 = -P, di = 0; k3 < P; k3++) {
          for(int k2 = -P; k2 < P; k2++) {
            int shiftK2 = (k2 + P);

            for(int k1 = -P; k1 < P; k1++, di++) {
              int shiftK1 = (k1 + P);

              WgArr[zi + zs][yi + ys][xi + xs][2*di] = 0.0;
              WgArr[zi + zs][yi + ys][xi + xs][(2*di) + 1] = 0.0;

              for(int j3 = 0; j3 < ptGridSizeWithinBox; j3++) {
                double pz = az + ptGridOff + (ptGridH*(static_cast<double>(j3)));

                double theta = lambda*(static_cast<double>(k3)*(cz - pz));

                double rVal = tmp2R[shiftK2][shiftK1][j3];
                double cVal = tmp2C[shiftK2][shiftK1][j3];

                WgArr[zi + zs][yi + ys][xi + xs][2*di] += ( (rVal*cos(theta)) - (cVal*sin(theta)) );
                WgArr[zi + zs][yi + ys][xi + xs][(2*di) + 1] += ( (rVal*sin(theta)) + (cVal*cos(theta)) );

              }//end for j3
            }//end for k1
          }//end for k2
        }//end for k3

      }//end for xi
    }//end for yi
  }//end for zi

  DAVecRestoreArrayDOF(da, Wglobal, &WgArr);

  PetscLogEventEnd(s2wEvent, 0, 0, 0, 0);

  PetscLogEventBegin(w2lEvent, 0, 0, 0, 0);

  //Exchange ghosts
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
  sweepW2L(WlArr, WgArr, xs, ys, zs, nx, ny, nz, Ne, h, K, P, lambda);


  DAVecRestoreArrayDOF(da, Wlocal, &WlArr);

  PetscLogEventEnd(w2lEvent, 0, 0, 0, 0);

  //Loop over local boxes and execute L2T in each box

  PetscLogEventBegin(l2tEvent, 0, 0, 0, 0);

  const double C0 = ( pow((0.5/sqrt(__PI__)), 3.0)*
      pow((static_cast<double>(L)/static_cast<double>(P)), 3.0) );

  std::vector<std::vector<double> > results(nx*ny*nz);

  for(PetscInt zi = 0, boxId = 0; zi < nz; zi++) {
    for(PetscInt yi = 0; yi < ny; yi++) {
      for(PetscInt xi = 0; xi < nx; xi++, boxId++) {

        //Anchor of the box
        double ax =  h*(static_cast<double>(xi + xs));
        double ay =  h*(static_cast<double>(yi + ys));
        double az =  h*(static_cast<double>(zi + zs));

        //Center of the box
        double cx =  ax + (h*0.5);
        double cy =  ay + (h*0.5);
        double cz =  az + (h*0.5);

        //Tensor Product Acceleration

        //Stage - 1

        tmp1R.resize(ptGridSizeWithinBox);
        tmp1C.resize(ptGridSizeWithinBox);
        for(unsigned int k1 = 0; k1 < ptGridSizeWithinBox; k1++) {
          tmp1R[k1].resize(2*P);
          tmp1C[k1].resize(2*P);

          double px = ax + ptGridOff + (ptGridH*(static_cast<double>(k1)));

          for(int j3 = -P, di = 0; j3 < P; j3++) {
            int shiftJ3 = (j3 + P);

            tmp1R[k1][shiftJ3].resize(2*P);
            tmp1C[k1][shiftJ3].resize(2*P);

            for(int j2 = -P; j2 < P; j2++) {
              int shiftJ2 = (j2 + P);

              double rSum = 0.0;
              double cSum = 0.0;

              for(int j1 = -P; j1 < P; j1++, di++) {
                double theta = lambda*(static_cast<double>(j1)*(px - cx)) ;

                double a = WgArr[zi + zs][yi + ys][xi + xs][2*di];
                double b = WgArr[zi + zs][yi + ys][xi + xs][(2*di) + 1];
                double c = cos(theta);
                double d = sin(theta);
                double factor = exp(-lambda*lambda*static_cast<double>( (j1*j1) + (j2*j2) + (j3*j3) )/4.0);

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

          double py = ay + ptGridOff + (ptGridH*(static_cast<double>(k2)));

          for(unsigned int k1 = 0; k1 < ptGridSizeWithinBox; k1++) {
            tmp2R[k2][k1].resize(2*P);
            tmp2C[k2][k1].resize(2*P);

            for(int j3 = -P; j3 < P; j3++) {
              int shiftJ3 = (j3 + P);

              tmp2R[k2][k1][shiftJ3] = 0.0;
              tmp2C[k2][k1][shiftJ3] = 0.0;

              for(int j2 = -P; j2 < P; j2++) {
                int shiftJ2 = (j2 + P);

                double theta = lambda*(static_cast<double>(j2)*(py - cy)) ;

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

        results[boxId].resize(ptGridSizeWithinBox*ptGridSizeWithinBox*ptGridSizeWithinBox);

        for(unsigned int k3 = 0, pt = 0; k3 < ptGridSizeWithinBox; k3++) {
          double pz = az + ptGridOff + (ptGridH*(static_cast<double>(k3)));

          for(unsigned int k2 = 0; k2 < ptGridSizeWithinBox; k2++) {
            for(unsigned int k1 = 0; k1 < ptGridSizeWithinBox; k1++, pt++) {

              results[boxId][pt] = 0.0;

              for(int j3 = -P; j3 < P; j3++) {
                int shiftJ3 = (j3 + P);

                double theta = lambda*(static_cast<double>(j3)*(pz - cz)) ;

                double a = tmp2R[k2][k1][shiftJ3];
                double b = tmp2C[k2][k1][shiftJ3];
                double c = cos(theta);
                double d = sin(theta);

                results[boxId][pt] += ( (a*c) - (b*d) );
              }//end for j3
            }//end for k1
          }//end for k2
        }//end for k3

      }//end for xi
    }//end for yi
  }//end for zi

  DAVecRestoreArrayDOF(da, Wglobal, &WgArr);

  PetscLogEventEnd(l2tEvent, 0, 0, 0, 0);

  VecDestroy(Wlocal);
  VecDestroy(Wglobal);

  DADestroy(da);

  if(writeOut) {
    char fname[256];
    sprintf(fname, "outType2_%d_%d.txt", rank, npes);
    FILE* fp = fopen(fname, "w");
    fprintf(fp, "%d\n", (results.size()));
    for(unsigned int i = 0; i < results.size(); i++) {
      fprintf(fp, "%d\n", results[i].size());
      for(unsigned int j = 0; j < results[i].size(); j++) {
        fprintf(fp, "%lf \n", results[i][j]);
      }//end for j
    }//end for i
    fclose(fp);
  }

  long long trueTotalPts;
  MPI_Reduce(&trueLocalNumPts, &trueTotalPts, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, comm);

  if(!rank) {
    std::cout<<"True Total NumPts: "<<trueTotalPts<<std::endl; 
  }

  PetscLogEventEnd(type2Event, 0, 0, 0, 0);

  PetscFunctionReturn(0);
}

void directW2L(PetscScalar**** WlArr, PetscScalar**** WgArr,
    int xs, int ys, int zs, int nx, int ny, int nz, 
    int Ne, double h, const int StencilWidth, const int P, const double lambda) {
  //Loop over local boxes and their Interaction lists and do a direct translation

  double theta, ct, st;
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

                    theta = lambda*h*( static_cast<double>(k1*(xx - xj) + k2*(yy - yj) + k3*(zz - zj) ) );
                    ct = cos(theta);
                    st = sin(theta);

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

