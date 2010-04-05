
#include "mpi.h"
#include "petsc.h"
#include "sys.h"
#include "parUtils.h"
#include "octUtils.h"
#include "TreeNode.h"
#include <cstdlib>
#include <vector>
#include "externVars.h"
#include "dendro.h"

#include "pfgtOctUtils.h"

#ifdef MPI_WTIME_IS_GLOBAL
#undef MPI_WTIME_IS_GLOBAL
#endif

PetscCookie fgtCookie;
PetscLogEvent fgtEvent;
PetscLogEvent s2wEvent;
PetscLogEvent s2wCommEvent;
PetscLogEvent w2dEvent;
PetscLogEvent d2dEvent;
PetscLogEvent w2lEvent;
PetscLogEvent d2lEvent;
PetscLogEvent l2tCommEvent;
PetscLogEvent l2tEvent;

void genGaussPts(int rank, unsigned int numOctPtsPerProc, std::vector<double> & pts);

void genSpherePts(int rank, int npes, unsigned int numOctPtsPerProc, std::vector<double> & pts);

void rescalePts(std::vector<double> & pts);

double gaussian(double mean, double std_deviation);

int main(int argc, char** argv) {

  PetscInitialize(&argc, &argv, NULL, NULL);
  ot::RegisterEvents();

  PetscCookieRegister("FGT", &fgtCookie);
  PetscLogEventRegister("FGT", fgtCookie, &fgtEvent);
  PetscLogEventRegister("S2W", fgtCookie, &s2wEvent);
  PetscLogEventRegister("S2Wcomm", fgtCookie, &s2wCommEvent);
  PetscLogEventRegister("W2D", fgtCookie, &w2dEvent);
  PetscLogEventRegister("D2D", fgtCookie, &d2dEvent);
  PetscLogEventRegister("W2L", fgtCookie, &w2lEvent);
  PetscLogEventRegister("D2L", fgtCookie, &d2lEvent);
  PetscLogEventRegister("L2Tcomm", fgtCookie, &l2tCommEvent);
  PetscLogEventRegister("L2T", fgtCookie, &l2tEvent);

  int npes, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &npes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(argc < 8) {
    if(!rank) {
      std::cout<<"Usage: exe octPtsType numOctPtsPerProc numFgtPtsPerDimPerOct fMag epsilon delta writeOut"<<std::endl;
    }
    PetscFinalize();
  }

  unsigned int octPtsType = atoi(argv[1]);
  unsigned int numOctPtsPerProc = atoi(argv[2]);
  unsigned int numFgtPtsPerDimPerOct = atoi(argv[3]);
  double fMag = atof(argv[4]);
  double epsilon = atof(argv[5]);  
  double delta = atof(argv[6]);
  int writeOut = atoi(argv[7]);

  int P, L, K;

  if(epsilon == 1.0e-3) {
    P = 6;
    L = 5;
    K = 3;
  } else if(epsilon == 1.0e-6) {
    P = 10;
    L = 7;
    K = 4;
  } else if(epsilon == 1.0e-9) {
    P = 16;
    L = 10;
    K = 5;
  } else if(epsilon == 1.0e-12) {
    P = 20;
    L = 11;
    K = 6;
  } else {
    if(!rank) {
      std::cout<<"Wrong epsilon!"<<std::endl;
    }
    assert(false);
    PetscFinalize();
  }

  //Generate gaussian distribution for the octree
  std::vector<double> pts;

  if(octPtsType == 0) {
    genGaussPts(rank, numOctPtsPerProc, pts);
  } else if(octPtsType == 1) {
    genSpherePts(rank, npes, numOctPtsPerProc, pts);
  } else {
    assert(false);
  }

  rescalePts(pts);

  unsigned int ptsLen = pts.size();
  unsigned int dim = 3;
  unsigned int maxDepth = 30;

  std::vector<ot::TreeNode> linOct;
  for(unsigned int i = 0; i < ptsLen; i += 3) {
    linOct.push_back( ot::TreeNode((unsigned int)(pts[i]*(double)(1u << maxDepth)),
          (unsigned int)(pts[i+1]*(double)(1u << maxDepth)),
          (unsigned int)(pts[i+2]*(double)(1u << maxDepth)),
          maxDepth, dim, maxDepth) );
  }//end for i

  par::removeDuplicates<ot::TreeNode>(linOct, false, MPI_COMM_WORLD);

  pts.resize(3*(linOct.size()));
  ptsLen = (3*(linOct.size()));
  for(int i = 0; i < linOct.size(); i++) {
    pts[3*i] = (((double)(linOct[i].getX())) + 0.5)/((double)(1u << maxDepth));
    pts[(3*i)+1] = (((double)(linOct[i].getY())) +0.5)/((double)(1u << maxDepth));
    pts[(3*i)+2] = (((double)(linOct[i].getZ())) +0.5)/((double)(1u << maxDepth));
  }//end for i

  unsigned int maxNumPts = 1;

  double gSize[3];
  gSize[0] = 1.0;
  gSize[1] = 1.0;
  gSize[2] = 1.0;

  //construct the octree
  linOct.clear();
  ot::points2Octree(pts, gSize, linOct, dim, maxDepth, maxNumPts, MPI_COMM_WORLD);
  pts.clear();

  //FGT
  pfgt(linOct, maxDepth, delta, fMag, numFgtPtsPerDimPerOct, P, L, K, writeOut);

  PetscFinalize();

}

void genSpherePts(int rank, int npes, unsigned int numOctPtsPerProc, std::vector<double> & pts) {
  const unsigned int numIntervals = static_cast<unsigned int>(floor(sqrt(0.5*
          static_cast<double>(npes)*static_cast<double>(numOctPtsPerProc))));

  const double  __PI__ = 3.14159265;

  unsigned int avgSize = (2*numIntervals*numIntervals)/npes;

  unsigned int stPtId = rank*avgSize;

  pts.resize( 3*avgSize );

  for(unsigned int i = 0; i < avgSize; i++) {
    unsigned int uId = (i + stPtId)/numIntervals;
    unsigned int vId = (i + stPtId)%numIntervals;

    double u = __PI__*static_cast<double>(uId)/static_cast<double>(numIntervals);
    double v = __PI__*static_cast<double>(vId)/static_cast<double>(numIntervals);

    pts[3*i] = cos(u)*cos(v);
    pts[(3*i) + 1] = cos(u)*sin(v);
    pts[(3*i) + 2] = sin(u);
  }//end for i
}

void genGaussPts(int rank, unsigned int numOctPtsPerProc, std::vector<double> & pts)
{
  const unsigned int seed = (0x12345678  + (76543*rank));
  srand48(seed);

  pts.resize(3*numOctPtsPerProc);
  for(unsigned int i = 0; i < (3*numOctPtsPerProc); i++) {
    pts[i]= gaussian(0.5, 0.16);
  }
}

double gaussian(double mean, double std_deviation) {
  static double t1 = 0, t2=0;
  double x1, x2, x3, r;

  using namespace std;

  // reuse previous calculations
  if(t1) {
    const double tmp = t1;
    t1 = 0;
    return mean + std_deviation * tmp;
  }
  if(t2) {
    const double tmp = t2;
    t2 = 0;
    return mean + std_deviation * tmp;
  }

  // pick randomly a point inside the unit disk
  do {
    x1 = 2 * drand48() - 1;
    x2 = 2 * drand48() - 1;
    x3 = 2 * drand48() - 1;
    r = x1 * x1 + x2 * x2 + x3*x3;
  } while(r >= 1);

  // Box-Muller transform
  r = sqrt(-2.0 * log(r) / r);

  // save for next call
  t1 = (r * x2);
  t2 = (r * x3);

  return mean + (std_deviation * r * x1);
}//end gaussian

void rescalePts(std::vector<double> & pts) 
{
  double minX = pts[0];
  double maxX = pts[0];

  double minY = pts[1];
  double maxY = pts[1];

  double minZ = pts[2];
  double maxZ = pts[2];

  for(unsigned int i = 0; i < pts.size(); i += 3) {
    double xPt = pts[i];
    double yPt = pts[i + 1];
    double zPt = pts[i + 2];

    if(xPt < minX) {
      minX = xPt;
    }

    if(xPt > maxX) {
      maxX = xPt;
    }

    if(yPt < minY) {
      minY = yPt;
    }

    if(yPt > maxY) {
      maxY = yPt;
    }

    if(zPt < minZ) {
      minZ = zPt;
    }

    if(zPt > maxZ) {
      maxZ = zPt;
    }
  }//end for i

  double xRange = (maxX - minX);
  double yRange = (maxY - minY);
  double zRange = (maxZ - minZ);

  for(unsigned int i = 0; i < pts.size();  i += 3) {
    double xPt = pts[i];
    double yPt = pts[i + 1];
    double zPt = pts[i + 2];

    pts[i] = 0.05 + (0.925*(xPt - minX)/xRange);
    pts[i + 1] = 0.05 + (0.925*(yPt - minY)/yRange);
    pts[i + 2] = 0.05 + (0.925*(zPt - minZ)/zRange);
  }//end for i

}


