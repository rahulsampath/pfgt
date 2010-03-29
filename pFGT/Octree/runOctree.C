
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

#define SEEDA 0x12345678 

#define SEEDB 76543

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

  if(argc < 7) {
    if(!rank) {
      std::cout<<"Usage: exe numOctPtsPerProc numFgtPtsPerProc fMag epsilon delta writeOut"<<std::endl;
    }
    PetscFinalize();
  }

  unsigned int numOctPtsPerProc = atoi(argv[1]);
  unsigned int numFgtPtsPerProc = atoi(argv[2]);
  double fMag = atof(argv[3]);
  double epsilon = atof(argv[4]);  
  double delta = atof(argv[5]);
  int writeOut = atoi(argv[6]);

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

  const unsigned int seed = (SEEDA + (SEEDB*rank));
  srand48(seed);

  //Generate gaussian distribution for the octree
  std::vector<double> pts;
  pts.resize(3*numOctPtsPerProc);
  for(unsigned int i = 0; i < (3*numOctPtsPerProc); i++) {
    pts[i]= gaussian(0.5, 0.16);
  }

  unsigned int ptsLen = pts.size();
  unsigned int dim = 3;
  unsigned int maxDepth = 30;

  std::vector<ot::TreeNode> linOct;
  for(unsigned int i = 0; i < ptsLen; i+=3) {
    if( (pts[i] > 0.0) && (pts[i+1] > 0.0) && (pts[i+2] > 0.0) &&
        ( ((unsigned int)(pts[i]*((double)(1u << maxDepth)))) < (1u << maxDepth))  &&
        ( ((unsigned int)(pts[i+1]*((double)(1u << maxDepth)))) < (1u << maxDepth))  &&
        ( ((unsigned int)(pts[i+2]*((double)(1u << maxDepth)))) < (1u << maxDepth)) ) 
    {
      linOct.push_back( ot::TreeNode((unsigned int)(pts[i]*(double)(1u << maxDepth)),
            (unsigned int)(pts[i+1]*(double)(1u << maxDepth)),
            (unsigned int)(pts[i+2]*(double)(1u << maxDepth)),
            maxDepth, dim, maxDepth) );
    }
  }

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

  long long totalNumOctants;
  long long localNumOctants = linOct.size();

  MPI_Reduce(&localNumOctants, &totalNumOctants, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if(!rank) {
    std::cout<<"Total Num Octs: "<< totalNumOctants << std::endl;
  }

  //FGT
  pfgt(linOct, maxDepth, delta, fMag, numFgtPtsPerProc, P, L, K, writeOut);

  PetscFinalize();

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


