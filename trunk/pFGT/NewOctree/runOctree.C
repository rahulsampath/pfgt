
#include "mpi.h"
#include "petsc.h"
#include "sys.h"
#include <cmath>
#include <cassert>
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
PetscLogEvent serialEvent;
PetscLogEvent fgtOctConEvent;
PetscLogEvent expandOnlyEvent;
PetscLogEvent expandHybridEvent;
PetscLogEvent directOnlyEvent;
PetscLogEvent directHybridEvent;

bool softEquals(double a, double b) {
  return ((fabs(a - b)) < 1.0e-14);
}

void genGaussPts(int rank, unsigned int numOctPtsPerProc, std::vector<double> & pts);

void rescalePts(std::vector<double> & pts);

double gaussian(double mean, double std_deviation);

int main(int argc, char** argv) {
  PetscInitialize(&argc, &argv, NULL, NULL);
  ot::RegisterEvents();

  PetscCookieRegister("FGT", &fgtCookie);
  PetscLogEventRegister("FGT", fgtCookie, &fgtEvent);
  PetscLogEventRegister("Serial", fgtCookie, &serialEvent);
  PetscLogEventRegister("FGToctCon", fgtCookie, &fgtOctConEvent);
  PetscLogEventRegister("Expand-O", fgtCookie, &expandOnlyEvent);
  PetscLogEventRegister("Expand-H", fgtCookie, &expandHybridEvent);
  PetscLogEventRegister("Direct-O", fgtCookie, &directOnlyEvent);
  PetscLogEventRegister("Direct-H", fgtCookie, &directHybridEvent);

  int npes, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &npes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(argc < 7) {
    if(!rank) {
      std::cout<<"Usage: exe numPtsPerProc fMag epsilon FgtLev DirectHfactor maxNumPts"<<std::endl;
    }
    PetscFinalize();
  }

  unsigned int numPtsPerProc = atoi(argv[1]);
  double fMag = atof(argv[2]);
  double epsilon = atof(argv[3]);  
  unsigned int FgtLev = atoi(argv[4]);
  double DirectHfactor = atof(argv[5]);
  unsigned int maxNumPts = atoi(argv[6]);
  const unsigned int dim = 3;
  const unsigned int maxDepth = 30;

  assert(FgtLev <= maxDepth);

  if(!rank) {
    std::cout<<"numPtsPerProc = "<<numPtsPerProc<<std::endl;
    std::cout<<"fMag = "<<fMag<<std::endl;
    std::cout<<"epsilon = "<<epsilon<<std::endl;
    std::cout<<"FgtLev = "<<FgtLev<<std::endl;
    std::cout<<"DirectHfactor = "<<DirectHfactor<<std::endl;
    std::cout<<"MaxNumPts = "<<maxNumPts<<std::endl;
  }

  int P, L, K;

  if(softEquals(epsilon, 1.0e-3)) {
    P = 6;
    L = 5;
    K = 3;
  } else if(softEquals(epsilon, 1.0e-6)) {
    P = 10;
    L = 7;
    K = 4;
  } else if(softEquals(epsilon, 1.0e-9)) {
    P = 16;
    L = 10;
    K = 5;
  } else if(softEquals(epsilon, 1.0e-12)) {
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
  genGaussPts(rank, numPtsPerProc, pts);
  rescalePts(pts);

  std::vector<ot::TreeNode> linOct;
  for(unsigned int i = 0; i < pts.size(); i += 3) {
    linOct.push_back( ot::TreeNode((unsigned int)(pts[i]*(double)(1u << maxDepth)),
          (unsigned int)(pts[i+1]*(double)(1u << maxDepth)),
          (unsigned int)(pts[i+2]*(double)(1u << maxDepth)),
          maxDepth, dim, maxDepth) );
  }//end for i

  par::removeDuplicates<ot::TreeNode>(linOct, false, MPI_COMM_WORLD);

  pts.resize(3*(linOct.size()));
  for(size_t i = 0; i < linOct.size(); i++) {
    pts[3*i] = (((double)(linOct[i].getX())) + 0.5)/((double)(1u << maxDepth));
    pts[(3*i)+1] = (((double)(linOct[i].getY())) +0.5)/((double)(1u << maxDepth));
    pts[(3*i)+2] = (((double)(linOct[i].getZ())) +0.5)/((double)(1u << maxDepth));
  }//end for i

  double gSize[3];
  gSize[0] = 1.0;
  gSize[1] = 1.0;
  gSize[2] = 1.0;

  linOct.clear();

  const unsigned int seed = (0x3456782  + (54763*rank));
  srand48(seed);

  //construct the octree
  int numPts = ((pts.size())/3);
  std::vector<double> sources(4*numPts);
  for(int i = 0; i < numPts; ++i) {
    sources[4*i] = pts[3*i];
    sources[(4*i) + 1] = pts[(3*i) + 1];
    sources[(4*i) + 2] = pts[(3*i) + 2];
    sources[(4*i) + 3] = (fMag*(drand48()));
  }//end i
  ot::points2Octree(pts, gSize, linOct, dim, maxDepth, maxNumPts, MPI_COMM_WORLD);
  pts.clear();

  //FGT
  pfgt(linOct, maxDepth, FgtLev, sources, P, L, K, DirectHfactor, MPI_COMM_WORLD);

  PetscFinalize();

}

void genGaussPts(int rank, unsigned int numPtsPerProc, std::vector<double> & pts)
{
  const unsigned int seed = (0x12345678  + (76543*rank));
  srand48(seed);

  pts.resize(3*numPtsPerProc);
  for(unsigned int i = 0; i < (3*numPtsPerProc); i++) {
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



