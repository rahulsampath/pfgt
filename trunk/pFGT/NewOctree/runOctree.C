
#include "mpi.h"
#include "petsc.h"
#include "sys/sys.h"
#include <cmath>
#include <cassert>
#include "par/parUtils.h"
#include "oct/octUtils.h"
#include "oct/TreeNode.h"
#include <cstdlib>
#include <vector>
#include "externVars.h"
#include "dendro.h"

#include "pfgtOctUtils.h"

PetscCookie fgtCookie;
PetscLogEvent pfgtMainEvent;
PetscLogEvent pfgtSetupEvent;
PetscLogEvent pfgtExpandEvent;
PetscLogEvent pfgtDirectEvent;
PetscLogEvent splitSourcesEvent;
PetscLogEvent s2wEvent;
PetscLogEvent l2tEvent;
PetscLogEvent w2lEvent;
PetscLogEvent d2dEvent;
PetscLogEvent w2dD2lExpandEvent;
PetscLogEvent w2dD2lDirectEvent;

bool softEquals(double a, double b) {
  return ((fabs(a - b)) < 1.0e-14);
}

void genGaussPts(int rank, unsigned int numOctPtsPerProc, std::vector<double> & pts);

void rescalePts(std::vector<double> & pts);

double gaussian(double mean, double std_deviation);

int main(int argc, char** argv) {
  PetscInitialize(&argc, &argv, NULL, NULL);
  ot::RegisterEvents();

  PetscCookieRegister("Fgt", &fgtCookie);
  PetscLogEventRegister("FgtMain", fgtCookie, &pfgtMainEvent);
  PetscLogEventRegister("FgtSetup", fgtCookie, &pfgtSetupEvent);
  PetscLogEventRegister("FgtExpand", fgtCookie, &pfgtExpandEvent);
  PetscLogEventRegister("FgtDirect", fgtCookie, &pfgtDirectEvent);
  PetscLogEventRegister("SplitSrc", fgtCookie, &splitSourcesEvent);
  PetscLogEventRegister("S2W", fgtCookie, &s2wEvent);
  PetscLogEventRegister("L2T", fgtCookie, &l2tEvent);
  PetscLogEventRegister("W2L", fgtCookie, &w2lEvent);
  PetscLogEventRegister("D2D", fgtCookie, &d2dEvent);
  PetscLogEventRegister("W2D2LE", fgtCookie, &w2dD2lExpandEvent);
  PetscLogEventRegister("W2D2LD", fgtCookie, &w2dD2lDirectEvent);

  int npes, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &npes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(argc < 6) {
    if(!rank) {
      std::cout<<"Usage: exe numPtsPerProc fMag epsilon FgtLev minPtsInFgt"<<std::endl;
    }
    PetscFinalize();
  }

  unsigned int numPtsPerProc = atoi(argv[1]);
  double fMag = atof(argv[2]);
  double epsilon = atof(argv[3]);  
  unsigned int FgtLev = atoi(argv[4]);
  unsigned int minPtsInFgt = atoi(argv[5]);

  //Fgt box size = sqrt(delta)
  const double hFgt = 1.0/(static_cast<double>(1u << FgtLev));

  //Kernel Bandwidth
  const double delta = hFgt*hFgt;

  assert(FgtLev <= __MAX_DEPTH__);

  if(!rank) {
    std::cout<<"numPtsPerProc = "<<numPtsPerProc<<std::endl;
    std::cout<<"fMag = "<<fMag<<std::endl;
    std::cout<<"epsilon = "<<epsilon<<std::endl;
    std::cout<<"FgtLev = "<<FgtLev<<std::endl;
    std::cout<<"minPtsInFgt = "<<minPtsInFgt<<std::endl;
    std::cout<<"delta = "<<delta<<std::endl;
  }

  int P, L;

  if(softEquals(epsilon, 1.0e-3)) {
    P = 6;
    L = 5;
  } else if(softEquals(epsilon, 1.0e-6)) {
    P = 10;
    L = 7;
  } else if(softEquals(epsilon, 1.0e-9)) {
    P = 16;
    L = 10;
  } else if(softEquals(epsilon, 1.0e-12)) {
    P = 20;
    L = 11;
  } else {
    if(!rank) {
      std::cout<<"Wrong epsilon!"<<std::endl;
    }
    assert(false);
    PetscFinalize();
  }

  int K = ceil(sqrt(-log(epsilon)) - 0.5);

  if(!rank) {
    std::cout<<"P = "<<P<<std::endl;
    std::cout<<"L = "<<L<<std::endl;
    std::cout<<"K = "<<K<<std::endl;
  }

  //Generate gaussian distribution of points in (0, 1)
  std::vector<double> pts;
  genGaussPts(rank, numPtsPerProc, pts);
  rescalePts(pts);

  std::vector<ot::TreeNode> linOct;
  for(unsigned int i = 0; i < pts.size(); i += 3) {
    unsigned int px = static_cast<unsigned int>(pts[i]*(__DTPMD__));
    unsigned int py = static_cast<unsigned int>(pts[i + 1]*(__DTPMD__));
    unsigned int pz = static_cast<unsigned int>(pts[i + 2]*(__DTPMD__));
    linOct.push_back( ot::TreeNode(px, py, pz, __MAX_DEPTH__, __DIM__, __MAX_DEPTH__) );
  }//end for i

  //Sort and Remove Duplicates
  par::removeDuplicates<ot::TreeNode>(linOct, false, MPI_COMM_WORLD);

  pts.resize(3*(linOct.size()));
  for(size_t i = 0; i < linOct.size(); i++) {
    pts[3*i] = (((double)(linOct[i].getX())) + 0.5)/(__DTPMD__);
    pts[(3*i)+1] = (((double)(linOct[i].getY())) +0.5)/(__DTPMD__);
    pts[(3*i)+2] = (((double)(linOct[i].getZ())) +0.5)/(__DTPMD__);
  }//end for i
  linOct.clear();

  const unsigned int seed = (0x3456782  + (54763*rank));
  srand48(seed);

  //Generate Sources
  int numPts = ((pts.size())/3);
  std::vector<double> sources(4*numPts);
  for(int i = 0; i < numPts; ++i) {
    sources[4*i] = pts[3*i];
    sources[(4*i) + 1] = pts[(3*i) + 1];
    sources[(4*i) + 2] = pts[(3*i) + 2];
    sources[(4*i) + 3] = (fMag*(drand48()));
  }//end i
  pts.clear();

  //Fgt
  pfgtMain(sources, minPtsInFgt, FgtLev, P, L, K, epsilon, MPI_COMM_WORLD);

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



