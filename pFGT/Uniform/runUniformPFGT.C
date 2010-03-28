
#include "mpi.h"
#include "petsc.h"
#include "petscsys.h"
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
#include "pfgtUtils.h"

#ifdef MPI_WTIME_IS_GLOBAL
#undef MPI_WTIME_IS_GLOBAL
#endif

PetscLogEvent type1Event;
PetscLogEvent type2Event;
PetscLogEvent s2wEvent;
PetscLogEvent w2lEvent;
PetscLogEvent l2tEvent;
PetscCookie fgtCookie;

#define SEEDA 0x12345678 

#define SEEDB 76543

int main(int argc, char ** argv ) {	

  PetscInitialize(&argc, &argv, NULL, NULL);

  PetscCookieRegister("FGT", &fgtCookie);
  PetscLogEventRegister("FGTtype1", fgtCookie, &type1Event);
  PetscLogEventRegister("FGTtype2", fgtCookie, &type2Event);
  PetscLogEventRegister("S2W", fgtCookie, &s2wEvent);
  PetscLogEventRegister("W2L", fgtCookie, &w2lEvent);
  PetscLogEventRegister("L2T", fgtCookie, &l2tEvent);

  int npes, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &npes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(argc < 6) {
    if(!rank) {
      std::cout<<"Usage: exe numPtsPerProc fMag epsilon delta writeOut"<<std::endl;
    }
    PetscFinalize();
  }

  const unsigned int seed = (SEEDA + (SEEDB*rank));
  srand48(seed);

  unsigned int numPtsPerProc = atoi(argv[1]);
  double fMag = atof(argv[2]);
  double epsilon = atof(argv[3]);  
  double delta = atof(argv[4]);
  int writeOut = atoi(argv[5]);

  int P;
  int L;
  int K;

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

  if( (static_cast<double>(npes)*static_cast<double>(numPtsPerProc)*sqrt(delta*log(1.0/epsilon))) <
      (4.0*static_cast<double>(P*P*P)) ) {
    if(!rank) {
      std::cout<<"Using Type-1"<<std::endl;
    }
    pfgtType1(delta, K, fMag, numPtsPerProc, writeOut);
  } else {
    if(!rank) {
      std::cout<<"Using Type-2"<<std::endl;
    }
    pfgtType2(delta, fMag, numPtsPerProc, P, L, K, writeOut);
  }

  PetscFinalize();

}//end main



