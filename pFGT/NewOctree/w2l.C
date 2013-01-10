
#include "petsc.h"
#include <vector>
#include "oct/TreeNode.h"
#include "pfgtOctUtils.h"
#include "par/parUtils.h"
#include "par/dtypes.h"

extern PetscLogEvent w2lEvent;
extern PetscLogEvent w2lSearchEvent;
extern PetscLogEvent w2lSortEvent;
extern PetscLogEvent w2lGenEvent;
extern PetscLogEvent w2lCoreEvent;

void w2l(std::vector<double> & localLlist, std::vector<double> & localWlist, 
    std::vector<ot::TreeNode> & fgtList, std::vector<ot::TreeNode> & fgtMins,
    const unsigned int FgtLev, const int P, const int L, const unsigned long long int K, MPI_Comm subComm) {
  PetscLogEventBegin(w2lEvent, 0, 0, 0, 0);

  int npes;
  MPI_Comm_size(subComm, &npes);

  const unsigned long long int cellsPerFgt = (1ull << (__MAX_DEPTH__ - FgtLev));

  PetscLogEventBegin(w2lGenEvent, 0, 0, 0, 0);
  PetscLogEventEnd(w2lGenEvent, 0, 0, 0, 0);

  PetscLogEventEnd(w2lEvent, 0, 0, 0, 0);
}


