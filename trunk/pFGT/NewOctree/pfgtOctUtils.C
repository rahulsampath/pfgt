
#include "mpi.h"
#include "pfgtOctUtils.h"

extern PetscLogEvent fgtEvent;

void pfgt(std::vector<ot::TreeNode> & linOct, const unsigned int maxDepth,
    const double delta, const double fMag, const unsigned int ptGridSizeWithinBox, 
    const int P, const int L, const int K, const int DirectHfactor, MPI_Comm commAll) {
  PetscLogEventBegin(fgtEvent, 0, 0, 0, 0);

  PetscLogEventEnd(fgtEvent, 0, 0, 0, 0);
}



