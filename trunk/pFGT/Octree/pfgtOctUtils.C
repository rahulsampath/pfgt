
#include "pfgtOctUtils.h"
#include <cmath>

#define EXPAND true

#define DIRECT false

PetscErrorCode pfgt(const std::vector<ot::TreeNode> & linOct, unsigned int maxDepth,
    double delta, double fMag, unsigned int numFgtPtsPerProc, 
    int P, int L, int K, int writeOut)
{
  MPI_Comm commAll = MPI_COMM_WORLD;

  const double hRg = sqrt(delta);

  //Mark octants
  std::vector<bool> octFlags(linOct.size());

  double hOctFac = 1.0/static_cast<double>(1u << maxDepth);

  for(unsigned int i = 0; i < octFlags.size(); i++) {
    unsigned int lev = linOct[i].getLevel();
    double hCurrOct = hOctFac*static_cast<double>(1u << (maxDepth - lev));

    if(hCurrOct <= hRg) {
      octFlags[i] = EXPAND;
    } else {
      octFlags[i] = DIRECT;
    }
  }//end for i

}

