
#include <iostream>
#include <cmath>
#include "mpi.h"
#include "pfgtOctUtils.h"

extern PetscLogEvent fgtEvent;

void pfgt(std::vector<ot::TreeNode> & linOct, const unsigned int maxDepth,
    const unsigned int NforDelta, const double fMag, const unsigned int ptGridSizeWithinBox, 
    const int P, const int L, const int K, const double DirectHfactor, MPI_Comm commAll) {
  PetscLogEventBegin(fgtEvent, 0, 0, 0, 0);

  int npesAll, rankAll;
  MPI_Comm_size(commAll, &npesAll);
  MPI_Comm_rank(commAll, &rankAll);

  //Kernel Bandwidth
  double delta = 1.0/(static_cast<double>(1u << ((maxDepth - NforDelta) << 1)));

  //FGT box size = sqrt(delta)
  const double hFgt = 1.0/(static_cast<double>(1u << (maxDepth - NforDelta)));

  const double hOctFac = 1.0/static_cast<double>(1u << maxDepth);

  //2P complex coefficients for each dimension.  
  const unsigned int Ndofs = 16*P*P*P;

  if(!rankAll) {
    std::cout<<"delta = "<<delta<<std::endl;
    std::cout<<"Ndofs = "<< Ndofs <<std::endl;
    std::cout<<"StencilWidth = "<< K <<std::endl;
  }

  //Split octree into 2 sets
  std::vector<ot::TreeNode> expandTree;
  std::vector<ot::TreeNode> directTree;

  for(size_t i = 0; i < linOct.size(); i++) {
    unsigned int lev = linOct[i].getLevel();
    double hCurrOct = hOctFac*static_cast<double>(1u << (maxDepth - lev));
    if( hCurrOct <= (hFgt*DirectHfactor) ) {
      expandTree.push_back(linOct[i]);
    } else {
      directTree.push_back(linOct[i]);
    }
  }//end for i
  linOct.clear();

  unsigned int localTreeSizes[2];
  unsigned int globalTreeSizes[2];
  localTreeSizes[0] = expandTree.size();
  localTreeSizes[1] = directTree.size();
  MPI_Allreduce(localTreeSizes, globalTreeSizes, 2, MPI_UNSIGNED, MPI_SUM, commAll);
  int npesExpand = (globalTreeSizes[0]*npesAll)/(globalTreeSizes[0] + globalTreeSizes[1]);
  int npesDirect = npesAll - npesExpand;

  MPI_Comm subComm;
  MPI_Group groupAll, subGroup;
  MPI_Comm_group(commAll, &groupAll);
  if(rankAll < npesExpand) {
    int* list = new int[npesExpand];
    for(int i = 0; i < npesExpand; i++) {
      list[i] = i;
    }//end for i
    MPI_Group_incl(groupAll, npesExpand, list, &subGroup);
    delete [] list;
  } else {
    int* list = new int[npesDirect];
    for(int i = 0; i < npesDirect; i++) {
      list[i] = npesExpand + i;
    }//end for i
    MPI_Group_incl(groupAll, npesDirect, list, &subGroup);
    delete [] list;
  }
  MPI_Group_free(&groupAll);
  MPI_Comm_create(commAll, subGroup, &subComm);
  MPI_Group_free(&subGroup);



  MPI_Comm_free(&subComm);

  PetscLogEventEnd(fgtEvent, 0, 0, 0, 0);
}



