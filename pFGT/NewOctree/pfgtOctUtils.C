
#include <iostream>
#include <cmath>
#include "mpi.h"
#include "pfgtOctUtils.h"
#include "parUtils.h"

extern PetscLogEvent fgtEvent;
extern PetscLogEvent expandEvent;
extern PetscLogEvent directEvent;

void pfgt(std::vector<ot::TreeNode> & linOct, const unsigned int maxDepth,
    const unsigned int NforDelta, const double fMag, const unsigned int ptGridSizeWithinBox, 
    const int P, const int L, const int K, const double DirectHfactor, MPI_Comm comm) {
  PetscLogEventBegin(fgtEvent, 0, 0, 0, 0);

  int npes, rank;
  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &rank);

  //Kernel Bandwidth
  double delta = 1.0/(static_cast<double>(1u << ((maxDepth - NforDelta) << 1)));

  //FGT box size = sqrt(delta)
  const double hFgt = 1.0/(static_cast<double>(1u << (maxDepth - NforDelta)));

  const double hOctFac = 1.0/static_cast<double>(1u << maxDepth);

  //2P complex coefficients for each dimension.  
  const unsigned int Ndofs = 16*P*P*P;

  if(!rank) {
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
  MPI_Allreduce(localTreeSizes, globalTreeSizes, 2, MPI_UNSIGNED, MPI_SUM, comm);
  int npesExpand = (globalTreeSizes[0]*npes)/(globalTreeSizes[0] + globalTreeSizes[1]);
  int npesDirect = npes - npesExpand;

  MPI_Comm subComm;
  MPI_Group group, subGroup;
  MPI_Comm_group(comm, &group);
  if(rank < npesExpand) {
    int* list = new int[npesExpand];
    for(int i = 0; i < npesExpand; i++) {
      list[i] = i;
    }//end for i
    MPI_Group_incl(group, npesExpand, list, &subGroup);
    delete [] list;
  } else {
    int* list = new int[npesDirect];
    for(int i = 0; i < npesDirect; i++) {
      list[i] = npesExpand + i;
    }//end for i
    MPI_Group_incl(group, npesDirect, list, &subGroup);
    delete [] list;
  }
  MPI_Group_free(&group);
  MPI_Comm_create(comm, subGroup, &subComm);
  MPI_Group_free(&subGroup);

  std::vector<ot::TreeNode> finalExpandTree;
  std::vector<ot::TreeNode> finalDirectTree;
  if(npesDirect == 0) {
    finalExpandTree = expandTree;
  } else if(npesExpand == 0) {
    finalDirectTree = directTree;
  } else {
    int avgExpand = (globalTreeSizes[0])/npesExpand;
    int extraExpand = (globalTreeSizes[0])%npesExpand; 
    int avgDirect = (globalTreeSizes[1])/npesDirect;
    int extraDirect = (globalTreeSizes[1])%npesDirect;
    if(rank < extraExpand) {
      par::scatterValues<ot::TreeNode>(expandTree, finalExpandTree, (avgExpand + 1), comm);
      par::scatterValues<ot::TreeNode>(directTree, finalDirectTree, 0, comm);
    } else if(rank < npesExpand) {
      par::scatterValues<ot::TreeNode>(expandTree, finalExpandTree, avgExpand, comm);
      par::scatterValues<ot::TreeNode>(directTree, finalDirectTree, 0, comm);
    } else if(rank < (npesExpand + extraDirect)) {
      par::scatterValues<ot::TreeNode>(expandTree, finalExpandTree, 0, comm);
      par::scatterValues<ot::TreeNode>(directTree, finalDirectTree, (avgDirect + 1), comm);
    } else {
      par::scatterValues<ot::TreeNode>(expandTree, finalExpandTree, 0, comm);
      par::scatterValues<ot::TreeNode>(directTree, finalDirectTree, avgDirect, comm);
    }
  }
  expandTree.clear();
  directTree.clear();

  if(rank < npesExpand) {
    pfgtExpand(finalExpandTree, subComm, comm);
  } else {
    pfgtDirect(finalDirectTree, subComm, comm);
  }

  MPI_Comm_free(&subComm);

  PetscLogEventEnd(fgtEvent, 0, 0, 0, 0);
}

void pfgtExpand(std::vector<ot::TreeNode> & expandTree, MPI_Comm subComm, MPI_Comm comm) {
  PetscLogEventBegin(expandEvent, 0, 0, 0, 0);

  PetscLogEventEnd(expandEvent, 0, 0, 0, 0);
}

void pfgtDirect(std::vector<ot::TreeNode> & directTree, MPI_Comm subComm, MPI_Comm comm) {
  PetscLogEventBegin(directEvent, 0, 0, 0, 0);

  PetscLogEventEnd(directEvent, 0, 0, 0, 0);
}


