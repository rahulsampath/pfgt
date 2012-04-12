
#include <iostream>
#include <cmath>
#include "mpi.h"
#include "pfgtOctUtils.h"
#include "parUtils.h"
#include "dtypes.h"

extern PetscLogEvent fgtEvent;
extern PetscLogEvent serialEvent;
extern PetscLogEvent fgtOctConEvent;
extern PetscLogEvent expandOnlyEvent;
extern PetscLogEvent expandHybridEvent;
extern PetscLogEvent directOnlyEvent;
extern PetscLogEvent directHybridEvent;

void pfgt(std::vector<ot::TreeNode> & linOct, const unsigned int maxDepth,
    const unsigned int FgtLev, std::vector<double> & sources,  
    const int P, const int L, const int K, const double DirectHfactor, MPI_Comm comm) {
  PetscLogEventBegin(fgtEvent, 0, 0, 0, 0);

  int npes, rank;
  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &rank);

  //Kernel Bandwidth
  const double delta = 1.0/(static_cast<double>(1u << (FgtLev << 1)));

  //FGT box size = sqrt(delta)
  const double hFgt = 1.0/(static_cast<double>(1u << FgtLev));

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
    double hCurrOct = 1.0/(static_cast<double>(1u << lev));
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

  if(globalTreeSizes[0] == 0) {
    pfgtOnlyDirect(sources, directTree, comm);
  } else if(globalTreeSizes[1] == 0) {
    pfgtOnlyExpand(sources, expandTree, maxDepth, FgtLev, comm);
  } else if(npes == 1) {
    pfgtSerial(sources, directTree, expandTree, maxDepth, FgtLev);
  } else {
    int npesExpand = (globalTreeSizes[0]*npes)/(globalTreeSizes[0] + globalTreeSizes[1]);
    assert(npesExpand < npes);
    if(npesExpand < 1) {
      npesExpand = 1;
    }
    int npesDirect = npes - npesExpand;
    assert(npesDirect > 0);

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

    int avgExpand = (globalTreeSizes[0])/npesExpand;
    int extraExpand = (globalTreeSizes[0])%npesExpand; 
    int avgDirect = (globalTreeSizes[1])/npesDirect;
    int extraDirect = (globalTreeSizes[1])%npesDirect;

    std::vector<ot::TreeNode> finalExpandTree;
    std::vector<ot::TreeNode> finalDirectTree;

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

    expandTree.clear();
    directTree.clear();

    if(rank < npesExpand) {
      pfgtHybridExpand(sources, finalExpandTree, maxDepth, FgtLev, delta, hFgt, subComm, comm);
    } else {
      pfgtHybridDirect(sources, finalDirectTree, FgtLev, subComm, comm);
    }

    MPI_Comm_free(&subComm);
  }

  PetscLogEventEnd(fgtEvent, 0, 0, 0, 0);
}

void pfgtOnlyDirect(std::vector<double> & sources, std::vector<ot::TreeNode> & directTree, MPI_Comm comm) {
  PetscLogEventBegin(directOnlyEvent, 0, 0, 0, 0);

  PetscLogEventEnd(directOnlyEvent, 0, 0, 0, 0);
}

void pfgtOnlyExpand(std::vector<double> & sources, std::vector<ot::TreeNode> & expandTree, 
    const unsigned int maxDepth, const unsigned int FgtLev, MPI_Comm comm) {
  PetscLogEventBegin(expandOnlyEvent, 0, 0, 0, 0);

  PetscLogEventEnd(expandOnlyEvent, 0, 0, 0, 0);
}

void pfgtSerial(std::vector<double> & sources, std::vector<ot::TreeNode> & directTree, 
    std::vector<ot::TreeNode> & expandTree, const unsigned int maxDepth, const unsigned int FgtLev) {
  PetscLogEventBegin(serialEvent, 0, 0, 0, 0);

  PetscLogEventEnd(serialEvent, 0, 0, 0, 0);
}

void pfgtHybridExpand(std::vector<double> & sources, std::vector<ot::TreeNode> & expandTree,
    const unsigned int maxDepth, const unsigned int FgtLev, const double delta, 
    const double hFgt, MPI_Comm subComm, MPI_Comm comm) {
  PetscLogEventBegin(expandHybridEvent, 0, 0, 0, 0);

  assert(!(expandTree.empty()));

  std::vector<ot::TreeNode> fgtList;
  createFGToctree(fgtList, expandTree, FgtLev, subComm);

  std::vector<ot::TreeNode> fgtMins;
  computeFGTminsHybridExpand(fgtMins, fgtList, subComm, comm);

  PetscLogEventEnd(expandHybridEvent, 0, 0, 0, 0);
}

void pfgtHybridDirect(std::vector<double> & sources, std::vector<ot::TreeNode> & directTree, 
    const unsigned int FgtLev, MPI_Comm subComm, MPI_Comm comm) {
  PetscLogEventBegin(directHybridEvent, 0, 0, 0, 0);

  std::vector<ot::TreeNode> fgtMins;
  computeFGTminsHybridDirect(fgtMins, comm);

  PetscLogEventEnd(directHybridEvent, 0, 0, 0, 0);
}

void computeFGTminsHybridExpand(std::vector<ot::TreeNode> & fgtMins, std::vector<ot::TreeNode> & fgtList,
    MPI_Comm subComm, MPI_Comm comm) {
  int subNpes;
  MPI_Comm_size(subComm, &subNpes);

  int subRank;
  MPI_Comm_rank(subComm, &subRank);

  int rank;
  MPI_Comm_rank(comm, &rank);

  assert(rank == subRank);

  ot::TreeNode firstFgt;
  if(!(fgtList.empty())) {
    firstFgt = fgtList[0];
  }

  ot::TreeNode* recvBuf = NULL;
  if(rank == 0) {
    fgtMins.resize(subNpes);
    recvBuf = &(fgtMins[0]);
  }

  MPI_Gather(&firstFgt, 1, par::Mpi_datatype<ot::TreeNode>::value(),
      recvBuf, 1, par::Mpi_datatype<ot::TreeNode>::value(), 0, subComm);

  int fgtMinSize;
  if(rank == 0) {
    std::vector<ot::TreeNode> tmpMins;
    for(int i = 0; i < fgtMins.size(); ++i) {
      if(fgtMins[i].getDim()) {
        fgtMins[i].setWeight(i);
        tmpMins.push_back(fgtMins[i]);
      }
    }//end i
    fgtMins = tmpMins;
    fgtMinSize = fgtMins.size();
  }

  MPI_Bcast(&fgtMinSize, 1, MPI_INT, 0, comm);

  if(rank) {
    fgtMins.resize(fgtMinSize);
  }

  MPI_Bcast(&(fgtMins[0]), fgtMinSize, par::Mpi_datatype<ot::TreeNode>::value(), 0, comm);
}

void computeFGTminsHybridDirect(std::vector<ot::TreeNode> & fgtMins, MPI_Comm comm) {
  int fgtMinSize;
  MPI_Bcast(&fgtMinSize, 1, MPI_INT, 0, comm);

  fgtMins.resize(fgtMinSize);
  MPI_Bcast(&(fgtMins[0]), fgtMinSize, par::Mpi_datatype<ot::TreeNode>::value(), 0, comm);
}

void createFGToctree(std::vector<ot::TreeNode> & fgtList, std::vector<ot::TreeNode> & expandTree,
    const unsigned int FgtLev, MPI_Comm subComm) {
  PetscLogEventBegin(fgtOctConEvent, 0, 0, 0, 0);

  std::vector<ot::TreeNode> tmpFgtListA;
  std::vector<ot::TreeNode> tmpFgtListB;
  for(size_t i = 0; i < expandTree.size(); ++i) {
    unsigned int lev = expandTree[i].getLevel();
    if(lev > FgtLev) {
      tmpFgtListA.push_back(expandTree[i].getAncestor(FgtLev));
    } else {
      tmpFgtListB.push_back(expandTree[i]);
    }
  }//end i

  seq::makeVectorUnique<ot::TreeNode>(tmpFgtListA, true);

  int aIdx = 0;
  int bIdx = 0;
  while( (aIdx < tmpFgtListA.size()) && (bIdx < tmpFgtListB.size()) ) {
    if(tmpFgtListA[aIdx] < tmpFgtListB[bIdx]) {
      fgtList.push_back(tmpFgtListA[aIdx]);
      ++aIdx;
    } else {
      fgtList.push_back(tmpFgtListB[bIdx]);
      ++bIdx;
    }
  }
  for(; aIdx < tmpFgtListA.size(); ++aIdx) {
    fgtList.push_back(tmpFgtListA[aIdx]);
  }
  for(; bIdx < tmpFgtListB.size(); ++bIdx) {
    fgtList.push_back(tmpFgtListB[bIdx]);
  }

  assert(!(fgtList.empty()));

  int rank, npes;
  MPI_Comm_rank(subComm, &rank);
  MPI_Comm_size(subComm, &npes);

  ot::TreeNode tmpOct;

  MPI_Request recvRequest, sendRequest;
  if(rank > 0) {
    MPI_Irecv(&tmpOct, 1, par::Mpi_datatype<ot::TreeNode>::value(),
        (rank - 1), 1, subComm, &recvRequest);
  }

  if(rank < (npes - 1)) {
    MPI_Isend(&(fgtList[fgtList.size() - 1]), 1, par::Mpi_datatype<ot::TreeNode>::value(),
        (rank + 1), 1, subComm, &sendRequest);
  }

  if(rank > 0) {
    MPI_Status status;
    MPI_Wait(&recvRequest, &status);
  }

  if(rank < (npes - 1)) {
    MPI_Status status;
    MPI_Wait(&sendRequest, &status);
  }

  if(rank > 0) {
    if(tmpOct == fgtList[0]) {
      fgtList.erase(fgtList.begin());
    }
  }

  PetscLogEventEnd(fgtOctConEvent, 0, 0, 0, 0);
}


