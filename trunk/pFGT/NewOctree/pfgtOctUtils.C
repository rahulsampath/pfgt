
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include "mpi.h"
#include "pfgtOctUtils.h"
#include "par/parUtils.h"
#include "par/dtypes.h"

#include "colors.h"

extern PetscLogEvent pfgtMainEvent;
extern PetscLogEvent pfgtSetupEvent;
extern PetscLogEvent pfgtExpandEvent;
extern PetscLogEvent pfgtDirectEvent;

void pfgtMain(std::vector<double>& sources, const unsigned int minPtsInFgt, const unsigned int FgtLev,
    const int P, const int L, const int K, const double epsilon, MPI_Comm comm) {
  PetscLogEventBegin(pfgtMainEvent, 0, 0, 0, 0);

  std::vector<double> expandSources;
  std::vector<double> directSources;
  std::vector<ot::TreeNode> fgtList;

  int npesExpand, avgExpand, extraExpand;
  MPI_Comm subComm = MPI_COMM_NULL;

  bool singleType = false;

  pfgtSetup(expandSources, directSources, fgtList, singleType, npesExpand, avgExpand, 
      extraExpand, subComm, sources, minPtsInFgt, FgtLev, comm);

  int rank;
  MPI_Comm_rank(comm, &rank);

  if(!rank) {
    std::cout << "Expand Num Procs = " << npesExpand <<std::endl;
  }

  if(rank < npesExpand) {
    pfgtExpand(expandSources, fgtList, FgtLev, P, L, K, avgExpand,
        extraExpand, subComm, comm, singleType);
  } else {
    pfgtDirect(directSources, FgtLev, P, L, K, epsilon,
        subComm, comm, singleType);
  }

  if(subComm != MPI_COMM_NULL) {
    MPI_Comm_free(&subComm);
  }

  PetscLogEventEnd(pfgtMainEvent, 0, 0, 0, 0);
}

void pfgtSetup(std::vector<double>& expandSources, std::vector<double>& directSources, std::vector<ot::TreeNode>& fgtList,
    bool & singleType, int & npesExpand, int & avgExpand, int & extraExpand, MPI_Comm & subComm,
    std::vector<double>& sources, const unsigned int minPtsInFgt, const unsigned int FgtLev, MPI_Comm comm) {
  PetscLogEventBegin(pfgtSetupEvent, 0, 0, 0, 0);

  splitSources(expandSources, directSources, fgtList, sources, minPtsInFgt, FgtLev, comm);
  sources.clear();

  int localSizes[3];
  int globalSizes[3];
  localSizes[0] = (expandSources.size())/4;
  localSizes[1] = (directSources.size())/4;
  localSizes[2] = fgtList.size();

  MPI_Allreduce(localSizes, globalSizes, 3, MPI_INT, MPI_SUM, comm);

  int rank;
  MPI_Comm_rank(comm, &rank);

  if(!rank) {
    std::cout<<"Total Number of Expand Pts = "<<(globalSizes[0])<<std::endl;
    std::cout<<"Total Number of Direct Pts = "<<(globalSizes[1])<<std::endl;
    std::cout<<"Total Number of FGT boxes = "<<(globalSizes[2])<<std::endl;
  }

  std::vector<double> tmpExpandSources;
  int srcCnt = 0;
  for(int i = 0; i < fgtList.size(); ++i) {
#ifdef DEBUG
    assert(fgtList[i].getWeight() > 0);
#endif
    {
      tmpExpandSources.push_back(expandSources[srcCnt]);
      tmpExpandSources.push_back(expandSources[srcCnt + 1]);
      tmpExpandSources.push_back(expandSources[srcCnt + 2]);
      tmpExpandSources.push_back(expandSources[srcCnt + 3]);
      tmpExpandSources.push_back(fgtList[i].getWeight());
      srcCnt += 4;
    }
    for(int j = 1; j < fgtList[i].getWeight(); ++j) {
      tmpExpandSources.push_back(expandSources[srcCnt]);
      tmpExpandSources.push_back(expandSources[srcCnt + 1]);
      tmpExpandSources.push_back(expandSources[srcCnt + 2]);
      tmpExpandSources.push_back(expandSources[srcCnt + 3]);
      tmpExpandSources.push_back(0);
      srcCnt += 4;
    }//end j
  }//end i
  swap(expandSources, tmpExpandSources);
  tmpExpandSources.clear();
  fgtList.clear();

  int npes;
  MPI_Comm_size(comm, &npes);

  if(globalSizes[0] == 0) {
    //Only Direct
    singleType = true;
    npesExpand = 0;
    if(!rank) {
      std::cout<<"NOTE: ONLY DIRECT!"<<std::endl;
    }
  } else if(globalSizes[1] == 0) {
    //Only Expand
    singleType = true;
    npesExpand = npes;
    if(!rank) {
      std::cout<<"NOTE: ONLY EXPAND!"<<std::endl;
    }
  } else if(npes == 1) {
    //Serial
    std::cout<<"THIS CASE (Serial + Hybrid) IS NOT SUPPORTED!"<<std::endl;
    assert(false);
  } else {
    //Both Expand and Direct
    //NOTE: The following heuristic may need to be modified!
    singleType = false;
    npesExpand = (globalSizes[0]*npes)/(globalSizes[0] + globalSizes[1]);
#ifdef DEBUG
    assert(npesExpand < npes);
#endif
    if(npesExpand < 1) {
      npesExpand = 1;
    }
  }

  int npesDirect = npes - npesExpand;

  if(singleType) {
    MPI_Comm_dup(comm, &subComm);
  } else {
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
  }

  avgExpand = 0;
  extraExpand = 0;
  if(npesExpand > 0) {
    avgExpand = (globalSizes[0])/npesExpand;
    extraExpand = (globalSizes[0])%npesExpand; 
  }

  int avgDirect = 0;
  int extraDirect = 0;
  if(npesDirect > 0) {
    avgDirect = (globalSizes[1])/npesDirect;
    extraDirect = (globalSizes[1])%npesDirect;
  }

  std::vector<double> finalExpandSources;

  if(npesExpand > 0) {
    if(rank < extraExpand) {
      par::scatterValues<double>(expandSources, finalExpandSources, (5*(avgExpand + 1)), comm);
    } else if(rank < npesExpand) {
      par::scatterValues<double>(expandSources, finalExpandSources, (5*avgExpand), comm);
    } else {
      par::scatterValues<double>(expandSources, finalExpandSources, 0, comm);
    }
  }

  std::vector<double> finalDirectSources;

  if(npesDirect > 0) {
    if(rank < npesExpand) {
      par::scatterValues<double>(directSources, finalDirectSources, 0, comm);
    } else if(rank < (npesExpand + extraDirect)) {
      par::scatterValues<double>(directSources, finalDirectSources, (4*(avgDirect + 1)), comm);
    } else {
      par::scatterValues<double>(directSources, finalDirectSources, (4*avgDirect), comm);
    }
  }

  swap(directSources, finalDirectSources);
  finalDirectSources.clear();

  expandSources.clear();
  for(int i = 0; i < finalExpandSources.size(); i += 5) {
    int flag = static_cast<int>(finalExpandSources[i + 4]);
    if(flag > 0) {
      unsigned int px = static_cast<unsigned int>(finalExpandSources[i]*(__DTPMD__));
      unsigned int py = static_cast<unsigned int>(finalExpandSources[i + 1]*(__DTPMD__));
      unsigned int pz = static_cast<unsigned int>(finalExpandSources[i + 2]*(__DTPMD__));
      ot::TreeNode pt(px, py, pz, __MAX_DEPTH__, __DIM__, __MAX_DEPTH__);
      ot::TreeNode box = pt.getAncestor(FgtLev);
      box.setWeight(flag);
      fgtList.push_back(box);
    }
    expandSources.push_back(finalExpandSources[i]);
    expandSources.push_back(finalExpandSources[i + 1]);
    expandSources.push_back(finalExpandSources[i + 2]);
    expandSources.push_back(finalExpandSources[i + 3]);
  }//end i
  finalExpandSources.clear();

  PetscLogEventEnd(pfgtSetupEvent, 0, 0, 0, 0);
}

void pfgtExpand(std::vector<double> & expandSources, std::vector<ot::TreeNode> & fgtList, const unsigned int FgtLev,
    const int P, const int L, const int K, const int avgExpand, const int extraExpand, 
    MPI_Comm subComm, MPI_Comm comm, bool singleType) {
  PetscLogEventBegin(pfgtExpandEvent, 0, 0, 0, 0);

  int subNpes;
  MPI_Comm_size(subComm, &subNpes);
  int subRank;
  MPI_Comm_rank(subComm, &subRank);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::vector<ot::TreeNode> fgtMins;
  computeFgtMinsExpand(fgtMins, fgtList, subComm, comm);

#ifdef DEBUG
  assert(!(expandSources.empty()));
#endif

  int sumFgtWts = 0;
  for(int i = 0; i < fgtList.size(); ++i) {
    sumFgtWts += fgtList[i].getWeight();
  }//end i

  int numExpandPts = (expandSources.size())/4;

  int numPtsInRemoteFgt = 0;
  if(fgtList.empty()) {
    numPtsInRemoteFgt = numExpandPts;
  } else {
    for( ; numPtsInRemoteFgt < numExpandPts; ++numPtsInRemoteFgt) {
      unsigned int px = static_cast<unsigned int>(expandSources[4*numPtsInRemoteFgt]*(__DTPMD__));
      unsigned int py = static_cast<unsigned int>(expandSources[(4*numPtsInRemoteFgt)+1]*(__DTPMD__));
      unsigned int pz = static_cast<unsigned int>(expandSources[(4*numPtsInRemoteFgt)+2]*(__DTPMD__));
      ot::TreeNode pt(px, py, pz, __MAX_DEPTH__, __DIM__, __MAX_DEPTH__);
      if(pt >= fgtList[0]) {
        break;
      }
    }//end for
  }

  int excessWt = sumFgtWts + numPtsInRemoteFgt - numExpandPts;
#ifdef DEBUG
  assert(excessWt >= 0);
#endif
  if(!(fgtList.empty())) {
    int lastWt = fgtList[fgtList.size() - 1].getWeight();
#ifdef DEBUG
    assert(lastWt > excessWt);
#endif
    fgtList[fgtList.size() - 1].setWeight(lastWt - excessWt);
  }

  int remoteFgtOwner = -1;
  ot::TreeNode remoteFgt;
  if(numPtsInRemoteFgt > 0) {
    computeRemoteFgt(remoteFgt, remoteFgtOwner, FgtLev, expandSources, fgtMins);
  }
#ifdef DEBUG
  assert(remoteFgtOwner < subRank);
#endif

  //Complex coefficients: [-P, P]x[-P, P]x[0, P] 
  //Coeff[-K1, -K2, -K3] = ComplexConjugate(Coeff[K1, K2, K3])
  const unsigned int TwoPplus1 = (2*P) + 1;
  const unsigned int numWcoeffs = 2*TwoPplus1*TwoPplus1*(P + 1);

  int* s2wSendCnts = NULL;
  int* s2wSendDisps = NULL;
  int* s2wRecvCnts = NULL;
  int* s2wRecvDisps = NULL;

  createS2WcommInfo(s2wSendCnts, s2wSendDisps, s2wRecvCnts, s2wRecvDisps, 
      remoteFgtOwner, numWcoeffs, excessWt, avgExpand, extraExpand, subComm);

  std::vector<double> localWlist( (numWcoeffs*(fgtList.size())), 0.0);
  s2w(localWlist, expandSources, remoteFgt, remoteFgtOwner, numPtsInRemoteFgt, fgtList,
      fgtMins, FgtLev, P, L, s2wSendCnts, s2wSendDisps, s2wRecvCnts, s2wRecvDisps, subComm);

  std::vector<double> localLlist( (localWlist.size()), 0.0);
  w2l(localLlist, localWlist, fgtList, fgtMins, FgtLev, P, L, K, subComm);

  if(!singleType) {
    w2dAndD2lExpand(localLlist, localWlist, fgtList, P, comm);
  }

  std::vector<double> results(((expandSources.size())/4), 0.0);
  l2t(results, localLlist, expandSources, remoteFgt, remoteFgtOwner, numPtsInRemoteFgt, 
      fgtList, fgtMins, FgtLev, P, L, s2wSendCnts, s2wSendDisps, s2wRecvCnts, s2wRecvDisps, subComm);

  destroyS2WcommInfo(s2wSendCnts, s2wSendDisps, s2wRecvCnts, s2wRecvDisps); 

#ifdef _WRITE_SOLN
  std::cout << rank << GRN" : Expand - writing "NRM << subRank << "/" << subNpes << std::endl; 
  char fname[256];
  sprintf(fname, "expand.%d.res", rank);
  std::ofstream out(fname, std::ios::binary);
  out.write((const char*)&(*(results.begin())),results.size()*sizeof(double)); 
  out.close();
#endif

  PetscLogEventEnd(pfgtExpandEvent, 0, 0, 0, 0);
}

void pfgtDirect(std::vector<double> & directSources, const unsigned int FgtLev, const int P, const int L,
    const int K, const double epsilon, MPI_Comm subComm, MPI_Comm comm, bool singleType) {
  PetscLogEventBegin(pfgtDirectEvent, 0, 0, 0, 0);

  int subNpes;
  MPI_Comm_size(subComm, &subNpes);
  int subRank;
  MPI_Comm_rank(subComm, &subRank);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef DEBUG
  assert(!(directSources.empty()));
#endif

  std::vector<ot::TreeNode> directNodes( directSources.size()/4 );
  for(size_t i = 0; i < directNodes.size(); ++i) {
    unsigned int px = static_cast<unsigned int>(directSources[(4*i) + 0]*(__DTPMD__));
    unsigned int py = static_cast<unsigned int>(directSources[(4*i) + 1]*(__DTPMD__));
    unsigned int pz = static_cast<unsigned int>(directSources[(4*i) + 2]*(__DTPMD__));
    directNodes[i] = ot::TreeNode(px, py, pz, __MAX_DEPTH__, __DIM__, __MAX_DEPTH__);
  }//end i

  std::vector<ot::TreeNode> directMins(subNpes);
  MPI_Allgather(&(directNodes[0]), 1, par::Mpi_datatype<ot::TreeNode>::value(),
      &(directMins[0]), 1, par::Mpi_datatype<ot::TreeNode>::value(), subComm);

  std::vector<ot::TreeNode> fgtMins;
  if(!singleType) {
    computeFgtMinsDirect(fgtMins, comm);
  }

  std::vector<double> results(directNodes.size(), 0.0);
  d2d(results, directSources, directNodes, directMins, FgtLev, epsilon, subComm);

  if(!singleType) {
    w2dAndD2lDirect(results, directSources, fgtMins, FgtLev, P, L, K, epsilon, comm);
  }

#ifdef _WRITE_SOLN
  std::cout << rank << RED" : Direct - writing "NRM << subRank << "/" << subNpes << std::endl; 
  char fname[256];
  sprintf(fname, "direct.%d.res", rank);
  std::ofstream out(fname, std::ios::binary);
  out.write((const char*)&(*(results.begin())),results.size()*sizeof(double)); 
  out.close();
#endif

  PetscLogEventEnd(pfgtDirectEvent, 0, 0, 0, 0);
}

void createS2WcommInfo(int*& sendCnts, int*& sendDisps, int*& recvCnts, int*& recvDisps, 
    const int remoteFgtOwner, const unsigned int numWcoeffs, const int excessWt,
    const int avgExpand, const int extraExpand, MPI_Comm subComm) {
  int npes;
  MPI_Comm_size(subComm, &npes);

  int rank;
  MPI_Comm_rank(subComm, &rank);

  sendCnts = new int[npes];
  recvCnts = new int[npes]; 
  sendDisps = new int[npes];
  recvDisps = new int[npes]; 

  for(int i = 0; i < npes; ++i) {
    sendCnts[i] = 0;
    recvCnts[i] = 0;
  }//end i

  if(remoteFgtOwner >= 0) {
    sendCnts[remoteFgtOwner] = numWcoeffs;
  }

  for(int i = (rank + 1), leftOver = excessWt; i < npes; ++i) {
    if(leftOver > 0) {
      recvCnts[i] = numWcoeffs;
      if(i < extraExpand) {
        leftOver = leftOver - (avgExpand + 1);
      } else {
        leftOver = leftOver - avgExpand;
      }
    } else {
      break;
    }
  }//end i

  sendDisps[0] = 0;
  recvDisps[0] = 0;
  for(int i = 1; i < npes; ++i) {
    sendDisps[i] = sendDisps[i - 1] + sendCnts[i - 1];
    recvDisps[i] = recvDisps[i - 1] + recvCnts[i - 1];
  }//end i
}

void destroyS2WcommInfo(int* sendCnts, int* sendDisps, int* recvCnts, int* recvDisps) {
  delete [] sendCnts;
  delete [] sendDisps;
  delete [] recvCnts;
  delete [] recvDisps;
}

void computeRemoteFgt(ot::TreeNode & remoteFgt, int & remoteFgtOwner, const unsigned int FgtLev,
    std::vector<double> & sources, std::vector<ot::TreeNode> & fgtMins) {
  unsigned int px = static_cast<unsigned int>(sources[0]*(__DTPMD__));
  unsigned int py = static_cast<unsigned int>(sources[1]*(__DTPMD__));
  unsigned int pz = static_cast<unsigned int>(sources[2]*(__DTPMD__));
  ot::TreeNode ptOct(px, py, pz, __MAX_DEPTH__, __DIM__, __MAX_DEPTH__);
  remoteFgt = ptOct.getAncestor(FgtLev);
  unsigned int retIdx;
  seq::maxLowerBound(fgtMins, remoteFgt, retIdx, NULL, NULL);
  remoteFgtOwner = fgtMins[retIdx].getWeight();
}

void computeFgtMinsExpand(std::vector<ot::TreeNode> & fgtMins, std::vector<ot::TreeNode> & fgtList,
    MPI_Comm subComm, MPI_Comm comm) {
  int subNpes;
  MPI_Comm_size(subComm, &subNpes);

  int subRank;
  MPI_Comm_rank(subComm, &subRank);

  int rank;
  MPI_Comm_rank(comm, &rank);

#ifdef DEBUG
  assert(rank == subRank);
#endif

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
    for(unsigned int i = 0; i < fgtMins.size(); ++i) {
      if(fgtMins[i].getDim()) {
        fgtMins[i].setWeight(i);
        tmpMins.push_back(fgtMins[i]);
      }
    }//end i
    swap(fgtMins, tmpMins);
    fgtMinSize = fgtMins.size();
  }

  MPI_Bcast(&fgtMinSize, 1, MPI_INT, 0, comm);

  if(rank) {
    fgtMins.resize(fgtMinSize);
  }

  MPI_Bcast(&(fgtMins[0]), fgtMinSize, par::Mpi_datatype<ot::TreeNode>::value(), 0, comm);
}

void computeFgtMinsDirect(std::vector<ot::TreeNode> & fgtMins, MPI_Comm comm) {
  int fgtMinSize;
  MPI_Bcast(&fgtMinSize, 1, MPI_INT, 0, comm);

  fgtMins.resize(fgtMinSize);
  MPI_Bcast(&(fgtMins[0]), fgtMinSize, par::Mpi_datatype<ot::TreeNode>::value(), 0, comm);
}



