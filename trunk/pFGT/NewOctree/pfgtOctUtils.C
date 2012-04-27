
#include <iostream>
#include <cmath>
#include <algorithm>
#include "mpi.h"
#include "pfgtOctUtils.h"
#include "parUtils.h"
#include "dtypes.h"

extern PetscLogEvent fgtEvent;
extern PetscLogEvent s2wEvent;
extern PetscLogEvent fgtOctConEvent;
extern PetscLogEvent expandHybridEvent;
extern PetscLogEvent directHybridEvent;

void pfgt(std::vector<double>& sources, const unsigned int minPtsInFgt, const unsigned int FgtLev,
    const int P, const int L, const int K, MPI_Comm comm) {
  PetscLogEventBegin(fgtEvent, 0, 0, 0, 0);

  int npes, rank;
  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &rank);

  //FGT box size = sqrt(delta)
  const double hFgt = 1.0/(static_cast<double>(1u << FgtLev));

  //Kernel Bandwidth
  const double delta = hFgt*hFgt;

  if(!rank) {
    std::cout<<"delta = "<<delta<<std::endl;
  }

  std::vector<double> expandSources;
  std::vector<double> directSources;
  std::vector<ot::TreeNode> fgtList;
  splitSources(sources, minPtsInFgt, FgtLev, expandSources, directSources, fgtList, comm);
  sources.clear();

  /*
  //Split octree and sources into 2 sets
  //Remove empty octants
  std::vector<ot::TreeNode> expandTree;
  std::vector<ot::TreeNode> directTree;
  int ptsCnt = 0;
  for(size_t i = 0; i < linOct.size(); i++) {
  unsigned int lev = linOct[i].getLevel();
  double hCurrOct = 1.0/(static_cast<double>(1u << lev));
  bool isExpand = false;
  if( hCurrOct <= (hFgt*DirectHfactor) ) {
  isExpand = true;
  }
  linOct[i].setWeight(0);
  while(ptsCnt < numPts) {
  unsigned int px = static_cast<unsigned int>(sources[4*ptsCnt]*(__DTPMD__));
  unsigned int py = static_cast<unsigned int>(sources[(4*ptsCnt)+1]*(__DTPMD__));
  unsigned int pz = static_cast<unsigned int>(sources[(4*ptsCnt)+2]*(__DTPMD__));
  ot::TreeNode tmpOct(px, py, pz, __MAX_DEPTH__, __DIM__, __MAX_DEPTH__);
  assert(tmpOct >= linOct[i]);
  if((tmpOct == linOct[i]) || (linOct[i].isAncestor(tmpOct))) {
  if(isExpand) {
  expandSources.push_back(sources[4*ptsCnt]);
  expandSources.push_back(sources[(4*ptsCnt) + 1]);
  expandSources.push_back(sources[(4*ptsCnt) + 2]);
  expandSources.push_back(sources[(4*ptsCnt) + 3]);
  } else {
  directSources.push_back(sources[4*ptsCnt]);
  directSources.push_back(sources[(4*ptsCnt) + 1]);
  directSources.push_back(sources[(4*ptsCnt) + 2]);
  directSources.push_back(sources[(4*ptsCnt) + 3]);
  }
  linOct[i].addWeight(1);
  ++ptsCnt;
  } else {
  break;
  }
  }//end while
  if(linOct[i].getWeight() > 0) {
  if(isExpand) {
  expandTree.push_back(linOct[i]);
  } else {
  directTree.push_back(linOct[i]);
  }
  }
  }//end for i
  linOct.clear();

  unsigned int localTreeSizes[2];
  unsigned int globalTreeSizes[2];
  localTreeSizes[0] = expandTree.size();
  localTreeSizes[1] = directTree.size();
  MPI_Allreduce(localTreeSizes, globalTreeSizes, 2, MPI_UNSIGNED, MPI_SUM, comm);

  if(globalTreeSizes[0] == 0) {
  //Only Direct
  if(!rank) {
  std::cout<<"THIS CASE IS NOT SUPPORTED!"<<std::endl;
  }
  assert(false);
  } else if(globalTreeSizes[1] == 0) {
  //Only Expand
  if(!rank) {
  std::cout<<"THIS CASE IS NOT SUPPORTED!"<<std::endl;
  }
  assert(false);
  } else if(npes == 1) {
  //Serial
  if(!rank) {
  std::cout<<"THIS CASE IS NOT SUPPORTED!"<<std::endl;
  }
  assert(false);
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

  int numRecvExpandPts = 0;
  for(int i = 0; i < finalExpandTree.size(); ++i) {
    numRecvExpandPts += (finalExpandTree[i].getWeight());
  }//end for i

  int numRecvDirectPts = 0;
  for(int i = 0; i < finalDirectTree.size(); ++i) {
    numRecvDirectPts += (finalDirectTree[i].getWeight());
  }//end for i

  std::vector<double> finalExpandSources;
  std::vector<double> finalDirectSources;

  par::scatterValues<double>(expandSources, finalExpandSources, (4*numRecvExpandPts), comm);
  par::scatterValues<double>(directSources, finalDirectSources, (4*numRecvDirectPts), comm);

  expandSources.clear();
  directSources.clear();

  if(rank < npesExpand) {
    pfgtHybridExpand(finalExpandSources, finalExpandTree, P, L, FgtLev, delta, hFgt, subComm, comm);
  } else {
    pfgtHybridDirect(finalDirectSources, finalDirectTree, FgtLev, subComm, comm);
  }

  MPI_Comm_free(&subComm);
}

*/

PetscLogEventEnd(fgtEvent, 0, 0, 0, 0);
}

void pfgtHybridExpand(std::vector<double> & expandSources, std::vector<ot::TreeNode> & expandTree,
    const int P, const int L, const unsigned int FgtLev, const double delta, 
    const double hFgt, MPI_Comm subComm, MPI_Comm comm) {
  PetscLogEventBegin(expandHybridEvent, 0, 0, 0, 0);

  assert(!(expandTree.empty()));

  std::vector<ot::TreeNode> fgtList;
  createFGToctree(fgtList, expandTree, FgtLev, subComm);

  std::vector<ot::TreeNode> fgtMins;
  computeFGTminsHybridExpand(fgtMins, fgtList, subComm, comm);

  unsigned int numPtsInRemoteFGT;
  computeNumPtsInFGT(expandSources, fgtList, numPtsInRemoteFGT);

  ot::TreeNode remoteFGT;
  if(numPtsInRemoteFGT > 0) {
    assert((expandTree[0].getLevel()) > FgtLev);
    remoteFGT = expandTree[0].getAncestor(FgtLev);
  }

  std::vector<double> wVec;
  s2w(wVec, expandSources, remoteFGT, numPtsInRemoteFGT, fgtList, fgtMins, P, L, FgtLev, hFgt, subComm);

  PetscLogEventEnd(expandHybridEvent, 0, 0, 0, 0);
}

void computeNumPtsInFGT(std::vector<double> & sources, std::vector<ot::TreeNode> & fgtList, 
    unsigned int & numPtsInRemoteFGT) {

  int numPts = ((sources.size())/4);

  if(fgtList.empty()) {
    numPtsInRemoteFGT = numPts;
  } else {
    int ptsCnt = 0;
    while(ptsCnt < numPts) {
      unsigned int px = static_cast<unsigned int>(sources[4*ptsCnt]*(__DTPMD__));
      unsigned int py = static_cast<unsigned int>(sources[(4*ptsCnt)+1]*(__DTPMD__));
      unsigned int pz = static_cast<unsigned int>(sources[(4*ptsCnt)+2]*(__DTPMD__));
      ot::TreeNode tmpOct(px, py, pz, __MAX_DEPTH__, __DIM__, __MAX_DEPTH__);
      if(tmpOct < fgtList[0]) {
        ++ptsCnt;
      } else {
        break;
      }
    }//end while

    numPtsInRemoteFGT = ptsCnt;

    for(int i = 0; i < fgtList.size(); ++i) {
      fgtList[i].setWeight(0);
      while(ptsCnt < numPts) {
        unsigned int px = static_cast<unsigned int>(sources[4*ptsCnt]*(__DTPMD__));
        unsigned int py = static_cast<unsigned int>(sources[(4*ptsCnt)+1]*(__DTPMD__));
        unsigned int pz = static_cast<unsigned int>(sources[(4*ptsCnt)+2]*(__DTPMD__));
        ot::TreeNode tmpOct(px, py, pz, __MAX_DEPTH__, __DIM__, __MAX_DEPTH__);
        assert(tmpOct >= fgtList[i]);
        if( (tmpOct == fgtList[i]) || (fgtList[i].isAncestor(tmpOct)) ) {
          fgtList[i].addWeight(1);
          ++ptsCnt;
        } else {
          break;
        }
      }//end while
    }//end i
  }

}

void s2w(std::vector<double> & localWlist, std::vector<double> & sources, ot::TreeNode remoteFGT, 
    const unsigned int numPtsInRemoteFGT, std::vector<ot::TreeNode> & fgtList, 
    std::vector<ot::TreeNode> & fgtMins, const int P, const int L, 
    const unsigned int FgtLev, const double hFgt, MPI_Comm subComm) {
  PetscLogEventBegin(s2wEvent, 0, 0, 0, 0);

  int rank;
  MPI_Comm_rank(subComm, &rank);

  int remoteFGTowner = -1;
  if(numPtsInRemoteFGT > 0) {
    unsigned int retIdx;
    seq::maxLowerBound(fgtMins, remoteFGT, retIdx, NULL, NULL);
    remoteFGTowner = fgtMins[retIdx].getWeight();
    assert(remoteFGTowner < rank);
  }

  int subNpes;
  MPI_Comm_size(subComm, &subNpes);

  //2P complex coefficients for each dimension.  
  const unsigned int numWcoeffs = 16*P*P*P;

  int* sendCnts = new int[subNpes];
  for(int i = 0; i < subNpes; ++i) {
    sendCnts[i] = 0;
  }//end i
  if(remoteFGTowner >= 0) {
    sendCnts[remoteFGTowner] = numWcoeffs;
  }

  int* recvCnts = new int[subNpes]; 

  MPI_Alltoall(sendCnts, 1, MPI_INT, recvCnts, 1, MPI_INT, subComm);

  int* sendDisps = new int[subNpes];
  int* recvDisps = new int[subNpes]; 

  sendDisps[0] = 0;
  recvDisps[0] = 0;
  for(int i = 1; i < subNpes; ++i) {
    sendDisps[i] = sendDisps[i - 1] + sendCnts[i - 1];
    recvDisps[i] = recvDisps[i - 1] + recvCnts[i - 1];
  }//end i

  const double LbyP = static_cast<double>(L)/static_cast<double>(P);
  const double ImExpZfactor = LbyP/hFgt;

  std::vector<double> sendWlist;
  if(remoteFGTowner >= 0) {
    sendWlist.resize(numWcoeffs, 0.0);
    double cx = (0.5*hFgt) + ((static_cast<double>(remoteFGT.getX()))/(__DTPMD__));
    double cy = (0.5*hFgt) + ((static_cast<double>(remoteFGT.getY()))/(__DTPMD__));
    double cz = (0.5*hFgt) + ((static_cast<double>(remoteFGT.getZ()))/(__DTPMD__));
    for(int i = 0; i < numPtsInRemoteFGT; ++i) {
      double px = sources[4*i];
      double py = sources[(4*i)+1];
      double pz = sources[(4*i)+2];
      double pf = sources[(4*i)+3];
      for(int k3 = -P, di = 0; k3 < P; k3++) {
        double thetaZ = ImExpZfactor*(static_cast<double>(k3)*(cz - pz));
        for(int k2 = -P; k2 < P; k2++) {
          double thetaY = ImExpZfactor*(static_cast<double>(k2)*(cy - py));
          for(int k1 = -P; k1 < P; k1++, di++) {
            double thetaX = ImExpZfactor*(static_cast<double>(k1)*(cx - px));
            double theta = (thetaX + thetaY + thetaZ);
            sendWlist[2*di] += (pf*cos(theta));
            sendWlist[(2*di) + 1] += (pf*sin(theta));
          }//end for k1
        }//end for k2
      }//end for k3
    }//end i
  }

  std::vector<double> recvWlist(recvDisps[subNpes - 1] + recvCnts[subNpes - 1]);

  double* sendBuf = NULL;
  if(!(sendWlist.empty())) {
    sendBuf = &(sendWlist[0]);
  }
  double* recvBuf = NULL;
  if(!(recvWlist.empty())) {
    recvBuf = &(recvWlist[0]);
  }
  MPI_Alltoallv(sendBuf, sendCnts, sendDisps, MPI_DOUBLE,
      recvBuf, recvCnts, recvDisps, MPI_DOUBLE, subComm);

  delete [] sendCnts;
  delete [] sendDisps;
  delete [] recvCnts;
  delete [] recvDisps;

  localWlist.resize( (numWcoeffs*(fgtList.size())), 0.0);

  for(int i = 0; i < recvWlist.size(); i += numWcoeffs) {
    for(int d = 0; d < numWcoeffs; ++d) {
      localWlist[(numWcoeffs*(fgtList.size() - 1)) + d] += recvWlist[i + d];
    }//end d
  }//end i

  for(int i = 0, ptsIdx = numPtsInRemoteFGT; i < fgtList.size(); ++i) {
    double cx = (0.5*hFgt) + ((static_cast<double>(fgtList[i].getX()))/(__DTPMD__));
    double cy = (0.5*hFgt) + ((static_cast<double>(fgtList[i].getY()))/(__DTPMD__));
    double cz = (0.5*hFgt) + ((static_cast<double>(fgtList[i].getZ()))/(__DTPMD__));
    for(int j = 0; j < fgtList[i].getWeight(); ++j, ++ptsIdx) {
      double px = sources[4*ptsIdx];
      double py = sources[(4*ptsIdx)+1];
      double pz = sources[(4*ptsIdx)+2];
      double pf = sources[(4*ptsIdx)+3];
      for(int k3 = -P, di = 0; k3 < P; k3++) {
        double thetaZ = ImExpZfactor*(static_cast<double>(k3)*(cz - pz));
        for(int k2 = -P; k2 < P; k2++) {
          double thetaY = ImExpZfactor*(static_cast<double>(k2)*(cy - py));
          for(int k1 = -P; k1 < P; k1++, di++) {
            double thetaX = ImExpZfactor*(static_cast<double>(k1)*(cx - px));
            double theta = (thetaX + thetaY + thetaZ);
            localWlist[(numWcoeffs*i) + (2*di)] += (pf*cos(theta));
            localWlist[(numWcoeffs*i) + (2*di) + 1] += (pf*sin(theta));
          }//end for k1
        }//end for k2
      }//end for k3
    }//end j
  }//end i


  PetscLogEventEnd(s2wEvent, 0, 0, 0, 0);
}

void pfgtHybridDirect(std::vector<double> & directSources, std::vector<ot::TreeNode> & directTree, 
    const unsigned int FgtLev, MPI_Comm subComm, MPI_Comm comm) {
  PetscLogEventBegin(directHybridEvent, 0, 0, 0, 0);

  assert(!(directTree.empty()));

  std::vector<ot::TreeNode> directMins;
  computeMins(directMins, directTree, subComm);

  std::vector<ot::TreeNode> fgtMins;
  computeFGTminsHybridDirect(fgtMins, comm);

  PetscLogEventEnd(directHybridEvent, 0, 0, 0, 0);
}

void computeMins(std::vector<ot::TreeNode> & mins, std::vector<ot::TreeNode> & subTree, MPI_Comm subComm) {
  int subNpes;
  MPI_Comm_size(subComm, &subNpes);

  mins.resize(subNpes);

  MPI_Allgather(&(subTree[0]), 1, par::Mpi_datatype<ot::TreeNode>::value(),
      &(mins[0]), 1, par::Mpi_datatype<ot::TreeNode>::value(), subComm);
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
    for(unsigned int i = 0; i < fgtMins.size(); ++i) {
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
      std::vector<ot::TreeNode> subTree;
      subTree.push_back(expandTree[i]);
      while(true) {
        std::vector<ot::TreeNode> tmpSubTree;
        for(int j = 0; j < subTree.size(); ++j) {
          if((subTree[j].getLevel()) < FgtLev) {
            subTree[j].addChildren(tmpSubTree);
          } else {
            assert((subTree[j].getLevel()) == FgtLev);
            tmpSubTree.push_back(subTree[j]);
          }
        }//end j
        if((tmpSubTree.size()) == (subTree.size())) {
          break;
        } else {
          swap(subTree, tmpSubTree);
        }
      }//end while
      tmpFgtListB.insert(tmpFgtListB.end(), subTree.begin(), subTree.end());
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

void splitSources(std::vector<double>& sources, const unsigned int minPtsInFgt, 
    const unsigned int FgtLev, std::vector<double>& expandSources, std::vector<double>& directSources, 
    std::vector<ot::TreeNode>& fgtList, MPI_Comm comm) {

  int numPts = ((sources.size())/4);

  assert(!(sources.empty()));
  assert(fgtList.empty());
  {
    unsigned int px = static_cast<unsigned int>(sources[0]*(__DTPMD__));
    unsigned int py = static_cast<unsigned int>(sources[1]*(__DTPMD__));
    unsigned int pz = static_cast<unsigned int>(sources[2]*(__DTPMD__));
    ot::TreeNode ptOct(px, py, pz, __MAX_DEPTH__, __DIM__, __MAX_DEPTH__);
    ot::TreeNode newFgt = ptOct.getAncestor(FgtLev);
    fgtList.push_back(newFgt);
  }

  for(int i = 1; i < numPts; ++i) {
    unsigned int px = static_cast<unsigned int>(sources[4*i]*(__DTPMD__));
    unsigned int py = static_cast<unsigned int>(sources[(4*i)+1]*(__DTPMD__));
    unsigned int pz = static_cast<unsigned int>(sources[(4*i)+2]*(__DTPMD__));
    ot::TreeNode ptOct(px, py, pz, __MAX_DEPTH__, __DIM__, __MAX_DEPTH__);
    ot::TreeNode newFgt = ptOct.getAncestor(FgtLev);
    if(fgtList[fgtList.size() - 1] == newFgt) {
      fgtList[fgtList.size() - 1].addWeight(1);
    } else {
      fgtList.push_back(newFgt);
    }
  }//end for i

  assert(!(fgtList.empty()));

  int rank;
  int npes;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &npes);

  int localFlag = 0;
  if( (rank > 0) && (rank < (npes - 1)) && ((fgtList.size()) == 1) ) {
    localFlag = 1;
  }

  int globalFlag;
  MPI_Allreduce(&localFlag, &globalFlag, 1, MPI_INT, MPI_SUM, comm);

  int prevRank = rank - 1;
  int nextRank = rank + 1;

  if(globalFlag > 0) {
    int gatherSendBuf = 0;
    if( (rank > 0) && (rank < (npes - 1)) && (fgtList.size() == 1) ) {
      gatherSendBuf = sources.size();
    }

    int* gatherList = new int[npes];

    MPI_Allgather((&gatherSendBuf), 1, MPI_INT, gatherList, 1, MPI_INT, comm);

    if(rank > 0) {
      while(gatherList[prevRank] > 0) {
        --prevRank;
      }//end while
    }

    if(rank < (npes - 1)) {
      while(gatherList[nextRank] > 0) {
        ++nextRank;
      }//end while
    }

    int* sendFgtCnts = new int[npes];
    int* recvFgtCnts = new int[npes];

    int* sendSourceCnts = new int[npes];
    int* recvSourceCnts = new int[npes];

    for(int i = 0; i < npes; ++i) {
      sendFgtCnts[i] = 0;
      recvFgtCnts[i] = 0;
      sendSourceCnts[i] = 0;
      recvSourceCnts[i] = 0;
    }//end i

    if(gatherSendBuf > 0) {
      sendFgtCnts[prevRank] = 1;
      sendSourceCnts[prevRank] = gatherSendBuf;
    }
    for(int i = rank + 1; i < nextRank; ++i) {
      recvFgtCnts[i] = 1;
      recvSourceCnts[i] = gatherList[i];
    }//end i

    delete [] gatherList;

    int* sendFgtDisps = new int[npes];
    int* recvFgtDisps = new int[npes];
    sendFgtDisps[0] = 0;
    recvFgtDisps[0] = 0;
    for(int i = 1; i < npes; ++i) {
      sendFgtDisps[i] = sendFgtDisps[i - 1] + sendFgtCnts[i - 1];
      recvFgtDisps[i] = recvFgtDisps[i - 1] + recvFgtCnts[i - 1];
    }//end i

    std::vector<ot::TreeNode> tmpFgtList(recvFgtDisps[npes - 1] + recvFgtCnts[npes - 1]);

    ot::TreeNode* recvFgtBuf = NULL;
    if(!(tmpFgtList.empty())) {
      recvFgtBuf = (&(tmpFgtList[0]));
    }

    MPI_Alltoallv( (&(fgtList[0])), sendFgtCnts, sendFgtDisps, par::Mpi_datatype<ot::TreeNode>::value(),
        recvFgtBuf, recvFgtCnts, recvFgtDisps, par::Mpi_datatype<ot::TreeNode>::value(), comm);

    if(gatherSendBuf > 0) {
      fgtList.clear();
    } else {
      for(int i = 0; i < tmpFgtList.size(); ++i) {
        if(tmpFgtList[i] == fgtList[fgtList.size() - 1]) {
          fgtList[fgtList.size() - 1].addWeight(tmpFgtList[i].getWeight());
        } else {
          fgtList.push_back(tmpFgtList[i]);
        }
      }//end i
    }

    delete [] sendFgtCnts;
    delete [] recvFgtCnts;
    delete [] sendFgtDisps;
    delete [] recvFgtDisps;

    int* sendSourceDisps = new int[npes];
    int* recvSourceDisps = new int[npes];
    sendSourceDisps[0] = 0;
    recvSourceDisps[0] = 0;
    for(int i = 1; i < npes; ++i) {
      sendSourceDisps[i] = sendSourceDisps[i - 1] + sendSourceCnts[i - 1];
      recvSourceDisps[i] = recvSourceDisps[i - 1] + recvSourceCnts[i - 1];
    }//end i

    std::vector<double> tmpSources(recvSourceDisps[npes - 1] + recvSourceCnts[npes - 1]);

    double* recvSourceBuf = NULL;
    if(!(tmpSources.empty())) {
      recvSourceBuf = (&(tmpSources[0]));
    }

    MPI_Alltoallv( (&(sources[0])), sendSourceCnts, sendSourceDisps, MPI_DOUBLE,
        recvSourceBuf, recvSourceCnts, recvSourceDisps, MPI_DOUBLE, comm);

    if(gatherSendBuf > 0) {
      sources.clear();
    } else {
      if(!(tmpSources.empty())) {
        sources.insert(sources.end(), tmpSources.begin(), tmpSources.end());
      }
    }

    delete [] sendSourceCnts;
    delete [] recvSourceCnts;
    delete [] sendSourceDisps;
    delete [] recvSourceDisps;
  }

  if(!(fgtList.empty())) {
    ot::TreeNode prevFgt;
    ot::TreeNode nextFgt;
    ot::TreeNode firstFgt = fgtList[0];
    ot::TreeNode lastFgt = fgtList[fgtList.size() - 1];
    MPI_Request recvPrevReq;
    MPI_Request recvNextReq;
    MPI_Request sendFirstReq;
    MPI_Request sendLastReq;
    if(rank > 0) {
      MPI_Irecv(&prevFgt, 1, par::Mpi_datatype<ot::TreeNode>::value(),
          prevRank, 1, comm, &recvPrevReq);
      MPI_Isend(&firstFgt, 1, par::Mpi_datatype<ot::TreeNode>::value(),
          prevRank, 2, comm, &sendFirstReq);
    }
    if(rank < (npes - 1)) {
      MPI_Irecv(&nextFgt, 1, par::Mpi_datatype<ot::TreeNode>::value(),
          nextRank, 2, comm, &recvNextReq);
      MPI_Isend(&lastFgt, 1, par::Mpi_datatype<ot::TreeNode>::value(),
          nextRank, 1, comm, &sendLastReq);
    }

    if(rank > 0) {
      MPI_Status status;
      MPI_Wait(&recvPrevReq, &status);
      MPI_Wait(&sendFirstReq, &status);
    }
    if(rank < (npes - 1)) {
      MPI_Status status;
      MPI_Wait(&recvNextReq, &status);
      MPI_Wait(&sendLastReq, &status);
    }

    bool removeFirst = false;
    bool addToLast = false;
    if(rank > 0) {
      if(prevFgt == firstFgt) {
        removeFirst = true;
      }
    }
    if(rank < (npes - 1)) {
      if(nextFgt == lastFgt) {
        addToLast = true;
      }
    }

    MPI_Request recvPtsReq;
    if(addToLast) {
      sources.resize(4*(numPts + (nextFgt.getWeight())));
      fgtList[fgtList.size() - 1].addWeight(nextFgt.getWeight());
      MPI_Irecv((&(sources[4*numPts])), (4*(nextFgt.getWeight())), MPI_DOUBLE, nextRank,
          3, comm, &recvPtsReq);
    }
    if(removeFirst) {
      MPI_Send((&(sources[0])), (4*(firstFgt.getWeight())), MPI_DOUBLE, prevRank, 3, comm);
      fgtList.erase(fgtList.begin());
    }
    if(addToLast) {
      MPI_Status status;
      MPI_Wait(&recvPtsReq, &status);
    }
    if(removeFirst) {
      sources.erase(sources.begin(), sources.begin() + (4*(firstFgt.getWeight())));
    }
  } 

  assert(expandSources.empty());
  assert(directSources.empty());
  std::vector<ot::TreeNode> dummyList;
  int sourceIdx = 0;
  for(size_t i = 0; i < fgtList.size(); ++i) {
    if((fgtList[i].getWeight()) < minPtsInFgt) {
      directSources.insert(directSources.end(), (sources.begin() + sourceIdx),
          (sources.begin() + sourceIdx + (4*(fgtList[i].getWeight()))));
    } else {
      dummyList.push_back(fgtList[i]);
      expandSources.insert(expandSources.end(), (sources.begin() + sourceIdx), 
          (sources.begin() + sourceIdx + (4*(fgtList[i].getWeight()))));
    }
    sourceIdx += (4*(fgtList[i].getWeight()));
  }//end i
  swap(dummyList, fgtList);
  assert((sources.size()) == ((directSources.size()) + (expandSources.size())));
}



