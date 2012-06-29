
#include <iostream>
#include <cmath>
#include <algorithm>
#include "mpi.h"
#include "pfgtOctUtils.h"
#include "par/parUtils.h"
#include "par/dtypes.h"

extern PetscLogEvent pfgtMainEvent;
extern PetscLogEvent pfgtExpandEvent;
extern PetscLogEvent pfgtDirectEvent;
extern PetscLogEvent splitSourcesEvent;
extern PetscLogEvent s2wEvent;
extern PetscLogEvent l2tEvent;
extern PetscLogEvent w2lEvent;
extern PetscLogEvent d2dEvent;
extern PetscLogEvent w2dD2lExpandEvent;
extern PetscLogEvent w2dD2lDirectEvent;

void pfgtMain(std::vector<double>& sources, const unsigned int minPtsInFgt, const unsigned int FgtLev,
    const int P, const int L, const int K, const double epsilon, MPI_Comm comm) {
  PetscLogEventBegin(pfgtMainEvent, 0, 0, 0, 0);

  int npes, rank;
  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &rank);

  std::vector<double> expandSources;
  std::vector<double> directSources;
  std::vector<ot::TreeNode> fgtList;
  splitSources(sources, minPtsInFgt, FgtLev, expandSources, directSources, fgtList, comm);
  sources.clear();

  std::vector<double> tmpExpandSources;
  int srcCnt = 0;
  for(int i = 0; i < fgtList.size(); ++i) {
    assert(fgtList[i].getWeight() > 0);
    {
      tmpExpandSources.push_back(expandSources[(4*srcCnt)]);
      tmpExpandSources.push_back(expandSources[(4*srcCnt) + 1]);
      tmpExpandSources.push_back(expandSources[(4*srcCnt) + 2]);
      tmpExpandSources.push_back(expandSources[(4*srcCnt) + 3]);
      tmpExpandSources.push_back(fgtList[i].getWeight());
      srcCnt++;
    }
    for(int j = 1; j < fgtList[i].getWeight(); ++j) {
      tmpExpandSources.push_back(expandSources[(4*srcCnt)]);
      tmpExpandSources.push_back(expandSources[(4*srcCnt) + 1]);
      tmpExpandSources.push_back(expandSources[(4*srcCnt) + 2]);
      tmpExpandSources.push_back(expandSources[(4*srcCnt) + 3]);
      tmpExpandSources.push_back(0);
      srcCnt++;
    }//end j
  }//end i
  swap(expandSources, tmpExpandSources);
  tmpExpandSources.clear();
  fgtList.clear();

  int localSizes[2];
  int globalSizes[2];
  localSizes[0] = (expandSources.size())/5;
  localSizes[1] = (directSources.size())/4;
  MPI_Allreduce(localSizes, globalSizes, 2, MPI_INT, MPI_SUM, comm);

  if(globalSizes[0] == 0) {
    //Only Direct
    if(!rank) {
      std::cout<<"THIS CASE (Only Direct Points) IS NOT SUPPORTED!"<<std::endl;
    }
    assert(false);
  } else if(globalSizes[1] == 0) {
    //Only Expand
    if(!rank) {
      std::cout<<"THIS CASE (Only Expand Points) IS NOT SUPPORTED!"<<std::endl;
    }
    assert(false);
  } else if(npes == 1) {
    //Serial
    if(!rank) {
      std::cout<<"THIS CASE (Serial) IS NOT SUPPORTED!"<<std::endl;
    }
    assert(false);
  } else {
    int npesExpand = (globalSizes[0]*npes)/(globalSizes[0] + globalSizes[1]);
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

    int avgExpand = (globalSizes[0])/npesExpand;
    int extraExpand = (globalSizes[0])%npesExpand; 
    int avgDirect = (globalSizes[1])/npesDirect;
    int extraDirect = (globalSizes[1])%npesDirect;

    std::vector<double> finalExpandSources;
    std::vector<double> finalDirectSources;

    if(rank < extraExpand) {
      par::scatterValues<double>(expandSources, finalExpandSources, (5*(avgExpand + 1)), comm);
      par::scatterValues<double>(directSources, finalDirectSources, 0, comm);
    } else if(rank < npesExpand) {
      par::scatterValues<double>(expandSources, finalExpandSources, (5*avgExpand), comm);
      par::scatterValues<double>(directSources, finalDirectSources, 0, comm);
    } else if(rank < (npesExpand + extraDirect)) {
      par::scatterValues<double>(expandSources, finalExpandSources, 0, comm);
      par::scatterValues<double>(directSources, finalDirectSources, (4*(avgDirect + 1)), comm);
    } else {
      par::scatterValues<double>(expandSources, finalExpandSources, 0, comm);
      par::scatterValues<double>(directSources, finalDirectSources, (4*avgDirect), comm);
    }

    expandSources.clear();
    directSources.clear();

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

    if(rank < npesExpand) {
      pfgtExpand(expandSources, fgtList, FgtLev, P, L, K, avgExpand, extraExpand, subComm, comm);
    } else {
      pfgtDirect(finalDirectSources, FgtLev, P, L, K, epsilon, subComm, comm);
    }

    MPI_Comm_free(&subComm);
  }

  PetscLogEventEnd(pfgtMainEvent, 0, 0, 0, 0);
}

void pfgtExpand(std::vector<double> & expandSources, std::vector<ot::TreeNode> & fgtList,
    const unsigned int FgtLev, const int P, const int L, const int K, 
    const int avgExpand, const int extraExpand, MPI_Comm subComm, MPI_Comm comm) {
  PetscLogEventBegin(pfgtExpandEvent, 0, 0, 0, 0);

  std::vector<ot::TreeNode> fgtMins;
  computeFgtMinsExpand(fgtMins, fgtList, subComm, comm);

  int subRank;
  MPI_Comm_rank(subComm, &subRank);

  assert(!(expandSources.empty()));

  int sumFgtWts = 0;
  for(int i = 0; i < fgtList.size(); ++i) {
    sumFgtWts += fgtList[i].getWeight();
  }//end i

  int numExpandPts = (expandSources.size())/4;

  int numPtsInRemoteFgt = 0;
  for( ; numPtsInRemoteFgt < numExpandPts; ++numPtsInRemoteFgt) {
    unsigned int px = static_cast<unsigned int>(expandSources[4*numPtsInRemoteFgt]*(__DTPMD__));
    unsigned int py = static_cast<unsigned int>(expandSources[(4*numPtsInRemoteFgt)+1]*(__DTPMD__));
    unsigned int pz = static_cast<unsigned int>(expandSources[(4*numPtsInRemoteFgt)+2]*(__DTPMD__));
    ot::TreeNode pt(px, py, pz, __MAX_DEPTH__, __DIM__, __MAX_DEPTH__);
    if(pt >= fgtList[0]) {
      break;
    }
  }//end for

  int excessWt = sumFgtWts + numPtsInRemoteFgt - numExpandPts;
  assert(excessWt >= 0);
  if(!(fgtList.empty())) {
    int lastWt = fgtList[fgtList.size() - 1].getWeight();
    assert(lastWt > excessWt);
    fgtList[fgtList.size() - 1].setWeight(lastWt - excessWt);
  }

  int remoteFgtOwner = -1;
  ot::TreeNode remoteFgt;
  if(numPtsInRemoteFgt > 0) {
    computeRemoteFgt(remoteFgt, remoteFgtOwner, FgtLev, expandSources, fgtMins);
  }
  assert(remoteFgtOwner < subRank);

  //2P complex coefficients for each dimension.  
  const unsigned int numWcoeffs = 16*P*P*P;

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

  //w2dAndD2lExpand(localLlist, localWlist, fgtList, P, comm);

  std::vector<double> results(((expandSources.size())/4), 0.0);
  l2t(results, localLlist, expandSources, remoteFgt, remoteFgtOwner, numPtsInRemoteFgt, 
      fgtList, fgtMins, FgtLev, P, L, s2wSendCnts, s2wSendDisps, s2wRecvCnts, s2wRecvDisps, subComm);

  destroyS2WcommInfo(s2wSendCnts, s2wSendDisps, s2wRecvCnts, s2wRecvDisps); 

  PetscLogEventEnd(pfgtExpandEvent, 0, 0, 0, 0);
}

void pfgtDirect(std::vector<double> & directSources, const unsigned int FgtLev,
    const int P, const int L, const int K, const double epsilon, MPI_Comm subComm, MPI_Comm comm) {
  PetscLogEventBegin(pfgtDirectEvent, 0, 0, 0, 0);

  int subNpes;
  MPI_Comm_size(subComm, &subNpes);

  assert(!(directSources.empty()));

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
  computeFgtMinsDirect(fgtMins, comm);

  std::vector<double> results(directNodes.size(), 0.0);
  d2d(results, directSources, directNodes, directMins, FgtLev, epsilon, subComm);

  //w2dAndD2lDirect(results, directSources, fgtMins, FgtLev, P, L, K, epsilon, comm);

  PetscLogEventEnd(pfgtDirectEvent, 0, 0, 0, 0);
}

void s2w(std::vector<double> & localWlist, std::vector<double> & sources,  
    const ot::TreeNode remoteFgt, const int remoteFgtOwner, const int numPtsInRemoteFgt,
    std::vector<ot::TreeNode> & fgtList, std::vector<ot::TreeNode> & fgtMins,
    const unsigned int FgtLev, const int P, const int L,
    int* sendCnts, int* sendDisps, int* recvCnts, int* recvDisps, MPI_Comm subComm) {
  PetscLogEventBegin(s2wEvent, 0, 0, 0, 0);

  int npes;
  MPI_Comm_size(subComm, &npes);

  //Fgt box size = sqrt(delta)
  const double hFgt = 1.0/(static_cast<double>(1u << FgtLev));

  const double LbyP = static_cast<double>(L)/static_cast<double>(P);
  const double ImExpZfactor = LbyP/hFgt;

  //2P complex coefficients for each dimension.  
  const unsigned int numWcoeffs = 16*P*P*P;

  std::vector<double> sendWlist;
  if(remoteFgtOwner >= 0) {
    sendWlist.resize(numWcoeffs, 0.0);
    double cx = (0.5*hFgt) + ((static_cast<double>(remoteFgt.getX()))/(__DTPMD__));
    double cy = (0.5*hFgt) + ((static_cast<double>(remoteFgt.getY()))/(__DTPMD__));
    double cz = (0.5*hFgt) + ((static_cast<double>(remoteFgt.getZ()))/(__DTPMD__));
    for(int i = 0; i < numPtsInRemoteFgt; ++i) {
      double px = sources[4*i];
      double py = sources[(4*i)+1];
      double pz = sources[(4*i)+2];
      double pf = sources[(4*i)+3];
      for(int k3 = -P, di = 0; k3 < P; k3++) {
        double thetaZ = (static_cast<double>(k3))*(cz - pz);
        for(int k2 = -P; k2 < P; k2++) {
          double thetaY = (static_cast<double>(k2))*(cy - py);
          for(int k1 = -P; k1 < P; k1++, di++) {
            double thetaX = (static_cast<double>(k1))*(cx - px);
            double theta = ImExpZfactor*(thetaX + thetaY + thetaZ);
            sendWlist[2*di] += (pf*cos(theta));
            sendWlist[(2*di) + 1] += (pf*sin(theta));
          }//end for k1
        }//end for k2
      }//end for k3
    }//end i
  }

  std::vector<double> recvWlist(recvDisps[npes - 1] + recvCnts[npes - 1]);

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

  for(size_t i = 0; i < recvWlist.size(); i += numWcoeffs) {
    for(unsigned int d = 0; d < numWcoeffs; ++d) {
      localWlist[(numWcoeffs*(fgtList.size() - 1)) + d] += recvWlist[i + d];
    }//end d
  }//end i

  for(int i = 0, ptsIdx = numPtsInRemoteFgt; i < fgtList.size(); ++i) {
    double cx = (0.5*hFgt) + ((static_cast<double>(fgtList[i].getX()))/(__DTPMD__));
    double cy = (0.5*hFgt) + ((static_cast<double>(fgtList[i].getY()))/(__DTPMD__));
    double cz = (0.5*hFgt) + ((static_cast<double>(fgtList[i].getZ()))/(__DTPMD__));
    for(int j = 0; j < fgtList[i].getWeight(); ++j, ++ptsIdx) {
      double px = sources[4*ptsIdx];
      double py = sources[(4*ptsIdx)+1];
      double pz = sources[(4*ptsIdx)+2];
      double pf = sources[(4*ptsIdx)+3];
      for(int k3 = -P, di = 0; k3 < P; k3++) {
        double thetaZ = (static_cast<double>(k3))*(cz - pz);
        for(int k2 = -P; k2 < P; k2++) {
          double thetaY = (static_cast<double>(k2))*(cy - py);
          for(int k1 = -P; k1 < P; k1++, di++) {
            double thetaX = (static_cast<double>(k1))*(cx - px);
            double theta = ImExpZfactor*(thetaX + thetaY + thetaZ);
            localWlist[(numWcoeffs*i) + (2*di)] += (pf*cos(theta));
            localWlist[(numWcoeffs*i) + (2*di) + 1] += (pf*sin(theta));
          }//end for k1
        }//end for k2
      }//end for k3
    }//end j
  }//end i

  PetscLogEventEnd(s2wEvent, 0, 0, 0, 0);
}

void l2t(std::vector<double> & results, std::vector<double> & localLlist, std::vector<double> & sources, 
    const ot::TreeNode remoteFgt, const int remoteFgtOwner, const int numPtsInRemoteFgt,
    std::vector<ot::TreeNode> & fgtList, std::vector<ot::TreeNode> & fgtMins,
    const unsigned int FgtLev, const int P, const int L,
    int* sendCnts, int* sendDisps, int* recvCnts, int* recvDisps, MPI_Comm subComm) {
  PetscLogEventBegin(l2tEvent, 0, 0, 0, 0);

  int npes;
  MPI_Comm_size(subComm, &npes);

  //Fgt box size = sqrt(delta)
  const double hFgt = 1.0/(static_cast<double>(1u << FgtLev));

  const double LbyP = static_cast<double>(L)/static_cast<double>(P);
  const double ImExpZfactor = LbyP/hFgt;
  const double ReExpZfactor = -0.25*LbyP*LbyP;
  const double C0 = (0.125*LbyP*LbyP*LbyP/(__SQRT_PI__*__SQRT_PI__*__SQRT_PI__));

  //2P complex coefficients for each dimension.  
  const unsigned int numWcoeffs = 16*P*P*P;

  std::vector<double> sendLlist(recvDisps[npes - 1] + recvCnts[npes - 1]);

  for(size_t i = 0; i < sendLlist.size(); i += numWcoeffs) {
    for(unsigned int d = 0; d < numWcoeffs; ++d) {
      sendLlist[i + d] = localLlist[(numWcoeffs*(fgtList.size() - 1)) + d];
    }//end d
  }//end i

  std::vector<double> recvLlist;
  if(remoteFgtOwner >= 0) {
    recvLlist.resize(numWcoeffs);
  }

  double* sendBuf = NULL;
  if(!(sendLlist.empty())) {
    sendBuf = &(sendLlist[0]);
  }
  double* recvBuf = NULL;
  if(!(recvLlist.empty())) {
    recvBuf = &(recvLlist[0]);
  }
  MPI_Alltoallv(sendBuf, recvCnts, recvDisps, MPI_DOUBLE,
      recvBuf, sendCnts, sendDisps, MPI_DOUBLE, subComm);

  if(remoteFgtOwner >= 0) {
    double cx = (0.5*hFgt) + ((static_cast<double>(remoteFgt.getX()))/(__DTPMD__));
    double cy = (0.5*hFgt) + ((static_cast<double>(remoteFgt.getY()))/(__DTPMD__));
    double cz = (0.5*hFgt) + ((static_cast<double>(remoteFgt.getZ()))/(__DTPMD__));
    for(int i = 0; i < numPtsInRemoteFgt; ++i) {
      double px = sources[4*i];
      double py = sources[(4*i)+1];
      double pz = sources[(4*i)+2];
      for(int k3 = -P, di = 0; k3 < P; k3++) {
        double thetaZ = (static_cast<double>(k3))*(pz - cz);
        for(int k2 = -P; k2 < P; k2++) {
          double thetaY = (static_cast<double>(k2))*(py - cy);
          for(int k1 = -P; k1 < P; k1++, di++) {
            double thetaX = (static_cast<double>(k1))*(px - cx);
            double theta = ImExpZfactor*(thetaX + thetaY + thetaZ);
            double factor = C0*exp(ReExpZfactor*(static_cast<double>((k1*k1) + (k2*k2) + (k3*k3))));
            double a = recvLlist[2*di];
            double b = recvLlist[(2*di) + 1];
            double c = cos(theta);
            double d = sin(theta);
            results[i] += (factor*( (a*c) - (b*d) ));
          }//end for k1
        }//end for k2
      }//end for k3
    }//end i
  }

  for(int i = 0, ptsIdx = numPtsInRemoteFgt; i < fgtList.size(); ++i) {
    double cx = (0.5*hFgt) + ((static_cast<double>(fgtList[i].getX()))/(__DTPMD__));
    double cy = (0.5*hFgt) + ((static_cast<double>(fgtList[i].getY()))/(__DTPMD__));
    double cz = (0.5*hFgt) + ((static_cast<double>(fgtList[i].getZ()))/(__DTPMD__));
    for(int j = 0; j < fgtList[i].getWeight(); ++j, ++ptsIdx) {
      double px = sources[4*ptsIdx];
      double py = sources[(4*ptsIdx)+1];
      double pz = sources[(4*ptsIdx)+2];
      for(int k3 = -P, di = 0; k3 < P; k3++) {
        double thetaZ = (static_cast<double>(k3))*(pz - cz);
        for(int k2 = -P; k2 < P; k2++) {
          double thetaY = (static_cast<double>(k2))*(py - cy);
          for(int k1 = -P; k1 < P; k1++, di++) {
            double thetaX = (static_cast<double>(k1))*(px - cx);
            double theta = ImExpZfactor*(thetaX + thetaY + thetaZ);
            double factor = C0*exp(ReExpZfactor*(static_cast<double>((k1*k1) + (k2*k2) + (k3*k3))));
            double a = localLlist[(numWcoeffs*i) + (2*di)];
            double b = localLlist[(numWcoeffs*i) + (2*di) + 1];
            double c = cos(theta);
            double d = sin(theta);
            results[ptsIdx] += (factor*( (a*c) - (b*d) ));
          }//end for k1
        }//end for k2
      }//end for k3
    }//end j
  }//end i

  PetscLogEventEnd(l2tEvent, 0, 0, 0, 0);
}

void w2l(std::vector<double> & localLlist, std::vector<double> & localWlist, 
    std::vector<ot::TreeNode> & fgtList, std::vector<ot::TreeNode> & fgtMins,
    const unsigned int FgtLev, const int P, const int L, const int K, MPI_Comm subComm) {
  PetscLogEventBegin(w2lEvent, 0, 0, 0, 0);

  int npes;
  MPI_Comm_size(subComm, &npes);

  //Fgt box size = sqrt(delta)
  const double hFgt = 1.0/(static_cast<double>(1u << FgtLev));

  const unsigned int cellsPerFgt = (1u << (__MAX_DEPTH__ - FgtLev));

  //2P complex coefficients for each dimension.  
  const unsigned int numWcoeffs = 16*P*P*P;

  const double LbyP = static_cast<double>(L)/static_cast<double>(P);
  const double ImExpZfactor = LbyP/hFgt;

  std::vector<ot::TreeNode> tmpBoxes;
  std::vector<double> tmpVals;

  for(size_t i = 0; i < fgtList.size(); ++i) {
    unsigned int bAx = fgtList[i].getX();
    unsigned int bAy = fgtList[i].getY();
    unsigned int bAz = fgtList[i].getZ();
    double bx = (0.5*hFgt) + ((static_cast<double>(bAx))/(__DTPMD__));
    double by = (0.5*hFgt) + ((static_cast<double>(bAy))/(__DTPMD__));
    double bz = (0.5*hFgt) + ((static_cast<double>(bAz))/(__DTPMD__));
    unsigned int dAxs, dAxe, dAys, dAye, dAzs, dAze;
    if(bAx >= (K*cellsPerFgt)) {
      dAxs = bAx - (K*cellsPerFgt);
    } else {
      dAxs = 0; 
    }
    if((bAx + ((K + 1)*cellsPerFgt)) <= (__ITPMD__)) {
      dAxe = bAx + (K*cellsPerFgt);
    } else {
      dAxe = (__ITPMD__) - cellsPerFgt;
    }
    if(bAy >= (K*cellsPerFgt)) {
      dAys = bAy - (K*cellsPerFgt);
    } else {
      dAys = 0; 
    }
    if((bAy + ((K + 1)*cellsPerFgt)) <= (__ITPMD__)) {
      dAye = bAy + (K*cellsPerFgt);
    } else {
      dAye = (__ITPMD__) - cellsPerFgt;
    }
    if(bAz >= (K*cellsPerFgt)) {
      dAzs = bAz - (K*cellsPerFgt);
    } else {
      dAzs = 0; 
    }
    if((bAz + ((K + 1)*cellsPerFgt)) <= (__ITPMD__)) {
      dAze = bAz + (K*cellsPerFgt);
    } else {
      dAze = (__ITPMD__) - cellsPerFgt;
    }
    for(int dAz = dAzs; dAz <= dAze; dAz += cellsPerFgt) {
      for(int dAy = dAys; dAy <= dAye; dAy += cellsPerFgt) {
        for(int dAx = dAxs; dAx <= dAxe; dAx += cellsPerFgt) {
          ot::TreeNode boxD(dAx, dAy, dAz, FgtLev, __DIM__, __MAX_DEPTH__);
          boxD.setWeight(tmpBoxes.size());
          tmpBoxes.push_back(boxD);
          double dx = (0.5*hFgt) + ((static_cast<double>(dAx))/(__DTPMD__));
          double dy = (0.5*hFgt) + ((static_cast<double>(dAy))/(__DTPMD__));
          double dz = (0.5*hFgt) + ((static_cast<double>(dAz))/(__DTPMD__));
          for(int k3 = -P, di = 0; k3 < P; ++k3) {
            double thetaZ = (static_cast<double>(k3))*(dz - bz);
            for(int k2 = -P; k2 < P; ++k2) {
              double thetaY = (static_cast<double>(k2))*(dy - by);
              for(int k1 = -P; k1 < P; ++k1, ++di) {
                double thetaX = (static_cast<double>(k1))*(dx - bx);
                double theta = ImExpZfactor*(thetaX + thetaY + thetaZ);
                double a = localWlist[(numWcoeffs*i) + (2*di)];
                double b = localWlist[(numWcoeffs*i) + (2*di) + 1];
                double c = cos(theta);
                double d = sin(theta);
                double reVal = ((a*c) - (b*d));
                double imVal = ((a*d) + (b*c));
                tmpVals.push_back(reVal);
                tmpVals.push_back(imVal);
              }//end for k1
            }//end for k2
          }//end for k3
        }//end dAx
      }//end dAy
    }//end dAz
  }//end i

  std::vector<ot::TreeNode> sendBoxList;
  std::vector<double> sendLlist;

  if(!(tmpBoxes.empty())) {
    std::sort(tmpBoxes.begin(), tmpBoxes.end());

    sendLlist.insert(sendLlist.end(), tmpVals.begin() + (numWcoeffs*(tmpBoxes[0].getWeight())),
        tmpVals.begin() + (numWcoeffs*(tmpBoxes[0].getWeight() + 1)));
    sendBoxList.push_back(tmpBoxes[0]);
    for(size_t i = 1; i < tmpBoxes.size(); ++i) {
      if(tmpBoxes[i] == tmpBoxes[i - 1]) {
        for(int d = 0; d < numWcoeffs; ++d) {
          sendLlist[sendLlist.size() - 1 - d] += tmpVals[(numWcoeffs*(tmpBoxes[i].getWeight() + 1)) - 1 - d];
        }//end d
      } else {
        sendLlist.insert(sendLlist.end(), tmpVals.begin() + (numWcoeffs*(tmpBoxes[i].getWeight())),
            tmpVals.begin() + (numWcoeffs*(tmpBoxes[i].getWeight() + 1)));
        sendBoxList.push_back(tmpBoxes[i]);
      }
    }//end i
  }

  tmpBoxes.clear();
  tmpVals.clear();

  int* sendCnts = new int[npes];
  int* sendDisps = new int[npes];
  int* recvCnts = new int[npes];
  int* recvDisps = new int[npes];

  for(int i = 0; i < npes; ++i) {
    sendCnts[i] = 0;
  }//end i 

  //Performance Improvement: This binary search can be avoided by using the
  //fact that sendBoxList is sorted.
  for(size_t i = 0; i < sendBoxList.size(); ++i) {
    unsigned int retIdx;
    bool found = seq::maxLowerBound(fgtMins, sendBoxList[i], retIdx, NULL, NULL);
    if(found) {
      ++(sendCnts[fgtMins[retIdx].getWeight()]);
    }
  }//end i

  MPI_Alltoall(sendCnts, 1, MPI_INT, recvCnts, 1, MPI_INT, subComm);

  sendDisps[0] = 0;
  recvDisps[0] = 0;
  for(int i = 1; i < npes; ++i) {
    sendDisps[i] = sendDisps[i - 1] + sendCnts[i - 1];
    recvDisps[i] = recvDisps[i - 1] + recvCnts[i - 1];
  }//end i

  std::vector<ot::TreeNode> recvBoxList(recvDisps[npes - 1] + recvCnts[npes - 1]);

  ot::TreeNode* sendBuf1 = NULL;
  if(!(sendBoxList.empty())) {
    sendBuf1 = &(sendBoxList[0]);
  }

  ot::TreeNode* recvBuf1 = NULL;
  if(!(recvBoxList.empty())) {
    recvBuf1 = &(recvBoxList[0]);
  }

  MPI_Alltoallv(sendBuf1, sendCnts, sendDisps, par::Mpi_datatype<ot::TreeNode>::value(),
      recvBuf1, recvCnts, recvDisps, par::Mpi_datatype<ot::TreeNode>::value(), subComm);

  for(int i = 0; i < npes; ++i) {
    sendCnts[i] *= numWcoeffs;
    sendDisps[i] *= numWcoeffs;
    recvCnts[i] *= numWcoeffs;
    recvDisps[i] *= numWcoeffs;
  }//end i 

  std::vector<double> recvLlist(recvDisps[npes - 1] + recvCnts[npes - 1]);

  double* sendBuf2 = NULL;
  if(!(sendLlist.empty())) {
    sendBuf2 = &(sendLlist[0]);
  }

  double* recvBuf2 = NULL;
  if(!(recvLlist.empty())) {
    recvBuf2 = &(recvLlist[0]);
  }

  MPI_Alltoallv(sendBuf2, sendCnts, sendDisps, MPI_DOUBLE,
      recvBuf2, recvCnts, recvDisps, MPI_DOUBLE, subComm);

  delete [] sendCnts;
  delete [] sendDisps;
  delete [] recvCnts;
  delete [] recvDisps;

  //Performance Improvement: This binary search can be avoided by making use of
  //the fact that recvBoxList is sorted within each processor chunk.
  for(size_t i = 0; i < recvBoxList.size(); ++i) {
    unsigned int retIdx;
    bool found = seq::BinarySearch(&(fgtList[0]), fgtList.size(), recvBoxList[i], &retIdx);
    if(found) {
      for(int d = 0; d < numWcoeffs; ++d) {
        localLlist[(numWcoeffs*retIdx) + d] += recvLlist[(numWcoeffs*i) + d];
      }//end d
    }
  }//end i

  PetscLogEventEnd(w2lEvent, 0, 0, 0, 0);
}

void d2d(std::vector<double> & results, std::vector<double> & sources,
    std::vector<ot::TreeNode> & nodes, std::vector<ot::TreeNode> & directMins,
    const unsigned int FgtLev, const double epsilon, MPI_Comm subComm) {
  PetscLogEventBegin(d2dEvent, 0, 0, 0, 0);

  int npes;
  MPI_Comm_size(subComm, &npes);

  //Fgt box size = sqrt(delta)
  const double hFgt = 1.0/(static_cast<double>(1u << FgtLev));

  const double Iwidth = hFgt*(sqrt(-log(epsilon)));

  const double IwidthSqr = Iwidth*Iwidth;

  const double delta = hFgt*hFgt;

  int* sendCnts = new int[npes];
  int* sendDisps = new int[npes];
  int* recvCnts = new int[npes];
  int* recvDisps = new int[npes];

  for(int i = 0; i < npes; ++i) {
    sendCnts[i] = 0;
  }//end i

  //Performance Improvement: All the searches can be avoided by making use of
  //the fact that sources is sorted and so the minPts are sorted and the maxPts
  //are sorted. Also, each processor's chunk of the sendList is sorted, i.e.
  //tmpSendList[i] is sorted for all i. Although, sendList may not be sorted.
  //recvList will be sorted.

  std::vector<std::vector<double> > tmpSendList(npes);

  for(size_t i = 0; i < sources.size(); i += 4) {
    unsigned int uiMinPt[3];
    unsigned int uiMaxPt[3];
    for(int d = 0; d < 3; ++d) {
      double minPt, maxPt;
      minPt = sources[i + d] - Iwidth;
      if(minPt < 0.0) {
        minPt = 0.0;
      }
      maxPt = sources[i + d] + Iwidth;
      if(maxPt > 1.0) {
        maxPt = 1.0;
      }
      uiMinPt[d] = static_cast<unsigned int>(floor(minPt*(__DTPMD__)));
      uiMaxPt[d] = static_cast<unsigned int>(ceil(maxPt*(__DTPMD__)));
    }//end d

    ot::TreeNode minNode(uiMinPt[0], uiMinPt[1], uiMinPt[2], __MAX_DEPTH__, __DIM__, __MAX_DEPTH__);
    ot::TreeNode maxNode((uiMaxPt[0] - 1), (uiMaxPt[1] - 1), (uiMaxPt[2] - 1), __MAX_DEPTH__, __DIM__, __MAX_DEPTH__);

    unsigned int minIdx;
    bool foundMin = seq::maxLowerBound<ot::TreeNode>(directMins, minNode, minIdx, NULL, NULL);

    if(!foundMin) {
      minIdx = 0;
    }

    unsigned int maxIdx;
    bool foundMax = seq::maxLowerBound<ot::TreeNode>(directMins, maxNode, maxIdx, NULL, NULL);

    //maxPt >= currPt and currPt is a direct point
    assert(foundMax);

    for(int j = minIdx; j <= maxIdx; ++j) {
      for(int d = 0; d < 4; ++d) {
        tmpSendList[j].push_back(sources[i + d]);
      }//end d
      sendCnts[j] += 4;
    }//end j
  }//end i

  MPI_Alltoall(sendCnts, 1, MPI_INT, recvCnts, 1, MPI_INT, subComm);

  sendDisps[0] = 0;
  recvDisps[0] = 0;
  for(int i = 1; i < npes; ++i) {
    sendDisps[i] = sendDisps[i - 1] + sendCnts[i - 1];
    recvDisps[i] = recvDisps[i - 1] + recvCnts[i - 1];
  }//end i

  std::vector<double> sendList(sendDisps[npes - 1] + sendCnts[npes - 1]);

  for(int i = 0; i < npes; ++i) {
    for(int j = 0; j < sendCnts[i]; ++j) {
      sendList[sendDisps[i] + j] = tmpSendList[i][j];
    }//end j
  }//end i

  tmpSendList.clear();

  std::vector<double> recvList(recvDisps[npes - 1] + recvCnts[npes - 1]);

  double* sendBuf = NULL;
  if(!(sendList.empty())) {
    sendBuf = &(sendList[0]);
  }

  double* recvBuf = NULL;
  if(!(recvList.empty())) {
    recvBuf = &(recvList[0]);
  }

  MPI_Alltoallv(sendBuf, sendCnts, sendDisps, MPI_DOUBLE,
      recvBuf, recvCnts, recvDisps, MPI_DOUBLE, subComm);

  sendList.clear();

  delete [] sendCnts;
  delete [] sendDisps;
  delete [] recvCnts;
  delete [] recvDisps;

  for(size_t i = 0; i < recvList.size(); i += 4) {
    unsigned int uiMinPt[3];
    unsigned int uiMaxPt[3];
    for(int d = 0; d < 3; ++d) {
      double minPt, maxPt;
      minPt = recvList[i + d] - Iwidth;
      if(minPt < 0.0) {
        minPt = 0.0;
      }
      maxPt = recvList[i + d] + Iwidth;
      if(maxPt > 1.0) {
        maxPt = 1.0;
      }
      uiMinPt[d] = static_cast<unsigned int>(floor(minPt*(__DTPMD__)));
      uiMaxPt[d] = static_cast<unsigned int>(ceil(maxPt*(__DTPMD__)));
    }//end d

    ot::TreeNode minNode(uiMinPt[0], uiMinPt[1], uiMinPt[2], __MAX_DEPTH__, __DIM__, __MAX_DEPTH__);
    ot::TreeNode maxNode((uiMaxPt[0] - 1), (uiMaxPt[1] - 1), (uiMaxPt[2] - 1), __MAX_DEPTH__, __DIM__, __MAX_DEPTH__);

    unsigned int minIdx;
    bool foundMin = seq::maxLowerBound<ot::TreeNode>(nodes, minNode, minIdx, NULL, NULL);

    if(!foundMin) {
      minIdx = 0;
    }

    unsigned int maxIdx;
    bool foundMax = seq::maxLowerBound<ot::TreeNode>(nodes, maxNode, maxIdx, NULL, NULL);

    //This source point was sent only to those procs whose directMin <= maxPt
    assert(foundMax);

    for(int j = minIdx; j <= maxIdx; ++j) {
      double distSqr = 0.0;
      for(int d = 0; d < 3; ++d) {
        distSqr += ((sources[(4*j) + d] - recvList[i + d])*(sources[(4*j) + d] - recvList[i + d]));
      }//end d
      if(distSqr < IwidthSqr) {
        results[j] += (recvList[i + 3]*exp(-distSqr/delta));
      }
    }//end j
  }//end i

  PetscLogEventEnd(d2dEvent, 0, 0, 0, 0);
}

void w2dAndD2lExpand(std::vector<double> & localLlist, std::vector<double> & localWlist, 
    std::vector<ot::TreeNode> & fgtList, const int P, MPI_Comm comm) {
  PetscLogEventBegin(w2dD2lExpandEvent, 0, 0, 0, 0);

  int npes;
  MPI_Comm_size(comm, &npes);

  //2P complex coefficients for each dimension.  
  const unsigned int numWcoeffs = 16*P*P*P;

  int* sendCnts = new int[npes];
  int* sendDisps = new int[npes];
  for(int i = 0; i < npes; ++i) {
    sendCnts[i] = 0;
    sendDisps[i] = 0;
  }//end i 

  int* recvCnts = new int[npes];

  MPI_Alltoall(sendCnts, 1, MPI_INT, recvCnts, 1, MPI_INT, comm);

  int* recvDisps = new int[npes];
  recvDisps[0] = 0;
  for(int i = 1; i < npes; ++i) {
    recvDisps[i] = recvDisps[i - 1] + recvCnts[i - 1];
  }//end i

  std::vector<ot::TreeNode> recvBoxList(recvDisps[npes - 1] + recvCnts[npes - 1]);

  ot::TreeNode* recvBoxListPtr = NULL;
  if(!(recvBoxList.empty())) {
    recvBoxListPtr = &(recvBoxList[0]);
  }

  MPI_Alltoallv(NULL, sendCnts, sendDisps, par::Mpi_datatype<ot::TreeNode>::value(),
      recvBoxListPtr, recvCnts, recvDisps, par::Mpi_datatype<ot::TreeNode>::value(), comm);

  delete [] sendCnts;
  delete [] sendDisps;

  //Performance Improvement: We can use the fact that each processor's chunk in
  //the recvBoxList is sorted and avoid the searches.
  std::vector<int> recvBoxIds(recvBoxList.size(), -1);
  for(int i = 0; i < recvBoxList.size(); ++i) { 
    unsigned int retIdx;
    bool found = seq::BinarySearch(&(fgtList[0]), fgtList.size(), recvBoxList[i], &retIdx);
    if(found) {
      recvBoxIds[i] = retIdx;
    }
  }//end i

  //Performance Improvement (Probably necessary when sources != targets) Incur
  //a synchronization penalty and remove invalid boxes. This will reduce the
  //message size for the subsequent communication. Also mark the boxes as d2l
  //candidates, w2d candidates or both. This info could be used to further reduce the
  //message size for the subsequent communication.   

  for(int i = 0; i < npes; ++i) {
    recvCnts[i] *= numWcoeffs;
    recvDisps[i] *= numWcoeffs;
  }//end i

  std::vector<double> sendWlist((recvDisps[npes - 1] + recvCnts[npes - 1]), 0.0);
  for(int i = 0; i < recvBoxIds.size(); ++i) {
    if(recvBoxIds[i] >= 0) {
      for(int d = 0; d < numWcoeffs; ++d) {
        sendWlist[(numWcoeffs*i) + d] = localWlist[(numWcoeffs*recvBoxIds[i]) + d];
      }//end d
    }
  }//end i

  std::vector<double> recvLlist(recvDisps[npes - 1] + recvCnts[npes - 1]);

  double* sendWlistPtr = NULL;
  if(!(sendWlist.empty())) {
    sendWlistPtr = &(sendWlist[0]);
  }

  double* recvLlistPtr = NULL;
  if(!(recvLlist.empty())) {
    recvLlistPtr = &(recvLlist[0]);
  }

  MPI_Alltoallv(sendWlistPtr, recvCnts, recvDisps, par::Mpi_datatype<ot::TreeNode>::value(),
      recvLlistPtr, recvCnts, recvDisps, par::Mpi_datatype<ot::TreeNode>::value(), comm);

  for(int i = 0; i < recvBoxIds.size(); ++i) {
    if(recvBoxIds[i] >= 0) {
      for(int d = 0; d < numWcoeffs; ++d) {
        localLlist[(numWcoeffs*recvBoxIds[i]) + d] += recvLlist[(numWcoeffs*i) + d];
      }//end d
    }
  }//end i

  delete [] recvCnts;
  delete [] recvDisps;

  PetscLogEventEnd(w2dD2lExpandEvent, 0, 0, 0, 0);
}

void w2dAndD2lDirect(std::vector<double> & results, std::vector<double> & sources,
    std::vector<ot::TreeNode> & fgtMins, const unsigned int FgtLev, 
    const int P, const int L, const int K, const double epsilon, MPI_Comm comm) {
  PetscLogEventBegin(w2dD2lDirectEvent, 0, 0, 0, 0);

  int npes;
  MPI_Comm_size(comm, &npes);

  //2P complex coefficients for each dimension.  
  const unsigned int numWcoeffs = 16*P*P*P;

  const unsigned int cellsPerFgt = (1u << (__MAX_DEPTH__ - FgtLev));

  const unsigned int twoPowFgtLev = (1u << FgtLev);

  const double invHfgt =  static_cast<double>(twoPowFgtLev);

  //Fgt box size = sqrt(delta)
  const double hFgt = 1.0/invHfgt;
  const double delta = hFgt*hFgt;

  const double ptIwidth = hFgt*(sqrt(-log(epsilon)));
  const double ptIwidthSqr = ptIwidth*ptIwidth;

  const double LbyP = static_cast<double>(L)/static_cast<double>(P);
  const double ImExpZfactor = LbyP/hFgt;
  const double ReExpZfactor = -0.25*LbyP*LbyP;
  const double C0 = (0.125*LbyP*LbyP*LbyP/(__SQRT_PI__*__SQRT_PI__*__SQRT_PI__));

  std::vector<ot::TreeNode> tmpSendBoxList;

  for(int i = 0; i < sources.size(); i += 4) {
    unsigned int uiMinPt1[3];
    unsigned int uiMaxPt1[3];
    unsigned int uiMinPt2[3];
    unsigned int uiMaxPt2[3];
    for(int d = 0; d < 3; ++d) {
      double minPt1 = sources[i + d] - ptIwidth;
      double maxPt1 = sources[i + d] + ptIwidth;
      double minVal2 = ((__DTPMD__)*sources[i + d]) - static_cast<double>((K + 1)*cellsPerFgt);
      double maxVal2 = ((__DTPMD__)*sources[i + d]) + static_cast<double>(K*cellsPerFgt);
      if(minPt1 < 0.0) {
        minPt1 = 0.0;
      }
      if(maxPt1 > 1.0) {
        maxPt1 = 1.0;
      }
      if(minVal2 < 0.0) {
        minVal2 = 0.0;
      }
      if(maxVal2 > (__DTPMD__)) {
        maxVal2 = (__DTPMD__);
      }
      uiMinPt1[d] = static_cast<unsigned int>(floor(minPt1*invHfgt));
      uiMaxPt1[d] = static_cast<unsigned int>(ceil(maxPt1*invHfgt));
      uiMinPt2[d] = 1 + static_cast<unsigned int>(floor(minVal2));
      uiMaxPt2[d] = static_cast<unsigned int>(ceil(maxVal2));
    }//end d
    std::vector<ot::TreeNode> selectedBoxes;
    //Target box is in interaction list of source point.
    for(int zi = uiMinPt1[2]; zi < uiMaxPt1[2]; ++zi) {
      for(int yi = uiMinPt1[1]; yi < uiMaxPt1[1]; ++yi) {
        for(int xi = uiMinPt1[0]; xi < uiMaxPt1[0]; ++xi) {
          ot::TreeNode tmpBox((xi*cellsPerFgt), (yi*cellsPerFgt), (zi*cellsPerFgt),
              FgtLev, __DIM__, __MAX_DEPTH__);
          selectedBoxes.push_back(tmpBox);
        }//end xi
      }//end yi
    }//end zi
    //Target point is in interaction list of source box.
    for(int zi = uiMinPt2[2]; zi < uiMaxPt2[2]; zi += cellsPerFgt) {
      for(int yi = uiMinPt2[1]; yi < uiMaxPt2[1]; yi += cellsPerFgt) {
        for(int xi = uiMinPt2[0]; xi < uiMaxPt2[0]; xi += cellsPerFgt) {
          ot::TreeNode tmpBox(xi, yi, zi, FgtLev, __DIM__, __MAX_DEPTH__);
          selectedBoxes.push_back(tmpBox);
        }//end xi
      }//end yi
    }//end zi
    seq::makeVectorUnique(selectedBoxes, false);
    for(int j = 0; j < selectedBoxes.size(); ++j) {
      selectedBoxes[j].setWeight(i);
      tmpSendBoxList.push_back(selectedBoxes[j]);
    }//end j
  }//end i

  std::vector<ot::TreeNode> sendBoxList;
  std::vector<std::vector<unsigned int> > box2PtMap;

  if(!(tmpSendBoxList.empty())) {
    std::sort((&(tmpSendBoxList[0])), (&(tmpSendBoxList[0])) + tmpSendBoxList.size());
    sendBoxList.push_back(tmpSendBoxList[0]);
    unsigned int ptId = tmpSendBoxList[0].getWeight();
    std::vector<unsigned int> tmpPtIdVec(1, ptId);
    box2PtMap.push_back(tmpPtIdVec);
  }

  for(int i = 1; i < tmpSendBoxList.size(); ++i) {
    unsigned int ptId = tmpSendBoxList[i].getWeight();
    if(tmpSendBoxList[i] == sendBoxList[sendBoxList.size() - 1]) {
      box2PtMap[box2PtMap.size() - 1].push_back(ptId);
    } else {
      sendBoxList.push_back(tmpSendBoxList[i]);
      std::vector<unsigned int> tmpPtIdVec(1, ptId);
      box2PtMap.push_back(tmpPtIdVec);
    }
  }//end i

  tmpSendBoxList.clear();

  int* sendCnts = new int[npes];
  int* recvDisps = new int[npes]; 
  for(int i = 0; i < npes; ++i) {
    sendCnts[i] = 0;
    recvDisps[i] = 0;
  }//end i

  //Performance Improvement: We could make use of the fact that sendBoxList is
  //sorted and avoid the searches.
  for(int i = 0; i < sendBoxList.size(); ++i) {
    unsigned int idx;
    bool found = seq::maxLowerBound<ot::TreeNode>(fgtMins, sendBoxList[i], idx, NULL, NULL);
    if(found) {
      ++(sendCnts[fgtMins[idx].getWeight()]);
    }
  }//end i

  int* recvCnts = new int[npes];

  MPI_Alltoall(sendCnts, 1, MPI_INT, recvCnts, 1, MPI_INT, comm);

  int* sendDisps = new int[npes];
  sendDisps[0] = 0;
  for(int i = 1; i < npes; ++i) {
    sendDisps[i] = sendDisps[i - 1] + sendCnts[i - 1];
  }//end i

  ot::TreeNode* sendBoxListPtr = NULL;
  if(!(sendBoxList.empty())) {
    sendBoxListPtr = &(sendBoxList[0]);
  }

  MPI_Alltoallv(sendBoxListPtr, sendCnts, sendDisps, par::Mpi_datatype<ot::TreeNode>::value(),
      NULL, recvCnts, recvDisps, par::Mpi_datatype<ot::TreeNode>::value(), comm);

  delete [] recvCnts;
  delete [] recvDisps;

  //Performance Improvement (Probably necessary when sources != targets) Incur
  //a synchronization penalty and remove invalid boxes. This will reduce the
  //message size for the subsequent communication. Also mark the boxes as d2l
  //candidates, w2d candidates or both. This info could be used to further reduce the
  //message size for the subsequent communication.   

  for(int i = 0; i < npes; ++i) {
    sendCnts[i] *= numWcoeffs;
    sendDisps[i] *= numWcoeffs;
  }//end i

  std::vector<double> sendLlist((sendDisps[npes - 1] + sendCnts[npes - 1]), 0.0);
  for(int i = 0; i < sendBoxList.size(); ++i) {
    double cx = (0.5*hFgt) + ((static_cast<double>(sendBoxList[i].getX()))/(__DTPMD__));
    double cy = (0.5*hFgt) + ((static_cast<double>(sendBoxList[i].getY()))/(__DTPMD__));
    double cz = (0.5*hFgt) + ((static_cast<double>(sendBoxList[i].getZ()))/(__DTPMD__));
    for(int j = 0; j < box2PtMap[i].size(); ++j) {
      double px = sources[box2PtMap[i][j]];
      double py = sources[box2PtMap[i][j] + 1];
      double pz = sources[box2PtMap[i][j] + 2];
      double pf = sources[box2PtMap[i][j] + 3];
      for(int k3 = -P, di = 0; k3 < P; k3++) {
        double thetaZ = (static_cast<double>(k3))*(cz - pz);
        for(int k2 = -P; k2 < P; k2++) {
          double thetaY = (static_cast<double>(k2))*(cy - py);
          for(int k1 = -P; k1 < P; k1++, di++) {
            double thetaX = (static_cast<double>(k1))*(cx - px);
            double theta = ImExpZfactor*(thetaX + thetaY + thetaZ);
            sendLlist[(numWcoeffs*i) + (2*di)] += (pf*cos(theta));
            sendLlist[(numWcoeffs*i) + (2*di) + 1] += (pf*sin(theta));
          }//end for k1
        }//end for k2
      }//end for k3
    }//end j
  }//end i

  std::vector<double> recvWlist(sendDisps[npes - 1] + sendCnts[npes - 1]);

  double* sendLlistPtr = NULL;
  if(!(sendLlist.empty())) {
    sendLlistPtr = &(sendLlist[0]);
  }

  double* recvWlistPtr = NULL;
  if(!(recvWlist.empty())) {
    recvWlistPtr = &(recvWlist[0]);
  }

  MPI_Alltoallv(sendLlistPtr, sendCnts, sendDisps, par::Mpi_datatype<ot::TreeNode>::value(),
      recvWlistPtr, sendCnts, sendDisps, par::Mpi_datatype<ot::TreeNode>::value(), comm);

  delete [] sendCnts;
  delete [] sendDisps;

  for(int i = 0; i < sendBoxList.size(); ++i) {
    double cx = (0.5*hFgt) + ((static_cast<double>(sendBoxList[i].getX()))/(__DTPMD__));
    double cy = (0.5*hFgt) + ((static_cast<double>(sendBoxList[i].getY()))/(__DTPMD__));
    double cz = (0.5*hFgt) + ((static_cast<double>(sendBoxList[i].getZ()))/(__DTPMD__));
    for(int j = 0; j < box2PtMap[i].size(); ++j) {
      double px = sources[box2PtMap[i][j]];
      double py = sources[box2PtMap[i][j] + 1];
      double pz = sources[box2PtMap[i][j] + 2];
      for(int k3 = -P, di = 0; k3 < P; k3++) {
        double thetaZ = (static_cast<double>(k3))*(pz - cz);
        for(int k2 = -P; k2 < P; k2++) {
          double thetaY = (static_cast<double>(k2))*(py - cy);
          for(int k1 = -P; k1 < P; k1++, di++) {
            double thetaX = (static_cast<double>(k1))*(px - cx);
            double theta = ImExpZfactor*(thetaX + thetaY + thetaZ);
            double factor = C0*exp(ReExpZfactor*(static_cast<double>((k1*k1) + (k2*k2) + (k3*k3))));
            double a = recvWlist[(numWcoeffs*i) + (2*di)];
            double b = recvWlist[(numWcoeffs*i) + (2*di) + 1];
            double c = cos(theta);
            double d = sin(theta);
            results[(box2PtMap[i][j])/4] += (factor*( (a*c) - (b*d) ));
          }//end for k1
        }//end for k2
      }//end for k3
    }//end j
  }//end i

  PetscLogEventEnd(w2dD2lDirectEvent, 0, 0, 0, 0);
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

void splitSources(std::vector<double>& sources, const unsigned int minPtsInFgt, 
    const unsigned int FgtLev, std::vector<double>& expandSources, std::vector<double>& directSources, 
    std::vector<ot::TreeNode>& fgtList, MPI_Comm comm) {
  PetscLogEventBegin(splitSourcesEvent, 0, 0, 0, 0);

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
      numPts = ((sources.size())/4);
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

  PetscLogEventEnd(splitSourcesEvent, 0, 0, 0, 0);
}




