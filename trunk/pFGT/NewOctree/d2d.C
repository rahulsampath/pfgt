
#include "petsc.h"
#include <vector>
#include "oct/TreeNode.h"
#include "pfgtOctUtils.h"
#include "par/parUtils.h"
#include "par/dtypes.h"

extern PetscLogEvent d2dEvent;

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
      double minPt = sources[i + d] - Iwidth;
      if(minPt < 0.0) {
        minPt = 0.0;
      }
      double maxPt = sources[i + d] + Iwidth;
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
#ifdef DEBUG
    assert(foundMax);
#endif

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
    double x1 = recvList[i];
    double x2 = recvList[i + 1];
    double x3 = recvList[i + 2];
    double fx = recvList[i + 3];
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
#ifdef DEBUG
    assert(foundMax);
#endif

    for(int j = minIdx; j <= maxIdx; ++j) {
      int sOff = 4*j;
      double y1 = sources[sOff];
      double y2 = sources[sOff + 1];
      double y3 = sources[sOff + 2];
      double distSqr = ( ((x1-y1)*(x1-y1)) + ((x2-y2)*(x2-y2)) + ((x3-y3)*(x3-y3)) );
      if(distSqr < IwidthSqr) {
        results[j] += (fx*exp(-distSqr/delta));
      }
    }//end j
  }//end i

  PetscLogEventEnd(d2dEvent, 0, 0, 0, 0);
}


