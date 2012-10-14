
#include "petsc.h"
#include <vector>
#include "oct/TreeNode.h"
#include "pfgtOctUtils.h"
#include "par/parUtils.h"
#include "par/dtypes.h"

extern PetscLogEvent w2lEvent;
extern PetscLogEvent w2lSearchEvent;

void w2l(std::vector<double> & localLlist, std::vector<double> & localWlist, 
    std::vector<ot::TreeNode> & fgtList, std::vector<ot::TreeNode> & fgtMins,
    const unsigned int FgtLev, const int P, const int L, const unsigned long long int K, MPI_Comm subComm) {
  PetscLogEventBegin(w2lEvent, 0, 0, 0, 0);

  int npes;
  MPI_Comm_size(subComm, &npes);

  const unsigned long long int cellsPerFgt = (1ull << (__MAX_DEPTH__ - FgtLev));

  std::vector<ot::TreeNode> tmpBoxes;
  for(size_t i = 0; i < fgtList.size(); ++i) {
    unsigned long long int bAx = fgtList[i].getX();
    unsigned long long int bAy = fgtList[i].getY();
    unsigned long long int bAz = fgtList[i].getZ();
    unsigned long long int dAxs, dAxe, dAys, dAye, dAzs, dAze;
    if( bAx >= (K*cellsPerFgt) ) {
      dAxs = bAx - (K*cellsPerFgt);
    } else {
      dAxs = 0; 
    }
    if( (bAx + ((K + 1ull)*cellsPerFgt)) <= (__ITPMD__) ) {
      dAxe = bAx + (K*cellsPerFgt);
    } else {
      dAxe = (__ITPMD__) - cellsPerFgt;
    }
    if( bAy >= (K*cellsPerFgt) ) {
      dAys = bAy - (K*cellsPerFgt);
    } else {
      dAys = 0; 
    }
    if( (bAy + ((K + 1ull)*cellsPerFgt)) <= (__ITPMD__) ) {
      dAye = bAy + (K*cellsPerFgt);
    } else {
      dAye = (__ITPMD__) - cellsPerFgt;
    }
    if( bAz >= (K*cellsPerFgt) ) {
      dAzs = bAz - (K*cellsPerFgt);
    } else {
      dAzs = 0; 
    }
    if( (bAz + ((K + 1ull)*cellsPerFgt)) <= (__ITPMD__) ) {
      dAze = bAz + (K*cellsPerFgt);
    } else {
      dAze = (__ITPMD__) - cellsPerFgt;
    }
    for(unsigned int dAz = dAzs; dAz <= dAze; dAz += cellsPerFgt) {
      for(unsigned int dAy = dAys; dAy <= dAye; dAy += cellsPerFgt) {
        for(unsigned int dAx = dAxs; dAx <= dAxe; dAx += cellsPerFgt) {
          ot::TreeNode boxD(dAx, dAy, dAz, FgtLev, __DIM__, __MAX_DEPTH__);
          boxD.setWeight(i);
          tmpBoxes.push_back(boxD);
        }//end dAx
      }//end dAy
    }//end dAz
  }//end i

  std::vector<ot::TreeNode> sendBoxList;
  std::vector<unsigned int> sendBoxToMyBoxMap(tmpBoxes.size());
  if(!(tmpBoxes.empty())) {
    std::sort(tmpBoxes.begin(), tmpBoxes.end());

    sendBoxToMyBoxMap[0] = tmpBoxes[0].getWeight();
    sendBoxList.push_back(tmpBoxes[0]);
    sendBoxList[sendBoxList.size() - 1].setWeight(1);
    for(size_t i = 1; i < tmpBoxes.size(); ++i) {
      sendBoxToMyBoxMap[i] = tmpBoxes[i].getWeight();
      if(tmpBoxes[i] == tmpBoxes[i - 1]) {
        sendBoxList[sendBoxList.size() - 1].addWeight(1);
      } else {
        sendBoxList.push_back(tmpBoxes[i]);
        sendBoxList[sendBoxList.size() - 1].setWeight(1);
      }
    }//end i
  }
  tmpBoxes.clear();

  //Complex coefficients: [-P, P]x[-P, P]x[0, P] 
  //Coeff[-K1, -K2, -K3] = ComplexConjugate(Coeff[K1, K2, K3])
  const unsigned int TwoPplus1 = (2*P) + 1;
  const unsigned int numWcoeffs = 2*TwoPplus1*TwoPplus1*(P + 1);

  //Fgt box size = sqrt(delta)
  const double hFgt = 1.0/(static_cast<double>(1u << FgtLev));

  const double LbyP = static_cast<double>(L)/static_cast<double>(P);
  const double ImExpZfactor = LbyP/hFgt;

#ifdef DEBUG
  assert(P >= 1);
#endif
  std::vector<double> c1(P);
  std::vector<double> c2(P);
  std::vector<double> c3(P);
  std::vector<double> s1(P);
  std::vector<double> s2(P);
  std::vector<double> s3(P);

  double* c1Arr = (&(c1[0])) - 1;
  double* c2Arr = (&(c2[0])) - 1;
  double* c3Arr = (&(c3[0])) - 1;
  double* s1Arr = (&(s1[0])) - 1;
  double* s2Arr = (&(s2[0])) - 1;
  double* s3Arr = (&(s3[0])) - 1;

  std::vector<double> sendLlist((numWcoeffs*(sendBoxList.size())), 0);

  for(int i = 0, cnt = 0; i < sendBoxList.size(); ++i) {
    for(int j = 0; j < sendBoxList[i].getWeight(); ++j, ++cnt) {
    }//end j
  }//end i

  int* sendCnts = new int[npes];
  int* sendDisps = new int[npes];
  int* recvCnts = new int[npes];
  int* recvDisps = new int[npes];

  for(int i = 0; i < npes; ++i) {
    sendCnts[i] = 0;
  }//end i 

  //Performance Improvement: This binary search can be avoided by using the
  //fact that sendBoxList is sorted.
  PetscLogEventBegin(w2lSearchEvent, 0, 0, 0, 0);
  for(size_t i = 0; i < sendBoxList.size(); ++i) {
    unsigned int retIdx;
    bool found = seq::maxLowerBound(fgtMins, sendBoxList[i], retIdx, NULL, NULL);
    if(found) {
      ++(sendCnts[fgtMins[retIdx].getWeight()]);
    }
  }//end i
  PetscLogEventEnd(w2lSearchEvent, 0, 0, 0, 0);

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
  PetscLogEventBegin(w2lSearchEvent, 0, 0, 0, 0);
  for(size_t i = 0; i < recvBoxList.size(); ++i) {
    unsigned int retIdx;
    bool found = seq::BinarySearch(&(fgtList[0]), fgtList.size(), recvBoxList[i], &retIdx);
    if(found) {
      for(int d = 0; d < numWcoeffs; ++d) {
        localLlist[(numWcoeffs*retIdx) + d] += recvLlist[(numWcoeffs*i) + d];
      }//end d
    }
  }//end i
  PetscLogEventEnd(w2lSearchEvent, 0, 0, 0, 0);

  PetscLogEventEnd(w2lEvent, 0, 0, 0, 0);
}



