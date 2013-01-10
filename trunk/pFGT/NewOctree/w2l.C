
#include "petsc.h"
#include <vector>
#include "oct/TreeNode.h"
#include "pfgtOctUtils.h"
#include "par/parUtils.h"
#include "par/dtypes.h"

extern PetscLogEvent w2lEvent;
extern PetscLogEvent w2lSearchEvent;
extern PetscLogEvent w2lGenEvent;
extern PetscLogEvent w2lCoreEvent;

void w2l(std::vector<double> & localLlist, std::vector<double> & localWlist, 
    std::vector<ot::TreeNode> & fgtList, std::vector<ot::TreeNode> & fgtMins,
    const unsigned int FgtLev, const int P, const int L, const unsigned long long int K, MPI_Comm subComm) {
  PetscLogEventBegin(w2lEvent, 0, 0, 0, 0);

  int npes;
  MPI_Comm_size(subComm, &npes);

  //Fgt box size = sqrt(delta)
  const double hFgt = 1.0/(static_cast<double>(1u << FgtLev));

  const unsigned long long int cellsPerFgt = (1ull << (__MAX_DEPTH__ - FgtLev));

  //Complex coefficients: [-P, P]x[-P, P]x[0, P] 
  //Coeff[-K1, -K2, -K3] = ComplexConjugate(Coeff[K1, K2, K3])
  const unsigned int TwoPplus1 = (2*P) + 1;
  const unsigned int numWcoeffs = 2*TwoPplus1*TwoPplus1*(P + 1);

  //Generate a list of W (source) boxes that will
  //contribute to my L (target) boxes. Identify their owners.
  PetscLogEventBegin(w2lGenEvent, 0, 0, 0, 0);
  std::vector<std::vector<ot::TreeNode> > tmpBoxList(npes);
  PetscLogEventEnd(w2lGenEvent, 0, 0, 0, 0);

  //Send candidate W boxes to their respective owners.
  std::vector<ot::TreeNode> sendBoxList;
  int* sendCnts = new int[npes];
  for(int i = 0; i < npes; ++i) {
    sendCnts[i] = tmpBoxList[i].size();
    sendBoxList.insert(sendBoxList.end(), tmpBoxList[i].begin(), tmpBoxList[i].end());
  }//end i
  tmpBoxList.clear();

  int* recvCnts = new int[npes];
  MPI_Alltoall(sendCnts, 1, MPI_INT, recvCnts, 1, MPI_INT, subComm);

  int* sendDisps = new int[npes];
  int* recvDisps = new int[npes];
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

  //Find the local indices of the received candidate W boxes.
  //Prepare the list of Wcoeffs for each box. 
  //Flag if the box does not exist.
  PetscLogEventBegin(w2lSearchEvent, 0, 0, 0, 0);
  std::vector<double> sendWlist;
  std::vector<int> sendFlags(recvBoxList.size(), 1);
  for(int i = 0; i < recvBoxList.size(); ++i) {
    unsigned int retIdx;
    bool found = seq::BinarySearch(&(fgtList[0]), fgtList.size(), recvBoxList[i], &retIdx);
    if(found) {
      sendWlist.insert(sendWlist.end(), (localWlist.begin() + (numWcoeffs*retIdx)),
          (localWlist.begin() + (numWcoeffs*(retIdx + 1))));
    } else {
      sendFlags[i] = 0;
    }
  }//end i
  PetscLogEventEnd(w2lSearchEvent, 0, 0, 0, 0);

  recvBoxList.clear();

  //Send the flags by reversing the communication.
  std::vector<int> recvFlags(sendBoxList.size());

  int* sendBuf2 = NULL;
  if(!(sendFlags.empty())) {
    sendBuf2 = &(sendFlags[0]);
  }

  int* recvBuf2 = NULL;
  if(!(recvFlags.empty())) {
    recvBuf2 = &(recvFlags[0]);
  }

  MPI_Alltoallv(sendBuf2, recvCnts, recvDisps, MPI_INT,
      recvBuf2, sendCnts, sendDisps, MPI_INT, subComm);

  //Adjust the send/recv Cnts and Disps for the missing boxes.
  for(int i = 0; i < npes; ++i) {
    int oldSendSz = sendCnts[i];
    for(int j = 0; j < oldSendSz; ++j) {
      if(recvFlags[sendDisps[i] + j] == 0) {
        --(sendCnts[i]);
      }
    }//end j
    int oldRecvSz = recvCnts[i];
    for(int j = 0; j < oldRecvSz; ++j) {
      if(sendFlags[recvDisps[i] + j] == 0) {
        --(recvCnts[i]);
      }
    }//end j
  }//end i

  sendFlags.clear();

  std::vector<ot::TreeNode> tmpSendBoxList;
  for(int i = 0; i < recvFlags.size(); ++i) {
    if(recvFlags[i] != 0) {
      tmpSendBoxList.push_back(sendBoxList[i]);
    }
  }//end i
  swap(tmpSendBoxList, sendBoxList);
  tmpSendBoxList.clear();

  recvFlags.clear();

  for(int i = 0; i < npes; ++i) {
    sendCnts[i] *= numWcoeffs;
    recvCnts[i] *= numWcoeffs;
  }//end i

  sendDisps[0] = 0;
  recvDisps[0] = 0;
  for(int i = 1; i < npes; ++i) {
    sendDisps[i] = sendDisps[i - 1] + sendCnts[i - 1];
    recvDisps[i] = recvDisps[i - 1] + recvCnts[i - 1];
  }//end i

  //Send the list of Wcoeffs.
  std::vector<double> recvWlist(sendDisps[npes - 1] + sendCnts[npes - 1]);

  double* sendBuf3 = NULL;
  if(!(sendWlist.empty())) {
    sendBuf3 = &(sendWlist[0]);
  }

  double* recvBuf3 = NULL;
  if(!(recvWlist.empty())) {
    recvBuf3 = &(recvWlist[0]);
  }

  MPI_Alltoallv(sendBuf3, recvCnts, recvDisps, MPI_DOUBLE,
      recvBuf3, sendCnts, sendDisps, MPI_DOUBLE, subComm);

  sendWlist.clear();

  delete [] recvCnts;
  delete [] recvDisps;

  for(int i = 0; i < npes; ++i) {
    sendCnts[i] /= numWcoeffs;
    sendDisps[i] /= numWcoeffs;
  }//end i

  const double LbyP = static_cast<double>(L)/static_cast<double>(P);
  const double ImExpZfactor = LbyP/hFgt;

  PetscLogEventBegin(w2lCoreEvent, 0, 0, 0, 0);
  PetscLogEventEnd(w2lCoreEvent, 0, 0, 0, 0);

  delete [] sendCnts;
  delete [] sendDisps;

  PetscLogEventEnd(w2lEvent, 0, 0, 0, 0);
}


