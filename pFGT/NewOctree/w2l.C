
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

  const unsigned long long int cellsPerFgt = (1ull << (__MAX_DEPTH__ - FgtLev));

  std::vector<std::vector<ot::TreeNode> > tmpBoxList(npes);
  //Generate a list of W (source) boxes that will
  //contribute to my L (target) boxes. Identify their owners.
  PetscLogEventBegin(w2lGenEvent, 0, 0, 0, 0);
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
  PetscLogEventEnd(w2lSearchEvent, 0, 0, 0, 0);

  //Send the flags by reversing the communication.

  //Adjust the send/recv Cnts and Disps for the missing boxes.

  //Send the list of Wcoeffs.

  //Complex coefficients: [-P, P]x[-P, P]x[0, P] 
  //Coeff[-K1, -K2, -K3] = ComplexConjugate(Coeff[K1, K2, K3])
  const unsigned int TwoPplus1 = (2*P) + 1;
  const unsigned int numWcoeffs = 2*TwoPplus1*TwoPplus1*(P + 1);

  //Fgt box size = sqrt(delta)
  const double hFgt = 1.0/(static_cast<double>(1u << FgtLev));

  const double LbyP = static_cast<double>(L)/static_cast<double>(P);
  const double ImExpZfactor = LbyP/hFgt;

  PetscLogEventBegin(w2lCoreEvent, 0, 0, 0, 0);
  PetscLogEventEnd(w2lCoreEvent, 0, 0, 0, 0);

  delete [] sendCnts;
  delete [] sendDisps;
  delete [] recvCnts;
  delete [] recvDisps;

  PetscLogEventEnd(w2lEvent, 0, 0, 0, 0);
}


