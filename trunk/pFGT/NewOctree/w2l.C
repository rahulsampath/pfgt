
#include "petsc.h"
#include <vector>
#include "oct/TreeNode.h"
#include "pfgtOctUtils.h"
#include "par/parUtils.h"
#include "par/dtypes.h"

extern PetscLogEvent w2lEvent;

void w2l(std::vector<double> & localLlist, std::vector<double> & localWlist, 
    std::vector<ot::TreeNode> & fgtList, std::vector<ot::TreeNode> & fgtMins,
    const unsigned int FgtLev, const int P, const int L, const int K, MPI_Comm subComm) {
  PetscLogEventBegin(w2lEvent, 0, 0, 0, 0);

  int npes;
  MPI_Comm_size(subComm, &npes);

  //Fgt box size = sqrt(delta)
  const double hFgt = 1.0/(static_cast<double>(1u << FgtLev));

  const unsigned int cellsPerFgt = (1u << (__MAX_DEPTH__ - FgtLev));

  //Complex coefficients: [-P, P]x[-P, P]x[0, P] 
  //Coeff[-K1, -K2, -K3] = ComplexConjugate(Coeff[K1, K2, K3])
  const unsigned int TwoPplus1 = (2*P) + 1;
  const unsigned int numWcoeffs = 2*TwoPplus1*TwoPplus1*(P + 1);

  const double LbyP = static_cast<double>(L)/static_cast<double>(P);
  const double ImExpZfactor = LbyP/hFgt;

  const unsigned int TwoP = 2*P;
  std::vector<double> c1(TwoP);
  std::vector<double> c2(TwoP);
  std::vector<double> c3(TwoP);
  std::vector<double> s1(TwoP);
  std::vector<double> s2(TwoP);
  std::vector<double> s3(TwoP);

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
    if( static_cast<unsigned long long int>(bAx) >= static_cast<unsigned long long int>(
          static_cast<unsigned long long int>(K)*static_cast<unsigned long long int>(cellsPerFgt)) ) {
      dAxs = bAx - (K*cellsPerFgt);
    } else {
      dAxs = 0; 
    }
    if( static_cast<unsigned long long int>(static_cast<unsigned long long int>(bAx) + 
          static_cast<unsigned long long int>(static_cast<unsigned long long int>(K + 1)*
            static_cast<unsigned long long int>(cellsPerFgt)))
        <= static_cast<unsigned long long int>(__ITPMD__) ) {
      dAxe = bAx + (K*cellsPerFgt);
    } else {
      dAxe = (__ITPMD__) - cellsPerFgt;
    }
    if( static_cast<unsigned long long int>(bAy) >= static_cast<unsigned long long int>(
          static_cast<unsigned long long int>(K)*static_cast<unsigned long long int>(cellsPerFgt)) ) {
      dAys = bAy - (K*cellsPerFgt);
    } else {
      dAys = 0; 
    }
    if( static_cast<unsigned long long int>(static_cast<unsigned long long int>(bAy) + 
          static_cast<unsigned long long int>(static_cast<unsigned long long int>(K + 1)*
            static_cast<unsigned long long int>(cellsPerFgt)))
        <= static_cast<unsigned long long int>(__ITPMD__) ) {
      dAye = bAy + (K*cellsPerFgt);
    } else {
      dAye = (__ITPMD__) - cellsPerFgt;
    }
    if( static_cast<unsigned long long int>(bAz) >= static_cast<unsigned long long int>(
          static_cast<unsigned long long int>(K)*static_cast<unsigned long long int>(cellsPerFgt)) ) {
      dAzs = bAz - (K*cellsPerFgt);
    } else {
      dAzs = 0; 
    }
    if( static_cast<unsigned long long int>(static_cast<unsigned long long int>(bAz) +
          static_cast<unsigned long long int>(static_cast<unsigned long long int>(K + 1)*
            static_cast<unsigned long long int>(cellsPerFgt)))
        <= static_cast<unsigned long long int>(__ITPMD__) ) {
      dAze = bAz + (K*cellsPerFgt);
    } else {
      dAze = (__ITPMD__) - cellsPerFgt;
    }
    for(unsigned int dAz = dAzs; dAz <= dAze; dAz += cellsPerFgt) {
      for(unsigned int dAy = dAys; dAy <= dAye; dAy += cellsPerFgt) {
        for(unsigned int dAx = dAxs; dAx <= dAxe; dAx += cellsPerFgt) {
          ot::TreeNode boxD(dAx, dAy, dAz, FgtLev, __DIM__, __MAX_DEPTH__);
          boxD.setWeight(tmpBoxes.size());
          tmpBoxes.push_back(boxD);
          double px = (0.5*hFgt) + ((static_cast<double>(dAx))/(__DTPMD__)) - bx;
          double py = (0.5*hFgt) + ((static_cast<double>(dAy))/(__DTPMD__)) - by;
          double pz = (0.5*hFgt) + ((static_cast<double>(dAz))/(__DTPMD__)) - bz;

          for(int kk = -P, di = 0; kk < P; ++kk, ++di) {
            c1[di] = cos(ImExpZfactor*static_cast<double>(kk)*px);
            s1[di] = sin(ImExpZfactor*static_cast<double>(kk)*px);
            c2[di] = cos(ImExpZfactor*static_cast<double>(kk)*py);
            s2[di] = sin(ImExpZfactor*static_cast<double>(kk)*py);
            c3[di] = cos(ImExpZfactor*static_cast<double>(kk)*pz);
            s3[di] = sin(ImExpZfactor*static_cast<double>(kk)*pz);
          }//end kk

          for(int k3 = -P, d3 = 0, di = 0; k3 < P; ++d3, ++k3) {
            for(int k2 = -P, d2 = 0; k2 < P; ++d2, ++k2) {
              for(int k1 = -P, d1 = 0; k1 < P; ++d1, ++k1, ++di) {
                double tmp1 =  ((c1[d1])*(c2[d2])) - ((s1[d1])*(s2[d2]));
                double tmp2 =  ((s1[d1])*(c2[d2])) + ((s2[d2])*(c1[d1]));
                double a = localWlist[(numWcoeffs*i) + (2*di)];
                double b = localWlist[(numWcoeffs*i) + (2*di) + 1];
                double c = ((c3[d3])*tmp1) - ((s3[d3])*tmp2);
                double d = ((s3[d3])*tmp1) + ((c3[d3])*tmp2); 
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


