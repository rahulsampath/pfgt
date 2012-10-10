
#include "petsc.h"
#include <vector>
#include "oct/TreeNode.h"
#include "pfgtOctUtils.h"
#include "par/parUtils.h"
#include "par/dtypes.h"

extern PetscLogEvent w2dD2lExpandEvent;
extern PetscLogEvent w2dD2lDirectEvent;
extern PetscLogEvent w2dD2lEsearchEvent;
extern PetscLogEvent w2dD2lDsearchEvent;

void w2dAndD2lExpand(std::vector<double> & localLlist, std::vector<double> & localWlist, 
    std::vector<ot::TreeNode> & fgtList, const int P, MPI_Comm comm) {
  PetscLogEventBegin(w2dD2lExpandEvent, 0, 0, 0, 0);

  int npes;
  MPI_Comm_size(comm, &npes);

  //Complex coefficients: [-P, P]x[-P, P]x[0, P] 
  //Coeff[-K1, -K2, -K3] = ComplexConjugate(Coeff[K1, K2, K3])
  const unsigned int TwoPplus1 = (2*P) + 1;
  const unsigned int numWcoeffs = 2*TwoPplus1*TwoPplus1*(P + 1);

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

  std::vector<int> recvBoxIds(recvBoxList.size(), -1);

  //Performance Improvement: We can use the fact that each processor's chunk in
  //the recvBoxList is sorted and avoid the searches.
  PetscLogEventBegin(w2dD2lEsearchEvent, 0, 0, 0, 0);
  for(int i = 0; i < recvBoxList.size(); ++i) { 
    unsigned int retIdx;
    bool found = seq::BinarySearch(&(fgtList[0]), fgtList.size(), recvBoxList[i], &retIdx);
    if(found) {
      recvBoxIds[i] = retIdx;
    }
  }//end i
  PetscLogEventEnd(w2dD2lEsearchEvent, 0, 0, 0, 0);

  recvBoxList.clear();

  int* recvBoxIdsPtr = NULL;
  if(!(recvBoxIds.empty())) {
    recvBoxIdsPtr = &(recvBoxIds[0]);
  }

  MPI_Alltoallv(recvBoxIdsPtr, recvCnts, recvDisps, MPI_INT,
      NULL, sendCnts, sendDisps, MPI_INT, comm);

  delete [] sendCnts;
  delete [] sendDisps;

  for(int i = 0; i < npes; ++i) {
    int numNotFound = 0;
    for(int j = 0; j < recvCnts[i]; ++j) {
      if(recvBoxIds[recvDisps[i] + j] < 0) {
        ++numNotFound;
      }
    }//end j
    recvCnts[i] = recvCnts[i] - numNotFound;
  }//end i

  for(int i = 0; i < npes; ++i) {
    recvCnts[i] *= numWcoeffs;
  }//end i

  recvDisps[0] = 0;
  for(int i = 1; i < npes; ++i) {
    recvDisps[i] = recvDisps[i - 1] + recvCnts[i - 1];
  }//end i

  std::vector<int> tmpBoxIds;
  for(int i = 0; i < recvBoxIds.size(); ++i) {
    if(recvBoxIds[i] >= 0) {
      tmpBoxIds.push_back(recvBoxIds[i]);
    }
  }//end i
  swap(recvBoxIds, tmpBoxIds);
  tmpBoxIds.clear();

  std::vector<double> sendWlist((recvDisps[npes - 1] + recvCnts[npes - 1]), 0.0);

  for(int i = 0; i < recvBoxIds.size(); ++i) {
    for(int d = 0; d < numWcoeffs; ++d) {
      sendWlist[(numWcoeffs*i) + d] = localWlist[(numWcoeffs*(recvBoxIds[i])) + d];
    }//end d
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

  MPI_Alltoallv(sendWlistPtr, recvCnts, recvDisps, MPI_DOUBLE,
      recvLlistPtr, recvCnts, recvDisps, MPI_DOUBLE, comm);

  sendWlist.clear();

  delete [] recvCnts;
  delete [] recvDisps;

  for(int i = 0; i < recvBoxIds.size(); ++i) {
    for(int d = 0; d < numWcoeffs; ++d) {
      localLlist[(numWcoeffs*(recvBoxIds[i])) + d] += recvLlist[(numWcoeffs*i) + d];
    }//end d
  }//end i

  recvBoxIds.clear();
  recvLlist.clear(); 

  PetscLogEventEnd(w2dD2lExpandEvent, 0, 0, 0, 0);
}

void w2dAndD2lDirect(std::vector<double> & results, std::vector<double> & sources,
    std::vector<ot::TreeNode> & fgtMins, const unsigned int FgtLev, 
    const int P, const int L, const int K, const double epsilon, MPI_Comm comm) {
  PetscLogEventBegin(w2dD2lDirectEvent, 0, 0, 0, 0);

  int npes;
  MPI_Comm_size(comm, &npes);

  //Complex coefficients: [-P, P]x[-P, P]x[0, P] 
  //Coeff[-K1, -K2, -K3] = ComplexConjugate(Coeff[K1, K2, K3])
  const unsigned int TwoPplus1 = (2*P) + 1;
  const unsigned int numWcoeffs = 2*TwoPplus1*TwoPplus1*(P + 1);

  const unsigned int cellsPerFgt = (1u << (__MAX_DEPTH__ - FgtLev));

  const unsigned int twoPowFgtLev = (1u << FgtLev);

  const double invHfgt =  static_cast<double>(twoPowFgtLev);

  //Fgt box size = sqrt(delta)
  const double hFgt = 1.0/invHfgt;
  const double delta = hFgt*hFgt;

  const double ptIwidth = hFgt*(sqrt(-log(epsilon)));
  const double ptIwidthSqr = ptIwidth*ptIwidth;

  std::vector<ot::TreeNode> tmpSendBoxList;

  for(int i = 0; i < sources.size(); i += 4) {
    unsigned int uiMinPt1[3];
    unsigned int uiMaxPt1[3];
    unsigned int uiMinPt2[3];
    unsigned int uiMaxPt2[3];
    for(int d = 0; d < 3; ++d) {
      double minPt1 = sources[i + d] - ptIwidth;
      double maxPt1 = sources[i + d] + ptIwidth;
      double minVal2 = ((__DTPMD__)*sources[i + d]) - (static_cast<double>(K + 1)*static_cast<double>(cellsPerFgt));
      double maxVal2 = ((__DTPMD__)*sources[i + d]) + (static_cast<double>(K)*static_cast<double>(cellsPerFgt));
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
    for(unsigned int zi = uiMinPt1[2]; zi < uiMaxPt1[2]; ++zi) {
      for(unsigned int yi = uiMinPt1[1]; yi < uiMaxPt1[1]; ++yi) {
        for(unsigned int xi = uiMinPt1[0]; xi < uiMaxPt1[0]; ++xi) {
          ot::TreeNode tmpBox((xi*cellsPerFgt), (yi*cellsPerFgt), (zi*cellsPerFgt),
              FgtLev, __DIM__, __MAX_DEPTH__);
          selectedBoxes.push_back(tmpBox);
        }//end xi
      }//end yi
    }//end zi
    //Target point is in interaction list of source box.
    for(unsigned int zi = uiMinPt2[2]; zi < uiMaxPt2[2]; zi += cellsPerFgt) {
      for(unsigned int yi = uiMinPt2[1]; yi < uiMaxPt2[1]; yi += cellsPerFgt) {
        for(unsigned int xi = uiMinPt2[0]; xi < uiMaxPt2[0]; xi += cellsPerFgt) {
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

  //Performance Improvement: We could avoid this sort if we move the
  //construction of sendBoxList and box2PtMap into the above loop. This will
  //also reduce the temporary storage required for tmpSendBoxList.
  if(!(tmpSendBoxList.empty())) {
    std::sort((&(tmpSendBoxList[0])), (&(tmpSendBoxList[0])) + tmpSendBoxList.size());
  }

  int* sendCnts = new int[npes];
  int* recvDisps = new int[npes]; 
  for(int i = 0; i < npes; ++i) {
    sendCnts[i] = 0;
    recvDisps[i] = 0;
  }//end i

  std::vector<ot::TreeNode> sendBoxList;
  std::vector<std::vector<unsigned int> > box2PtMap;

  //Performance Improvement: We could make use of the fact that tmpSendBoxList is
  //sorted and avoid the searches.
  for(int i = 0; i < tmpSendBoxList.size(); ++i) {
    unsigned int ptId = tmpSendBoxList[i].getWeight();
    unsigned int idx;
    bool foundNew = false;
    if(sendBoxList.empty()) {
      PetscLogEventBegin(w2dD2lDsearchEvent, 0, 0, 0, 0);
      foundNew = seq::maxLowerBound<ot::TreeNode>(fgtMins, tmpSendBoxList[i], idx, NULL, NULL);
      PetscLogEventEnd(w2dD2lDsearchEvent, 0, 0, 0, 0);
    } else {
      if(tmpSendBoxList[i] == sendBoxList[sendBoxList.size() - 1]) {
        box2PtMap[box2PtMap.size() - 1].push_back(ptId);
      } else {
        PetscLogEventBegin(w2dD2lDsearchEvent, 0, 0, 0, 0);
        foundNew = seq::maxLowerBound<ot::TreeNode>(fgtMins, tmpSendBoxList[i], idx, NULL, NULL);
        PetscLogEventEnd(w2dD2lDsearchEvent, 0, 0, 0, 0);
      }
    }
    if(foundNew) {
      ++(sendCnts[fgtMins[idx].getWeight()]);
      sendBoxList.push_back(tmpSendBoxList[i]);
      std::vector<unsigned int> tmpPtIdVec(1, ptId);
      box2PtMap.push_back(tmpPtIdVec);
    }
  }//end i

  tmpSendBoxList.clear();

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

  std::vector<int> foundFlags(sendBoxList.size());

  int* foundFlagsBuf = NULL;
  if(!(foundFlags.empty())) {
    foundFlagsBuf = &(foundFlags[0]);
  }

  MPI_Alltoallv(NULL, recvCnts, recvDisps, MPI_INT,
      foundFlagsBuf, sendCnts, sendDisps, MPI_INT, comm);

  delete [] recvCnts;
  delete [] recvDisps;

  for(int i = 0; i < npes; ++i) {
    int numNotFound = 0;
    for(int j = 0; j < sendCnts[i]; ++j) {
      if(foundFlags[sendDisps[i] + j] < 0) {
        ++numNotFound;
      }
    }//end j
    sendCnts[i] = sendCnts[i] - numNotFound;
  }//end i

  for(int i = 0; i < npes; ++i) {
    sendCnts[i] *= numWcoeffs;
  }//end i

  sendDisps[0] = 0;
  for(int i = 1; i < npes; ++i) {
    sendDisps[i] = sendDisps[i - 1] + sendCnts[i - 1];
  }//end i

  std::vector<int> foundIds;
  for(int i = 0; i < foundFlags.size(); ++i) {
    if(foundFlags[i] >= 0) {
      foundIds.push_back(i);
    }
  }//end i
  foundFlags.clear();

  const double LbyP = static_cast<double>(L)/static_cast<double>(P);
  const double ImExpZfactor = LbyP/hFgt;

  const unsigned int TwoP = 2*P;
  std::vector<double> c1(TwoP);
  std::vector<double> c2(TwoP);
  std::vector<double> c3(TwoP);
  std::vector<double> s1(TwoP);
  std::vector<double> s2(TwoP);
  std::vector<double> s3(TwoP);

  std::vector<double> sendLlist((sendDisps[npes - 1] + sendCnts[npes - 1]), 0.0);

  for(int i = 0; i < foundIds.size(); ++i) {
    double* arr = &(sendLlist[numWcoeffs*i]); 
    int boxId = foundIds[i];
    double cx = (0.5*hFgt) + ((static_cast<double>(sendBoxList[boxId].getX()))/(__DTPMD__));
    double cy = (0.5*hFgt) + ((static_cast<double>(sendBoxList[boxId].getY()))/(__DTPMD__));
    double cz = (0.5*hFgt) + ((static_cast<double>(sendBoxList[boxId].getZ()))/(__DTPMD__));
    for(int j = 0; j < box2PtMap[boxId].size(); ++j) {
      unsigned int sOff = box2PtMap[boxId][j];
      double px = cx - sources[sOff];
      double py = cy - sources[sOff + 1];
      double pz = cz - sources[sOff + 2];
      double pf = sources[sOff + 3];

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
            double cosTh = ( ((c3[d3])*tmp1) - ((s3[d3])*tmp2) );
            double sinTh = ( ((s3[d3])*tmp1) + ((c3[d3])*tmp2) ); 
            int cOff = 2*di;
            arr[cOff] += (pf * cosTh);
            arr[cOff + 1] += (pf * sinTh);
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

  MPI_Alltoallv(sendLlistPtr, sendCnts, sendDisps, MPI_DOUBLE,
      recvWlistPtr, sendCnts, sendDisps, MPI_DOUBLE, comm);

  sendLlist.clear();

  delete [] sendCnts;
  delete [] sendDisps;

  const double ReExpZfactor = -0.25*LbyP*LbyP;
  const double C0 = (0.5*LbyP/(__SQRT_PI__));
  std::vector<double> fac(TwoP);
  for(int kk = -P, di = 0; kk < P; ++di, ++kk) {
    fac[di] = C0*exp(ReExpZfactor*(static_cast<double>(kk*kk)));
  }//end kk

  for(int i = 0; i < foundIds.size(); ++i) {
    double* recvWarr = &(recvWlist[numWcoeffs*i]);
    int boxId = foundIds[i];
    double cx = (0.5*hFgt) + ((static_cast<double>(sendBoxList[boxId].getX()))/(__DTPMD__));
    double cy = (0.5*hFgt) + ((static_cast<double>(sendBoxList[boxId].getY()))/(__DTPMD__));
    double cz = (0.5*hFgt) + ((static_cast<double>(sendBoxList[boxId].getZ()))/(__DTPMD__));
    for(int j = 0; j < box2PtMap[boxId].size(); ++j) {
      unsigned int sOff = box2PtMap[boxId][j];
      double px = sources[sOff] - cx;
      double py = sources[sOff + 1] - cy;
      double pz = sources[sOff + 2] - cz;

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
            double c = ((c3[d3])*tmp1) - ((s3[d3])*tmp2);
            double d = ((s3[d3])*tmp1) + ((c3[d3])*tmp2); 
            int cOff = 2*di;
            double a = recvWarr[cOff];
            double b = recvWarr[cOff + 1];
            results[(box2PtMap[boxId][j])/4] += ((fac[d3])*(fac[d2])*(fac[d1])*( (a*c) - (b*d) ));
          }//end for k1
        }//end for k2
      }//end for k3
    }//end j
  }//end i

  PetscLogEventEnd(w2dD2lDirectEvent, 0, 0, 0, 0);
}



