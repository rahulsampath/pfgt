
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
extern PetscLogEvent w2dD2lDsortEvent;
extern PetscLogEvent w2dD2lDgenEvent;
extern PetscLogEvent w2dD2lDcore1Event;
extern PetscLogEvent w2dD2lDcore2Event;

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
    const int P, const int L, const unsigned long long int K, const double epsilon, MPI_Comm comm) {
  PetscLogEventBegin(w2dD2lDirectEvent, 0, 0, 0, 0);

  int npes;
  MPI_Comm_size(comm, &npes);

  //Complex coefficients: [-P, P]x[-P, P]x[0, P] 
  //Coeff[-K1, -K2, -K3] = ComplexConjugate(Coeff[K1, K2, K3])
  const unsigned int TwoPplus1 = (2*P) + 1;
  const unsigned int numWcoeffs = 2*TwoPplus1*TwoPplus1*(P + 1);

  const unsigned long long int cellsPerFgt = (1ull << (__MAX_DEPTH__ - FgtLev));

  const unsigned int twoPowFgtLev = (1u << FgtLev);

  const double invHfgt =  static_cast<double>(twoPowFgtLev);

  //Fgt box size = sqrt(delta)
  const double hFgt = 1.0/invHfgt;
  const double delta = hFgt*hFgt;

  const double ptIwidth = hFgt*(sqrt(-log(epsilon)));
  const double ptIwidthSqr = ptIwidth*ptIwidth;

  std::vector<ot::TreeNode> tmpSendBoxList;

  PetscLogEventBegin(w2dD2lDgenEvent, 0, 0, 0, 0);
  for(int i = 0; i < sources.size(); i += 4) {
    unsigned long long int uiMinPt1[3];
    unsigned long long int uiMaxPt1[3];
    unsigned long long int uiMinPt2[3];
    unsigned long long int uiMaxPt2[3];
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
      uiMinPt1[d] = static_cast<unsigned long long int>(std::floor(minPt1*invHfgt));
      uiMaxPt1[d] = static_cast<unsigned long long int>(std::ceil(maxPt1*invHfgt));
      uiMinPt2[d] = 1ull + static_cast<unsigned long long int>(std::floor(minVal2));
      uiMaxPt2[d] = static_cast<unsigned long long int>(std::ceil(maxVal2));
    }//end d
    std::vector<ot::TreeNode> selectedBoxes;
    //Target box is in interaction list of source point.
    for(unsigned long long int zi = uiMinPt1[2]; zi < uiMaxPt1[2]; ++zi) {
      for(unsigned long long int yi = uiMinPt1[1]; yi < uiMaxPt1[1]; ++yi) {
        for(unsigned long long int xi = uiMinPt1[0]; xi < uiMaxPt1[0]; ++xi) {
          ot::TreeNode tmpBox((xi*cellsPerFgt), (yi*cellsPerFgt), (zi*cellsPerFgt),
              FgtLev, __DIM__, __MAX_DEPTH__);
          selectedBoxes.push_back(tmpBox);
        }//end xi
      }//end yi
    }//end zi
    //Target point is in interaction list of source box.
    for(unsigned long long int zi = uiMinPt2[2]; zi < uiMaxPt2[2]; zi += cellsPerFgt) {
      for(unsigned long long int yi = uiMinPt2[1]; yi < uiMaxPt2[1]; yi += cellsPerFgt) {
        for(unsigned long long int xi = uiMinPt2[0]; xi < uiMaxPt2[0]; xi += cellsPerFgt) {
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
  PetscLogEventEnd(w2dD2lDgenEvent, 0, 0, 0, 0);

  //Performance Improvement: We could avoid this sort if we move the
  //construction of sendBoxList and box2PtMap into the above loop. This will
  //also reduce the temporary storage required for tmpSendBoxList.
  PetscLogEventBegin(w2dD2lDsortEvent, 0, 0, 0, 0);
  if(!(tmpSendBoxList.empty())) {
    std::sort((&(tmpSendBoxList[0])), (&(tmpSendBoxList[0])) + tmpSendBoxList.size());
  }
  PetscLogEventEnd(w2dD2lDsortEvent, 0, 0, 0, 0);

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
  PetscLogEventBegin(w2dD2lDsearchEvent, 0, 0, 0, 0);
  for(int i = 0; i < tmpSendBoxList.size(); ++i) {
    unsigned int ptId = tmpSendBoxList[i].getWeight();
    unsigned int idx;
    bool foundNew = false;
    if(sendBoxList.empty()) {
      foundNew = seq::maxLowerBound<ot::TreeNode>(fgtMins, tmpSendBoxList[i], idx, NULL, NULL);
    } else {
      if(tmpSendBoxList[i] == sendBoxList[sendBoxList.size() - 1]) {
        box2PtMap[box2PtMap.size() - 1].push_back(ptId);
      } else {
        foundNew = seq::maxLowerBound<ot::TreeNode>(fgtMins, tmpSendBoxList[i], idx, NULL, NULL);
      }
    }
    if(foundNew) {
      ++(sendCnts[fgtMins[idx].getWeight()]);
      sendBoxList.push_back(tmpSendBoxList[i]);
      std::vector<unsigned int> tmpPtIdVec(1, ptId);
      box2PtMap.push_back(tmpPtIdVec);
    }
  }//end i
  PetscLogEventEnd(w2dD2lDsearchEvent, 0, 0, 0, 0);

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

  std::vector<double> sendLlist((sendDisps[npes - 1] + sendCnts[npes - 1]), 0.0);

  PetscLogEventBegin(w2dD2lDcore1Event, 0, 0, 0, 0);
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

      double argX = ImExpZfactor*px; 
      double argY = ImExpZfactor*py; 
      double argZ = ImExpZfactor*pz; 
      c1[0] = cos(argX);
      s1[0] = sin(argX);
      c2[0] = cos(argY);
      s2[0] = sin(argY);
      c3[0] = cos(argZ);
      s3[0] = sin(argZ);
      for(int curr = 1; curr < P; ++curr) {
        int prev = curr - 1;
        c1[curr] = (c1[prev] * c1[0]) - (s1[prev] * s1[0]);
        s1[curr] = (s1[prev] * c1[0]) + (c1[prev] * s1[0]);
        c2[curr] = (c2[prev] * c2[0]) - (s2[prev] * s2[0]);
        s2[curr] = (s2[prev] * c2[0]) + (c2[prev] * s2[0]);
        c3[curr] = (c3[prev] * c3[0]) - (s3[prev] * s3[0]);
        s3[curr] = (s3[prev] * c3[0]) + (c3[prev] * s3[0]);
      }//end curr

      {
        //k3 = 0
        //cosZ = 1
        //sinZ = 0
        //zId = k3 = 0
        //yOff = zId*TwoPplus1 = 0
        {
          //k2 = 0
          //cosY = 1
          //sinY = 0
          //yId = P + k2 = P
          //xOff = (yOff + yId)*TwoPplus1  
          int xOff = P*TwoPplus1;
          //cosYplusZ = (cosY*cosZ) - (sinY*sinZ) = 1
          //sinYplusZ = (sinY*cosZ) + (cosY*sinZ) = 0
          {
            //k1 = 0
            //cosX = 1
            //sinX = 0
            //cosTh = (cosX*cosYplusZ) - (sinX*sinYplusZ) = 1
            //sinTh = (sinX*cosYplusZ) + (cosX*sinYplusZ) = 0
            //xId = P + k1 = P
            int cOff = 2*(xOff + P);
            arr[cOff] += pf;
          }//k1 = 0
          for(int k1 = 1; k1 <= P; ++k1) {
            double cosX = c1Arr[k1];
            double sinX = s1Arr[k1];
            //cXcYZ = cosX*cosYplusZ = cosX
            //sXcYZ = sinX*cosYplusZ = sinX
            //sXsYZ = sinX*sinYplusZ = 0
            //cXsYZ = cosX*sinYplusZ = 0

            //+ve k1
            int xId = P + k1;
            int cOff = 2*(xOff + xId);
            //cosTh = cXcYZ - sXsYZ = cosX
            //sinTh = sXcYZ + cXsYZ = sinX
            arr[cOff] += (pf * cosX);
            arr[cOff + 1] += (pf * sinX);

            //-ve k1
            xId = P - k1;
            cOff = 2*(xOff + xId);
            //cosTh = cXcYZ + sXsYZ = cosX
            //sinTh = cXsYZ - sXcYZ = -sinX
            arr[cOff] += (pf * cosX);
            arr[cOff + 1] -= (pf * sinX);
          }//end k1
        }//k2 = 0 
        for(int k2 = 1; k2 <= P; ++k2) {
          double cosY = c2Arr[k2];
          double sinY = s2Arr[k2];
          //cYcZ = cosY*cosZ = cosY
          //sYcZ = sinY*cosZ = sinY
          //cYsZ = cosY*sinZ = 0
          //sYsZ = sinY*sinZ = 0

          //+ve k2
          int yId = P + k2;
          //xOff = (yOff + yId)*TwoPplus1
          int xOff = yId*TwoPplus1;
          //cosYplusZ = cYcZ - sYsZ = cosY
          //sinYplusZ = sYcZ + cYsZ = sinY
          {
            //k1 = 0
            //cosX = 1
            //sinX = 0
            //xId = P + k1 = P
            int cOff = 2*(xOff + P);
            //cosTh = (cosX*cosYplusZ) - (sinX*sinYplusZ) = cosY
            //sinTh = (sinX*cosYplusZ) + (cosX*sinYplusZ) = sinY
            arr[cOff] += (pf * cosY);
            arr[cOff + 1] += (pf * sinY);
          }//k1 = 0
          for(int k1 = 1; k1 <= P; ++k1) {
            double cosX = c1Arr[k1];
            double sinX = s1Arr[k1];
            //cXcYZ = cosX*cosYplusZ
            double cXcYZ = cosX*cosY;
            //sXsYZ = sinX*sinYplusZ
            double sXsYZ = sinX*sinY;
            //sXcYZ = sinX*cosYplusZ
            double sXcYZ = sinX*cosY;
            //cXsYZ = cosX*sinYplusZ
            double cXsYZ = cosX*sinY;

            //+ve k1
            int xId = P + k1;
            int cOff = 2*(xOff + xId);
            double cosTh = cXcYZ - sXsYZ;
            double sinTh = sXcYZ + cXsYZ;
            arr[cOff] += (pf * cosTh);
            arr[cOff + 1] += (pf * sinTh);

            //-ve k1
            xId = P - k1;
            cOff = 2*(xOff + xId);
            cosTh = cXcYZ + sXsYZ;
            sinTh = cXsYZ - sXcYZ;
            arr[cOff] += (pf * cosTh);
            arr[cOff + 1] += (pf * sinTh);
          }//end k1

          //-ve k2
          yId = P - k2;
          //xOff = (yOff + yId)*TwoPplus1
          xOff = yId*TwoPplus1;
          //cosYplusZ = cYcZ + sYsZ = cosY
          //sinYplusZ = cYsZ - sYcZ = -sinY
          {
            //k1 = 0
            //cosX = 1
            //sinX = 0
            //xId = P + k1 = P
            int cOff = 2*(xOff + P);
            //cosTh = (cosX*cosYplusZ) - (sinX*sinYplusZ) = cosY
            //sinTh = (sinX*cosYplusZ) + (cosX*sinYplusZ) = -sinY
            arr[cOff] += (pf * cosY);
            arr[cOff + 1] -= (pf * sinY);
          }//k1 = 0
          for(int k1 = 1; k1 <= P; ++k1) {
            double cosX = c1Arr[k1];
            double sinX = s1Arr[k1];
            //cXcYZ = cosX*cosYplusZ
            double cXcYZ = cosX*cosY;
            //sXsYZ = sinX*sinYplusZ
            double sXsYZ = -(sinX*sinY);
            //sXcYZ = sinX*cosYplusZ
            double sXcYZ = sinX*cosY;
            //cXsYZ = cosX*sinYplusZ
            double cXsYZ = -(cosX*sinY);

            //+ve k1
            int xId = P + k1;
            int cOff = 2*(xOff + xId);
            double cosTh = cXcYZ - sXsYZ;
            double sinTh = sXcYZ + cXsYZ;
            arr[cOff] += (pf * cosTh);
            arr[cOff + 1] += (pf * sinTh);

            //-ve k1
            xId = P - k1;
            cOff = 2*(xOff + xId);
            cosTh = cXcYZ + sXsYZ;
            sinTh = cXsYZ - sXcYZ;
            arr[cOff] += (pf * cosTh);
            arr[cOff + 1] += (pf * sinTh);
          }//end k1
        }//end k2
      }//k3 = 0
      for(int k3 = 1; k3 <= P; ++k3) {
        double cosZ = c3Arr[k3];
        double sinZ = s3Arr[k3];
        int zId = k3;
        int yOff = zId*TwoPplus1;
        {
          //k2 = 0
          //cosY = 1
          //sinY = 0
          //yId = P + k2 = P
          //xOff = (yOff + yId)*TwoPplus1
          int xOff = (yOff + P)*TwoPplus1;
          //cosYplusZ = (cosY*cosZ) - (sinY*sinZ) = cosZ
          //sinYplusZ = (sinY*cosZ) + (cosY*sinZ) = sinZ
          {
            //k1 = 0
            //cosX = 1
            //sinX = 0
            //xId = P + k1 = P
            int cOff = 2*(xOff + P);
            //cosTh = (cosX*cosYplusZ) - (sinX*sinYplusZ) = cosZ
            //sinTh = (sinX*cosYplusZ) + (cosX*sinYplusZ) = sinZ
            arr[cOff] += (pf * cosZ);
            arr[cOff + 1] += (pf * sinZ);
          }//k1 = 0
          for(int k1 = 1; k1 <= P; ++k1) {
            double cosX = c1Arr[k1];
            double sinX = s1Arr[k1];
            //cXcYZ = cosX*cosYplusZ
            double cXcYZ = cosX*cosZ;
            //sXsYZ = sinX*sinYplusZ
            double sXsYZ = sinX*sinZ;
            //sXcYZ = sinX*cosYplusZ
            double sXcYZ = sinX*cosZ;
            //cXsYZ = cosX*sinYplusZ
            double cXsYZ = cosX*sinZ;

            //+ve k1
            int xId = P + k1;
            int cOff = 2*(xOff + xId);
            double cosTh = cXcYZ - sXsYZ;
            double sinTh = sXcYZ + cXsYZ;
            arr[cOff] += (pf * cosTh);
            arr[cOff + 1] += (pf * sinTh);

            //-ve k1
            xId = P - k1;
            cOff = 2*(xOff + xId);
            cosTh = cXcYZ + sXsYZ;
            sinTh = cXsYZ - sXcYZ;
            arr[cOff] += (pf * cosTh);
            arr[cOff + 1] += (pf * sinTh);
          }//end k1
        }//k2 = 0 
        for(int k2 = 1; k2 <= P; ++k2) {
          double cosY = c2Arr[k2];
          double sinY = s2Arr[k2];
          double cYcZ = cosY*cosZ;
          double cYsZ = cosY*sinZ;
          double sYsZ = sinY*sinZ;
          double sYcZ = sinY*cosZ;

          //+ve k2
          int yId = P + k2;
          int xOff = (yOff + yId)*TwoPplus1;
          double cosYplusZ = cYcZ - sYsZ;
          double sinYplusZ = sYcZ + cYsZ;
          {
            //k1 = 0
            //cosX = 1
            //sinX = 0
            //xId = P + k1 = P
            int cOff = 2*(xOff + P);
            //cosTh = (cosX*cosYplusZ) - (sinX*sinYplusZ) = cosYplusZ
            //sinTh = (sinX*cosYplusZ) + (cosX*sinYplusZ) = sinYplusZ
            arr[cOff] += (pf * cosYplusZ);
            arr[cOff + 1] += (pf * sinYplusZ);
          }//k1 = 0
          for(int k1 = 1; k1 <= P; ++k1) {
            double cosX = c1Arr[k1];
            double sinX = s1Arr[k1];
            double cXcYZ = cosX*cosYplusZ;
            double sXsYZ = sinX*sinYplusZ;
            double sXcYZ = sinX*cosYplusZ;
            double cXsYZ = cosX*sinYplusZ;

            //+ve k1
            int xId = P + k1;
            int cOff = 2*(xOff + xId);
            double cosTh = cXcYZ - sXsYZ;
            double sinTh = sXcYZ + cXsYZ;
            arr[cOff] += (pf * cosTh);
            arr[cOff + 1] += (pf * sinTh);

            //-ve k1
            xId = P - k1;
            cOff = 2*(xOff + xId);
            cosTh = cXcYZ + sXsYZ;
            sinTh = cXsYZ - sXcYZ;
            arr[cOff] += (pf * cosTh);
            arr[cOff + 1] += (pf * sinTh);
          }//end k1

          //-ve k2
          yId = P - k2;
          xOff = (yOff + yId)*TwoPplus1;
          cosYplusZ = cYcZ + sYsZ;
          sinYplusZ = cYsZ - sYcZ;
          {
            //k1 = 0
            //cosX = 1
            //sinX = 0
            //xId = P + k1 = P
            int cOff = 2*(xOff + P);
            //cosTh = (cosX*cosYplusZ) - (sinX*sinYplusZ) = cosYplusZ
            //sinTh = (sinX*cosYplusZ) + (cosX*sinYplusZ) = sinYplusZ
            arr[cOff] += (pf * cosYplusZ);
            arr[cOff + 1] += (pf * sinYplusZ);
          }//k1 = 0
          for(int k1 = 1; k1 <= P; ++k1) {
            double cosX = c1Arr[k1];
            double sinX = s1Arr[k1];
            double cXcYZ = cosX*cosYplusZ;
            double sXsYZ = sinX*sinYplusZ;
            double sXcYZ = sinX*cosYplusZ;
            double cXsYZ = cosX*sinYplusZ;

            //+ve k1
            int xId = P + k1;
            int cOff = 2*(xOff + xId);
            double cosTh = cXcYZ - sXsYZ;
            double sinTh = sXcYZ + cXsYZ;
            arr[cOff] += (pf * cosTh);
            arr[cOff + 1] += (pf * sinTh);

            //-ve k1
            xId = P - k1;
            cOff = 2*(xOff + xId);
            cosTh = cXcYZ + sXsYZ;
            sinTh = cXsYZ - sXcYZ;
            arr[cOff] += (pf * cosTh);
            arr[cOff + 1] += (pf * sinTh);
          }//end k1
        }//end k2
      }//end k3
    }//end j
  }//end i
  PetscLogEventEnd(w2dD2lDcore1Event, 0, 0, 0, 0);

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
  std::vector<double> fac(P);
  double* facArr = (&(fac[0])) - 1;
  for(int k = 1; k <= P; ++k) {
    facArr[k] = C0*exp(ReExpZfactor*(static_cast<double>(k*k)));
  }//end k

  PetscLogEventBegin(w2dD2lDcore2Event, 0, 0, 0, 0);
  for(int i = 0; i < foundIds.size(); ++i) {
    double* arr = &(recvWlist[numWcoeffs*i]);
    int boxId = foundIds[i];
    double cx = (0.5*hFgt) + ((static_cast<double>(sendBoxList[boxId].getX()))/(__DTPMD__));
    double cy = (0.5*hFgt) + ((static_cast<double>(sendBoxList[boxId].getY()))/(__DTPMD__));
    double cz = (0.5*hFgt) + ((static_cast<double>(sendBoxList[boxId].getZ()))/(__DTPMD__));
    for(int j = 0; j < box2PtMap[boxId].size(); ++j) {
      unsigned int sOff = box2PtMap[boxId][j];
      double px = sources[sOff] - cx;
      double py = sources[sOff + 1] - cy;
      double pz = sources[sOff + 2] - cz;

      double argX = ImExpZfactor*px; 
      double argY = ImExpZfactor*py; 
      double argZ = ImExpZfactor*pz; 
      c1[0] = cos(argX);
      s1[0] = sin(argX);
      c2[0] = cos(argY);
      s2[0] = sin(argY);
      c3[0] = cos(argZ);
      s3[0] = sin(argZ);
      for(int curr = 1; curr < P; ++curr) {
        int prev = curr - 1;
        c1[curr] = (c1[prev] * c1[0]) - (s1[prev] * s1[0]);
        s1[curr] = (s1[prev] * c1[0]) + (c1[prev] * s1[0]);
        c2[curr] = (c2[prev] * c2[0]) - (s2[prev] * s2[0]);
        s2[curr] = (s2[prev] * c2[0]) + (c2[prev] * s2[0]);
        c3[curr] = (c3[prev] * c3[0]) - (s3[prev] * s3[0]);
        s3[curr] = (s3[prev] * c3[0]) + (c3[prev] * s3[0]);
      }//end curr

      double outVal = 0;
      {
        //k3 = 0
        //cosZ = 1
        //sinZ = 0
        //zId = k3 = 0
        //yOff = zId*TwoPplus1 = 0
        {
          //k2 = 0
          //cosY = 1
          //sinY = 0
          //yId = P + k2 = P
          //xOff = (yOff + yId)*TwoPplus1  
          int xOff = P*TwoPplus1;
          //cosYplusZ = (cosY*cosZ) - (sinY*sinZ) = 1
          //sinYplusZ = (sinY*cosZ) + (cosY*sinZ) = 0
          {
            //k1 = 0
            //cosX = 1
            //sinX = 0
            //cosTh = (cosX*cosYplusZ) - (sinX*sinYplusZ) = 1
            //sinTh = (sinX*cosYplusZ) + (cosX*sinYplusZ) = 0
            //xId = P + k1 = P
            int cOff = 2*(xOff + P);
            outVal += (C0 * C0 * C0 * arr[cOff]);
          }//k1 = 0
          for(int k1 = 1; k1 <= P; ++k1) {
            double cosX = c1Arr[k1];
            double sinX = s1Arr[k1];
            //cXcYZ = cosX*cosYplusZ = cosX
            //sXcYZ = sinX*cosYplusZ = sinX
            //sXsYZ = sinX*sinYplusZ = 0
            //cXsYZ = cosX*sinYplusZ = 0

            //+ve k1
            int xId = P + k1;
            int cOff = 2*(xOff + xId);
            //cosTh = cXcYZ - sXsYZ = cosX
            //sinTh = sXcYZ + cXsYZ = sinX
            outVal += (C0 * C0 * facArr[k1] * ( (arr[cOff] * cosX) - (arr[cOff + 1] * sinX) ));

            //-ve k1
            xId = P - k1;
            cOff = 2*(xOff + xId);
            //cosTh = cXcYZ + sXsYZ = cosX
            //sinTh = cXsYZ - sXcYZ = -sinX
            outVal += (C0 * C0 * facArr[k1] * ( (arr[cOff] * cosX) + (arr[cOff + 1] * sinX) ));
          }//end k1
        }//k2 = 0 
        for(int k2 = 1; k2 <= P; ++k2) {
          double cosY = c2Arr[k2];
          double sinY = s2Arr[k2];
          //cYcZ = cosY*cosZ = cosY
          //sYcZ = sinY*cosZ = sinY
          //cYsZ = cosY*sinZ = 0
          //sYsZ = sinY*sinZ = 0

          //+ve k2
          int yId = P + k2;
          //xOff = (yOff + yId)*TwoPplus1
          int xOff = yId*TwoPplus1;
          //cosYplusZ = cYcZ - sYsZ = cosY
          //sinYplusZ = sYcZ + cYsZ = sinY
          {
            //k1 = 0
            //cosX = 1
            //sinX = 0
            //xId = P + k1 = P
            int cOff = 2*(xOff + P);
            //cosTh = (cosX*cosYplusZ) - (sinX*sinYplusZ) = cosY
            //sinTh = (sinX*cosYplusZ) + (cosX*sinYplusZ) = sinY
            outVal += (C0 * facArr[k2] * C0 * ( (arr[cOff] * cosY) - (arr[cOff + 1] * sinY) ));
          }//k1 = 0
          for(int k1 = 1; k1 <= P; ++k1) {
            double cosX = c1Arr[k1];
            double sinX = s1Arr[k1];
            //cXcYZ = cosX*cosYplusZ
            double cXcYZ = cosX*cosY;
            //sXsYZ = sinX*sinYplusZ
            double sXsYZ = sinX*sinY;
            //sXcYZ = sinX*cosYplusZ
            double sXcYZ = sinX*cosY;
            //cXsYZ = cosX*sinYplusZ
            double cXsYZ = cosX*sinY;

            //+ve k1
            int xId = P + k1;
            int cOff = 2*(xOff + xId);
            double cosTh = cXcYZ - sXsYZ;
            double sinTh = sXcYZ + cXsYZ;
            outVal += (C0 * facArr[k2] * facArr[k1] * ( (arr[cOff] * cosTh) - (arr[cOff + 1] * sinTh) ));

            //-ve k1
            xId = P - k1;
            cOff = 2*(xOff + xId);
            cosTh = cXcYZ + sXsYZ;
            sinTh = cXsYZ - sXcYZ;
            outVal += (C0 * facArr[k2] * facArr[k1] * ( (arr[cOff] * cosTh) - (arr[cOff + 1] * sinTh) ));
          }//end k1

          //-ve k2
          yId = P - k2;
          //xOff = (yOff + yId)*TwoPplus1
          xOff = yId*TwoPplus1;
          //cosYplusZ = cYcZ + sYsZ = cosY
          //sinYplusZ = cYsZ - sYcZ = -sinY
          {
            //k1 = 0
            //cosX = 1
            //sinX = 0
            //xId = P + k1 = P
            int cOff = 2*(xOff + P);
            //cosTh = (cosX*cosYplusZ) - (sinX*sinYplusZ) = cosY
            //sinTh = (sinX*cosYplusZ) + (cosX*sinYplusZ) = -sinY
            outVal += (C0 * facArr[k2] * C0 * ( (arr[cOff] * cosY) + (arr[cOff + 1] * sinY) ));
          }//k1 = 0
          for(int k1 = 1; k1 <= P; ++k1) {
            double cosX = c1Arr[k1];
            double sinX = s1Arr[k1];
            //cXcYZ = cosX*cosYplusZ
            double cXcYZ = cosX*cosY;
            //sXsYZ = sinX*sinYplusZ
            double sXsYZ = -(sinX*sinY);
            //sXcYZ = sinX*cosYplusZ
            double sXcYZ = sinX*cosY;
            //cXsYZ = cosX*sinYplusZ
            double cXsYZ = -(cosX*sinY);

            //+ve k1
            int xId = P + k1;
            int cOff = 2*(xOff + xId);
            double cosTh = cXcYZ - sXsYZ;
            double sinTh = sXcYZ + cXsYZ;
            outVal += (C0 * facArr[k2] * facArr[k1] * ( (arr[cOff] * cosTh) - (arr[cOff + 1] * sinTh) ));

            //-ve k1
            xId = P - k1;
            cOff = 2*(xOff + xId);
            cosTh = cXcYZ + sXsYZ;
            sinTh = cXsYZ - sXcYZ;
            outVal += (C0 * facArr[k2] * facArr[k1] * ( (arr[cOff] * cosTh) - (arr[cOff + 1] * sinTh) ));
          }//end k1
        }//end k2
      }//k3 = 0
      for(int k3 = 1; k3 <= P; ++k3) {
        double cosZ = c3Arr[k3];
        double sinZ = s3Arr[k3];
        int zId = k3;
        int yOff = zId*TwoPplus1;
        {
          //k2 = 0
          //cosY = 1
          //sinY = 0
          //yId = P + k2 = P
          //xOff = (yOff + yId)*TwoPplus1
          int xOff = (yOff + P)*TwoPplus1;
          //cosYplusZ = (cosY*cosZ) - (sinY*sinZ) = cosZ
          //sinYplusZ = (sinY*cosZ) + (cosY*sinZ) = sinZ
          {
            //k1 = 0
            //cosX = 1
            //sinX = 0
            //xId = P + k1 = P
            int cOff = 2*(xOff + P);
            //cosTh = (cosX*cosYplusZ) - (sinX*sinYplusZ) = cosZ
            //sinTh = (sinX*cosYplusZ) + (cosX*sinYplusZ) = sinZ
            outVal += (2.0 * facArr[k3] * C0 * C0 * ( (arr[cOff] * cosZ) - (arr[cOff + 1] * sinZ) ));
          }//k1 = 0
          for(int k1 = 1; k1 <= P; ++k1) {
            double cosX = c1Arr[k1];
            double sinX = s1Arr[k1];
            //cXcYZ = cosX*cosYplusZ
            double cXcYZ = cosX*cosZ;
            //sXsYZ = sinX*sinYplusZ
            double sXsYZ = sinX*sinZ;
            //sXcYZ = sinX*cosYplusZ
            double sXcYZ = sinX*cosZ;
            //cXsYZ = cosX*sinYplusZ
            double cXsYZ = cosX*sinZ;

            //+ve k1
            int xId = P + k1;
            int cOff = 2*(xOff + xId);
            double cosTh = cXcYZ - sXsYZ;
            double sinTh = sXcYZ + cXsYZ;
            outVal += (2.0 * facArr[k3] * C0 * facArr[k1] * ( (arr[cOff] * cosTh) - (arr[cOff + 1] * sinTh) ));

            //-ve k1
            xId = P - k1;
            cOff = 2*(xOff + xId);
            cosTh = cXcYZ + sXsYZ;
            sinTh = cXsYZ - sXcYZ;
            outVal += (2.0 * facArr[k3] * C0 * facArr[k1] * ( (arr[cOff] * cosTh) - (arr[cOff + 1] * sinTh) ));
          }//end k1
        }//k2 = 0 
        for(int k2 = 1; k2 <= P; ++k2) {
          double cosY = c2Arr[k2];
          double sinY = s2Arr[k2];
          double cYcZ = cosY*cosZ;
          double cYsZ = cosY*sinZ;
          double sYsZ = sinY*sinZ;
          double sYcZ = sinY*cosZ;

          //+ve k2
          int yId = P + k2;
          int xOff = (yOff + yId)*TwoPplus1;
          double cosYplusZ = cYcZ - sYsZ;
          double sinYplusZ = sYcZ + cYsZ;
          {
            //k1 = 0
            //cosX = 1
            //sinX = 0
            //xId = P + k1 = P
            int cOff = 2*(xOff + P);
            //cosTh = (cosX*cosYplusZ) - (sinX*sinYplusZ) = cosYplusZ
            //sinTh = (sinX*cosYplusZ) + (cosX*sinYplusZ) = sinYplusZ
            outVal += (2.0 * facArr[k3] * facArr[k2] * C0 * ( (arr[cOff] * cosYplusZ) - (arr[cOff + 1] * sinYplusZ) ));
          }//k1 = 0
          for(int k1 = 1; k1 <= P; ++k1) {
            double cosX = c1Arr[k1];
            double sinX = s1Arr[k1];
            double cXcYZ = cosX*cosYplusZ;
            double sXsYZ = sinX*sinYplusZ;
            double sXcYZ = sinX*cosYplusZ;
            double cXsYZ = cosX*sinYplusZ;

            //+ve k1
            int xId = P + k1;
            int cOff = 2*(xOff + xId);
            double cosTh = cXcYZ - sXsYZ;
            double sinTh = sXcYZ + cXsYZ;
            outVal += (2.0 * facArr[k3] * facArr[k2] * facArr[k1] * ( (arr[cOff] * cosTh) - (arr[cOff + 1] * sinTh) ));

            //-ve k1
            xId = P - k1;
            cOff = 2*(xOff + xId);
            cosTh = cXcYZ + sXsYZ;
            sinTh = cXsYZ - sXcYZ;
            outVal += (2.0 * facArr[k3] * facArr[k2] * facArr[k1] * ( (arr[cOff] * cosTh) - (arr[cOff + 1] * sinTh) ));
          }//end k1

          //-ve k2
          yId = P - k2;
          xOff = (yOff + yId)*TwoPplus1;
          cosYplusZ = cYcZ + sYsZ;
          sinYplusZ = cYsZ - sYcZ;
          {
            //k1 = 0
            //cosX = 1
            //sinX = 0
            //xId = P + k1 = P
            int cOff = 2*(xOff + P);
            //cosTh = (cosX*cosYplusZ) - (sinX*sinYplusZ) = cosYplusZ
            //sinTh = (sinX*cosYplusZ) + (cosX*sinYplusZ) = sinYplusZ
            outVal += (2.0 * facArr[k3] * facArr[k2] * C0 * ( (arr[cOff] * cosYplusZ) - (arr[cOff + 1] * sinYplusZ) ));
          }//k1 = 0
          for(int k1 = 1; k1 <= P; ++k1) {
            double cosX = c1Arr[k1];
            double sinX = s1Arr[k1];
            double cXcYZ = cosX*cosYplusZ;
            double sXsYZ = sinX*sinYplusZ;
            double sXcYZ = sinX*cosYplusZ;
            double cXsYZ = cosX*sinYplusZ;

            //+ve k1
            int xId = P + k1;
            int cOff = 2*(xOff + xId);
            double cosTh = cXcYZ - sXsYZ;
            double sinTh = sXcYZ + cXsYZ;
            outVal += (2.0 * facArr[k3] * facArr[k2] * facArr[k1] * ( (arr[cOff] * cosTh) - (arr[cOff + 1] * sinTh) ));

            //-ve k1
            xId = P - k1;
            cOff = 2*(xOff + xId);
            cosTh = cXcYZ + sXsYZ;
            sinTh = cXsYZ - sXcYZ;
            outVal += (2.0 * facArr[k3] * facArr[k2] * facArr[k1] * ( (arr[cOff] * cosTh) - (arr[cOff + 1] * sinTh) ));
          }//end k1
        }//end k2
      }//end k3
      results[sOff/4] += outVal;
    }//end j
  }//end i
  PetscLogEventEnd(w2dD2lDcore2Event, 0, 0, 0, 0);

  PetscLogEventEnd(w2dD2lDirectEvent, 0, 0, 0, 0);
}



