
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
    for(unsigned long long int dAz = dAzs; dAz <= dAze; dAz += cellsPerFgt) {
      for(unsigned long long int dAy = dAys; dAy <= dAye; dAy += cellsPerFgt) {
        for(unsigned long long int dAx = dAxs; dAx <= dAxe; dAx += cellsPerFgt) {
          ot::TreeNode boxD(dAx, dAy, dAz, FgtLev, __DIM__, __MAX_DEPTH__);
          boxD.setWeight(i);
          tmpBoxes.push_back(boxD);
        }//end dAx
      }//end dAy
    }//end dAz
  }//end i

  if(!(tmpBoxes.empty())) {
    std::sort(tmpBoxes.begin(), tmpBoxes.end());
  }

  int* sendCnts = new int[npes];
  for(int i = 0; i < npes; ++i) {
    sendCnts[i] = 0;
  }//end i 

  //Performance Improvement: This binary search can be avoided by using the
  //fact that tmpBoxes is sorted.
  PetscLogEventBegin(w2lSearchEvent, 0, 0, 0, 0);
  std::vector<ot::TreeNode> sendBoxList;
  std::vector<unsigned int> sendBoxToMyBoxMap;
  for(size_t i = 0; i < tmpBoxes.size(); ++i) {
    unsigned int myBoxId = tmpBoxes[i].getWeight();
    unsigned int idx;
    bool foundNew = false;
    if(sendBoxList.empty()) {
      foundNew = seq::maxLowerBound<ot::TreeNode>(fgtMins, tmpBoxes[i], idx, NULL, NULL);
    } else {
      if(tmpBoxes[i] == sendBoxList[sendBoxList.size() - 1]) {
        sendBoxToMyBoxMap.push_back(myBoxId);
        sendBoxList[sendBoxList.size() - 1].addWeight(1);
      } else {
        foundNew = seq::maxLowerBound<ot::TreeNode>(fgtMins, tmpBoxes[i], idx, NULL, NULL);
      }
    }
    if(foundNew) {
      ++(sendCnts[fgtMins[idx].getWeight()]);
      sendBoxToMyBoxMap.push_back(myBoxId);
      sendBoxList.push_back(tmpBoxes[i]);
      sendBoxList[sendBoxList.size() - 1].setWeight(1);
    }
  }//end i
  tmpBoxes.clear();
  PetscLogEventEnd(w2lSearchEvent, 0, 0, 0, 0);

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

  PetscLogEventBegin(w2lSearchEvent, 0, 0, 0, 0);
  //Performance Improvement: This binary search can be avoided by making use of
  //the fact that recvBoxList is sorted within each processor chunk.
  std::vector<int> foundIdx((recvBoxList.size()), -1);
  for(size_t i = 0; i < recvBoxList.size(); ++i) {
    unsigned int retIdx;
    bool found = seq::BinarySearch(&(fgtList[0]), fgtList.size(), recvBoxList[i], &retIdx);
    if(found) {
      foundIdx[i] = retIdx;
    }
  }//end i
  PetscLogEventEnd(w2lSearchEvent, 0, 0, 0, 0);
  recvBoxList.clear();

  std::vector<int> recvFoundIdx(sendBoxList.size());

  int* sendBuf2 = NULL;
  if(!(foundIdx.empty())) {
    sendBuf2 = &(foundIdx[0]);
  }

  int* recvBuf2 = NULL; 
  if(!(recvFoundIdx.empty())) {
    recvBuf2 = &(recvFoundIdx[0]);
  }

  MPI_Alltoallv(sendBuf2, recvCnts, recvDisps, MPI_INT,
      recvBuf2, sendCnts, sendDisps, MPI_INT, subComm);

  std::vector<int> tmpFoundIdx;
  for(int i = 0; i < npes; ++i) {
    int numNotFound = 0;
    for(int j = 0; j < recvCnts[i]; ++j) {
      if(foundIdx[recvDisps[i] + j] < 0) {
        ++numNotFound;
      } else {
        tmpFoundIdx.push_back(foundIdx[recvDisps[i] + j]);
      }
    }//end j
    recvCnts[i] -= numNotFound;
  }//end i
  swap(tmpFoundIdx, foundIdx);
  tmpFoundIdx.clear();

  std::vector<ot::TreeNode> tmpSendBoxList;
  std::vector<unsigned int> tmpSendBoxToMyBoxMap;
  for(int i = 0, cnt = 0; i < npes; ++i) {
    int numNotFound = 0;
    for(int j = 0; j < sendCnts[i]; ++j) {
      int boxId = sendDisps[i] + j;
      if(recvFoundIdx[boxId] < 0) {
        ++numNotFound;
      } else {
        tmpSendBoxList.push_back(sendBoxList[boxId]);
        tmpSendBoxToMyBoxMap.insert(tmpSendBoxToMyBoxMap.end(), (sendBoxToMyBoxMap.begin() + cnt), 
            (sendBoxToMyBoxMap.begin() + cnt + (sendBoxList[boxId].getWeight())));
      }
      cnt += (sendBoxList[boxId].getWeight());
    }//end j
    sendCnts[i] -= numNotFound;
  }//end i
  swap(tmpSendBoxList, sendBoxList);
  swap(tmpSendBoxToMyBoxMap, sendBoxToMyBoxMap);
  tmpSendBoxList.clear();
  tmpSendBoxToMyBoxMap.clear();
  recvFoundIdx.clear();

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
    double* tmpArr = &(sendLlist[i*numWcoeffs]);
    unsigned long long int dAx = sendBoxList[i].getX();
    unsigned long long int dAy = sendBoxList[i].getY();
    unsigned long long int dAz = sendBoxList[i].getZ();
    double dx = (0.5*hFgt) + ((static_cast<double>(dAx))/(__DTPMD__));
    double dy = (0.5*hFgt) + ((static_cast<double>(dAy))/(__DTPMD__));
    double dz = (0.5*hFgt) + ((static_cast<double>(dAz))/(__DTPMD__));
    for(int j = 0; j < sendBoxList[i].getWeight(); ++j, ++cnt) {
      unsigned int wId = sendBoxToMyBoxMap[cnt];
      double* localWarr = &(localWlist[wId*numWcoeffs]);
      unsigned long long int bAx = fgtList[wId].getX();
      unsigned long long int bAy = fgtList[wId].getY();
      unsigned long long int bAz = fgtList[wId].getZ();
      double bx = (0.5*hFgt) + ((static_cast<double>(bAx))/(__DTPMD__));
      double by = (0.5*hFgt) + ((static_cast<double>(bAy))/(__DTPMD__));
      double bz = (0.5*hFgt) + ((static_cast<double>(bAz))/(__DTPMD__));

      double px = dx - bx;
      double py = dy - by;
      double pz = dz - bz;

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
            //tmpArr[cOff] += (localWarr[cOff] * cosTh) - (localWarr[cOff + 1] * sinTh)
            tmpArr[cOff] += localWarr[cOff];
            //tmpArr[cOff + 1] += (localWarr[cOff] * sinTh) + (localWarr[cOff + 1] * cosTh)
            tmpArr[cOff + 1] += localWarr[cOff + 1];
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
            //tmpArr[cOff] += (localWarr[cOff] * cosTh) - (localWarr[cOff + 1] * sinTh)
            tmpArr[cOff] += (localWarr[cOff] * cosX) - (localWarr[cOff + 1] * sinX);
            //tmpArr[cOff + 1] += (localWarr[cOff] * sinTh) + (localWarr[cOff + 1] * cosTh)
            tmpArr[cOff + 1] += (localWarr[cOff] * sinX) + (localWarr[cOff + 1] * cosX);

            //-ve k1
            xId = P - k1;
            cOff = 2*(xOff + xId);
            //cosTh = cXcYZ + sXsYZ = cosX
            //sinTh = cXsYZ - sXcYZ = -sinX
            //tmpArr[cOff] += (localWarr[cOff] * cosTh) - (localWarr[cOff + 1] * sinTh)
            tmpArr[cOff] += (localWarr[cOff] * cosX) + (localWarr[cOff + 1] * sinX);
            //tmpArr[cOff + 1] += (localWarr[cOff] * sinTh) + (localWarr[cOff + 1] * cosTh)
            tmpArr[cOff + 1] += (localWarr[cOff + 1] * cosX) - (localWarr[cOff] * sinX);
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
            //tmpArr[cOff] += (localWarr[cOff] * cosTh) - (localWarr[cOff + 1] * sinTh)
            tmpArr[cOff] += (localWarr[cOff] * cosY) - (localWarr[cOff + 1] * sinY);
            //tmpArr[cOff + 1] += (localWarr[cOff] * sinTh) + (localWarr[cOff + 1] * cosTh)
            tmpArr[cOff + 1] += (localWarr[cOff] * sinY) + (localWarr[cOff + 1] * cosY);
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
            tmpArr[cOff] += (localWarr[cOff] * cosTh) - (localWarr[cOff + 1] * sinTh);
            tmpArr[cOff + 1] += (localWarr[cOff] * sinTh) + (localWarr[cOff + 1] * cosTh);

            //-ve k1
            xId = P - k1;
            cOff = 2*(xOff + xId);
            cosTh = cXcYZ + sXsYZ;
            sinTh = cXsYZ - sXcYZ;
            tmpArr[cOff] += (localWarr[cOff] * cosTh) - (localWarr[cOff + 1] * sinTh);
            tmpArr[cOff + 1] += (localWarr[cOff] * sinTh) + (localWarr[cOff + 1] * cosTh);
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
            //tmpArr[cOff] += (localWarr[cOff] * cosTh) - (localWarr[cOff + 1] * sinTh)
            tmpArr[cOff] += (localWarr[cOff] * cosY) + (localWarr[cOff + 1] * sinY);
            //tmpArr[cOff + 1] += (localWarr[cOff] * sinTh) + (localWarr[cOff + 1] * cosTh)
            tmpArr[cOff + 1] += (localWarr[cOff + 1] * cosY) - (localWarr[cOff] * sinY);
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
            tmpArr[cOff] += (localWarr[cOff] * cosTh) - (localWarr[cOff + 1] * sinTh);
            tmpArr[cOff + 1] += (localWarr[cOff] * sinTh) + (localWarr[cOff + 1] * cosTh);

            //-ve k1
            xId = P - k1;
            cOff = 2*(xOff + xId);
            cosTh = cXcYZ + sXsYZ;
            sinTh = cXsYZ - sXcYZ;
            tmpArr[cOff] += (localWarr[cOff] * cosTh) - (localWarr[cOff + 1] * sinTh);
            tmpArr[cOff + 1] += (localWarr[cOff] * sinTh) + (localWarr[cOff + 1] * cosTh);
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
            //tmpArr[cOff] += (localWarr[cOff] * cosTh) - (localWarr[cOff + 1] * sinTh)
            tmpArr[cOff] += (localWarr[cOff] * cosZ) - (localWarr[cOff + 1] * sinZ);
            //tmpArr[cOff + 1] += (localWarr[cOff] * sinTh) + (localWarr[cOff + 1] * cosTh)
            tmpArr[cOff + 1] += (localWarr[cOff] * sinZ) + (localWarr[cOff + 1] * cosZ);
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
            tmpArr[cOff] += (localWarr[cOff] * cosTh) - (localWarr[cOff + 1] * sinTh);
            tmpArr[cOff + 1] += (localWarr[cOff] * sinTh) + (localWarr[cOff + 1] * cosTh);

            //-ve k1
            xId = P - k1;
            cOff = 2*(xOff + xId);
            cosTh = cXcYZ + sXsYZ;
            sinTh = cXsYZ - sXcYZ;
            tmpArr[cOff] += (localWarr[cOff] * cosTh) - (localWarr[cOff + 1] * sinTh);
            tmpArr[cOff + 1] += (localWarr[cOff] * sinTh) + (localWarr[cOff + 1] * cosTh);
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
            //tmpArr[cOff] += (localWarr[cOff] * cosTh) - (localWarr[cOff + 1] * sinTh)
            tmpArr[cOff] += (localWarr[cOff] * cosYplusZ) - (localWarr[cOff + 1] * sinYplusZ);
            //tmpArr[cOff + 1] += (localWarr[cOff] * sinTh) + (localWarr[cOff + 1] * cosTh)
            tmpArr[cOff + 1] += (localWarr[cOff] * sinYplusZ) + (localWarr[cOff + 1] * cosYplusZ);
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
            tmpArr[cOff] += (localWarr[cOff] * cosTh) - (localWarr[cOff + 1] * sinTh);
            tmpArr[cOff + 1] += (localWarr[cOff] * sinTh) + (localWarr[cOff + 1] * cosTh);

            //-ve k1
            xId = P - k1;
            cOff = 2*(xOff + xId);
            cosTh = cXcYZ + sXsYZ;
            sinTh = cXsYZ - sXcYZ;
            tmpArr[cOff] += (localWarr[cOff] * cosTh) - (localWarr[cOff + 1] * sinTh);
            tmpArr[cOff + 1] += (localWarr[cOff] * sinTh) + (localWarr[cOff + 1] * cosTh);
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
            //tmpArr[cOff] += (localWarr[cOff] * cosTh) - (localWarr[cOff + 1] * sinTh)
            tmpArr[cOff] += (localWarr[cOff] * cosYplusZ) - (localWarr[cOff + 1] * sinYplusZ);
            //tmpArr[cOff + 1] += (localWarr[cOff] * sinTh) + (localWarr[cOff + 1] * cosTh)
            tmpArr[cOff + 1] += (localWarr[cOff] * sinYplusZ) + (localWarr[cOff + 1] * cosYplusZ);
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
            tmpArr[cOff] += (localWarr[cOff] * cosTh) - (localWarr[cOff + 1] * sinTh);
            tmpArr[cOff + 1] += (localWarr[cOff] * sinTh) + (localWarr[cOff + 1] * cosTh);

            //-ve k1
            xId = P - k1;
            cOff = 2*(xOff + xId);
            cosTh = cXcYZ + sXsYZ;
            sinTh = cXsYZ - sXcYZ;
            tmpArr[cOff] += (localWarr[cOff] * cosTh) - (localWarr[cOff + 1] * sinTh);
            tmpArr[cOff + 1] += (localWarr[cOff] * sinTh) + (localWarr[cOff + 1] * cosTh);
          }//end k1
        }//end k2
      }//end k3
    }//end j
  }//end i
  sendBoxList.clear();

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

  std::vector<double> recvLlist(recvDisps[npes - 1] + recvCnts[npes - 1]);

  double* sendBuf3 = NULL;
  if(!(sendLlist.empty())) {
    sendBuf3 = &(sendLlist[0]);
  }

  double* recvBuf3 = NULL;
  if(!(recvLlist.empty())) {
    recvBuf3 = &(recvLlist[0]);
  }

  MPI_Alltoallv(sendBuf3, sendCnts, sendDisps, MPI_DOUBLE,
      recvBuf3, recvCnts, recvDisps, MPI_DOUBLE, subComm);

  delete [] sendCnts;
  delete [] sendDisps;
  delete [] recvCnts;
  delete [] recvDisps;

  for(size_t i = 0; i < foundIdx.size(); ++i) {
    for(int d = 0; d < numWcoeffs; ++d) {
      localLlist[(numWcoeffs*(foundIdx[i])) + d] += recvLlist[(numWcoeffs*i) + d];
    }//end d
  }//end i

  PetscLogEventEnd(w2lEvent, 0, 0, 0, 0);
}



