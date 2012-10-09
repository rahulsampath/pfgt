
#include "petsc.h"
#include <vector>
#include "oct/TreeNode.h"
#include "pfgtOctUtils.h"

extern PetscLogEvent l2tEvent;

void l2t(std::vector<double> & results, std::vector<double> & localLlist, std::vector<double> & sources, 
    const ot::TreeNode remoteFgt, const int remoteFgtOwner, const int numPtsInRemoteFgt,
    std::vector<ot::TreeNode> & fgtList, std::vector<ot::TreeNode> & fgtMins,
    const unsigned int FgtLev, const int P, const int L,
    int* sendCnts, int* sendDisps, int* recvCnts, int* recvDisps, MPI_Comm subComm) {
  PetscLogEventBegin(l2tEvent, 0, 0, 0, 0);

  int npes;
  MPI_Comm_size(subComm, &npes);

  //Complex coefficients: [-P, P]x[-P, P]x[0, P] 
  //Coeff[-K1, -K2, -K3] = ComplexConjugate(Coeff[K1, K2, K3])
  const unsigned int TwoPplus1 = (2*P) + 1;
  const unsigned int numWcoeffs = 2*TwoPplus1*TwoPplus1*(P + 1);

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

  //Fgt box size = sqrt(delta)
  const double hFgt = 1.0/(static_cast<double>(1u << FgtLev));

  const double LbyP = static_cast<double>(L)/static_cast<double>(P);
  const double ImExpZfactor = LbyP/hFgt;
  const double ReExpZfactor = -0.25*LbyP*LbyP;
  const double C0 = (0.5*LbyP/(__SQRT_PI__));

#ifdef DEBUG
  assert(P >= 1);
#endif
  std::vector<double> fac(P);
  double* facArr = (&(fac[0])) - 1;
  for(int k = 1; k <= P; ++k) {
    facArr[k] = C0*exp(ReExpZfactor*(static_cast<double>(k*k)));
  }//end k

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

  if(remoteFgtOwner >= 0) {
    double cx = (0.5*hFgt) + ((static_cast<double>(remoteFgt.getX()))/(__DTPMD__));
    double cy = (0.5*hFgt) + ((static_cast<double>(remoteFgt.getY()))/(__DTPMD__));
    double cz = (0.5*hFgt) + ((static_cast<double>(remoteFgt.getZ()))/(__DTPMD__));
    for(int i = 0; i < numPtsInRemoteFgt; ++i) {
      unsigned int sOff = 4*i;
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

      {
        //k3 = k2 = k1 = 0
        //cosZ = 1
        //sinZ = 0
        //zId = k3 = 0
        //yOff = zId*TwoPplus1 = 0
        //cosY = 1
        //sinY = 0
        //cYcZ = cosY*cosZ = cosZ = 1
        //cYsZ = cosY*sinZ = sinZ = 0
        //sYsZ = sinY*sinZ = 0
        //sYcZ = sinY*cosZ = 0
        //yId = P + k2 = P
        //xOff = (yOff + yId)*TwoPplus1  
        int xOff = P*TwoPplus1;
        //cosYplusZ = cYcZ - sYsZ = 1
        //sinYplusZ = sYcZ + cYsZ = 0
        //cosX = 1
        //sinX = 0
        //cXcYZ = cosX*cosYplusZ = cosYplusZ = 1
        //cXsYZ = cosX*sinYplusZ = sinYplusZ = 0
        //sXsYZ = sinX*sinYplusZ = 0
        //sXcYZ = sinX*cosYplusZ = 0
        //xId = P + k1 = P
        int cOff = 2*(xOff + P);
        //cosTh = cXcYZ - sXsYZ = cosYplusZ = 1
        //sinTh = sXcYZ + cXsYZ = sinYplusZ = 0
      }
      {
        //k3 = k2 = 0
        //cosZ = 1
        //sinZ = 0
        //zId = k3 = 0
        //yOff = zId*TwoPplus1 = 0
        //cosY = 1
        //sinY = 0
        //cYcZ = cosY*cosZ = cosZ = 1
        //cYsZ = cosY*sinZ = sinZ = 0
        //sYsZ = sinY*sinZ = 0
        //sYcZ = sinY*cosZ = 0
        //yId = P + k2 = P
        //xOff = (yOff + yId)*TwoPplus1  
        int xOff = P*TwoPplus1;
        //cosYplusZ = cYcZ - sYsZ = 1
        //sinYplusZ = sYcZ + cYsZ = 0
        for(int k1 = 1; k1 <= P; ++k1) {
          double cosX = c1Arr[k1];
          double sinX = s1Arr[k1];
          double cXcYZ = cosX*cosYplusZ;
          double cXsYZ = cosX*sinYplusZ;
          double sXsYZ = sinX*sinYplusZ;
          double sXcYZ = sinX*cosYplusZ;

          int xId = P + k1;
          int cOff = 2*(xOff + xId);
          double cosTh = cXcYZ - sXsYZ;
          double sinTh = sXcYZ + cXsYZ;
        }//end k1
      }
      {
        //k3 = 0
        //cosZ = 1
        //sinZ = 0
        //zId = k3 = 0
        //yOff = zId*TwoPplus1 = 0
        for(int k2 = 1; k2 <= P; ++k2) {
          double cosY = c2Arr[k2];
          double sinY = s2Arr[k2];
          double cYcZ = cosY*cosZ;
          double cYsZ = cosY*sinZ;
          double sYsZ = sinY*sinZ;
          double sYcZ = sinY*cosZ;

          int yId = P + k2;
          int xOff = (yOff + yId)*TwoPplus1;
          double cosYplusZ = cYcZ - sYsZ;
          double sinYplusZ = sYcZ + cYsZ;
          {
            //k1 = 0
            //cosX = 1
            //sinX = 0
            //cXcYZ = cosX*cosYplusZ = cosYplusZ
            //cXsYZ = cosX*sinYplusZ = sinYplusZ
            //sXsYZ = sinX*sinYplusZ = 0
            //sXcYZ = sinX*cosYplusZ = 0
            //xId = P + k1 = P
            int cOff = 2*(xOff + P);
            //cosTh = cXcYZ - sXsYZ = cosYplusZ
            //sinTh = sXcYZ + cXsYZ = sinYplusZ
          }
          for(int k1 = 1; k1 <= P; ++k1) {
            double cosX = c1Arr[k1];
            double sinX = s1Arr[k1];
            double cXcYZ = cosX*cosYplusZ;
            double cXsYZ = cosX*sinYplusZ;
            double sXsYZ = sinX*sinYplusZ;
            double sXcYZ = sinX*cosYplusZ;

            //+ve k1
            int xId = P + k1;
            int cOff = 2*(xOff + xId);
            double cosTh = cXcYZ - sXsYZ;
            double sinTh = sXcYZ + cXsYZ;

            //-ve k1
            xId = P - k1;
            cOff = 2*(xOff + xId);
            cosTh = cXcYZ + sXsYZ;
            sinTh = cXsYZ - sXcYZ;
          }//end k1
        }//end k2
      }
      for(int k3 = 1; k3 <= P; ++k3) {
        double cosZ = c3Arr[k3];
        double sinZ = s3Arr[k3];
        int zId = k3;
        int yOff = zId*TwoPplus1;
        {
          //k2 = 0
          //cosY = 1
          //sinY = 0
          //cYcZ = cosY*cosZ = cosZ
          //cYsZ = cosY*sinZ = sinZ
          //sYsZ = sinY*sinZ = 0
          //sYcZ = sinY*cosZ = 0
          //yId = P + k2 = P
          //xOff = (yOff + yId)*TwoPplus1
          int xOff = (yOff + P)*TwoPplus1;
          //cosYplusZ = cYcZ - sYsZ = cosZ
          //sinYplusZ = sYcZ + cYsZ = sinZ
          {
            //k1 = 0
            //cosX = 1
            //sinX = 0
            //cXcYZ = cosX*cosYplusZ = cosYplusZ
            //cXsYZ = cosX*sinYplusZ = sinYplusZ
            //sXsYZ = sinX*sinYplusZ = 0
            //sXcYZ = sinX*cosYplusZ = 0
            //xId = P + k1 = P
            int cOff = 2*(xOff + P);
            //cosTh = cXcYZ - sXsYZ = cosYplusZ
            //sinTh = sXcYZ + cXsYZ = sinYplusZ
          }
          for(int k1 = 1; k1 <= P; ++k1) {
            double cosX = c1Arr[k1];
            double sinX = s1Arr[k1];
            double cXcYZ = cosX*cosYplusZ;
            double cXsYZ = cosX*sinYplusZ;
            double sXsYZ = sinX*sinYplusZ;
            double sXcYZ = sinX*cosYplusZ;

            //+ve k1
            int xId = P + k1;
            int cOff = 2*(xOff + xId);
            double cosTh = cXcYZ - sXsYZ;
            double sinTh = sXcYZ + cXsYZ;

            //-ve k1
            xId = P - k1;
            cOff = 2*(xOff + xId);
            cosTh = cXcYZ + sXsYZ;
            sinTh = cXsYZ - sXcYZ;
          }//end k1
        }
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
            //cXcYZ = cosX*cosYplusZ = cosYplusZ
            //cXsYZ = cosX*sinYplusZ = sinYplusZ
            //sXsYZ = sinX*sinYplusZ = 0
            //sXcYZ = sinX*cosYplusZ = 0
            //xId = P + k1 = P
            int cOff = 2*(xOff + P);
            //cosTh = cXcYZ - sXsYZ = cosYplusZ
            //sinTh = sXcYZ + cXsYZ = sinYplusZ
          }
          for(int k1 = 1; k1 <= P; ++k1) {
            double cosX = c1Arr[k1];
            double sinX = s1Arr[k1];
            double cXcYZ = cosX*cosYplusZ;
            double cXsYZ = cosX*sinYplusZ;
            double sXsYZ = sinX*sinYplusZ;
            double sXcYZ = sinX*cosYplusZ;

            //+ve k1
            int xId = P + k1;
            int cOff = 2*(xOff + xId);
            double cosTh = cXcYZ - sXsYZ;
            double sinTh = sXcYZ + cXsYZ;

            //-ve k1
            xId = P - k1;
            cOff = 2*(xOff + xId);
            cosTh = cXcYZ + sXsYZ;
            sinTh = cXsYZ - sXcYZ;
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
            //cXcYZ = cosX*cosYplusZ = cosYplusZ
            //cXsYZ = cosX*sinYplusZ = sinYplusZ
            //sXsYZ = sinX*sinYplusZ = 0
            //sXcYZ = sinX*cosYplusZ = 0
            //xId = P + k1 = P
            int cOff = 2*(xOff + P);
            //cosTh = cXcYZ - sXsYZ = cosYplusZ
            //sinTh = sXcYZ + cXsYZ = sinYplusZ
          }
          for(int k1 = 1; k1 <= P; ++k1) {
            double cosX = c1Arr[k1];
            double sinX = s1Arr[k1];
            double cXcYZ = cosX*cosYplusZ;
            double cXsYZ = cosX*sinYplusZ;
            double sXsYZ = sinX*sinYplusZ;
            double sXcYZ = sinX*cosYplusZ;

            //+ve k1
            int xId = P + k1;
            int cOff = 2*(xOff + xId);
            double cosTh = cXcYZ - sXsYZ;
            double sinTh = sXcYZ + cXsYZ;

            //-ve k1
            xId = P - k1;
            cOff = 2*(xOff + xId);
            cosTh = cXcYZ + sXsYZ;
            sinTh = cXsYZ - sXcYZ;
          }//end k1
        }//end k2
      }//end k3

      /*
         for(int k3 = -P, d3 = 0, di = 0; k3 < P; ++d3, ++k3) {
         for(int k2 = -P, d2 = 0; k2 < P; ++d2, ++k2) {
         for(int k1 = -P, d1 = 0; k1 < P; ++d1, ++k1, ++di) {
         double tmp1 =  ((c1[d1])*(c2[d2])) - ((s1[d1])*(s2[d2]));
         double tmp2 =  ((s1[d1])*(c2[d2])) + ((s2[d2])*(c1[d1]));
         double cosTh = ((c3[d3])*tmp1) - ((s3[d3])*tmp2);
         double sinTh = ((s3[d3])*tmp1) + ((c3[d3])*tmp2); 
         int cOff = 2*di;
         results[i] += (facArr[k3] * facArr[k2] * facArr[k1] * ( (recvLlist[cOff] * cosTh) - (recvLlist[cOff + 1] * sinTh) ));
         }//end for k1
         }//end for k2
         }//end for k3
         */
    }//end i
  }

  for(int i = 0, ptsIdx = numPtsInRemoteFgt; i < fgtList.size(); ++i) {
    double* localLarr = &(localLlist[numWcoeffs*i]);
    double cx = (0.5*hFgt) + ((static_cast<double>(fgtList[i].getX()))/(__DTPMD__));
    double cy = (0.5*hFgt) + ((static_cast<double>(fgtList[i].getY()))/(__DTPMD__));
    double cz = (0.5*hFgt) + ((static_cast<double>(fgtList[i].getZ()))/(__DTPMD__));
    for(int j = 0; j < fgtList[i].getWeight(); ++j, ++ptsIdx) {
      unsigned int sOff = 4*ptsIdx;
      double px = sources[sOff] - cx;
      double py = sources[sOff + 1] - cy;
      double pz = sources[sOff + 2] - cx;

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

      /*
         for(int k3 = -P, d3 = 0, di = 0; k3 < P; ++d3, ++k3) {
         for(int k2 = -P, d2 = 0; k2 < P; ++d2, ++k2) {
         for(int k1 = -P, d1 = 0; k1 < P; ++d1, ++k1, ++di) {
         double tmp1 =  ((c1[d1])*(c2[d2])) - ((s1[d1])*(s2[d2]));
         double tmp2 =  ((s1[d1])*(c2[d2])) + ((s2[d2])*(c1[d1]));
         double cosTh = ((c3[d3])*tmp1) - ((s3[d3])*tmp2);
         double sinTh = ((s3[d3])*tmp1) + ((c3[d3])*tmp2); 
         int cOff = 2*di;
         results[ptsIdx] += (facArr[k3] * facArr[k2] * facArr[k1] * ( (localLarr[cOff] * cosTh) - (localLarr[cOff + 1] * sinTh) ));
         }//end for k1
         }//end for k2
         }//end for k3
         */
    }//end j
  }//end i

  PetscLogEventEnd(l2tEvent, 0, 0, 0, 0);
}


