
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

  const unsigned int TwoP = 2*P;
  std::vector<double> fac(TwoP);
  for(int kk = -P, di = 0; kk < P; ++di, ++kk) {
    fac[di] = C0*exp(ReExpZfactor*(static_cast<double>(kk*kk)));
  }//end kk

  std::vector<double> c1(TwoP);
  std::vector<double> c2(TwoP);
  std::vector<double> c3(TwoP);
  std::vector<double> s1(TwoP);
  std::vector<double> s2(TwoP);
  std::vector<double> s3(TwoP);

  if(remoteFgtOwner >= 0) {
    double cx = (0.5*hFgt) + ((static_cast<double>(remoteFgt.getX()))/(__DTPMD__));
    double cy = (0.5*hFgt) + ((static_cast<double>(remoteFgt.getY()))/(__DTPMD__));
    double cz = (0.5*hFgt) + ((static_cast<double>(remoteFgt.getZ()))/(__DTPMD__));
    for(int i = 0; i < numPtsInRemoteFgt; ++i) {
      unsigned int sOff = 4*i;
      double px = sources[sOff] - cx;
      double py = sources[sOff + 1] - cy;
      double pz = sources[sOff + 2] - cz;

      for (int kk = -P, di = 0; kk < P; ++kk, ++di) {
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
            double a = recvLlist[cOff];
            double b = recvLlist[cOff + 1];
            results[i] += ((fac[d3])*(fac[d2])*(fac[d1])*( (a*c) - (b*d) ));
          }//end for k1
        }//end for k2
      }//end for k3
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

      for (int kk = -P, di = 0; kk < P; ++kk, ++di) {
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
            double a = localLarr[cOff];
            double b = localLarr[cOff + 1];
            results[ptsIdx] += ((fac[d3])*(fac[d2])*(fac[d1])*( (a*c) - (b*d) ));
          }//end for k1
        }//end for k2
      }//end for k3
    }//end j
  }//end i

  PetscLogEventEnd(l2tEvent, 0, 0, 0, 0);
}


