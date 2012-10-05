
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include "mpi.h"
#include "pfgtOctUtils.h"
#include "par/parUtils.h"
#include "par/dtypes.h"

#include "colors.h"

extern PetscLogEvent pfgtMainEvent;
extern PetscLogEvent pfgtSetupEvent;
extern PetscLogEvent pfgtExpandEvent;
extern PetscLogEvent pfgtDirectEvent;
extern PetscLogEvent w2lEvent;
extern PetscLogEvent w2dD2lExpandEvent;
extern PetscLogEvent w2dD2lDirectEvent;

void pfgtMain(std::vector<double>& sources, const unsigned int minPtsInFgt, const unsigned int FgtLev,
    const int P, const int L, const int K, const double epsilon, MPI_Comm comm) {
  PetscLogEventBegin(pfgtMainEvent, 0, 0, 0, 0);

  std::vector<double> expandSources;
  std::vector<double> directSources;
  std::vector<ot::TreeNode> fgtList;

  int npesExpand, avgExpand, extraExpand;
  MPI_Comm subComm = MPI_COMM_NULL;

  bool singleType = false;

  pfgtSetup(expandSources, directSources, fgtList, singleType, npesExpand, avgExpand, 
      extraExpand, subComm, sources, minPtsInFgt, FgtLev, comm);

  int rank;
  MPI_Comm_rank(comm, &rank);

  if(!rank) {
    std::cout << GRN"Expand Num Procs = "NRM << npesExpand <<std::endl;
  }

  if(rank < npesExpand) {
    pfgtExpand(expandSources, fgtList, FgtLev, P, L, K, avgExpand,
        extraExpand, subComm, comm, singleType);
  } else {
    pfgtDirect(directSources, FgtLev, P, L, K, epsilon,
        subComm, comm, singleType);
  }

  if(subComm != MPI_COMM_NULL) {
    MPI_Comm_free(&subComm);
  }

  PetscLogEventEnd(pfgtMainEvent, 0, 0, 0, 0);
}

void pfgtSetup(std::vector<double>& expandSources, std::vector<double>& directSources, std::vector<ot::TreeNode>& fgtList,
    bool & singleType, int & npesExpand, int & avgExpand, int & extraExpand, MPI_Comm & subComm,
    std::vector<double>& sources, const unsigned int minPtsInFgt, const unsigned int FgtLev, MPI_Comm comm) {
  PetscLogEventBegin(pfgtSetupEvent, 0, 0, 0, 0);

  splitSources(expandSources, directSources, fgtList, sources, minPtsInFgt, FgtLev, comm);
  sources.clear();

  int localSizes[3];
  int globalSizes[3];
  localSizes[0] = (expandSources.size())/4;
  localSizes[1] = (directSources.size())/4;
  localSizes[2] = fgtList.size();

  MPI_Allreduce(localSizes, globalSizes, 3, MPI_INT, MPI_SUM, comm);

  int rank;
  MPI_Comm_rank(comm, &rank);

  if(!rank) {
    std::cout<<"Total Number of Expand Pts = "<<(globalSizes[0])<<std::endl;
    std::cout<<"Total Number of Direct Pts = "<<(globalSizes[1])<<std::endl;
    std::cout<<"Total Number of FGT boxes = "<<(globalSizes[2])<<std::endl;
  }

  std::vector<double> tmpExpandSources;
  int srcCnt = 0;
  for(int i = 0; i < fgtList.size(); ++i) {
#ifdef DEBUG
    assert(fgtList[i].getWeight() > 0);
#endif
    {
      tmpExpandSources.push_back(expandSources[srcCnt]);
      tmpExpandSources.push_back(expandSources[srcCnt + 1]);
      tmpExpandSources.push_back(expandSources[srcCnt + 2]);
      tmpExpandSources.push_back(expandSources[srcCnt + 3]);
      tmpExpandSources.push_back(fgtList[i].getWeight());
      srcCnt += 4;
    }
    for(int j = 1; j < fgtList[i].getWeight(); ++j) {
      tmpExpandSources.push_back(expandSources[srcCnt]);
      tmpExpandSources.push_back(expandSources[srcCnt + 1]);
      tmpExpandSources.push_back(expandSources[srcCnt + 2]);
      tmpExpandSources.push_back(expandSources[srcCnt + 3]);
      tmpExpandSources.push_back(0);
      srcCnt += 4;
    }//end j
  }//end i
  swap(expandSources, tmpExpandSources);
  tmpExpandSources.clear();
  fgtList.clear();

  int npes;
  MPI_Comm_size(comm, &npes);

  if(globalSizes[0] == 0) {
    //Only Direct
    singleType = true;
    npesExpand = 0;
    if(!rank) {
      std::cout<<"NOTE: ONLY DIRECT!"<<std::endl;
    }
  } else if(globalSizes[1] == 0) {
    //Only Expand
    singleType = true;
    npesExpand = npes;
    if(!rank) {
      std::cout<<"NOTE: ONLY EXPAND!"<<std::endl;
    }
  } else if(npes == 1) {
    //Serial
    std::cout<<"THIS CASE (Serial + Hybrid) IS NOT SUPPORTED!"<<std::endl;
    assert(false);
  } else {
    //Both Expand and Direct
    //NOTE: The following heuristic may need to be modified!
    singleType = false;
    npesExpand = (globalSizes[0]*npes)/(globalSizes[0] + globalSizes[1]);
#ifdef DEBUG
    assert(npesExpand < npes);
#endif
    if(npesExpand < 1) {
      npesExpand = 1;
    }
  }

  int npesDirect = npes - npesExpand;

  if(singleType) {
    MPI_Comm_dup(comm, &subComm);
  } else {
    MPI_Group group, subGroup;
    MPI_Comm_group(comm, &group);
    if(rank < npesExpand) {
      int* list = new int[npesExpand];
      for(int i = 0; i < npesExpand; i++) {
        list[i] = i;
      }//end for i
      MPI_Group_incl(group, npesExpand, list, &subGroup);
      delete [] list;
    } else {
      int* list = new int[npesDirect];
      for(int i = 0; i < npesDirect; i++) {
        list[i] = npesExpand + i;
      }//end for i
      MPI_Group_incl(group, npesDirect, list, &subGroup);
      delete [] list;
    }
    MPI_Group_free(&group);
    MPI_Comm_create(comm, subGroup, &subComm);
    MPI_Group_free(&subGroup);
  }

  avgExpand = 0;
  extraExpand = 0;
  if(npesExpand > 0) {
    avgExpand = (globalSizes[0])/npesExpand;
    extraExpand = (globalSizes[0])%npesExpand; 
  }

  int avgDirect = 0;
  int extraDirect = 0;
  if(npesDirect > 0) {
    avgDirect = (globalSizes[1])/npesDirect;
    extraDirect = (globalSizes[1])%npesDirect;
  }

  std::vector<double> finalExpandSources;

  if(npesExpand > 0) {
    if(rank < extraExpand) {
      par::scatterValues<double>(expandSources, finalExpandSources, (5*(avgExpand + 1)), comm);
    } else if(rank < npesExpand) {
      par::scatterValues<double>(expandSources, finalExpandSources, (5*avgExpand), comm);
    } else {
      par::scatterValues<double>(expandSources, finalExpandSources, 0, comm);
    }
  }

  std::vector<double> finalDirectSources;

  if(npesDirect > 0) {
    if(rank < npesExpand) {
      par::scatterValues<double>(directSources, finalDirectSources, 0, comm);
    } else if(rank < (npesExpand + extraDirect)) {
      par::scatterValues<double>(directSources, finalDirectSources, (4*(avgDirect + 1)), comm);
    } else {
      par::scatterValues<double>(directSources, finalDirectSources, (4*avgDirect), comm);
    }
  }

  swap(directSources, finalDirectSources);
  finalDirectSources.clear();

  expandSources.clear();
  for(int i = 0; i < finalExpandSources.size(); i += 5) {
    int flag = static_cast<int>(finalExpandSources[i + 4]);
    if(flag > 0) {
      unsigned int px = static_cast<unsigned int>(finalExpandSources[i]*(__DTPMD__));
      unsigned int py = static_cast<unsigned int>(finalExpandSources[i + 1]*(__DTPMD__));
      unsigned int pz = static_cast<unsigned int>(finalExpandSources[i + 2]*(__DTPMD__));
      ot::TreeNode pt(px, py, pz, __MAX_DEPTH__, __DIM__, __MAX_DEPTH__);
      ot::TreeNode box = pt.getAncestor(FgtLev);
      box.setWeight(flag);
      fgtList.push_back(box);
    }
    expandSources.push_back(finalExpandSources[i]);
    expandSources.push_back(finalExpandSources[i + 1]);
    expandSources.push_back(finalExpandSources[i + 2]);
    expandSources.push_back(finalExpandSources[i + 3]);
  }//end i
  finalExpandSources.clear();

  PetscLogEventEnd(pfgtSetupEvent, 0, 0, 0, 0);
}

void pfgtExpand(std::vector<double> & expandSources, std::vector<ot::TreeNode> & fgtList, const unsigned int FgtLev,
    const int P, const int L, const int K, const int avgExpand, const int extraExpand, 
    MPI_Comm subComm, MPI_Comm comm, bool singleType) {
  PetscLogEventBegin(pfgtExpandEvent, 0, 0, 0, 0);

  int subNpes;
  MPI_Comm_size(subComm, &subNpes);
  int subRank;
  MPI_Comm_rank(subComm, &subRank);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::cout << rank << GRN" : Expand - start "NRM << subRank << "/" << subNpes << std::endl; 

  std::vector<ot::TreeNode> fgtMins;
  computeFgtMinsExpand(fgtMins, fgtList, subComm, comm);

  std::cout << rank << GRN" : Expand - fgtmins "NRM << subRank << "/" << subNpes << std::endl; 

#ifdef DEBUG
  assert(!(expandSources.empty()));
#endif

  int sumFgtWts = 0;
  for(int i = 0; i < fgtList.size(); ++i) {
    sumFgtWts += fgtList[i].getWeight();
  }//end i

  int numExpandPts = (expandSources.size())/4;

  int numPtsInRemoteFgt = 0;
  if(fgtList.empty()) {
    numPtsInRemoteFgt = numExpandPts;
  } else {
    for( ; numPtsInRemoteFgt < numExpandPts; ++numPtsInRemoteFgt) {
      unsigned int px = static_cast<unsigned int>(expandSources[4*numPtsInRemoteFgt]*(__DTPMD__));
      unsigned int py = static_cast<unsigned int>(expandSources[(4*numPtsInRemoteFgt)+1]*(__DTPMD__));
      unsigned int pz = static_cast<unsigned int>(expandSources[(4*numPtsInRemoteFgt)+2]*(__DTPMD__));
      ot::TreeNode pt(px, py, pz, __MAX_DEPTH__, __DIM__, __MAX_DEPTH__);
      if(pt >= fgtList[0]) {
        break;
      }
    }//end for
  }

  int excessWt = sumFgtWts + numPtsInRemoteFgt - numExpandPts;
#ifdef DEBUG
  assert(excessWt >= 0);
#endif
  if(!(fgtList.empty())) {
    int lastWt = fgtList[fgtList.size() - 1].getWeight();
#ifdef DEBUG
    assert(lastWt > excessWt);
#endif
    fgtList[fgtList.size() - 1].setWeight(lastWt - excessWt);
  }

  int remoteFgtOwner = -1;
  ot::TreeNode remoteFgt;
  if(numPtsInRemoteFgt > 0) {
    computeRemoteFgt(remoteFgt, remoteFgtOwner, FgtLev, expandSources, fgtMins);
  }
#ifdef DEBUG
  assert(remoteFgtOwner < subRank);
#endif

  //Complex coefficients: [-P, P]x[-P, P]x[0, P] 
  //Coeff[-K1, -K2, -K3] = ComplexConjugate(Coeff[K1, K2, K3])
  const unsigned int TwoPplus1 = (2*P) + 1;
  const unsigned int numWcoeffs = 2*TwoPplus1*TwoPplus1*(P + 1);

  int* s2wSendCnts = NULL;
  int* s2wSendDisps = NULL;
  int* s2wRecvCnts = NULL;
  int* s2wRecvDisps = NULL;

  createS2WcommInfo(s2wSendCnts, s2wSendDisps, s2wRecvCnts, s2wRecvDisps, 
      remoteFgtOwner, numWcoeffs, excessWt, avgExpand, extraExpand, subComm);

  std::cout << rank << GRN" : Expand - pre-s2w "NRM << subRank << "/" << subNpes << std::endl; 

  std::vector<double> localWlist( (numWcoeffs*(fgtList.size())), 0.0);
  s2w(localWlist, expandSources, remoteFgt, remoteFgtOwner, numPtsInRemoteFgt, fgtList,
      fgtMins, FgtLev, P, L, s2wSendCnts, s2wSendDisps, s2wRecvCnts, s2wRecvDisps, subComm);
  std::cout << rank << GRN" : Expand - s2w "NRM << subRank << "/" << subNpes << std::endl; 

  std::vector<double> localLlist( (localWlist.size()), 0.0);
  w2l(localLlist, localWlist, fgtList, fgtMins, FgtLev, P, L, K, subComm);

  std::cout << rank << GRN" : Expand - w2l "NRM << subRank << "/" << subNpes << std::endl; 

  if(!singleType) {
    w2dAndD2lExpand(localLlist, localWlist, fgtList, P, comm);
  }
  std::cout << rank << GRN" : Expand - w2d+d2l"NRM << subRank << "/" << subNpes << std::endl; 

  std::vector<double> results(((expandSources.size())/4), 0.0);
  l2t(results, localLlist, expandSources, remoteFgt, remoteFgtOwner, numPtsInRemoteFgt, 
      fgtList, fgtMins, FgtLev, P, L, s2wSendCnts, s2wSendDisps, s2wRecvCnts, s2wRecvDisps, subComm);

  destroyS2WcommInfo(s2wSendCnts, s2wSendDisps, s2wRecvCnts, s2wRecvDisps); 
  std::cout << rank << GRN" : Expand - l2t "NRM << subRank << "/" << subNpes << std::endl; 

#ifdef _WRITE_SOLN
  std::cout << rank << GRN" : Expand - writing "NRM << subRank << "/" << subNpes << std::endl; 
  char fname[256];
  sprintf(fname, "expand.%d.res", rank);
  std::ofstream out(fname, std::ios::binary);
  out.write((const char*)&(*(results.begin())),results.size()*sizeof(double)); 
  out.close();
#endif

  std::cout << rank << GRN" : Expand - all_done "NRM << subRank << "/" << subNpes << std::endl; 

  PetscLogEventEnd(pfgtExpandEvent, 0, 0, 0, 0);
}

void pfgtDirect(std::vector<double> & directSources, const unsigned int FgtLev, const int P, const int L,
    const int K, const double epsilon, MPI_Comm subComm, MPI_Comm comm, bool singleType) {
  PetscLogEventBegin(pfgtDirectEvent, 0, 0, 0, 0);

  int subNpes;
  MPI_Comm_size(subComm, &subNpes);
  int subRank;
  MPI_Comm_rank(subComm, &subRank);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef DEBUG
  assert(!(directSources.empty()));
#endif

  std::cout << rank << RED" : Direct - start "NRM << subRank << "/" << subNpes << std::endl; 

  std::vector<ot::TreeNode> directNodes( directSources.size()/4 );
  for(size_t i = 0; i < directNodes.size(); ++i) {
    unsigned int px = static_cast<unsigned int>(directSources[(4*i) + 0]*(__DTPMD__));
    unsigned int py = static_cast<unsigned int>(directSources[(4*i) + 1]*(__DTPMD__));
    unsigned int pz = static_cast<unsigned int>(directSources[(4*i) + 2]*(__DTPMD__));
    directNodes[i] = ot::TreeNode(px, py, pz, __MAX_DEPTH__, __DIM__, __MAX_DEPTH__);
  }//end i

  std::vector<ot::TreeNode> directMins(subNpes);
  MPI_Allgather(&(directNodes[0]), 1, par::Mpi_datatype<ot::TreeNode>::value(),
      &(directMins[0]), 1, par::Mpi_datatype<ot::TreeNode>::value(), subComm);

  std::vector<ot::TreeNode> fgtMins;
  if(!singleType) {
    computeFgtMinsDirect(fgtMins, comm);
  }

  std::cout << rank << RED" : Direct - fgtMins "NRM << subRank << "/" << subNpes << std::endl; 

  std::vector<double> results(directNodes.size(), 0.0);
  d2d(results, directSources, directNodes, directMins, FgtLev, epsilon, subComm);

  std::cout << rank << RED" : Direct - d2d "NRM << subRank << "/" << subNpes << std::endl; 

  if(!singleType) {
    w2dAndD2lDirect(results, directSources, fgtMins, FgtLev, P, L, K, epsilon, comm);
  }

  std::cout << rank << RED" : Direct - w2d+d2l "NRM << subRank << "/" << subNpes << std::endl; 

#ifdef _WRITE_SOLN
  std::cout << rank << RED" : Direct - writing "NRM << subRank << "/" << subNpes << std::endl; 
  char fname[256];
  sprintf(fname, "direct.%d.res", rank);
  std::ofstream out(fname, std::ios::binary);
  out.write((const char*)&(*(results.begin())),results.size()*sizeof(double)); 
  out.close();
#endif

  std::cout << rank << RED" : Direct - all_done "NRM << subRank << "/" << subNpes << std::endl; 
  PetscLogEventEnd(pfgtDirectEvent, 0, 0, 0, 0);
}

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

  //Performance Improvement: We can use the fact that each processor's chunk in
  //the recvBoxList is sorted and avoid the searches.
  std::vector<int> recvBoxIds(recvBoxList.size(), -1);

  for(int i = 0; i < recvBoxList.size(); ++i) { 
    unsigned int retIdx;
    bool found = seq::BinarySearch(&(fgtList[0]), fgtList.size(), recvBoxList[i], &retIdx);
    if(found) {
      recvBoxIds[i] = retIdx;
    }
  }//end i

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

  if(!(tmpSendBoxList.empty())) {
    std::sort((&(tmpSendBoxList[0])), (&(tmpSendBoxList[0])) + tmpSendBoxList.size());
  }

  int* sendCnts = new int[npes];
  int* recvDisps = new int[npes]; 
  for(int i = 0; i < npes; ++i) {
    sendCnts[i] = 0;
    recvDisps[i] = 0;
  }//end i

  //Performance Improvement: We could make use of the fact that tmpSendBoxList is
  //sorted and avoid the searches.

  std::vector<ot::TreeNode> sendBoxList;
  std::vector<std::vector<unsigned int> > box2PtMap;

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
    int boxId = foundIds[i];
    double cx = (0.5*hFgt) + ((static_cast<double>(sendBoxList[boxId].getX()))/(__DTPMD__));
    double cy = (0.5*hFgt) + ((static_cast<double>(sendBoxList[boxId].getY()))/(__DTPMD__));
    double cz = (0.5*hFgt) + ((static_cast<double>(sendBoxList[boxId].getZ()))/(__DTPMD__));
    for(int j = 0; j < box2PtMap[boxId].size(); ++j) {
      double px = cx - sources[box2PtMap[boxId][j]];
      double py = cy - sources[box2PtMap[boxId][j] + 1];
      double pz = cz - sources[box2PtMap[boxId][j] + 2];
      double pf = sources[box2PtMap[boxId][j] + 3];

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
            sendLlist[(numWcoeffs*i) + (2*di)] += (pf * cosTh);
            sendLlist[(numWcoeffs*i) + (2*di) + 1] += (pf * sinTh);
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
    int boxId = foundIds[i];
    double cx = (0.5*hFgt) + ((static_cast<double>(sendBoxList[boxId].getX()))/(__DTPMD__));
    double cy = (0.5*hFgt) + ((static_cast<double>(sendBoxList[boxId].getY()))/(__DTPMD__));
    double cz = (0.5*hFgt) + ((static_cast<double>(sendBoxList[boxId].getZ()))/(__DTPMD__));
    for(int j = 0; j < box2PtMap[boxId].size(); ++j) {
      double px = sources[box2PtMap[boxId][j]] - cx;
      double py = sources[box2PtMap[boxId][j] + 1] - cy;
      double pz = sources[box2PtMap[boxId][j] + 2] - cz;

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
            double a = recvWlist[(numWcoeffs*i) + (2*di)];
            double b = recvWlist[(numWcoeffs*i) + (2*di) + 1];
            double c = ((c3[d3])*tmp1) - ((s3[d3])*tmp2);
            double d = ((s3[d3])*tmp1) + ((c3[d3])*tmp2); 
            results[(box2PtMap[boxId][j])/4] += ((fac[d3])*(fac[d2])*(fac[d1])*( (a*c) - (b*d) ));
          }//end for k1
        }//end for k2
      }//end for k3
    }//end j
  }//end i

  PetscLogEventEnd(w2dD2lDirectEvent, 0, 0, 0, 0);
}

void createS2WcommInfo(int*& sendCnts, int*& sendDisps, int*& recvCnts, int*& recvDisps, 
    const int remoteFgtOwner, const unsigned int numWcoeffs, const int excessWt,
    const int avgExpand, const int extraExpand, MPI_Comm subComm) {
  int npes;
  MPI_Comm_size(subComm, &npes);

  int rank;
  MPI_Comm_rank(subComm, &rank);

  sendCnts = new int[npes];
  recvCnts = new int[npes]; 
  sendDisps = new int[npes];
  recvDisps = new int[npes]; 

  for(int i = 0; i < npes; ++i) {
    sendCnts[i] = 0;
    recvCnts[i] = 0;
  }//end i

  if(remoteFgtOwner >= 0) {
    sendCnts[remoteFgtOwner] = numWcoeffs;
  }

  for(int i = (rank + 1), leftOver = excessWt; i < npes; ++i) {
    if(leftOver > 0) {
      recvCnts[i] = numWcoeffs;
      if(i < extraExpand) {
        leftOver = leftOver - (avgExpand + 1);
      } else {
        leftOver = leftOver - avgExpand;
      }
    } else {
      break;
    }
  }//end i

  sendDisps[0] = 0;
  recvDisps[0] = 0;
  for(int i = 1; i < npes; ++i) {
    sendDisps[i] = sendDisps[i - 1] + sendCnts[i - 1];
    recvDisps[i] = recvDisps[i - 1] + recvCnts[i - 1];
  }//end i
}

void destroyS2WcommInfo(int* sendCnts, int* sendDisps, int* recvCnts, int* recvDisps) {
  delete [] sendCnts;
  delete [] sendDisps;
  delete [] recvCnts;
  delete [] recvDisps;
}

void computeRemoteFgt(ot::TreeNode & remoteFgt, int & remoteFgtOwner, const unsigned int FgtLev,
    std::vector<double> & sources, std::vector<ot::TreeNode> & fgtMins) {
  unsigned int px = static_cast<unsigned int>(sources[0]*(__DTPMD__));
  unsigned int py = static_cast<unsigned int>(sources[1]*(__DTPMD__));
  unsigned int pz = static_cast<unsigned int>(sources[2]*(__DTPMD__));
  ot::TreeNode ptOct(px, py, pz, __MAX_DEPTH__, __DIM__, __MAX_DEPTH__);
  remoteFgt = ptOct.getAncestor(FgtLev);
  unsigned int retIdx;
  seq::maxLowerBound(fgtMins, remoteFgt, retIdx, NULL, NULL);
  remoteFgtOwner = fgtMins[retIdx].getWeight();
}

void computeFgtMinsExpand(std::vector<ot::TreeNode> & fgtMins, std::vector<ot::TreeNode> & fgtList,
    MPI_Comm subComm, MPI_Comm comm) {
  int subNpes;
  MPI_Comm_size(subComm, &subNpes);

  int subRank;
  MPI_Comm_rank(subComm, &subRank);

  int rank;
  MPI_Comm_rank(comm, &rank);

#ifdef DEBUG
  assert(rank == subRank);
#endif

  ot::TreeNode firstFgt;
  if(!(fgtList.empty())) {
    firstFgt = fgtList[0];
  }

  ot::TreeNode* recvBuf = NULL;
  if(rank == 0) {
    fgtMins.resize(subNpes);
    recvBuf = &(fgtMins[0]);
  }

  MPI_Gather(&firstFgt, 1, par::Mpi_datatype<ot::TreeNode>::value(),
      recvBuf, 1, par::Mpi_datatype<ot::TreeNode>::value(), 0, subComm);

  int fgtMinSize;
  if(rank == 0) {
    std::vector<ot::TreeNode> tmpMins;
    for(unsigned int i = 0; i < fgtMins.size(); ++i) {
      if(fgtMins[i].getDim()) {
        fgtMins[i].setWeight(i);
        tmpMins.push_back(fgtMins[i]);
      }
    }//end i
    swap(fgtMins, tmpMins);
    fgtMinSize = fgtMins.size();
  }

  MPI_Bcast(&fgtMinSize, 1, MPI_INT, 0, comm);

  if(rank) {
    fgtMins.resize(fgtMinSize);
  }

  MPI_Bcast(&(fgtMins[0]), fgtMinSize, par::Mpi_datatype<ot::TreeNode>::value(), 0, comm);
}

void computeFgtMinsDirect(std::vector<ot::TreeNode> & fgtMins, MPI_Comm comm) {
  int fgtMinSize;
  MPI_Bcast(&fgtMinSize, 1, MPI_INT, 0, comm);

  fgtMins.resize(fgtMinSize);
  MPI_Bcast(&(fgtMins[0]), fgtMinSize, par::Mpi_datatype<ot::TreeNode>::value(), 0, comm);
}



