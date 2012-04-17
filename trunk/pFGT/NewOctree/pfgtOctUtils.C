
#include <iostream>
#include <cmath>
#include <algorithm>
#include "mpi.h"
#include "pfgtOctUtils.h"
#include "parUtils.h"
#include "dtypes.h"

extern PetscLogEvent fgtEvent;
extern PetscLogEvent s2wEvent;
extern PetscLogEvent s2wCommEvent;
extern PetscLogEvent serialEvent;
extern PetscLogEvent fgtOctConEvent;
extern PetscLogEvent expandOnlyEvent;
extern PetscLogEvent expandHybridEvent;
extern PetscLogEvent directOnlyEvent;
extern PetscLogEvent directHybridEvent;

void pfgt(std::vector<ot::TreeNode> & linOct, const unsigned int FgtLev, std::vector<double> & sources,  
    const int P, const int L, const int K, const double DirectHfactor, MPI_Comm comm) {
  PetscLogEventBegin(fgtEvent, 0, 0, 0, 0);

  int npes, rank;
  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &rank);

  //Kernel Bandwidth
  const double delta = 1.0/(static_cast<double>(1u << (FgtLev << 1)));

  //FGT box size = sqrt(delta)
  const double hFgt = 1.0/(static_cast<double>(1u << FgtLev));

  //2P complex coefficients for each dimension.  
  const unsigned int Ndofs = 16*P*P*P;

  int numPts = ((sources.size())/4);

  if(!rank) {
    std::cout<<"delta = "<<delta<<std::endl;
    std::cout<<"Ndofs = "<< Ndofs <<std::endl;
    std::cout<<"StencilWidth = "<< K <<std::endl;
  }

  //Split octree and sources into 2 sets
  std::vector<double> expandSources;
  std::vector<double> directSources;
  std::vector<ot::TreeNode> expandTree;
  std::vector<ot::TreeNode> directTree;
  int ptsCnt = 0;
  for(size_t i = 0; i < linOct.size(); i++) {
    unsigned int lev = linOct[i].getLevel();
    double hCurrOct = 1.0/(static_cast<double>(1u << lev));
    linOct[i].setWeight(0);
    bool isExpand = false;
    if( hCurrOct <= (hFgt*DirectHfactor) ) {
      expandTree.push_back(linOct[i]);
      isExpand = true;
    } else {
      directTree.push_back(linOct[i]);
    }
    while(ptsCnt < numPts) {
      unsigned int px = (unsigned int)(sources[4*ptsCnt]*(double)(1u << __MAX_DEPTH__));
      unsigned int py = (unsigned int)(sources[(4*ptsCnt)+1]*(double)(1u << __MAX_DEPTH__));
      unsigned int pz = (unsigned int)(sources[(4*ptsCnt)+2]*(double)(1u << __MAX_DEPTH__));
      ot::TreeNode tmpOct(px, py, pz, __MAX_DEPTH__, __DIM__, __MAX_DEPTH__);
      if(tmpOct >= linOct[i]) {
        if((tmpOct == linOct[i]) || (linOct[i].isAncestor(tmpOct))) {
          if(isExpand) {
            expandTree[expandTree.size() - 1].addWeight(1);
            expandSources.push_back(sources[4*ptsCnt]);
            expandSources.push_back(sources[(4*ptsCnt) + 1]);
            expandSources.push_back(sources[(4*ptsCnt) + 2]);
            expandSources.push_back(sources[(4*ptsCnt) + 3]);
          } else {
            directTree[directTree.size() - 1].addWeight(1);
            directSources.push_back(sources[4*ptsCnt]);
            directSources.push_back(sources[(4*ptsCnt) + 1]);
            directSources.push_back(sources[(4*ptsCnt) + 2]);
            directSources.push_back(sources[(4*ptsCnt) + 3]);
          }
        } else {
          break;
        }
      }
      ++ptsCnt;
    }//end while
  }//end for i
  linOct.clear();
  sources.clear();

  unsigned int localTreeSizes[2];
  unsigned int globalTreeSizes[2];
  localTreeSizes[0] = expandTree.size();
  localTreeSizes[1] = directTree.size();
  MPI_Allreduce(localTreeSizes, globalTreeSizes, 2, MPI_UNSIGNED, MPI_SUM, comm);

  if(globalTreeSizes[0] == 0) {
    pfgtOnlyDirect(directSources, directTree, comm);
  } else if(globalTreeSizes[1] == 0) {
    pfgtOnlyExpand(expandSources, expandTree, FgtLev, comm);
  } else if(npes == 1) {
    pfgtSerial(directSources, expandSources, directTree, expandTree, FgtLev);
  } else {
    int npesExpand = (globalTreeSizes[0]*npes)/(globalTreeSizes[0] + globalTreeSizes[1]);
    assert(npesExpand < npes);
    if(npesExpand < 1) {
      npesExpand = 1;
    }
    int npesDirect = npes - npesExpand;
    assert(npesDirect > 0);

    MPI_Comm subComm;
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

    int avgExpand = (globalTreeSizes[0])/npesExpand;
    int extraExpand = (globalTreeSizes[0])%npesExpand; 
    int avgDirect = (globalTreeSizes[1])/npesDirect;
    int extraDirect = (globalTreeSizes[1])%npesDirect;

    std::vector<ot::TreeNode> finalExpandTree;
    std::vector<ot::TreeNode> finalDirectTree;

    if(rank < extraExpand) {
      par::scatterValues<ot::TreeNode>(expandTree, finalExpandTree, (avgExpand + 1), comm);
      par::scatterValues<ot::TreeNode>(directTree, finalDirectTree, 0, comm);
    } else if(rank < npesExpand) {
      par::scatterValues<ot::TreeNode>(expandTree, finalExpandTree, avgExpand, comm);
      par::scatterValues<ot::TreeNode>(directTree, finalDirectTree, 0, comm);
    } else if(rank < (npesExpand + extraDirect)) {
      par::scatterValues<ot::TreeNode>(expandTree, finalExpandTree, 0, comm);
      par::scatterValues<ot::TreeNode>(directTree, finalDirectTree, (avgDirect + 1), comm);
    } else {
      par::scatterValues<ot::TreeNode>(expandTree, finalExpandTree, 0, comm);
      par::scatterValues<ot::TreeNode>(directTree, finalDirectTree, avgDirect, comm);
    }

    expandTree.clear();
    directTree.clear();

    int numRecvExpandPts = 0;
    for(int i = 0; i < finalExpandTree.size(); ++i) {
      numRecvExpandPts += (finalExpandTree[i].getWeight());
    }//end for i

    int numRecvDirectPts = 0;
    for(int i = 0; i < finalDirectTree.size(); ++i) {
      numRecvDirectPts += (finalDirectTree[i].getWeight());
    }//end for i

    std::vector<double> finalExpandSources;
    std::vector<double> finalDirectSources;

    par::scatterValues<double>(expandSources, finalExpandSources, (4*numRecvExpandPts), comm);
    par::scatterValues<double>(directSources, finalDirectSources, (4*numRecvDirectPts), comm);

    expandSources.clear();
    directSources.clear();

    if(rank < npesExpand) {
      pfgtHybridExpand(finalExpandSources, finalExpandTree, P, L, FgtLev, delta, hFgt, subComm, comm);
    } else {
      pfgtHybridDirect(finalDirectSources, finalDirectTree, FgtLev, subComm, comm);
    }

    MPI_Comm_free(&subComm);
  }

  PetscLogEventEnd(fgtEvent, 0, 0, 0, 0);
}

void pfgtHybridExpand(std::vector<double> & expandSources, std::vector<ot::TreeNode> & expandTree,
    const int P, const int L, const unsigned int FgtLev, const double delta, 
    const double hFgt, MPI_Comm subComm, MPI_Comm comm) {
  PetscLogEventBegin(expandHybridEvent, 0, 0, 0, 0);

  assert(!(expandTree.empty()));

  std::vector<ot::TreeNode> expandMins;
  computeMins(expandMins, expandTree, subComm);

  std::vector<ot::TreeNode> fgtList;
  createFGToctree(fgtList, expandTree, FgtLev, subComm);

  std::vector<ot::TreeNode> fgtMins;
  computeFGTminsHybridExpand(fgtMins, fgtList, subComm, comm);

  PetscLogEventEnd(expandHybridEvent, 0, 0, 0, 0);
}

void s2wLocal(std::vector<double> & expandSources, std::vector<ot::TreeNode> & expandTree,
    const int P, const int L, const double hFgt) {
  PetscLogEventBegin(s2wEvent, 0, 0, 0, 0);

  const double LbyP = static_cast<double>(L)/static_cast<double>(P);
  const double ImExpZfactor = LbyP/hFgt; 

  std::vector<std::vector<unsigned int> > oct2fgtIdmap(expandTree.size());

  std::vector<unsigned int> uniqueOct2fgtIdmap;

  std::vector<std::vector<double> > Wfgt;

  for(unsigned int i = 0; i < expandTree.size(); i++) {
    unsigned int lev = expandTree[i].getLevel();
    double hCurrOct = 1.0/(static_cast<double>(1u << lev));

    //Anchor of the octant
    unsigned int anchX = expandTree[i].getX();
    unsigned int anchY = expandTree[i].getY();
    unsigned int anchZ = expandTree[i].getZ();

    double aOx =  hOctFac*(static_cast<double>(anchX));
    double aOy =  hOctFac*(static_cast<double>(anchY));
    double aOz =  hOctFac*(static_cast<double>(anchZ));

    bool fgtContainsOct = true;
    if(hCurrOct > hFgt) {
      fgtContainsOct = false;
    }

    double ptGridOff, ptGridH;
    unsigned int tmpPtGridSize;

    if(fgtContainsOct) {
      tmpPtGridSize = ptGridSizeWithinBox;
      ptGridH = 0.8*hCurrOct/(static_cast<double>(tmpPtGridSize) - 1.0);
      ptGridOff = 0.1*hCurrOct;
    } else {
      tmpPtGridSize = ptGridSizeWithinBox/numFgtPerDim;
      ptGridH = 0.8*hRg/(static_cast<double>(tmpPtGridSize) - 1.0);
      ptGridOff = 0.1*hRg;
    }

    unsigned int fgtStXid = static_cast<unsigned int>(floor(aOx/hRg));
    unsigned int fgtStYid = static_cast<unsigned int>(floor(aOy/hRg));
    unsigned int fgtStZid = static_cast<unsigned int>(floor(aOz/hRg));

    for(unsigned int fgtzid = fgtStZid; fgtzid < (fgtStZid + numFgtPerDim); fgtzid++) {
      for(unsigned int fgtyid = fgtStYid; fgtyid < (fgtStYid + numFgtPerDim); fgtyid++) {
        for(unsigned int fgtxid = fgtStXid; fgtxid < (fgtStXid + numFgtPerDim); fgtxid++) {

          unsigned int fgtId = ( (fgtzid*Ne*Ne) + (fgtyid*Ne) + fgtxid );

          oct2fgtIdmap[i].push_back(fgtId);

          //Anchor of the FGT box
          double aFx = hRg*static_cast<double>(fgtxid);
          double aFy = hRg*static_cast<double>(fgtyid);
          double aFz = hRg*static_cast<double>(fgtzid);

          //Center of the FGT box
          double cx =  aFx + halfH;
          double cy =  aFy + halfH;
          double cz =  aFz + halfH;

          //Anchor for the points
          double aPx, aPy, aPz;

          if(fgtContainsOct) {
            aPx = aOx;
            aPy = aOy;
            aPz = aOz;
          } else {
            aPx = aFx;
            aPy = aFy;
            aPz = aFz;
          }

          std::vector<double> octWvals(Ndofs);

          for(int k3 = -P, di = 0; k3 < P; k3++) {
            for(int k2 = -P; k2 < P; k2++) {

              for(int k1 = -P; k1 < 1; k1++, di++) {
                octWvals[2*di] = 0.0;
                octWvals[(2*di) + 1] = 0.0;

                for(int j3 = 0; j3 < tmpPtGridSize; j3++) {
                  double pz = aPz + ptGridOff + (ptGridH*(static_cast<double>(j3)));

                  double thetaZ = ImExpZfactor*(static_cast<double>(k3)*(cz - pz));

                  for(int j2 = 0; j2 < tmpPtGridSize; j2++) {
                    double py = aPy + ptGridOff + (ptGridH*(static_cast<double>(j2)));

                    double thetaY = ImExpZfactor*(static_cast<double>(k2)*(cy - py));

                    for(int j1 = 0; j1 < tmpPtGridSize; j1++) {
                      double px = aPx + ptGridOff + (ptGridH*(static_cast<double>(j1)));

                      double thetaX = ImExpZfactor*(static_cast<double>(k1)*(cx - px));

                      double theta = (thetaX + thetaY + thetaZ);

                      //Replace fMag by drand48() if you want
                      octWvals[2*di] += (fMag*cos(theta));
                      octWvals[(2*di) + 1] += (fMag*sin(theta));

                    }//end for j1
                  }//end for j2
                }//end for j3

              }//end for k1

              for(int k1 = 1; k1 < P; k1++, di++) {
                octWvals[2*di] = octWvals[2*(di - (2*k1))];
                octWvals[(2*di) + 1] = -octWvals[(2*(di - (2*k1))) + 1];
              }//end for k1

            }//end for k2
          }//end for k3

          unsigned int foundIdx;
          bool foundIt = seq::maxLowerBound<unsigned int>(uniqueOct2fgtIdmap, fgtId, foundIdx, 0, 0);

          if(foundIt) {
            if( uniqueOct2fgtIdmap[foundIdx] != fgtId ) {
              uniqueOct2fgtIdmap.insert( (uniqueOct2fgtIdmap.begin() + foundIdx + 1), fgtId );
              Wfgt.insert( (Wfgt.begin() + foundIdx + 1), octWvals );
            } else {
              for(int li = 0; li < Ndofs; li++) {
                Wfgt[foundIdx][li] += octWvals[li];
              }//end for li
            }
          } else {
            uniqueOct2fgtIdmap.insert( uniqueOct2fgtIdmap.begin(), fgtId );
            Wfgt.insert( Wfgt.begin(), octWvals );
          }

        }//end for fgtxid
      }//end for fgtyid
    }//end for fgtzid

  }//end for i

  for(unsigned int i = 0; i < numLocalExpandOcts; i++) {
    for(unsigned int j = 0; j < oct2fgtIdmap[i].size(); j++) {
      unsigned int fgtId = oct2fgtIdmap[i][j];
      unsigned int fgtIndex;

      bool foundIt = seq::BinarySearch<unsigned int>( (&(*(uniqueOct2fgtIdmap.begin()))),
          uniqueOct2fgtIdmap.size(), fgtId, &fgtIndex);

      oct2fgtIdmap[i][j] = fgtIndex;
    }//end for j
  }//end for i

  PetscLogEventEnd(s2wEvent, 0, 0, 0, 0);
}

void pfgtHybridDirect(std::vector<double> & directSources, std::vector<ot::TreeNode> & directTree, 
    const unsigned int FgtLev, MPI_Comm subComm, MPI_Comm comm) {
  PetscLogEventBegin(directHybridEvent, 0, 0, 0, 0);

  assert(!(directTree.empty()));

  std::vector<ot::TreeNode> directMins;
  computeMins(directMins, directTree, subComm);

  std::vector<ot::TreeNode> fgtMins;
  computeFGTminsHybridDirect(fgtMins, comm);

  PetscLogEventEnd(directHybridEvent, 0, 0, 0, 0);
}

void computeMins(std::vector<ot::TreeNode> & mins, std::vector<ot::TreeNode> & subTree, MPI_Comm subComm) {
  int subNpes;
  MPI_Comm_size(subComm, &subNpes);

  mins.resize(subNpes);

  MPI_Allgather(&(subTree[0]), 1, par::Mpi_datatype<ot::TreeNode>::value(),
      &(mins[0]), 1, par::Mpi_datatype<ot::TreeNode>::value(), subComm);
}

void computeFGTminsHybridExpand(std::vector<ot::TreeNode> & fgtMins, std::vector<ot::TreeNode> & fgtList,
    MPI_Comm subComm, MPI_Comm comm) {
  int subNpes;
  MPI_Comm_size(subComm, &subNpes);

  int subRank;
  MPI_Comm_rank(subComm, &subRank);

  int rank;
  MPI_Comm_rank(comm, &rank);

  assert(rank == subRank);

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
    for(int i = 0; i < fgtMins.size(); ++i) {
      if(fgtMins[i].getDim()) {
        fgtMins[i].setWeight(i);
        tmpMins.push_back(fgtMins[i]);
      }
    }//end i
    fgtMins = tmpMins;
    fgtMinSize = fgtMins.size();
  }

  MPI_Bcast(&fgtMinSize, 1, MPI_INT, 0, comm);

  if(rank) {
    fgtMins.resize(fgtMinSize);
  }

  MPI_Bcast(&(fgtMins[0]), fgtMinSize, par::Mpi_datatype<ot::TreeNode>::value(), 0, comm);
}

void computeFGTminsHybridDirect(std::vector<ot::TreeNode> & fgtMins, MPI_Comm comm) {
  int fgtMinSize;
  MPI_Bcast(&fgtMinSize, 1, MPI_INT, 0, comm);

  fgtMins.resize(fgtMinSize);
  MPI_Bcast(&(fgtMins[0]), fgtMinSize, par::Mpi_datatype<ot::TreeNode>::value(), 0, comm);
}

void createFGToctree(std::vector<ot::TreeNode> & fgtList, std::vector<ot::TreeNode> & expandTree,
    const unsigned int FgtLev, MPI_Comm subComm) {
  PetscLogEventBegin(fgtOctConEvent, 0, 0, 0, 0);

  std::vector<ot::TreeNode> tmpFgtListA;
  std::vector<ot::TreeNode> tmpFgtListB;
  for(size_t i = 0; i < expandTree.size(); ++i) {
    unsigned int lev = expandTree[i].getLevel();
    if(lev > FgtLev) {
      tmpFgtListA.push_back(expandTree[i].getAncestor(FgtLev));
    } else {
      tmpFgtListB.push_back(expandTree[i]);
    }
  }//end i

  seq::makeVectorUnique<ot::TreeNode>(tmpFgtListA, true);

  int aIdx = 0;
  int bIdx = 0;
  while( (aIdx < tmpFgtListA.size()) && (bIdx < tmpFgtListB.size()) ) {
    if(tmpFgtListA[aIdx] < tmpFgtListB[bIdx]) {
      fgtList.push_back(tmpFgtListA[aIdx]);
      ++aIdx;
    } else {
      fgtList.push_back(tmpFgtListB[bIdx]);
      ++bIdx;
    }
  }
  for(; aIdx < tmpFgtListA.size(); ++aIdx) {
    fgtList.push_back(tmpFgtListA[aIdx]);
  }
  for(; bIdx < tmpFgtListB.size(); ++bIdx) {
    fgtList.push_back(tmpFgtListB[bIdx]);
  }

  assert(!(fgtList.empty()));

  int rank, npes;
  MPI_Comm_rank(subComm, &rank);
  MPI_Comm_size(subComm, &npes);

  ot::TreeNode tmpOct;

  MPI_Request recvRequest, sendRequest;
  if(rank > 0) {
    MPI_Irecv(&tmpOct, 1, par::Mpi_datatype<ot::TreeNode>::value(),
        (rank - 1), 1, subComm, &recvRequest);
  }

  if(rank < (npes - 1)) {
    MPI_Isend(&(fgtList[fgtList.size() - 1]), 1, par::Mpi_datatype<ot::TreeNode>::value(),
        (rank + 1), 1, subComm, &sendRequest);
  }

  if(rank > 0) {
    MPI_Status status;
    MPI_Wait(&recvRequest, &status);
  }

  if(rank < (npes - 1)) {
    MPI_Status status;
    MPI_Wait(&sendRequest, &status);
  }

  if(rank > 0) {
    if(tmpOct == fgtList[0]) {
      fgtList.erase(fgtList.begin());
    }
  }

  PetscLogEventEnd(fgtOctConEvent, 0, 0, 0, 0);
}

void alignSources(std::vector<double> & sources, std::vector<ot::TreeNode> & linOct, MPI_Comm comm) {
  assert(!(linOct.empty()));

  int npes;
  MPI_Comm_size(comm, &npes);

  int numPts = ((sources.size())/4);

  std::vector<ot::TreeNode> mins(npes);
  MPI_Allgather(&(linOct[0]), 1, par::Mpi_datatype<ot::TreeNode>::value(),
      &(mins[0]), 1, par::Mpi_datatype<ot::TreeNode>::value(), comm);

  int* sendCnts = new int[npes];

  for(int i = 0; i < npes; ++i) {
    sendCnts[i] = 0;
  }//end i

  int minsCnt = 0;
  for(int i = 0; i < numPts; ++i) {
    unsigned int px = (unsigned int)(sources[4*i]*(double)(1u << __MAX_DEPTH__));
    unsigned int py = (unsigned int)(sources[(4*i)+1]*(double)(1u << __MAX_DEPTH__));
    unsigned int pz = (unsigned int)(sources[(4*i)+2]*(double)(1u << __MAX_DEPTH__));
    ot::TreeNode tmpOct(px, py, pz, __MAX_DEPTH__, __DIM__, __MAX_DEPTH__);
    while((minsCnt < npes) && (mins[minsCnt] <= tmpOct)) {
      minsCnt++;
    }
    minsCnt--;
    sendCnts[minsCnt] += 4;
  }//end i

  int* recvCnts = new int[npes];

  MPI_Alltoall(sendCnts, 1, MPI_INT, recvCnts, 1, MPI_INT, comm);

  int* sendDisps = new int[npes];
  int* recvDisps = new int[npes];

  sendDisps[0] = 0;
  recvDisps[0] = 0;
  for(int i = 1; i < npes; ++i) {
    sendDisps[i] = sendDisps[i - 1] + sendCnts[i - 1];
    recvDisps[i] = recvDisps[i - 1] + recvCnts[i - 1];
  }//end i

  std::vector<double> recvSources(recvDisps[npes - 1] + recvCnts[npes - 1]);

  MPI_Alltoallv(&(sources[0]), sendCnts, sendDisps, MPI_DOUBLE, 
      &(recvSources[0]), recvCnts, recvDisps, MPI_DOUBLE, comm);

  swap(sources, recvSources);

  delete [] sendCnts;
  delete [] recvCnts;

  delete [] sendDisps;
  delete [] recvDisps;
}

void pfgtOnlyDirect(std::vector<double> & directSources, std::vector<ot::TreeNode> & directTree, MPI_Comm comm) {
  PetscLogEventBegin(directOnlyEvent, 0, 0, 0, 0);

  //Not Implemented
  assert(false);

  PetscLogEventEnd(directOnlyEvent, 0, 0, 0, 0);
}

void pfgtOnlyExpand(std::vector<double> & expandSources, std::vector<ot::TreeNode> & expandTree, 
    const unsigned int FgtLev, MPI_Comm comm) {
  PetscLogEventBegin(expandOnlyEvent, 0, 0, 0, 0);

  //Not Implemented
  assert(false);

  PetscLogEventEnd(expandOnlyEvent, 0, 0, 0, 0);
}

void pfgtSerial(std::vector<double> & directSources, std::vector<double> & expandSources,
    std::vector<ot::TreeNode> & directTree, std::vector<ot::TreeNode> & expandTree, const unsigned int FgtLev) {
  PetscLogEventBegin(serialEvent, 0, 0, 0, 0);

  //Not Implemented
  assert(false);

  PetscLogEventEnd(serialEvent, 0, 0, 0, 0);
}

void s2wComm() {
}


