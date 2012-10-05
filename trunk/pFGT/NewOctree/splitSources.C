
#include "petsc.h"
#include <vector>
#include "oct/TreeNode.h"
#include "pfgtOctUtils.h"

extern PetscLogEvent splitSourcesEvent;

void splitSources(std::vector<double>& expandSources, std::vector<double>& directSources, 
    std::vector<ot::TreeNode>& fgtList, std::vector<double>& sources, const unsigned int minPtsInFgt, 
    const unsigned int FgtLev, MPI_Comm comm) {
  PetscLogEventBegin(splitSourcesEvent, 0, 0, 0, 0);

  int numPts = ((sources.size())/4);

#ifdef DEBUG
  assert(!(sources.empty()));
  assert(fgtList.empty());
#endif
  {
    unsigned int px = static_cast<unsigned int>(sources[0]*(__DTPMD__));
    unsigned int py = static_cast<unsigned int>(sources[1]*(__DTPMD__));
    unsigned int pz = static_cast<unsigned int>(sources[2]*(__DTPMD__));
    ot::TreeNode ptOct(px, py, pz, __MAX_DEPTH__, __DIM__, __MAX_DEPTH__);
    ot::TreeNode newFgt = ptOct.getAncestor(FgtLev);
    fgtList.push_back(newFgt);
  }

  for(int i = 1; i < numPts; ++i) {
    unsigned int px = static_cast<unsigned int>(sources[4*i]*(__DTPMD__));
    unsigned int py = static_cast<unsigned int>(sources[(4*i)+1]*(__DTPMD__));
    unsigned int pz = static_cast<unsigned int>(sources[(4*i)+2]*(__DTPMD__));
    ot::TreeNode ptOct(px, py, pz, __MAX_DEPTH__, __DIM__, __MAX_DEPTH__);
    ot::TreeNode newFgt = ptOct.getAncestor(FgtLev);
    if(fgtList[fgtList.size() - 1] == newFgt) {
      fgtList[fgtList.size() - 1].addWeight(1);
    } else {
      fgtList.push_back(newFgt);
    }
  }//end for i

#ifdef DEBUG
  assert(!(fgtList.empty()));
#endif

  int rank;
  int npes;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &npes);

  int localFlag = 0;
  if( (rank > 0) && (rank < (npes - 1)) && ((fgtList.size()) == 1) ) {
    localFlag = 1;
  }

  int globalFlag;
  MPI_Allreduce(&localFlag, &globalFlag, 1, MPI_INT, MPI_SUM, comm);

  int prevRank = rank - 1;
  int nextRank = rank + 1;

  if(globalFlag > 0) {
    int gatherSendBuf = 0;
    if( (rank > 0) && (rank < (npes - 1)) && (fgtList.size() == 1) ) {
      gatherSendBuf = sources.size();
    }

    int* gatherList = new int[npes];

    MPI_Allgather((&gatherSendBuf), 1, MPI_INT, gatherList, 1, MPI_INT, comm);

    if(rank > 0) {
      while(gatherList[prevRank] > 0) {
        --prevRank;
      }//end while
    }

    if(rank < (npes - 1)) {
      while(gatherList[nextRank] > 0) {
        ++nextRank;
      }//end while
    }

    int* sendFgtCnts = new int[npes];
    int* recvFgtCnts = new int[npes];

    int* sendSourceCnts = new int[npes];
    int* recvSourceCnts = new int[npes];

    for(int i = 0; i < npes; ++i) {
      sendFgtCnts[i] = 0;
      recvFgtCnts[i] = 0;
      sendSourceCnts[i] = 0;
      recvSourceCnts[i] = 0;
    }//end i

    if(gatherSendBuf > 0) {
      sendFgtCnts[prevRank] = 1;
      sendSourceCnts[prevRank] = gatherSendBuf;
    }
    for(int i = rank + 1; i < nextRank; ++i) {
      recvFgtCnts[i] = 1;
      recvSourceCnts[i] = gatherList[i];
    }//end i

    delete [] gatherList;

    int* sendFgtDisps = new int[npes];
    int* recvFgtDisps = new int[npes];
    sendFgtDisps[0] = 0;
    recvFgtDisps[0] = 0;
    for(int i = 1; i < npes; ++i) {
      sendFgtDisps[i] = sendFgtDisps[i - 1] + sendFgtCnts[i - 1];
      recvFgtDisps[i] = recvFgtDisps[i - 1] + recvFgtCnts[i - 1];
    }//end i

    std::vector<ot::TreeNode> tmpFgtList(recvFgtDisps[npes - 1] + recvFgtCnts[npes - 1]);

    ot::TreeNode* recvFgtBuf = NULL;
    if(!(tmpFgtList.empty())) {
      recvFgtBuf = (&(tmpFgtList[0]));
    }

    MPI_Alltoallv( (&(fgtList[0])), sendFgtCnts, sendFgtDisps, par::Mpi_datatype<ot::TreeNode>::value(),
        recvFgtBuf, recvFgtCnts, recvFgtDisps, par::Mpi_datatype<ot::TreeNode>::value(), comm);

    if(gatherSendBuf > 0) {
      fgtList.clear();
    } else {
      for(int i = 0; i < tmpFgtList.size(); ++i) {
        if(tmpFgtList[i] == fgtList[fgtList.size() - 1]) {
          fgtList[fgtList.size() - 1].addWeight(tmpFgtList[i].getWeight());
        } else {
          fgtList.push_back(tmpFgtList[i]);
        }
      }//end i
    }

    delete [] sendFgtCnts;
    delete [] recvFgtCnts;
    delete [] sendFgtDisps;
    delete [] recvFgtDisps;

    int* sendSourceDisps = new int[npes];
    int* recvSourceDisps = new int[npes];
    sendSourceDisps[0] = 0;
    recvSourceDisps[0] = 0;
    for(int i = 1; i < npes; ++i) {
      sendSourceDisps[i] = sendSourceDisps[i - 1] + sendSourceCnts[i - 1];
      recvSourceDisps[i] = recvSourceDisps[i - 1] + recvSourceCnts[i - 1];
    }//end i

    std::vector<double> tmpSources(recvSourceDisps[npes - 1] + recvSourceCnts[npes - 1]);

    double* recvSourceBuf = NULL;
    if(!(tmpSources.empty())) {
      recvSourceBuf = (&(tmpSources[0]));
    }

    MPI_Alltoallv( (&(sources[0])), sendSourceCnts, sendSourceDisps, MPI_DOUBLE,
        recvSourceBuf, recvSourceCnts, recvSourceDisps, MPI_DOUBLE, comm);

    if(gatherSendBuf > 0) {
      sources.clear();
    } else {
      if(!(tmpSources.empty())) {
        sources.insert(sources.end(), tmpSources.begin(), tmpSources.end());
      }
    }

    delete [] sendSourceCnts;
    delete [] recvSourceCnts;
    delete [] sendSourceDisps;
    delete [] recvSourceDisps;
  }

  if(!(fgtList.empty())) {
    ot::TreeNode prevFgt;
    ot::TreeNode nextFgt;
    ot::TreeNode firstFgt = fgtList[0];
    ot::TreeNode lastFgt = fgtList[fgtList.size() - 1];
    MPI_Request recvPrevReq;
    MPI_Request recvNextReq;
    MPI_Request sendFirstReq;
    MPI_Request sendLastReq;
    if(rank > 0) {
      MPI_Irecv(&prevFgt, 1, par::Mpi_datatype<ot::TreeNode>::value(),
          prevRank, 1, comm, &recvPrevReq);
      MPI_Isend(&firstFgt, 1, par::Mpi_datatype<ot::TreeNode>::value(),
          prevRank, 2, comm, &sendFirstReq);
    }
    if(rank < (npes - 1)) {
      MPI_Irecv(&nextFgt, 1, par::Mpi_datatype<ot::TreeNode>::value(),
          nextRank, 2, comm, &recvNextReq);
      MPI_Isend(&lastFgt, 1, par::Mpi_datatype<ot::TreeNode>::value(),
          nextRank, 1, comm, &sendLastReq);
    }

    if(rank > 0) {
      MPI_Status status;
      MPI_Wait(&recvPrevReq, &status);
      MPI_Wait(&sendFirstReq, &status);
    }
    if(rank < (npes - 1)) {
      MPI_Status status;
      MPI_Wait(&recvNextReq, &status);
      MPI_Wait(&sendLastReq, &status);
    }

    bool removeFirst = false;
    bool addToLast = false;
    if(rank > 0) {
      if(prevFgt == firstFgt) {
        removeFirst = true;
      }
    }
    if(rank < (npes - 1)) {
      if(nextFgt == lastFgt) {
        addToLast = true;
      }
    }

    MPI_Request recvPtsReq;
    if(addToLast) {
      numPts = ((sources.size())/4);
      sources.resize(4*(numPts + (nextFgt.getWeight())));
      fgtList[fgtList.size() - 1].addWeight(nextFgt.getWeight());
      MPI_Irecv((&(sources[4*numPts])), (4*(nextFgt.getWeight())), MPI_DOUBLE, nextRank,
          3, comm, &recvPtsReq);
    }
    if(removeFirst) {
      MPI_Send((&(sources[0])), (4*(firstFgt.getWeight())), MPI_DOUBLE, prevRank, 3, comm);
      fgtList.erase(fgtList.begin());
    }
    if(addToLast) {
      MPI_Status status;
      MPI_Wait(&recvPtsReq, &status);
    }
    if(removeFirst) {
      sources.erase(sources.begin(), sources.begin() + (4*(firstFgt.getWeight())));
    }
  } 

#ifdef DEBUG
  assert(expandSources.empty());
  assert(directSources.empty());
#endif
  std::vector<ot::TreeNode> dummyList;
  int sourceIdx = 0;
  for(size_t i = 0; i < fgtList.size(); ++i) {
    if((fgtList[i].getWeight()) < minPtsInFgt) {
      directSources.insert(directSources.end(), (sources.begin() + sourceIdx),
          (sources.begin() + sourceIdx + (4*(fgtList[i].getWeight()))));
    } else {
      dummyList.push_back(fgtList[i]);
      expandSources.insert(expandSources.end(), (sources.begin() + sourceIdx), 
          (sources.begin() + sourceIdx + (4*(fgtList[i].getWeight()))));
    }
    sourceIdx += (4*(fgtList[i].getWeight()));
  }//end i
  swap(dummyList, fgtList);
#ifdef DEBUG
  assert((sources.size()) == ((directSources.size()) + (expandSources.size())));
#endif

  PetscLogEventEnd(splitSourcesEvent, 0, 0, 0, 0);
}



