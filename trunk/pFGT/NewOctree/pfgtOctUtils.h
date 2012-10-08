
#ifndef __PFGT_OCT_UTILS__
#define __PFGT_OCT_UTILS__

#include "petsc.h"
#include "oct/TreeNode.h"
#include <vector>

#define __PI__ 3.1415926535897932

#define __SQRT_PI__ 1.7724538509055160

#define __DIM__ 3

#define __MAX_DEPTH__ 30

#define __ITPMD__  (1ull << __MAX_DEPTH__)

#define __DTPMD__  (static_cast<double>(__ITPMD__))

#define __COMP_MUL_RE(a, ai, b, bi) ( ((a)*(b)) - ((ai)*(bi)) )
#define __COMP_MUL_IM(a, ai, b, bi) ( ((a)*(bi)) + ((ai)*(b)) )

void pfgtMain(std::vector<double>& sources, const unsigned int minPtsInFgt, const unsigned int FgtLev,
    const int P, const int L, const int K, const double epsilon, MPI_Comm comm);

void pfgtSetup(std::vector<double>& expandSources, std::vector<double>& directSources, std::vector<ot::TreeNode>& fgtList,
    bool & singleType, int & npesExpand, int & avgExpand, int & extraExpand, MPI_Comm & subComm,
    std::vector<double>& sources, const unsigned int minPtsInFgt, const unsigned int FgtLev, MPI_Comm comm);

void splitSources(std::vector<double>& expandSources, std::vector<double>& directSources, 
    std::vector<ot::TreeNode>& fgtList, std::vector<double>& sources, const unsigned int minPtsInFgt, 
    const unsigned int FgtLev, MPI_Comm comm);

void pfgtExpand(std::vector<double> & expandSources, std::vector<ot::TreeNode> & fgtList, const unsigned int FgtLev, 
    const int P, const int L, const int K, const int avgExpand, const int extraExpand,
    MPI_Comm subComm, MPI_Comm comm, bool singleType);

void pfgtDirect(std::vector<double> & directSources, const unsigned int FgtLev,
    const int P, const int L, const int K, const double epsilon, 
    MPI_Comm subComm, MPI_Comm comm, bool singleType);

void computeFgtMinsExpand(std::vector<ot::TreeNode> & fgtMins, std::vector<ot::TreeNode> & fgtList,
    MPI_Comm subComm, MPI_Comm comm);

void computeFgtMinsDirect(std::vector<ot::TreeNode> & fgtMins, MPI_Comm comm);

void computeRemoteFgt(ot::TreeNode & remoteFgt, int & remoteFgtOwner, const unsigned int FgtLev,
    std::vector<double> & sources, std::vector<ot::TreeNode> & fgtMins);

void createS2WcommInfo(int*& sendCnts, int*& sendDisps, int*& recvCnts, int*& recvDisps, 
    const int remoteFgtOwner, const unsigned int numWcoeffs, const int excessWt,
    const int avgExpand, const int extraExpand, MPI_Comm subComm);

void destroyS2WcommInfo(int* sendCnts, int* sendDisps, int* recvCnts, int* recvDisps);

void s2w(std::vector<double> & localWlist, std::vector<double> & sources,  
    const ot::TreeNode remoteFgt, const int remoteFgtOwner, const int numPtsInRemoteFgt,
    std::vector<ot::TreeNode> & fgtList,  std::vector<ot::TreeNode> & fgtMins,
    const unsigned int FgtLev, const int P, const int L,
    int* sendCnts, int* sendDisps, int* recvCnts, int* recvDisps, MPI_Comm subComm);

void l2t(std::vector<double> & results, std::vector<double> & localLlist, std::vector<double> & sources,
    const ot::TreeNode remoteFgt, const int remoteFgtOwner, const int numPtsInRemoteFgt,
    std::vector<ot::TreeNode> & fgtList, std::vector<ot::TreeNode> & fgtMins,
    const unsigned int FgtLev, const int P, const int L,
    int* sendCnts, int* sendDisps, int* recvCnts, int* recvDisps, MPI_Comm subComm);

void w2l(std::vector<double> & localLlist, std::vector<double> & localWlist, 
    std::vector<ot::TreeNode> & fgtList, std::vector<ot::TreeNode> & fgtMins,
    const unsigned int FgtLev, const int P, const int L, const unsigned long long int K, MPI_Comm subComm);

void d2d(std::vector<double> & results, std::vector<double> & sources,
    std::vector<ot::TreeNode> & nodes, std::vector<ot::TreeNode> & directMins,
    const unsigned int FgtLev, const double epsilon, MPI_Comm subComm);

void w2dAndD2lExpand(std::vector<double> & localLlist, std::vector<double> & localWlist, 
    std::vector<ot::TreeNode> & fgtList, const int P, MPI_Comm comm);

void w2dAndD2lDirect(std::vector<double> & results, std::vector<double> & sources,
    std::vector<ot::TreeNode> & fgtMins, const unsigned int FgtLev, 
    const int P, const int L, const int K, const double epsilon, MPI_Comm comm);

#endif


