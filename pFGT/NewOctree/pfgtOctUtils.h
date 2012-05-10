
#ifndef __PFGT_OCT_UTILS__
#define __PFGT_OCT_UTILS__

#include "petsc.h"
#include "oct/TreeNode.h"
#include <vector>

#define __PI__ 3.1415926535897932

#define __DIM__ 3

#define __MAX_DEPTH__ 30

#define __ITPMD__  (1u << __MAX_DEPTH__)

#define __DTPMD__  (static_cast<double>(__ITPMD__))

#define __COMP_MUL_RE(a, ai, b, bi) ( ((a)*(b)) - ((ai)*(bi)) )
#define __COMP_MUL_IM(a, ai, b, bi) ( ((a)*(bi)) + ((ai)*(b)) )

void pfgt(std::vector<double>& sources, const unsigned int minPtsInFgt, const unsigned int FgtLev,
    const int P, const int L, const int K, MPI_Comm comm);

void splitSources(std::vector<double>& sources, const unsigned int minPtsInFgt, 
    const unsigned int FgtLev, std::vector<double>& expandSources, std::vector<double>& directSources, 
    std::vector<ot::TreeNode>& fgtList, MPI_Comm comm);

void pfgtHybridExpand(std::vector<double> & expandSources, int numPtsInRemoteFgt, 
    std::vector<ot::TreeNode> & fgtList, const unsigned int FgtLev, const int P,
    const int L, MPI_Comm subComm, MPI_Comm comm);

void pfgtHybridDirect(std::vector<double> & directSources, const unsigned int FgtLev,
    MPI_Comm subComm, MPI_Comm comm);

void computeFgtMinsHybridExpand(std::vector<ot::TreeNode> & fgtMins, std::vector<ot::TreeNode> & fgtList,
    MPI_Comm subComm, MPI_Comm comm);

void computeFgtMinsHybridDirect(std::vector<ot::TreeNode> & fgtMins, MPI_Comm comm);

void s2w(std::vector<double> & localWlist, std::vector<double> & sources,  
    const int numPtsInRemoteFgt, std::vector<ot::TreeNode> & fgtList, 
    std::vector<ot::TreeNode> & fgtMins, const unsigned int FgtLev,
    const int P, const int L, MPI_Comm subComm);

#endif


