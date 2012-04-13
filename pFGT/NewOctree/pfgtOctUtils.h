
#ifndef __PFGT_OCT_UTILS__
#define __PFGT_OCT_UTILS__

#include "petsc.h"
#include "TreeNode.h"
#include <vector>

#define __PI__ 3.1415926535897932

#define __DIM__ 3

#define __MAX_DEPTH__ 30

#define __COMP_MUL_RE(a, ai, b, bi) ( ((a)*(b)) - ((ai)*(bi)) )
#define __COMP_MUL_IM(a, ai, b, bi) ( ((a)*(bi)) + ((ai)*(b)) )

void alignSources(std::vector<double> & sources, std::vector<ot::TreeNode> & linOct, MPI_Comm comm);

void pfgt(std::vector<ot::TreeNode> & linOct, const unsigned int FgtLev, std::vector<double> & sources,
    const int P, const int L, const int K, const double DirectHfactor, MPI_Comm comm);

void pfgtOnlyDirect(std::vector<double> & directSources, std::vector<ot::TreeNode> & directTree, MPI_Comm comm);

void pfgtOnlyExpand(std::vector<double> & expandSources, std::vector<ot::TreeNode> & expandTree, 
    const unsigned int FgtLev, MPI_Comm comm);

void pfgtSerial(std::vector<double> & directSources, std::vector<double> & expandSources,
    std::vector<ot::TreeNode> & directTree, std::vector<ot::TreeNode> & expandTree, const unsigned int FgtLev);

void pfgtHybridExpand(std::vector<double> & expandSources, std::vector<ot::TreeNode> & expandTree, 
    const unsigned int FgtLev, const double delta, const double hFgt, MPI_Comm subComm, MPI_Comm comm);

void pfgtHybridDirect(std::vector<double> & directSources, std::vector<ot::TreeNode> & directTree,
    const unsigned int FgtLev, MPI_Comm subComm, MPI_Comm comm);

void createFGToctree(std::vector<ot::TreeNode> & fgtList, std::vector<ot::TreeNode> & expandTree,
    const unsigned int FgtLev, MPI_Comm subComm);

void computeFGTminsHybridExpand(std::vector<ot::TreeNode> & fgtMins, std::vector<ot::TreeNode> & fgtList,
    MPI_Comm subComm, MPI_Comm comm);

void computeFGTminsHybridDirect(std::vector<ot::TreeNode> & fgtMins, MPI_Comm comm);

void computeMins(std::vector<ot::TreeNode> & mins, std::vector<ot::TreeNode> & subTree, MPI_Comm subComm);

#endif


