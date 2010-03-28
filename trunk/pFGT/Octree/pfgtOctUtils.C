
#include "mpi.h"
#include "petscda.h"
#include "pfgtOctUtils.h"
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <iostream>

extern PetscLogEvent fgtEvent;
extern PetscLogEvent s2wEvent;
extern PetscLogEvent w2lEvent;
extern PetscLogEvent l2tEvent;

#define __PI__ 3.14159265

PetscErrorCode pfgt(std::vector<ot::TreeNode> & linOct, unsigned int maxDepth,
    double delta, double fMag, unsigned int numPtsPerProc, 
    int P, int L, int K, int writeOut)
{
  PetscFunctionBegin;

  PetscLogEventBegin(fgtEvent, 0, 0, 0, 0);

  MPI_Comm comm = MPI_COMM_WORLD;

  int npes, rank;
  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &rank);

  const double hRg = sqrt(delta);

  //Create DA for FGT boxes

  const unsigned int Ne = static_cast<unsigned int>(1.0/hRg);

  if(!rank) {
    std::cout<<"Ne = "<< Ne <<std::endl;
  }

  //2p complex coefficients for each dimension.  
  const unsigned int Ndofs = 16*P*P*P;

  if(!rank) {
    std::cout<<"Ndofs = "<<Ndofs<<std::endl;
  }

  if(!rank) {
    std::cout<<"StencilWidth = "<< K <<std::endl;
  }

  DA da;
  DACreate3d(comm, DA_NONPERIODIC, DA_STENCIL_BOX, Ne, Ne, Ne,
      PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, Ndofs, K,
      PETSC_NULL, PETSC_NULL, PETSC_NULL, &da);

  PetscInt xs, ys, zs, nx, ny, nz;
  DAGetCorners(da, &xs, &ys, &zs, &nx, &ny, &nz);

  //Split octree into 2 sets
  std::vector<ot::TreeNode> expandTree;
  std::vector<ot::TreeNode> directTree;

  const unsigned int numLocalOcts = linOct.size();

  const double hOctFac = 1.0/static_cast<double>(1u << maxDepth);

  for(unsigned int i = 0; i < numLocalOcts; i++) {
    unsigned int lev = linOct[i].getLevel();
    double hCurrOct = hOctFac*static_cast<double>(1u << (maxDepth - lev));

    if(hCurrOct <= hRg) {
      expandTree.push_back(linOct[i]);
    } else {
      directTree.push_back(linOct[i]);
    }
  }//end for i
  linOct.clear();

  const unsigned int numLocalExpandOcts = expandTree.size();
  const unsigned int numLocalDirectOcts = directTree.size();

  //Tensor-Product grid on each octant

  //Maximum value
  unsigned int numPtsPerBox = numPtsPerProc/numLocalOcts;

  //Tensor-Product Grid
  const unsigned int ptGridSizeWithinBox = static_cast<unsigned int>(floor(pow( numPtsPerBox, (1.0/3.0) )));
  long long trueLocalNumPts = ptGridSizeWithinBox*ptGridSizeWithinBox*ptGridSizeWithinBox*numLocalOcts;

  PetscLogEventBegin(s2wEvent, 0, 0, 0, 0);

  //Loop over local expand octants and execute S2W in each octant

  const double lambda = static_cast<double>(L)/(static_cast<double>(P)*sqrt(delta));

  std::vector<std::vector<std::vector<double> > > tmp1R;
  std::vector<std::vector<std::vector<double> > > tmp1C;

  std::vector<std::vector<std::vector<double> > > tmp2R;
  std::vector<std::vector<std::vector<double> > > tmp2C;

  for(unsigned int i = 0; i < numLocalExpandOcts; i++) {
    unsigned int lev = expandTree[i].getLevel();
    double hCurrOct = hOctFac*static_cast<double>(1u << (maxDepth - lev));

    double ptGridOff = 0.1*hCurrOct;
    double ptGridH = 0.8*hCurrOct/(static_cast<double>(ptGridSizeWithinBox) - 1.0);

    //Anchor of the octant
    unsigned int anchX = expandTree[i].getX();
    unsigned int anchY = expandTree[i].getY();
    unsigned int anchZ = expandTree[i].getZ();

    double aOx =  hOctFac*(static_cast<double>(anchX));
    double aOy =  hOctFac*(static_cast<double>(anchY));
    double aOz =  hOctFac*(static_cast<double>(anchZ));

    //Anchor of the FGT box
    double aFx = hRg*floor(aOx/hRg);
    double aFy = hRg*floor(aOy/hRg);
    double aFz = hRg*floor(aOz/hRg);

    //Center of the FGT box
    double halfH = (0.5*hRg);
    double cx =  aFx + halfH;
    double cy =  aFy + halfH;
    double cz =  aFz + halfH;

    //Tensor-Product Acceleration 


  }//end for i

  PetscLogEventEnd(s2wEvent, 0, 0, 0, 0);

  PetscLogEventBegin(w2lEvent, 0, 0, 0, 0);

  PetscLogEventEnd(w2lEvent, 0, 0, 0, 0);

  //Loop over local octants and execute L2T in each octant

  PetscLogEventBegin(l2tEvent, 0, 0, 0, 0);

  const double C0 = ( pow((0.5/sqrt(__PI__)), 3.0)*
      pow((static_cast<double>(L)/static_cast<double>(P)), 3.0) );

  std::vector<std::vector<double> > expandResults(numLocalExpandOcts);
  std::vector<std::vector<double> > directResults(numLocalDirectOcts);

  PetscLogEventEnd(l2tEvent, 0, 0, 0, 0);

  if(writeOut) {
    char fname[256];
    sprintf(fname, "OctExpandOutType2_%d_%d.txt", rank, npes);
    FILE* fp = fopen(fname, "w");
    fprintf(fp, "%d\n", (expandResults.size()));
    for(unsigned int i = 0; i < expandResults.size(); i++) {
      fprintf(fp, "%d\n", expandResults[i].size());
      for(unsigned int j = 0; j < expandResults[i].size(); j++) {
        fprintf(fp, "%lf \n", expandResults[i][j]);
      }//end for j
    }//end for i
    fclose(fp);
  }

  if(writeOut) {
    char fname[256];
    sprintf(fname, "OctDirectOutType2_%d_%d.txt", rank, npes);
    FILE* fp = fopen(fname, "w");
    fprintf(fp, "%d\n", (directResults.size()));
    for(unsigned int i = 0; i < directResults.size(); i++) {
      fprintf(fp, "%d\n", directResults[i].size());
      for(unsigned int j = 0; j < directResults[i].size(); j++) {
        fprintf(fp, "%lf \n", directResults[i][j]);
      }//end for j
    }//end for i
    fclose(fp);
  }

  long long trueTotalPts;
  MPI_Reduce(&trueLocalNumPts, &trueTotalPts, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, comm);

  if(!rank) {
    std::cout<<"True Total NumPts: "<<trueTotalPts<<std::endl; 
  }

  PetscLogEventEnd(fgtEvent, 0, 0, 0, 0);

  PetscFunctionReturn(0);

}

