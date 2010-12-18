
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <iostream>

int main(int argc, char ** argv ) {	

  if(argc < 2) {
    std::cout<<"Usage: exe delta"<<std::endl;
    exit(1);
  }

  double delta = atof(argv[1]);
  const double h = sqrt(delta);

  //Read sources

  FILE* fp = fopen("inpType2_0_1.txt", "r");

  int dummyInt;
  int ptGridSizeWithinBox, nx, ny, nz;

  fscanf(fp, "%d", &dummyInt);
  fscanf(fp, "%d", &dummyInt);
  fscanf(fp, "%d", &dummyInt);
  fscanf(fp, "%d", &nx);
  fscanf(fp, "%d", &ny);
  fscanf(fp, "%d", &nz);
  fscanf(fp, "%d", &ptGridSizeWithinBox);

  unsigned int trueLocalNumPts = ptGridSizeWithinBox*ptGridSizeWithinBox*ptGridSizeWithinBox*nx*ny*nz;

  std::vector<double> sources (trueLocalNumPts);

  for(unsigned int i = 0; i < trueLocalNumPts; i++) {
    fscanf(fp, "%lf", &(sources[i]));
  }//end for i

  fclose(fp);

  //Create points

  const double ptGridOff = 0.1*h;
  const double ptGridH = 0.8*h/(static_cast<double>(ptGridSizeWithinBox) - 1.0);

  std::vector<double> px (trueLocalNumPts);
  std::vector<double> py (trueLocalNumPts);
  std::vector<double> pz (trueLocalNumPts);

  for(int zi = 0, boxId = 0; zi < nz; zi++) {
    for(int yi = 0; yi < ny; yi++) {
      for(int xi = 0; xi < nx; xi++, boxId++) {

        unsigned int sourceOffset = (boxId*ptGridSizeWithinBox*ptGridSizeWithinBox*ptGridSizeWithinBox);

        //Anchor of the box
        double ax =  h*(static_cast<double>(xi));
        double ay =  h*(static_cast<double>(yi));
        double az =  h*(static_cast<double>(zi));

        for(int j3 = 0, ptId = 0; j3 < ptGridSizeWithinBox; j3++) {
          for(int j2 = 0; j2 < ptGridSizeWithinBox; j2++) {
            for(int j1 = 0; j1 < ptGridSizeWithinBox; j1++, ptId++) {
              px[sourceOffset + ptId] = ax + ptGridOff + (ptGridH*(static_cast<double>(j1)));
              py[sourceOffset + ptId] = ay + ptGridOff + (ptGridH*(static_cast<double>(j2)));
              pz[sourceOffset + ptId] = az + ptGridOff + (ptGridH*(static_cast<double>(j3)));
            }//end for j1
          }//end for j2
        }//end for j3

      }//end for xi
    }//end for yi
  }//end for zi

  //Direct Sum

  std::vector<double> directResults (trueLocalNumPts);

  for(int j = 0; j < trueLocalNumPts; j++) {
    directResults[j] = 0;
    for(int k = 0; k < trueLocalNumPts; k++) {
      directResults[j] += (sources[k]*exp(-( ((px[j] - px[k])*(px[j] - px[k])) 
              + ((py[j] - py[k])*(py[j] - py[k])) + ((pz[j] - pz[k])*(pz[j] - pz[k])) )/delta));
    }//end for k
  }//end for j

  //Free some memory

  px.clear();
  py.clear();
  pz.clear();
  sources.clear();

  //Read FGT results

  std::vector<double> fgtResults (trueLocalNumPts);

  fp = fopen("outType2_0_1.txt", "r");
  fscanf(fp, "%d", &dummyInt);
  for(int i = 0, k = 0; i < (nx*ny*nz); i++) {
    for(int j = 0; j < (ptGridSizeWithinBox*ptGridSizeWithinBox*ptGridSizeWithinBox); j++, k++) {
      fscanf(fp, "%lf", &(fgtResults[k]));
    }//end for j
  }//end for i
  fclose(fp);


  // Compute Error
  double maxErr = 0;

  for(int i = 0; i < trueLocalNumPts; i++) {
    double err = fabs(directResults[i] - fgtResults[i]);
    if(err > maxErr) {
      maxErr = err;
    }
  }//end for i

  std::cout<<"Max Error = "<<maxErr<<std::endl;


}



