
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

  int ptGridSizeWithinBox, nx, ny, nz, xs, ys, zs;

  fscanf(fp, "%d", &xs);
  fscanf(fp, "%d", &ys);
  fscanf(fp, "%d", &zs);
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
        double ax =  h*(static_cast<double>(xi + xs));
        double ay =  h*(static_cast<double>(yi + ys));
        double az =  h*(static_cast<double>(zi + zs));

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


  //Free some memory

  px.clear();
  py.clear();
  pz.clear();
  sources.clear();

  //Read FGT results

  std::vector<double> fgtResults (trueLocalNumPts);

  // Compute Error
  double maxErr = 0;

  std::cout<<"Max Error = "<<maxErr<<std::endl;


}



