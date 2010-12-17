
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <iostream>

int main(int argc, char ** argv ) {	

  if(argc < 2) {
    std::cout<<"Usage: exe delta"<<std:endl;
    exit(1);
  }

  double delta = atof(argv[1]);
  const double h = sqrt(delta);

  FILE* fp = fopen("inpType2_0_1.txt", "r");

  int ptGridSizeWithinBox, nx, ny, nz;

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

  const double ptGridOff = 0.1*h;
  const double ptGridH = 0.8*h/(static_cast<double>(ptGridSizeWithinBox) - 1.0);


}



