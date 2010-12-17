
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <iostream>

int main(int argc, char ** argv ) {	

  if(argc < 2) {
    std::cout<<"Usage: exe delta"<<std:endl;
    exit(1);
  }

  double delta = atof(argv[1]);
  const double h = sqrt(delta);

  FILE* fp = fopen();
  int ptGridSizeWithinBox, nx, ny, nz;

  unsigned int trueLocalNumPts = ptGridSizeWithinBox*ptGridSizeWithinBox*ptGridSizeWithinBox*nx*ny*nz;

  fclose(fp);

  const double ptGridOff = 0.1*h;
  const double ptGridH = 0.8*h/(static_cast<double>(ptGridSizeWithinBox) - 1.0);


}

