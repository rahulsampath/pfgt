  //Sequential W2L
  PetscScalar**** WlArr;
  DAVecGetArrayDOF(da, Wlocal, &WlArr);

  VecZeroEntries(Wglobal);
  DAVecGetArrayDOF(da, Wglobal, &WgArr);

  //Loop over local boxes and their Interaction lists and do a direct translation
  for(PetscInt zi = 0; zi < nz; zi++) {
    for(PetscInt yi = 0; yi < ny; yi++) {
      for(PetscInt xi = 0; xi < nx; xi++) {

        //Center of the box B
        double cBx =  h*(0.5 + static_cast<double>(xi + xs));
        double cBy =  h*(0.5 + static_cast<double>(yi + ys));
        double cBz =  h*(0.5 + static_cast<double>(zi + zs));

        //Bounds for Ilist of box B
        int Ixs = xi + xs - StencilWidth;
        int Ixe = xi + xs + StencilWidth;

        int Iys = yi + ys - StencilWidth;
        int Iye = yi + ys + StencilWidth;

        int Izs = zi + zs - StencilWidth;
        int Ize = zi + zs + StencilWidth;

        if(Ixs < 0) {
          Ixs = 0;
        }
        if(Ixe >= Ne) {
          Ixe = (Ne - 1);
        }

        if(Iys < 0) {
          Iys = 0;
        }
        if(Iye >= Ne) {
          Iye = (Ne - 1);
        }

        if(Izs < 0) {
          Izs = 0;
        }
        if(Ize >= Ne) {
          Ize = (Ne - 1);
        }

#ifdef __DEBUG__
        assert(Ixs >= gxs);
        assert(Iys >= gys);
        assert(Izs >= gzs);

        assert(Ixe < (gxs + gnx));
        assert(Iye < (gys + gny));
        assert(Ize < (gzs + gnz));
#endif

        //Loop over Ilist of box B
        for(int zj = Izs; zj <= Ize; zj++) {
          for(int yj = Iys; yj <= Iye; yj++) {
            for(int xj = Ixs; xj <= Ixe; xj++) {

              //Center of the box C
              double cCx =  h*(0.5 + static_cast<double>(xj));
              double cCy =  h*(0.5 + static_cast<double>(yj));
              double cCz =  h*(0.5 + static_cast<double>(zj));

              for(int k3 = -PforType2; k3 < PforType2; k3++) {
                int shiftK3 = (k3 + PforType2);

                for(int k2 = -PforType2; k2 < PforType2; k2++) {
                  int shiftK2 = (k2 + PforType2);

                  for(int k1 = -PforType2; k1 < PforType2; k1++) {
                    int shiftK1 = (k1 + PforType2);

                    int di = ( (4*shiftK3*PforType2*PforType2) +  (2*shiftK2*PforType2) + shiftK1 );

                    double theta = lambda*( (static_cast<double>(k1)*(cBx - cCx)) +
                        (static_cast<double>(k2)*(cBy - cCy)) +
                        (static_cast<double>(k3)*(cBz - cCz)) );

                    WgArr[zi + zs][yi + ys][xi + xs][2*di] += (WlArr[zj][yj][xj][2*di]*cos(theta));
                    WgArr[zi + zs][yi + ys][xi + xs][(2*di) + 1] += (WlArr[zj][yj][xj][(2*di) + 1]*sin(theta));

                  }//end for k1
                }//end for k2
              }//end for k3

            }//end for xj
          }//end for yj
        }//end for zj

      }//end for xi
    }//end for yi
  }//end for zi

  DAVecRestoreArrayDOF(da, Wlocal, &WlArr);

  PetscLogEventEnd(w2lEvent, 0, 0, 0, 0);


