
#include <cmath>
#include <vector>
#include <cassert>

/*
   computes the far field hermite expansion
   about the center cent due to the sources.

   cent(1),cent(2),cent(3) = center of the expansion
   sources - x,y,z, strength

   ninbox = number of sources
   delta  = gaussian width
   H = number of terms in expansion

output:
hexp[ (k*(H+1) + j)*(H+1) + i] = (i,j,k)th coefficient of far field expansion
due to sources
*/
void hermite_exp (double* cent, std::vector<double> &sources, unsigned int ninbox,
    unsigned int FgtLev, unsigned int H, std::vector<double> &hexp) {
  //Fgt box size = sqrt(delta)
  double dsq = (static_cast<double>(1u << FgtLev));

  std::vector<double> xp(H+1);
  std::vector<double> yp(H+1);
  std::vector<double> zp(H+1);

  assert((hexp.size()) == ((H + 1)*(H + 1)*(H + 1)));
  for (int i = 0; i < hexp.size(); ++i) {
    hexp[i] = 0.0;
  }//end i

  //! accumulate expansion due to each source.
  for(int i = 0; i < ninbox; ++i) {
    double x = (sources[4*i]   - cent[0])*dsq;
    double y = (sources[(4*i) + 1] - cent[1])*dsq;
    double z = (sources[(4*i) + 2] - cent[2])*dsq;
    double fy = sources[(4*i) + 3];
    //! compute the powers          
    xp[0] = 1.0;
    yp[0] = 1.0;
    zp[0] = 1.0;
    for(int k = 1; k <= H; ++k) {
      xp[k] = xp[k-1]*x/k;
      yp[k] = yp[k-1]*y/k;
      zp[k] = zp[k-1]*z/k;
    }
    for(int v = 0, di = 0; v <= H; ++v) {
      for(int j = 0; j <= H; ++j) {
        for(int k = 0; k <= H; ++k, ++di) {
          double prod = xp[k]*yp[j]*zp[v]; 
          hexp[di] += (prod * fy); 
        }//end k
      }//end j
    }//end v
  }//end i
}


/****************************************************************c
  converts hermite expansion into plane wave expansion. 
  zmul are pre-computed convertion factors ... can be generated using compute_conv_coeff()
  */
void hermite_to_pwave ( std::vector<double> &herm_exp, std::vector<double> &pw_exp, const int H,
    const int P, const double delta, std::vector<double> &zmul) {

  // note on sizes ... consider in relation to temporary work vectors 
  // herm_exp = (H+1)*(H+1)*(H+1)
  // pw_exp = (P+1 * 2P+1 * 2P+1)*2 - symmetric in z  
  assert((pw_exp.size()) == (2*(P + 1)*((2*P) + 1)*((2*P) + 1)));
  assert((herm_exp.size()) == ((H + 1)*(H + 1)*(H + 1)));

  // temporary work vectors 
  std::vector<double> ftemp(2*(P + 1)*(H + 1)*(H + 1));  
  std::vector<double> fftemp(2*(P + 1)*((2*P) + 1)*(H + 1)); 

  double zfac_re, zfac_im, mul;

  for (int k3 = 0, di = 0; k3 <= P; ++k3) {
    for (int j2 = 0; j2i <= H; ++j2) {
      for (int j1 = 0; j1 <= H; ++j1, di += 2) {

        ftemp[di] = 0.0;
        ftemp[di+1] = 0.0;
        // loop over j3 index               
        mul = 1.0;
        for (int j3 = 0; j3 <= H; ++j3) { 
          int dh = 2*((j3*(H+1) + j2)*(H+1)+j1);
          zfac_re = mul * zmul [(j3 + k3*(H+1))*2];
          zfac_im = mul * zmul [(j3 + k3*(H+1))*2 + 1];

          ftemp[di]   += zfac_re * herm_exp[dh];
          ftemp[di+1] += zfac_im * herm_exp[dh];

          mul = mul * (-1) ;
        } // j3
      } // j1
    } // j2
  } // k3

  for (int k3 = 0, di = 0; k3 <= P; ++k3) {
    for (int k2 = 0; k2 <= (2*P); ++k2) {
      for (int j1 = 0; j1 <= H; ++j1, di += 2) {
        fftemp[di] = 0.0;
        fftemp[di+1] = 0.0;
        // loop over j2 index               
        mul = 1.0;
        for (int j2=0; j2 <= H; ++j2) {
          int dh = 2*((k3*(H+1) + j2)*(H+1)+j1);
          zfac_re = mul * zmul [(j2 + k2*(H+1))*2];
          zfac_im = mul * zmul [(j2 + k2*(H+1))*2 + 1];

          fftemp[di]   += __COMP_MUL_RE(zfac_re, zfac_im, ftemp[dh], ftemp[dh+1]);
          fftemp[di+1] += __COMP_MUL_IM(zfac_re, zfac_im, ftemp[dh], ftemp[dh+1]);

          mul = mul * (-1) ;
        } // j2

      } // j1
    } // k2
  } // k3

  for (int k3 = 0, di = 0; k3 <= P; ++k3) {
    for (int k2 = 0; k2 <= (2*P); ++k2) {
      for (int k1 = 0; k1 <= (2*P); ++k1, di += 2) {
        pw_exp[di] = 0.0;
        pw_exp[di+1] = 0.0;
        // loop over j1 index               
        mul = 1.0;
        for (int j1=0; j1<=H; ++j1) {
          int dh = 2*((k3*(2*P+1) + k2)*(H+1)+j1);
          zfac_re = mul * zmul [(j1 + k1*(H+1))*2];
          zfac_im = mul * zmul [(j1 + k1*(H+1))*2 + 1];
          pw_exp[di]   += __COMP_MUL_RE(zfac_re, zfac_im, fftemp[dh], fftemp[dh+1]);
          pw_exp[di+1] += __COMP_MUL_IM(zfac_re, zfac_im, fftemp[dh], fftemp[dh+1]);

          mul = mul * (-1) ;
        } // j1
      } // k1
    } // k2
  } // k3
}

/* 
 * precompute the conversion factors from wave expansions to taylor exps
 *
 * P = number of terms in plane wave expansion
 * H = number of terms in the hermite expansion
 * L = interval of frequences for fourier integral
 *
 * zmul = double[ (2P+1) * (H+1)  *   2     ] -- stored as a 1d array
 *                                 complex
 */ 
void compute_conv_coeff(const int P, const double L, const int H, std::vector<double>& zmul) { 
  double rk; 
  double zfac_re, zfac_im;

  zmul.resize((2*P+1)*(H+1)*2);

  for (int k=-P, di=0; k<=P; ++k) {  
    rk = L * k; 
    rk /= P; 
    zfac_re = 1.0;
    zfac_im = 0.0;
    for (int j=0; j<=H; ++j, di+=2) {
      zmul[di] = zfac_re;
      zmul[di+1] = zfac_im;
      zfac_re = -1.0 * zfac_im*rk;
      zfac_im = zfac_re * rk;
    }
  }
}


