

#include "bremss_fits.h"

#include "decs.h"

// For planck_func
#include "maxwell_juettner.h"

#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>
#include <gsl/gsl_sf_expint.h>

double gffee(double Te, double nu);
double gffei(double Te, double nu);

#define THETAE_MIN 1.e-3

/* Emissivities */
double bremss_I(FitParams *params, int bremss_type)
{
  if (params->theta_e < THETAE_MIN) 
    return 0.;

  double Thetae = params->theta_e;
  double Te = Thetae * ME * CL * CL / KBOL;
  double nu = params->nu;
  double x = HPL * nu / (KBOL * Te);
  double Ne = params->electron_density;
  double jv;
  
  double efac = 0.;
  if (x < 1.e-3) {
    efac = (24. - 24.*x + 12.*x*x - 4.*x*x*x + x*x*x*x) / 24.;
  } else {
    efac = exp(-x);
  }

  // Bremss options
  // 0 No bremsstrahlung
  // 1 Rybicki and Lightman eq. 5.14a with eq. 5.25 corrective factor
  // 2 Piecewise formula
  // 3 van Hoof 2015 + Nozawa 2009
  // Bremss is only supported for thermal electrons
  if (bremss_type == 1) {
    // Method from Rybicki & Lightman, ultimately from Novikov & Thorne

    double rel = (1. + 4.4e-10*Te);
    double gff = 1.2;

    jv = 1./(4.*M_PI)*pow(2,5)*M_PI*pow(EE,6)/(3.*ME*pow(CL,3));
    jv *= pow(2.*M_PI/(3.*KBOL*ME),1./2.);
    jv *= pow(Te,-1./2.)*Ne*Ne;
    jv *= efac*rel*gff;

  } else if (bremss_type == 2) {
    // Svensson 1982 as used in e.g. Straub 2012
    double Fei=0., Fee=0., fei=0., fee=0.;

    double SOMMERFELD_ALPHA = 1. / 137.036;
    double eta = 0.5616;
    double gammaE = 0.577; // = - Log[0.5616] 
    double gff;

    if (x > 1) {
      gff = sqrt(3. / M_PI / x);
    } else {
      gff = sqrt(3.) / M_PI * log(4 / gammaE / x);
    }

    if (Thetae < 1) {
      Fei = 4. * sqrt(2.*Thetae/M_PI/M_PI/M_PI) * (1. + 1.781*pow(Thetae,1.34));
      Fee = 20./9./sqrt(M_PI) * (44. - 3.*M_PI*M_PI) * pow(Thetae,1.5);
      Fee *= (1. + 1.1*Thetae + Thetae*Thetae - 1.25*pow(Thetae,2.5));
    } else {
      Fei = 9.*Thetae/(2.*M_PI) * ( log(1.123 * Thetae + 0.48) + 1.5 );
      Fee = 24. * Thetae * ( log(2.*eta*Thetae) + 1.28 );
    }

    fei = Ne * Ne * SIGMA_THOMSON * SOMMERFELD_ALPHA * ME * CL * CL * CL * Fei;
    fee = Ne * Ne * RE * RE * SOMMERFELD_ALPHA * ME * CL * CL * CL * Fee;

    jv = (fei+fee) / (4.*M_PI) * HPL/KBOL/Te * efac * gff;

  } else if (bremss_type == 3) {
    // Nozawa 2009 - electron-electron
    jv  = 7.673889101895528e-44 * Ne * Ne * efac * gffee(Te, nu) * sqrt(Thetae);
    // van Hoof 2015 - electron-ion
    jv += 7.070090102391322e-44 * Ne * Ne * efac * gffei(Te, nu) / sqrt(Thetae);
  }

  return jv;
}

/* Absorptivities */
double bremss_I_abs(FitParams *params, int bremss_type)
{
  double ans = bremss_I(params, bremss_type) / planck_func(params);
  return ans;
}

// Below are bremsstrahlung routines that are used *only* if the van Hoof 2015 + Nozawa 2009 formulae (type 3) are used.
gsl_spline2d *bremss_spline;
gsl_interp_accel *bremss_xacc;
gsl_interp_accel *bremss_yacc;

/*
 * Electron-ion bremsstrahlung.
 * This code uses data from:
 * van Hoof, P.~A.~M., Ferland, G.~J., Williams, R.~J.~R., et al.\ 2015, \mnras, 449, 2112
 * Please cite their work if you use these electron-ion bremsstrahlung formulae with grmonty.
 */

// Sets the bremss splines as global variables so they can be used to interpolate the Gaunt factor
void init_bremss_spline()
{
  // This requires gff_ei_bremss.dat in ipole's directory in order to function
#if 0
  const size_t nx = 146;
  const size_t ny = 81;
  double *z = malloc(nx * ny * sizeof(double*));
  const double xa[] = {-16.,-15.8,-15.6,-15.4,-15.2,-15.,-14.8,-14.6,-14.4,-14.2,-14.,-13.8,-13.6,-13.4,-13.2,-13.,-12.8,-12.6,-12.4,-12.2,-12.,-11.8,-11.6,-11.4,-11.2,-11.,-10.8,-10.6,-10.4,-10.2,-10.,-9.8,-9.6,-9.4,-9.2,-9.,-8.8,-8.6,-8.4,-8.2,-8.,-7.8,-7.6,-7.4,-7.2,-7.,-6.8,-6.6,-6.4,-6.2,-6.,-5.8,-5.6,-5.4,-5.2,-5.,-4.8,-4.6,-4.4,-4.2,-4.,-3.8,-3.6,-3.4,-3.2,-3.,-2.8,-2.6,-2.4,-2.2,-2.,-1.8,-1.6,-1.4,-1.2,-1.,-0.8,-0.6,-0.4,-0.2,0.,0.2,0.4,0.6,0.8,1.,1.2,1.4,1.6,1.8,2.,2.2,2.4,2.6,2.8,3.,3.2,3.4,3.6,3.8,4.,4.2,4.4,4.6,4.8,5.,5.2,5.4,5.6,5.8,6.,6.2,6.4,6.6,6.8,7.,7.2,7.4,7.6,7.8,8.,8.2,8.4,8.6,8.8,9.,9.2,9.4,9.6,9.8,10.,10.2,10.4,10.6,10.8,11.,11.2,11.4,11.6,11.8,12.,12.2,12.4,12.6,12.8,13.};
  const double ya[] = {-6.,-5.8,-5.6,-5.4,-5.2,-5.,-4.8,-4.6,-4.4,-4.2,-4.,-3.8,-3.6,-3.4,-3.2,-3.,-2.8,-2.6,-2.4,-2.2,-2.,-1.8,-1.6,-1.4,-1.2,-1.,-0.8,-0.6,-0.4,-0.2,0.,0.2,0.4,0.6,0.8,1.,1.2,1.4,1.6,1.8,2.,2.2,2.4,2.6,2.8,3.,3.2,3.4,3.6,3.8,4.,4.2,4.4,4.6,4.8,5.,5.2,5.4,5.6,5.8,6.,6.2,6.4,6.6,6.8,7.,7.2,7.4,7.6,7.8,8.,8.2,8.4,8.6,8.8,9.,9.2,9.4,9.6,9.8,10.};
  double val;

  // Set spline up for quadratic interpolation between the tabulated VH15 data.
  const gsl_interp2d_type *T = gsl_interp2d_bilinear;

//  gsl_spline2d     *bremss_spline;
//  gsl_interp_accel *bremss_xacc;
//  gsl_interp_accel *bremss_yacc;


  bremss_spline = gsl_spline2d_alloc(T,nx,ny);
  bremss_xacc = gsl_interp_accel_alloc();
  bremss_yacc = gsl_interp_accel_alloc();

  // Data file where VH15 data is stored. Z01 means hydrogen, merged means combines relativistic and non-relativistic results
  FILE *myfile;
  myfile = fopen("gff_ei_bremss.dat","r");
  if (myfile == NULL) {
    fprintf(stderr, "Cannot find spline2d data: gff_ei_bremss.dat! Exiting.\n");
    exit(-1);
  }

  // Read file and assign it to spline.
  for (int i = 0; i < nx; i++)
  {
    for (int j = 0; j < ny; j++)
    {
      if (fscanf(myfile,"%lf",&val) != 1) exit(5);
      gsl_spline2d_set(bremss_spline, z, i, j, val);
    }
  }

  gsl_spline2d_init(bremss_spline,xa,ya,z,nx,ny);
#endif
}

// Returns the electron-ion Gaunt factor
double gffei(double Te, double nu)
{

  double gffeival,loggammasq,logu;

  // Make sure we don't go out of the domain.
  if      ( Te < 1.579e-5  ) Te = 1.579e-5;
  else if ( Te > 1.579e+11 ) Te = 1.579e+11;

  loggammasq = log10(157900. / Te);

  if      ( nu < (1.e-16) * KBOL * Te / HPL ) nu = (1.e-16) * KBOL * Te / HPL;
  else if ( nu > (1.e+13) * KBOL * Te / HPL ) nu = (1.e+13) * KBOL * Te / HPL;

  logu = log10(HPL * nu / KBOL / Te);

  gffeival = gsl_spline2d_eval(bremss_spline,logu,loggammasq,bremss_xacc,bremss_yacc);
  return gffeival;

}


/*
 * The following code is a very slightly modified version of the one released in:
 * Nozawa, S., Takahashi, K., Kohyama, Y., et al.\ 2009, \aap, 499, 661
 * Please cite their work if you use these electron-electron bremsstrahlung formulae with grmonty.
 */

double funce(double x)
{
	return gsl_sf_expint_Ei(-x);
}

double gi(double x, double tau)
{
	const double a[121] = {3.15847E0, 2.46819E-2, -2.11118E-2, 1.24009E-2, -5.41633E-3, 1.70070E-3, -3.05111E-4, -1.21721E-4, 1.77611E-4, -2.05480E-5, -3.58754E-5, -2.52430E0, 1.03924E-1, -8.53821E-2, 4.73623E-2, -1.91406E-2, 5.39773E-3, -7.26681E-4, -7.47266E-4, 8.73517E-4, -6.92284E-5, -1.80305E-4, 4.04877E-1, 1.98935E-1, -1.52444E-1, 7.51656E-2, -2.58034E-2, 4.13361E-3, 4.67015E-3, -2.20675E-3, -2.67582E-3, 2.95254E-5, 1.40751E-3, 6.13466E-1, 2.18843E-1, -1.45660E-1, 5.07201E-2, -2.23048E-3, -1.14273E-2, 1.24789E-2, -2.74351E-3, -4.57871E-3, -1.70374E-4, 2.06757E-3, 6.28867E-1, 1.20482E-1, -4.63705E-2, -2.25247E-2, 5.07325E-2, -3.23280E-2, -1.16976E-2, -1.00402E-3, 2.96622E-2, -5.43191E-4, -1.23098E-2, 3.29441E-1, -4.82390E-2, 8.16592E-2, -8.17151E-2, 5.94414E-2, -2.19399E-2, -1.13488E-2, -2.38863E-3, 1.89850E-2, 2.50978E-3, -8.81767E-3, -1.71486E-1, -1.20811E-1, 9.87296E-2, -4.59297E-2, -2.11247E-2, 1.76310E-2, 6.31446E-2, -2.28987E-3, -8.84093E-2, 4.45570E-3, 3.46210E-2, -3.68685E-1, -4.46133E-4, -3.24743E-2, 5.05096E-2, -5.05387E-2, 2.23352E-2, 1.33830E-2, 7.79323E-3, -2.93629E-2, -2.80083E-3, 1.23727E-2, -7.59200E-2, 8.88749E-2, -8.82637E-2, 5.58818E-2, 9.20453E-3, -4.59817E-3, -8.54735E-2, 7.98332E-3, 1.02966E-1, -5.68093E-3, -4.04801E-2, 1.60187E-1, 2.50320E-2, -7.52221E-3, -9.11885E-3, 1.67321E-2, -8.24286E-3, -6.47349E-3, -3.80435E-3, 1.38957E-2, 1.10618E-3, -5.68689E-3, 8.37729E-2, -1.28900E-2, 1.99419E-2, -1.71348E-2, -3.47663E-3, -3.90032E-4, 3.72266E-2, -4.25035E-3, -4.22093E-2, 2.33625E-3, 1.66733E-2};
	
	double logth = log10(tau);
	double logu  = log10(x  );
	double jfit = 0.;

	double the = ( logth + 2.65) / 1.35;
	double u   = ( logu  + 1.50) / 2.50;
	double uj;

	for (int j = 0; j < 11; j++){
		uj = pow(u, j);
		for (int i = 0; i < 11; i++){
			jfit += a[11*j + i] * pow(the, i) * uj;
		}
	}

	return sqrt(8. / 3. / M_PI) * jfit;

}

double gii(double x, double tau)
{
	const double a2[9] = {0.9217, -13.4988, 76.4539, -217.8301, 320.9753, -188.0667, -82.4161, 163.7191, -60.0248};
	const double a1[9] = {-9.3647, 95.9186, -397.0172, 842.9376, -907.3076, 306.8802, 291.2983, -299.0253, 76.3461};
	const double a0[9] = {-37.3698, 380.3659, -1489.8014, 2861.4150, -2326.3704, -691.6118, 2853.7893, -2040.7952, 492.5981};
	const double b1[9] = {-8.6991, 63.3830, -128.8939, -135.0312, 977.5838, -1649.9529, 1258.6812, -404.7461, 27.3354};
	const double b0[9] = {-11.6281, 125.6066, -532.7489, 1142.3873, -1156.8545, 75.0102, 996.8114, -888.1895, 250.1386};
	
	double aa2 = 0.;
	double aa1 = 0.;
	double aa0 = 0.;
	double bb1 = 0.;
	double bb0 = 0.;
	double powtau;

	double Ei = funce(x);

	for (int i = 0; i < 9; i++){
		powtau = pow(tau, i / 8.);
		aa2 += a2[i] * powtau;
		aa1 += a1[i] * powtau;
		aa0 += a0[i] * powtau;
		bb1 += b1[i] * powtau;
		bb0 += b0[i] * powtau;
	}

	return aa2 * x * x + aa1 * x + aa0 - exp(x) * Ei * ( bb1 * x + bb0 );

}

double giii(double x, double tau)
{
	double const a2[9] = {64.7512, -213.8956, 174.1432, 136.5088, -271.4899, 89.3210, 58.2584, -46.0807, 8.7301};
	double const a1[9] = {49.7139, -189.7746, 271.0298, -269.7807, 420.4812, -576.6247, 432.7790, -160.5365, 23.3925};
	double const a0[9] = {52.1633, -257.0313, 446.8161, -293.0585, 0.0000, 77.0474, -23.8718, 0.0000, 0.1997};
	double const b1[9] = {376.4322, -1223.3635, 628.6787, 2237.3946, -3828.8387, 2121.7933, -55.1667, -349.4321, 92.2059};
	double const b0[9] = {-8.5862, 34.1348, -116.3287, 296.5451, -393.4207, 237.5497, -30.6000, -27.6170, 8.8453};

	double aa2 = 0.;
	double aa1 = 0.;
	double aa0 = 0.;
	double bb1 = 0.;
	double bb0 = 0.;
	double powtau;

	double Ei = funce(x);

	for (int i = 0; i < 9; i++){
		powtau = pow(tau, i / 8.);
		aa2 += a2[i] * powtau;
		aa1 += a1[i] * powtau;
		aa0 += a0[i] * powtau;
		bb1 += b1[i] * powtau;
		bb0 += b0[i] * powtau;
	}

	return aa2 * x * x + aa1 * x + aa0 - exp(x) * Ei * ( bb1 * x + bb0 );

}

double giv(double x, double tau)
{
	double gam  = exp(0.5772156649);
	double con1 = 3. / ( 4. * M_PI * sqrt(tau) );
	double a1   = 28./3. + 2. * x      + x * x / 2.;
	double a2   =  8./3. + (4./3.) * x + x * x;
	double a3   =  8./3. - (4./3.) * x + x * x;

	double Ei = funce(x);
	double fx = a1 + 2. * a2 * log(2. * tau / gam) - exp(x) * Ei * a3;

	return con1 * fx;

}

double fcc(double x, double tau)
{
	double const p[7] = {-5.7752, 46.2097, -160.7280, 305.0070, -329.5420, 191.0770, -46.2718};
	double const q[7] = {30.5586, -248.2177, 874.1964, -1676.9028, 1828.8677, -1068.9366, 260.5656};
	double const r[7] = {-54.3272, 450.9676, -1616.5987, 3148.1061, -3478.3930, 2055.6693, -505.6789};
	double const s[7] = {36.2625, -310.0972, 1138.0531, -2260.8347, 2541.9361, -1525.2058, 380.0852};
	double const t[7] = {-8.4082, 74.7925, -282.9540, 576.3930, -661.9390, 404.2930, -102.2330};

	double ptau = 0.;
	double qtau = 0.;
	double rtau = 0.;
	double stau = 0.;
	double ttau = 0.;
	double powtau;

	for (int i = 0; i < 7; i++){
		powtau = pow(tau, i / 6.);
		ptau += p[i] * powtau;
		qtau += q[i] * powtau;
		rtau += r[i] * powtau;
		stau += s[i] * powtau;
		ttau += t[i] * powtau;
	}

	return 1. + ptau * pow(x, 2./8.) + qtau * pow(x, 3./8.) + rtau * pow(x, 4./8.) + stau * pow(x, 5./8.) + ttau * pow(x, 6./8.);

}

double gffee(double Te, double nu)
{
	double emass = 510.998902;
	double tau = KBOL * Te / ME / CL / CL;
	double x = HPL * nu / KBOL / Te;
	double gffeeval;

	if ( tau < ( 0.05 / emass ) ) tau = 0.05 / emass;

	if ( x < 1.e-4 )   x = 1.e-4;
	else if ( x > 10 ) x = 10.;

	if      ( tau >= ( 0.05  / emass ) && tau < ( 1.    / emass ) ) gffeeval = gi(x, tau);
	else if ( tau >  ( 1.    / emass ) && tau < ( 300.  / emass ) ) gffeeval = gii(x, tau) * fcc(x, tau);
	else if ( tau >  ( 300.  / emass ) && tau < ( 7000. / emass ) ) gffeeval = giii(x, tau);
	else if ( tau >  ( 7000. / emass ) )                            gffeeval = giv(x, tau);

	return gffeeval;

}
