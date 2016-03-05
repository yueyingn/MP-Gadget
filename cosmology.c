#include "cosmology.h"
#include "allvars.h"
/*Hubble function at scale factor a, in dimensions of All.Hubble*/
double hubble_function(double a)
{
  double hubble_a;
  /*Note OMEGAR is defined to be 0 if INCLUDE_RADIATION is not on*/
  hubble_a = All.Omega0 / (a * a * a) + OMEGAK / (a * a) + OMEGAR/(a*a*a*a) + All.OmegaLambda;
  hubble_a = All.Hubble * sqrt(hubble_a);
  return (hubble_a);
}