//Libraries
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <stdio.h>
#include <time.h>                      
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <vector>
#include <fstream>
#include <sstream> //ADDED BY JORGE
#include <map>
//End Libraries

using namespace std;

int main(int ac, char **av)
{
double alpha = 14.0;
double beta = 0.0025;
gsl_rng *generator = gsl_rng_alloc(gsl_rng_mt19937);// initializing random number
gsl_rng_set (generator, 312); //Define a seed for random values 
int n = 100000;
double *x = new double[n];
for (int i=0; i < n; i++) {
	x[i] = gsl_ran_gamma(generator, alpha, beta);
}
double sum = 0;
for (int i=0; i < n; i++) {
	sum = sum + x[i];
}
double mean = sum / n;

sum = 0;
for (int i=0; i < n; i++) {
	sum = sum + pow(x[i] - mean, 2);
}
double var = sum / n;

cout<<"E[X]: "<<(alpha*beta)<<"\n";
cout<<"mean: "<<mean<<"\n";
cout<<"V[X]: "<<(alpha*beta*beta)<<"\n";
cout<<"var: "<<var<<"\n";
}