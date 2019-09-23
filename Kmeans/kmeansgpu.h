#ifndef CUDARR
#define CUDARR


#define TH 256
#define OPERTH 4
#define sqr(x) ((x)*(x))

#include <stdio.h>
#ifdef __cplusplus
  extern "C"
#endif

int kmeansCUDA(	int  dim, float *H_X, int n, int k,	float *H_cluster_centroid, int iterations, int *H_cluster_assignment_final);


#endif
