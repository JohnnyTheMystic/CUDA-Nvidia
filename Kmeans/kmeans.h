#ifndef KMEANS_H
#define KMEANS_H

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

#define sqr(x) ((x)*(x))
#define MAX_ITER_NO_IMPR 10

void cluster_diag(int dim, int n, int k, float *X, int *cluster_assignment_index, float *cluster_centroid);
void random_init_centroid (float * cluster_centro_id, float * dataSetMatrix, int clusters, int rows, int columns);
void kmeans( int  dim, float *X, int n, int k, float *cluster_centroid, int iterations, int *cluster_assignment_final, int mode);



#endif
