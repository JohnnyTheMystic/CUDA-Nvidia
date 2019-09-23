//************************** kmeans.c *****************************
//*******************Developed by José M. Cecilia******************
//*******************Modified by JohhnyTheMystic*******************
//************************* September 2019***************************

#include "File.h"
#include "kmeans.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <omp.h>
#include "kmeansgpu.h"
#include <getopt.h>

#define sqr(x) ((x)*(x))
#define MAX_ITER_NO_IMPR 10



void menu (int * clusters, char ** fileName, int *iterations, int argc, char** argv, int * mode) {

    char c;

    int err=0;
    while ((c = getopt (argc, argv, "v:c:f:i:m:h")) != -1) {
	    switch (c) {
	        case 'v':
	            printf("K means algorithm v.1.0\n\n");
		    exit(0);
	        case 'c':
                *clusters = atoi(optarg);
                if (*clusters < 1) {
                    printf ("the minimum number of clusters is 1\n");
                    exit(0);
                }
                break;
    	    case 'f':
                *fileName = (char *) malloc (strlen(optarg)+1);
	            strcpy(*fileName,optarg);
               // printf ("El nombre del fichero es %s\n", *fileName);
                break;
    	    case 'i':
                *iterations = atoi (optarg);
                break;
            case 'm':
                *mode = atoi (optarg);
                break;
	        case 'h':
	        case '?':
	            printf("Usage:\trun -c number of clusters -f fichero.txt -i number of iterations -m mode [0=GPU | 1=CPU|>1 OMP_NUM_THREADS] [-h | -? HELP] \n");
		        printf("\t<Params>\n");
		        printf("\t\t-v\t\tOutput version information and exit\n");
	            exit(0);
	        default:
		        err = -1;
		        break;
        }
	    if (err==-1)
	        break;
    }

}


int main( int argc, char **argv ) {

    float *cluster_centroid;   // initial cluster centroids. The size is Clusters x rows
    int   *clustering_output;  // output
    int rows=0, columns=0, clusters=1;
    int mode = 1; //Executution mode: GPU 0, CPU 1, omp >1 -> Number of threads
    int iterations = 1000;
    float * dataSetMatrix=NULL;
    char *fileName=NULL;

    menu (&clusters, &fileName, &iterations, argc, argv, &mode);
   // printf ("El nombre del fichero es %s\n", fileName);
    //printf ("..............Loading data set...............\n ");
    // Get file size dataset
    getSizeFile( fileName, &rows, &columns);

    clustering_output = (int *) malloc (rows*sizeof(int));
    // Reserve dynamic memory for dataset matrix
    reserveDynamicMemoryForMatrix( &dataSetMatrix, rows, columns );

    // Set data in the dataset matrix
    setDataInMatrix( dataSetMatrix, fileName, rows, columns );

    //printf ("-------DataSet: \n");
    //printMatrix(dataSetMatrix, rows, columns);

    //printf ("..............Done..............\n ");
    cluster_centroid = (float *) malloc (clusters*columns*sizeof(float));

    //printf (".........Initial Prototypes: ................ \n");
    //printMatrix(cluster_centroid, clusters, columns);

 //   printf ("The number of instance: %d Variables: %d Clusters: %d and Iterations: %d\n", rows, columns,clusters, iterations);
    double ini = omp_get_wtime();

    switch (mode) {

        case 0: //CUDA execution of kmeans

            kmeansCUDA (columns, dataSetMatrix, rows, clusters, cluster_centroid, iterations, clustering_output);
            break;
        default: //Mode 0 sequential execution and >1
            kmeans (columns, dataSetMatrix, rows, clusters, cluster_centroid, iterations, clustering_output, mode);
            break;
    }

    double fin = omp_get_wtime();

    //cluster_diag (columns, rows, clusters, dataSetMatrix, clustering_output, cluster_centroid);

    printf ("Tiempo Total: %.3lf\n\n", fin-ini);
    //Free memory
    free (dataSetMatrix);
    free (cluster_centroid);
    free (clustering_output);


}






