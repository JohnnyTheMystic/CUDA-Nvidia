//************************** kmeans.c ***************************
//*******************Developed by JohnnyTheMystic*******************
//************************* September 2019************************

#include "kmeans.h"

void fail(const char * str) {
    fprintf(stderr,"%s", str);
    exit(-1);
}

/**
* calc_distance calculates the distance between a given point and a cluster
* @param int -dim: number of columns (variables) in the data set to be classified
* @param float * -: first arrray to calculate de distance
* @param float * -: Second array to calculate de distance
* @return float: Euclidean distance of two vectors
*/
float calc_distance(int dim, float *p1, float  *p2) {

    float distance_sq_sum = 0;

    for (int i = 0; i < dim; ++i)
      distance_sq_sum += sqr(p1[i] - p2[i]);

    return distance_sq_sum;

}

/**
* calc_all_distances computes the euclidean distances between centros ids and dataset points.
* @param int -dim: number of columns (variables) in the data set to be classified
* @param int -n: number of rows (points) in the data set to be classified
* @param int -k: number of clusters to be calculated
* @param float * -X: dataset to be classified
* @param float * -centroid: prototypes of each cluster.
* @param float * -distance_output[n][k] contains the distance between all elements * in the dataset and all clusters
* return void
*/
void calc_all_distances(int dim, int n, int k, float *X, float *centroid, float *distance_output) {

	//double ini = omp_get_wtime();
	#pragma omp parallel for
	for (int i = 0; i < n; ++i){ // for each point
		for (int j = 0; j < k; ++j) { // for each cluster
		// calculate distance between point and cluster centroid
			distance_output[i*k+j] = calc_distance(dim, &X[i*dim], &centroid[j*dim]);
		}
	}
	//double fin = omp_get_wtime();
	//printf("Tiempo CalcAllDistances OMP: %lf\n", fin-ini );
}


/**
* calc_total_distance calculates the clustering overall distance.
* @param int -dim: number of columns (variables) in the data set to be classified
* @param int -n: number of rows (points) in the data set to be classified
* @param int -k: number of clusters to be calculated
* @param float * -X: dataset to be classified
* @param float * -centroid: prototypes of each cluster.
* @param int * - cluster_assignment_index: current cluster assignment to each point
* @return float overall distance. This is what the algorithm tried to minimize
*/
float calc_total_distance(int dim, int n, int k, float *X, float *centroids, int *cluster_assignment_index) {

    // NOTE: a point with cluster assignment -1 is ignored
    float tot_D = 0;
	
	//double ini = omp_get_wtime();
    // for every point
	#pragma omp parallel for reduction (+:tot_D)
    for (int i = 0; i < n; ++i) {
        // which cluster is it in?
        int active_cluster = cluster_assignment_index[i];

       // sum distance
        if (active_cluster != -1)
            tot_D += calc_distance(dim, &X[i*dim], &centroids[active_cluster*dim]);
    }
	
	//double fin = omp_get_wtime();
	//printf("Tiempo Calc_Total_Distance OMP: %lf\n", fin-ini );

    return tot_D;
}


/**
* choose_all_clusters_from_distances obtains the closest cluster for each point.
* @param int -dim: number of columns (variables) in the data set to be classified
* @param int -n: number of rows (points) in the data set to be classified
* @param int -k: number of clusters to be calculated
* @param float * -distance_array[n][k] contains the distance between all elements * in the dataset and all clusters
* @param int* - cluster_assignment_index contains the assigned cluster to each point
* @return void
*/
void choose_all_clusters_from_distances(int dim, int n, int k, float *distance_array, int *cluster_assignment_index) {

	#pragma omp parallel for
    // for each point
    for (int i = 0; i < n; ++i) {
        int best_index = -1;
        float closest_distance = INFINITY;

        // for each cluster
        for (int j = 0; j < k; j++) {
           // distance between point and cluster centroid
            float cur_distance = distance_array[i*k+j];
            if (cur_distance < closest_distance) {
                best_index = j;
                closest_distance = cur_distance;
            }
        }

        // record in array
        cluster_assignment_index[i] = best_index;
    }
}


void calc_cluster_centroidsOMP(int dim, int n, int k, float *X, int *cluster_assignment_index, float *new_cluster_centroid) {

    int * cluster_member_count = (int *)malloc(k * sizeof(float));

	#pragma omp parallel for
	// initialize cluster centroid coordinate sums to zero
	for (int i = 0; i < k; ++i) {
		cluster_member_count[i] = 0;

		for (int j = 0; j < dim; ++j)
		{
			new_cluster_centroid[i*dim + j] = 0;
		}
	}

	// sum all points for every point
	#pragma omp parallel
    {

        float *P_new_cluster_centroid = (float *)malloc(k * dim * sizeof(float));
		int *P_cluster_member_count = (int *)malloc(k * sizeof(float));

		#pragma omp parallel for
		for (int i = 0; i < k; i++) {
			P_cluster_member_count[i] = 0;
			for (int j = 0; j < dim; j++) {
				P_new_cluster_centroid[i*dim + j] = 0;
			}
		}
		
		#pragma omp parallel for
		for (int i = 0; i < n; i++) {
			// which cluster is it in?
			int active_cluster = cluster_assignment_index[i];

			// update count of members in that cluster
			P_cluster_member_count[active_cluster]++;

			// sum point coordinates for finding centroid
			for (int j = 0; j < dim; j++)
			{
				P_new_cluster_centroid[active_cluster*dim + j] += X[i*dim + j];
			}
		}

		#pragma omp critical
		for (int i = 0; i < k; i++)	{
				cluster_member_count[i] += P_cluster_member_count[i];

				for (int j = 0; j < dim; j++)
				{
					new_cluster_centroid[i*dim + j] += P_new_cluster_centroid[i*dim + j];
				}
		}

		free(P_cluster_member_count);
		free(P_new_cluster_centroid);

	}


	// now divide each coordinate sum by number of members to find mean/centroid for each cluster
	for (int i = 0; i < k; ++i) {
		if (cluster_member_count[i] == 0) break; // Empy Cluster

    // for each dimension
		for (int j = 0; j < dim; j++) {
			new_cluster_centroid[i*dim + j] /= cluster_member_count[i];  /// XXXX will divide by zero here for any empty clusters!
		}
	}
}


/**
* calc_cluster_centroids calculates the new prototypes of all clusters
* @param int -dim: number of columns (variables) in the data set to be classified
* @param int -n: number of rows (points) in the data set to be classified
* @param int -k: number of clusters to be calculated
* @param float * -X: dataset to be classified
* @param int * - cluster_assigment_index:
* @param float * -new_cluster_centroid: it is the output with the new cluster prototypes
*/

void calc_cluster_centroids(int dim, int n, int k, float *X, int *cluster_assignment_index, float *new_cluster_centroid) {

    int * cluster_member_count = (int *) malloc (k*sizeof(float));

    // initialize cluster centroid coordinate sums to zero
    for (int i = 0; i < k; ++i)  {
        cluster_member_count[i] = 0;

        for (int j = 0; j < dim; ++j)
          new_cluster_centroid[i*dim + j] = 0;
    }

    // sum all points
    // for every point
    for (int i = 0; i < n; ++i) {

        // which cluster is it in?
        int active_cluster = cluster_assignment_index[i];
        // update count of members in that cluster
        cluster_member_count[active_cluster]++;

        // sum point coordinates for finding centroid
        for (int j = 0; j < dim; j++)

            new_cluster_centroid[active_cluster*dim + j] += X[i*dim + j];

    }

   // now divide each coordinate sum by number of members to find mean/centroid
   // for each cluster
    for (int i = 0; i < k; ++i) {

        if (cluster_member_count[i] == 0) {
           // printf("WARNING: Empty cluster %d! \n", i);
            cluster_member_count[i]=0.00005;
        }
        // for each dimension
        for (int j = 0; j < dim; j++)
            new_cluster_centroid[i*dim + j] /= cluster_member_count[i];  /// XXXX will divide by zero here for any empty clusters!
    }
}


/**
* get_cluster_member_count the member of each cluster
* @param int -n: number of rows (points) in the data set to be classified
* @param int -k: number of clusters to be calculated
* @param int* - cluster_assignment_index contains the assigned cluster to each point
* @param int * -cluster_member_count: count members of each cluster
*/
void get_cluster_member_count(int n, int k, int *cluster_assignment_index, int *cluster_member_count) {

   // initialize cluster member counts
    for (int i = 0; i < k; i++)
        cluster_member_count[i] = 0;

   // count members of each cluster
    for (int i = 0; i < n; i++)
        cluster_member_count[cluster_assignment_index[i]]++;
}


/**
* Visualize the number of members for all clusters
*
*/
void cluster_diag(int dim, int n, int k, float *X, int *cluster_assignment_index, float *cluster_centroid) {

    int * cluster_member_count = (int *) malloc (k*sizeof(int));

    get_cluster_member_count(n, k, cluster_assignment_index, cluster_member_count);

    printf("  Final clusters \n");
    for (int i = 0; i < k; ++i) {
         printf("\tcluster %d:  members: %8d, for the centroid (", i, cluster_member_count[i]);
        for (int j=0; j<dim;j++)
            printf ("%f, ", cluster_centroid[i*dim + j]);
        printf (")\n");
    }
}

void copy_assignment_array(int n, int *src, int *tgt) {

    for (int i = 0; i < n; i++)
        tgt[i] = src[i];

}


int assignment_change_count(int n, int a[], int b[]) {

    int change_count = 0;

    for (int i = 0; i < n; ++i)
        if (a[i] != b[i])
            change_count++;

    return change_count;
}


/**
* random_init_centroid chooses random prototypes that belong to the dataset. They are points of the dataset.
*@param float * -: cluster_centro_if: clustes id choosen
*@param float * -: dataSetMatrix
*@param int clusters: Number of cluster to be don.
*@param int rows in number of rows in the dataset; i.e. points
*@param int columns: number of columns. Point's dimension.
*@return void
*/
void random_init_centroid (float * cluster_centro_id, float * dataSetMatrix, int clusters, int rows, int columns) {

   srand(time(NULL));
   
   // Utilizar paralelismo en ésta parte no es eficiente ~0.000045 frente a ~0.000004

   //double ini = omp_get_wtime();
   for (int i=0; i<clusters; ++i) {

        int r = rand()%rows;
        for (int j=0; j<columns;++j) {
            cluster_centro_id[i*columns+j]=dataSetMatrix[r*columns+j];
        //    printf ("Los indices son  %d\n", r*columns+j);
        }

    }
	//double fin = omp_get_wtime();
	//printf("Tiempo Random Secuencial: %lf\n", fin-ini );
	
}


/*
* This is C source code for a simple implementation of the popular k-means clustering algorithm.
* It is based on the implementation in Matlab, which was in turn based on GAF Seber,
* Multivariate Observations, 1964, and H Spath, Cluster Dissection and Analysis: Theory, FORTRAN Programs, Examples.
* @param int -dim: number of columns (variables) in the data set to be classified
* @param float * -X: dataset to be classified
* @param int -n: number of rows (points) in the data set to be classified
* @param int -k: number of clusters to be calculated
* @param float * -cluster_centroid: Initial clusters prototypes or centros
* @param int iterations -: number of iterations
* @param int * cluster_assignment_final -: Output classitfication
*/
void kmeans(
            int  dim,	                    // dimension of data
            float *X,                        // pointer to data
            int   n,                        // number of elements
            int   k,                        // number of clusters
            float *cluster_centroid,         // initial cluster centroids
            int iterations,                   // Nunber of iterations to be performed
            int   *cluster_assignment_final,  // output
            int mode                      //number of threads in the execution
            ) {


//    printf ("The number of threads is %d\n", mode);
    omp_set_num_threads (mode);
    float  *dist = (float *) malloc(sizeof(float) * n * k);
    int   *cluster_assignment_cur  = (int *)malloc(sizeof(int) * n);
    int   *cluster_assignment_prev = (int *)malloc(sizeof(int) * n);
    float  *point_move_score        = (float *)malloc(sizeof(float) * n * k);


    if (!dist || !cluster_assignment_cur || !cluster_assignment_prev || !point_move_score)
        fail("Error allocating dist arrays\n");

    // Initial setup. Assignment Step
	double ini = omp_get_wtime();
    random_init_centroid (cluster_centroid, X, k, n, dim);
	double fin = omp_get_wtime();
    printf ("OMP-Time -- Random_Init_Centroid: %.5lf s\n", fin-ini);
	
	ini = omp_get_wtime();
    calc_all_distances(dim, n, k, X, cluster_centroid, dist);
    fin = omp_get_wtime();
    printf ("OMP-Time -- Calc_All_Distances 1a vez: %.5lf s\n", fin-ini);
	
	ini = omp_get_wtime();
	choose_all_clusters_from_distances(dim, n, k, dist, cluster_assignment_cur);
	fin = omp_get_wtime();
    printf ("OMP-Time -- Choose_All_Clusters_From_Distances 1a vez: %.5lf s\n", fin-ini);

    copy_assignment_array(n, cluster_assignment_cur, cluster_assignment_prev);

    //The initial quality is the one obtained from the random election
	ini = omp_get_wtime();
    float prev_totD = calc_total_distance(dim, n, k, X, cluster_centroid, cluster_assignment_cur);
	fin = omp_get_wtime();
	printf("OMP-Time -- Calc_Total_Distance totD 1a vez: %.5lf s\n", fin-ini );

	printf ("OMP -- Total Distance:\t\t%.4f\n", prev_totD);

    int numVariations =0;	
    // UPDATE STEP
   // for (int batch=0; (batch < iterations) && (numVariations <MAX_ITER_NO_IMPR); ++batch) {

	double ini_iter = omp_get_wtime();
	for (int batch=0; (batch < iterations); ++batch) {
        //printf("Batch step: %d \n", batch);
        //cluster_diag(dim, n, k, X, cluster_assignment_cur, cluster_centroid);

		ini = omp_get_wtime();
		// update cluster centroids. Update Step
		calc_cluster_centroidsOMP(dim, n, k, X, cluster_assignment_cur, cluster_centroid);
		if (batch == 1){
			fin = omp_get_wtime();
			printf("OMP-Time -- Calc_Cluster_CentroIds_Omp: %.5lf s\n", fin-ini );
		}
		
        float totD = calc_total_distance(dim, n, k, X, cluster_centroid, cluster_assignment_cur);

        // see if we've failed to improve
        if (totD >= prev_totD){
            // failed to improve - currently solution worse than previous
            // restore old assignments
            copy_assignment_array(n, cluster_assignment_prev, cluster_assignment_cur);

            // recalc centroids randomly
            random_init_centroid (cluster_centroid, X, k, n, dim);
            // calc_cluster_centroids(dim, n, k, X, cluster_assignment_cur, cluster_centroid);
            //printf("\tNegative progress made on this step - iteration completed (%.2f) \n", prev_totD-totD);
            //numVariations++; //To implement no convergence criteria
        }
        else { // We have made some improvements
            // save previous step
            copy_assignment_array(n, cluster_assignment_cur, cluster_assignment_prev);
           // move all points to nearest cluster
            calc_all_distances(dim, n, k, X, cluster_centroid, dist);
            choose_all_clusters_from_distances(dim, n, k, dist, cluster_assignment_cur);
            //check how many assignments are different
            //int change_count = assignment_change_count(n, cluster_assignment_cur, cluster_assignment_prev);
            //printf("\tIn the batch: %d, has changed: %d element to a different cluster with an improvement of %f \n", batch, change_count, prev_totD-totD);
            //fflush(stdout);
            prev_totD = totD;
        }
    }

//    cluster_diag(dim, n, k, X, cluster_assignment_cur, cluster_centroid);

	fin = omp_get_wtime();
	printf ("OMP -- Total Distance Mejorado:\t%.4lf\n", prev_totD);
	printf("OMP-Time -- %d Iteraciones: %.5lf s\n\n", iterations, fin-ini_iter );


    // write to output array
    copy_assignment_array(n, cluster_assignment_cur, cluster_assignment_final);

    free(dist);
    free(cluster_assignment_cur);
    free(cluster_assignment_prev);
    free(point_move_score);
  }






