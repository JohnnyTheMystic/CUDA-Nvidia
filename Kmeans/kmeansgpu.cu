//************************** kmeansgpu.cu ***************************
//*******************Developed by JohnnyTheMystic*******************
//************************* September 2019************************


#include "cuda.h"
#include "kmeansgpu.h"
#include <curand.h>
#include "device_functions.h"
#include <curand_kernel.h>
#include <time.h>
#include <math.h>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		const char * error = cudaGetErrorString(code);
		fprintf(stderr, "GPUassert: %s %s %d\n", error, file, line);
		if (abort) exit(code);
	}
}

__global__ void setup_kernel(curandState *state, unsigned long seed) {
	
    int index = threadIdx.x;
	curand_init(seed, index, 0, &state[index]);
    //__syncthreads();
}

__global__ void generate(curandState* globalState, float *result) {
	int ind = threadIdx.x;

	// copiar estado a la memoria local para mayor eficiencia
	curandState localState = globalState[ind];

	// generar número pseudoaleatorio
	float r = curand_uniform(&localState);

	//copiar state de regreso a memoria global
	globalState[ind] = localState;

	//almacenar resultados
	result[ind] = r;
	//printf("Resultado del random %f", result[ind]);
}



/**
* calc_all_distances computes the euclidean distances between centros ids and dataset points.
*/
__global__ void calc_all_distancesCUDA(int columns, int rows, int clusters, float *dataSetMatrix, float *centroid, float *distance_output) {

    int index = blockIdx.x*blockDim.x+threadIdx.x;
	
	if (index < rows){
		for (int j = 0; j < clusters; ++j) { // for each cluster

		// calculate distance between point and cluster centroid
		    for (int i = 0; i < columns; ++i){
				distance_output[index*clusters+j] += sqr(dataSetMatrix[index*columns+i] - centroid[j*columns+i]);
			}
		}
	}
}

/* __device__ inline void MyAtomicAdd (float *address, float value)
 {
   int oldval, newval, readback;
 
   oldval = __float_as_int(*address);
   newval = __float_as_int(__int_as_float(oldval) + value);
   while ((readback=atomicCAS((int *)address, oldval, newval)) != oldval) 
     {
      oldval = readback;
      newval = __float_as_int(__int_as_float(oldval) + value);
     }
 }*/


/**
* calc_total_distance calculates the clustering overall distance.
*/
__global__ void calc_total_distanceCUDA(int columns, int k, float *dataSetMatrix, float *centroids, int *cluster_assignment_index, float *totD) {

    int index = blockIdx.x*blockDim.x+threadIdx.x;

    // for every point
    // which cluster is it in?
    int active_cluster = cluster_assignment_index[index];
    // sum distance
    if (active_cluster != -1){
		for (int i = 0; i < columns; ++i){
			float raiz = (float) sqr(dataSetMatrix[index*columns+i] - centroids[active_cluster*columns+i]);
			atomicAdd(&totD[0], raiz);
		}
	}
}



double calc_distanceNoCUDA(int dim, float *p1, float  *p2) {

    double distance_sq_sum = 0;
    for (int i = 0; i < dim; ++i)
      distance_sq_sum += sqr(p1[i] - p2[i]);

    return distance_sq_sum;
}


double calc_total_distanceNoCUDA(int dim, int n, int k, float *X, float *centroids, int *cluster_assignment_index) {

    double tot_D = 0;
	
    // for every point
    for (int i = 0; i < n; ++i) {
        // which cluster is it in?
        int active_cluster = cluster_assignment_index[i];

       // sum distance
        if (active_cluster != -1)
            tot_D += calc_distanceNoCUDA(dim, &X[i*dim], &centroids[active_cluster*dim]);
    }
    return tot_D;
}




/**
* choose_all_clusters_from_distances obtains the closest cluster for each point.
*/
__global__ void choose_all_clusters_from_distancesCUDA(int rows, int clusters, float *distance_array, int *cluster_assignment_index) {

    // for each point
    int index = blockIdx.x*blockDim.x+threadIdx.x;
	
	if (index < rows){
		
        int best_index = -1;
        float closest_distance = INFINITY;

        // for each cluster
        for (int j = 0; j < clusters; j++) {
			
           // distance between point and cluster centroid
            float cur_distance = distance_array[index*clusters+j];
			
            if (cur_distance < closest_distance) {
                best_index = j;
                closest_distance = cur_distance;
            }
        }

        // record in array
        cluster_assignment_index[index] = best_index;
    }
}

/**
* calc_cluster_centroids calculates the new prototypes of all clusters
*/

__global__ void initialize_To_Zero (int dim, int *cluster_member_count, float *new_cluster_centroid) {
	
	int index = threadIdx.x;
	cluster_member_count[index] = 0;
	for (int j = 0; j < dim; ++j){
		new_cluster_centroid[index*dim + j] = 0;
	}
}

__global__ void which_Is_In (int dim, int *cluster_assignment_index, int *cluster_member_count, float *new_cluster_centroid, float *dataSetMatrix){
	
	int index = blockIdx.x*blockDim.x+threadIdx.x;

	// which cluster is it in?
	int active_cluster = cluster_assignment_index[index];

	// update count of members in that cluster
	atomicAdd(&cluster_member_count[active_cluster], 1);

	// sum point coordinates for finding centroid
	for (int j = 0; j < dim; j++)
	{
		atomicAdd(&new_cluster_centroid[active_cluster*dim + j], dataSetMatrix[index*dim + j]);
	}
}


__global__ void calc_cluster_centroidsCUDA(int dim, int *cluster_member_count, float *new_cluster_centroid) {

	int index = threadIdx.x;

	// now divide each coordinate sum by number of members to find mean/centroid for each cluster
	if (cluster_member_count[index] == 0){
        cluster_member_count[index]=0.00005;
	}

	// for each dimension
	for (int j = 0; j < dim; j++) {
		//printf("Cluster_Member_Count %d: %d\n", index, cluster_member_count[index]);
		new_cluster_centroid[index*dim + j] /= cluster_member_count[index];  /// XXXX will divide by zero here for any empty clusters!
	}
}



void copy_assignment_arrayNoCUDA(int n, int *src, int *tgt) {
    for (int i = 0; i < n; i++){
        tgt[i] = src[i];
	}
}



__global__ void random_init_centroidCUDA(float * randoms, float * cluster_centro_id, float * dataSetMatrix, int clusters, int rows, int columns) {
	
    //int index = blockIdx.x*blockDim.x+threadIdx.x;
	int index = threadIdx.x;
	int random = ((((int)(randoms[index]*100))/columns)*columns); // Ecuacion para conseguir el primer componente de la fila.
	//printf("%d\n",random);
	
	for (int i=0; i<columns; i++){
		cluster_centro_id[index*columns+i] = dataSetMatrix[random+i];
	}
}

__global__ void random_init_centroidCUDA_mil(float * randoms, float * cluster_centro_id, float * dataSetMatrix, int clusters, int rows, int columns) {
	
    //int index = blockIdx.x*blockDim.x+threadIdx.x;
	int index = threadIdx.x;
	int random = ((((int)(randoms[index]*1000))/columns)*columns); // Ecuacion para conseguir el primer componente de la fila.
	//printf("%d\n",random);
	
	for (int i=0; i<columns; i++){
		cluster_centro_id[index*columns+i] = dataSetMatrix[random+i];
	}
}





extern "C" int kmeansCUDA(	int  dim,		// columnas del fichero
							float *H_X,		// dataSetMatrix
							int n,		// numero de elementos (rows)
							int k,			// numero de clusters
							float *H_cluster_centroid,		// centroides de cada cluster (clusters*columnas)
							int iterations,					// repeticiones de mejora
							int *H_cluster_assignment_final		// resultado
							) {


	float *D_dist, *D_cluster_centroid, *D_dataSetMatrix;
	float *D_totD;
	double *D_prev_totD;
	int *D_cluster_assignment_cur, *D_cluster_assignment_prev, *D_cluster_member_count;
	
	cudaMalloc(&D_dist,(sizeof(float)*n * k));
	    float *dist = (float *) malloc(sizeof(float) * n * k);
    cudaMalloc(&D_cluster_assignment_cur, sizeof(int)*n);
	    int *cluster_assignment_cur  = (int *)malloc(sizeof(int) * n);
    cudaMalloc(&D_cluster_assignment_prev, sizeof(int)*n);
		int *cluster_assignment_prev  = (int *)malloc(sizeof(int) * n);
	cudaMalloc(&D_cluster_centroid,(k*dim*sizeof(float)));
	cudaMalloc(&D_dataSetMatrix,(n*dim*sizeof(float)));
	
	cudaMalloc(&D_cluster_member_count,(k*sizeof(float)));
	cudaMalloc(&D_totD, sizeof (float));
		float *total = (float *) malloc(sizeof(float));
	
	cudaEvent_t start, stop, start_iter;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&start_iter);
	float milliseconds = 0;


	//****** Definicion del Grid
	//int numBlocks = k;
	int numThreads = k;
	//dim3 block(numBlocks);
	dim3 threads(numThreads);
	//*****************************

	
	//******* ZONA RANDOM ******
	curandState* devStates;
	float *devResults;
	// reservando espacio para los states PRNG en el device
	cudaMalloc(&devStates, k * sizeof(curandState));
	// reservando espacio para el vector de resultados en device
	cudaMalloc(&devResults, k * sizeof(float));
	// setup semillas
	setup_kernel<<<1, threads>>>(devStates, time(0));
	// generar números aleatorios
	generate<<<1, threads>>>(devStates, devResults);
	//*****************************
	
	cudaMemcpy (D_cluster_centroid, H_cluster_centroid, (k*dim*sizeof(float)), cudaMemcpyHostToDevice);
	cudaMemcpy (D_dataSetMatrix, H_X, (n*dim*sizeof(float)), cudaMemcpyHostToDevice);
	
	cudaEventRecord(start);
	random_init_centroidCUDA <<< 1, threads >>> (devResults, D_cluster_centroid, D_dataSetMatrix, k, n, dim);
	cudaMemcpy (H_cluster_centroid, D_cluster_centroid, (k*dim*sizeof(float)), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf ("CUDA-Time -- Random_Init_Centroid: %.5lf s\n", milliseconds/1000);

	
	// ********** Definicion del segundo Grid -- Por cada punto un hilo
	int nBloques = (int) ceil(n/256);
	dim3 bloques(nBloques);
	dim3 hilos(256);
	// ******************************
	
	cudaEventRecord(start);
	calc_all_distancesCUDA <<< bloques, hilos >>> (dim, n, k, D_dataSetMatrix, D_cluster_centroid, D_dist);
	/*cudaMemcpy (dist, D_dist,(sizeof(float) * n * k), cudaMemcpyDeviceToHost);
	for (int i=0; i<20; i+=2){
		printf("%f %f\n", dist[i], dist[i+1]);
	}*/
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf ("CUDA-Time -- Calc_All_Distances 1a vez: %.5lf s\n", milliseconds/1000);

	cudaEventRecord(start);
	choose_all_clusters_from_distancesCUDA <<< bloques, hilos >>> (n, k, D_dist, D_cluster_assignment_cur);
	/*for (int i=0; i<n; i++){
		printf("%d\n", cluster_assignment_cur[i]);
	}*/
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf ("CUDA-Time -- Choose_All_Clusters_From_Distances 1a vez: %.5lf s\n", milliseconds/1000);
	
	cudaMemcpy (cluster_assignment_cur, D_cluster_assignment_cur,(sizeof(int) * n), cudaMemcpyDeviceToHost);
    copy_assignment_arrayNoCUDA(n, cluster_assignment_cur, cluster_assignment_prev);
	
	
	calc_total_distanceCUDA <<< bloques, hilos >>> (dim, k, D_dataSetMatrix, D_cluster_centroid, D_cluster_assignment_cur, D_totD);
	cudaMemcpy (total, D_totD, (sizeof(float)), cudaMemcpyDeviceToHost);
	printf("CUDA -- Total Distance Cuda:\t %.5lf\n", total[0]);
	
	cudaEventRecord(start);
    double H_prev_totD = calc_total_distanceNoCUDA(dim, n, k, H_X, H_cluster_centroid, cluster_assignment_cur);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf ("CUDA-Time -- Calc_Total_Distance totD 1a vez: %.5lf s\n", milliseconds/1000);
	
	
	printf ("CUDA -- Total Distance:\t\t  %.5lf\n", H_prev_totD);

	// Etapa 3 y 4
	cudaEventRecord(start_iter);
	cudaEventRecord(start);
	for (int batch=0; (batch < iterations); ++batch) {

        // update cluster centroids. Update Step
				
		//cudaEventRecord(start);
		initialize_To_Zero <<< 1, threads >>> (dim, D_cluster_member_count, D_cluster_centroid);
		which_Is_In <<< bloques, hilos >>> (dim, D_cluster_assignment_cur, D_cluster_member_count, D_cluster_centroid, D_dataSetMatrix);
		calc_cluster_centroidsCUDA <<< 1, threads >>> (dim, D_cluster_member_count, D_cluster_centroid);
		if (batch == 1){
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&milliseconds, start, stop);
			printf("CUDA-Time -- Calc_Cluster_CentroIds: %.5lf s\n",milliseconds/1000 );
		}
		cudaMemcpy (H_cluster_centroid, D_cluster_centroid, (k*dim*sizeof(float)), cudaMemcpyDeviceToHost);

        double H_totD = calc_total_distanceNoCUDA(dim, n, k, H_X, H_cluster_centroid, cluster_assignment_cur);

        // see if we've failed to improve
        if (H_totD >= H_prev_totD){
            // failed to improve - currently solution worse than previous restore old assignments
            copy_assignment_arrayNoCUDA(n, cluster_assignment_prev, cluster_assignment_cur);

            // recalc centroids randomly
			// setup semillas
			setup_kernel<<<1, threads>>>(devStates, time(0));
			// generar números aleatorios
			generate<<<1, threads>>>(devStates, devResults);
			if (batch%2) {
				random_init_centroidCUDA <<< 1, threads >>> (devResults, D_cluster_centroid, D_dataSetMatrix, k, n, dim);
			}else{
				random_init_centroidCUDA_mil <<< 1, threads >>> (devResults, D_cluster_centroid, D_dataSetMatrix, k, n, dim);
			}
        }
        else { // We have made some improvements
            // save previous step
			copy_assignment_arrayNoCUDA(n, cluster_assignment_cur, cluster_assignment_prev);
			// move all points to nearest cluster
			calc_all_distancesCUDA <<< bloques, hilos >>> (dim, n, k, D_dataSetMatrix, D_cluster_centroid, D_dist);
			choose_all_clusters_from_distancesCUDA <<< bloques, hilos >>> (n, k, D_dist, D_cluster_assignment_cur);
			
			cudaMemcpy (cluster_assignment_cur, D_cluster_assignment_cur,(sizeof(int) * n), cudaMemcpyDeviceToHost);

            H_prev_totD = H_totD;
        }
		//printf ("CUDA -- Total Distance Mejorado: %.4lf\n", H_prev_totD);
    }
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start_iter, stop);
	printf ("CUDA -- Total Distance Mejorado:  %lf\n", H_prev_totD);
	printf ("CUDA-Time -- %d Iteraciones: %lf s\n\n", iterations, milliseconds/1000);


	// ********** LIBERADO DE MEMORIA **********
	
	free (dist);
	free (cluster_assignment_cur);
	free (cluster_assignment_prev);
	free (total);
	
	cudaFree (D_dist);
	cudaFree (D_cluster_assignment_cur);
	cudaFree (D_cluster_assignment_prev);
	cudaFree (D_cluster_centroid);
	cudaFree (D_dataSetMatrix);
	cudaFree (D_prev_totD);
	cudaFree (D_totD);
	
	cudaFree(devStates);
	cudaFree(devResults);

    cudaDeviceReset();
    return 0;

}






















