#include "File.h"



/**
 * Reserve dynamic memory for matrix in 1D fashion
 * @param double*** -  Matrix for reserve dynamic memory in two dimensions
 * @param int - Number of rows to allocate
 * @param int - Number of columns to allocate
 */
 void reserveDynamicMemoryForMatrix( float **matrix, int rows, int columns ){
    // Define variables
    int i;

    // Reserve dynamic memory
    if (( *matrix = ( float * ) malloc( rows * columns * sizeof( float ) )) == NULL )
    {
        printf("ERROR 'Main.c': First malloc() is fail.\n");
        exit(1);
    }
 }

/**
 * Set the data in the matrix
 * @param float* - Input data matrix
 * @param char[] - File name
 * @param int - number of rows
 * @param int - number of columns
 * return void
 */
void setDataInMatrix(float * matrix, char * fileName, int rows, int columns ){

    // Define variables
    int i, j;
    float value;
    FILE *fp;

    // Open file with the name passed by argument
    fp = fopen( fileName, "r" );
    if ( NULL == fp ){
        printf("File not found.\n");
        exit(1);
    }

    fseek (fp, sizeof(int)*2, SEEK_SET);

    // Explore file and assign the value read to array
    for ( i = 0; i < rows; i++ ) {
        for ( j = 0; j < columns; j++ ) {
            if (fscanf(fp, "%f", &value) != 1) {
                fprintf (stderr, "%s\n", "There was a problem reading the file\n");
                exit (-1);
            }
            matrix[i*columns+j] = value;
           // printf ("%f, ", matrix[i*columns+j]);
        }
    //    printf ("\n");
    }
    fclose(fp); // Close file
}

/**
 * Print the matrix values by console
 * @param double** - Input data matrix
 * @param int - number of rows
 * @param int - number of columns
 * return void
 */
void printMatrix( float *matrix, int rows, int columns ){

    int i, j;

    // Print values of each element of the matrix
    for ( i = 0; i < rows; i++ ) {
        for ( j = 0; j < columns; j++ ) {
            // %lf double
            printf( "%f ", matrix[i*columns+j] );
        }
        printf("\n");
    }
}





/**
 * Get and list the files in dataset directory
 */
/*void listFileNameInDirectory( char path[] ){
	DIR *dir;
	struct dirent *ent;

	if ((dir = opendir (path)) != NULL) {
		printf("List of available dataset '%s' :\n", path);
	  	while ((ent = readdir (dir)) != NULL) {
	  		if( strcmp(ent->d_name,".") && strcmp(ent->d_name,"..")  )
	    		printf ("%s\n", ent->d_name);
	  	}
	  	closedir (dir);
	  	printf("\n");
	} else {
	  	printf("ERROR: Directory in path %s not found.\n", path);
	  	exit(1);
	}
}
*/
/**
 * Get file name by console
 * @param char[]
 * return void
 */
/*void getFileName( char fileName[] ){
	// Define variable
	char str[256];

	// Get the fail in dataset directory
	listFileNameInDirectory(DATASET_DIRECTORY);
	// Request the input file by console
	printf( "Please type the input file name (*.txt):\n" );
    scanf( "%s", str );
    // Compose structure of file name
   	sprintf( fileName, "%s%s", DATASET_DIRECTORY, str );
}

*/
/**
 * Get size file
 * @param char[] - File name
 * @param int* - number of rows in the data set
 * @param int* - number of columns in the data set
 */
void getSizeFile( char * fileName, int * rows, int * columns ){

    int counter;
    FILE *fp;

    if (fileName==NULL)
        fprintf (stderr, "%s\n", "Error at getSizeFile");

    // Open file with the name passed by argument
    fp = fopen( fileName, "r" );
    if ( NULL == fp ){
        fprintf(stderr, "File not found.\n");
        exit(1);
    }

    if ((fscanf(fp, "%d", rows) != 1) || (fscanf (fp, "%d", columns) !=1)) {

        fprintf(stderr, "Error reading the file.\n");
        exit(1);
    }


    fclose(fp); // Close file
}

/**
 * Create file
 */
/*void openFile( FILE **fp, char fileName[], char mode[] ){
	*fp = fopen( fileName, mode );
    if ( !*fp ){
        printf( "ERROR: Path '%s' not found.\n", fileName );
        exit(1);
    }
}*/

/**
 * Record the data set of the matrix in a file
 * @param double** - Data set
 * @param char* - File name
 * @param int - Start loop
 * @param int - End loop
 */
/*void recordFile( double **dataSetMatrix, char fileName[], int startLoop, int endLoop ){
	int i,j;
	FILE *fp;

	// Create file
	fp = fopen( fileName, "wb" );
	// Explore a first third part of the array and assign it to the new file
	for ( i = startLoop; i < endLoop; i++ )
	{
		for ( j = 0; j < DIMENSION; j++ )
		{
			fprintf( fp, "%lf  ", dataSetMatrix[i][j] );
		}
		fprintf(fp,"\n");
	}
	printf( "%s\n", fileName );
	fclose(fp); // Close file

}
*/
