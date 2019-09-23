#ifndef __FILE_H
#define __FILE_H

#include <stdio.h>
#include <stdlib.h>

// Reserve dynamic memory for dataset matrix
void reserveDynamicMemoryForMatrix( float ** dataSetMatrix, int rows, int columns );
// Set data in the dataset matrix
void setDataInMatrix( float * dataSetMatrix, char * fileName, int rows, int columns );

void printMatrix (float * matrix, int rows, int columns);

//void listFileNameInDirectory( char path[] );
//void getFileName( char fileName[] );
void getSizeFile( char fileName[], int *rows, int * columns );
//void openFile( FILE **fp, char fileName[], char mode[] );
//void recordFile(double **dataSetMatrix, char fileName[], int startLoop, int endLoop );

#endif
