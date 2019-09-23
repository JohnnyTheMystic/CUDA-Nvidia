#!/bin/bash

make clean; make


data=(`ls ./DataSet|sort -n`)
iter=( 250 500 1000 )
clusters=( 2 3 4 8 16 )
threads=( 0 1 8 16 32 )


for input in ${data[@]}
do
    echo Ejecutando Clasificacion para del fichero de datos: $input
    
    for i in ${iter[@]}
    do 
        for c in ${clusters[@]}
        do
			for proc in ${threads[@]}
			do
				echo Fichero:$input Iterations:$i Clusters $c Threads: $proc
				./run -f DataSet/$input -i $i -c $c -m $proc
			done
        done
    done
done 

