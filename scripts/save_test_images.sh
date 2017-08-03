#!/bin/bash


path='/home/inavarro/Desktop/Workspace/ml/TensorFlow/keras-frcnn/images/test_images/'

while IFS='' read -r line || [[ -n "$line" ]]; do  
	#"$path${line##/*/}"; 
	cp $line $path${line##/*/}; 
done < "$1"

