#!/bin/bash

path="/home/inavarro/Desktop/Workspace/ml/TensorFlow/wheelchair_files/ROI227_wc/"

while IFS='' read -r line || [[ -n "$line" ]]; do 
	newstr="$path$line"
	echo $newstr
	echo  "$newstr"  >> mod_paths_2.txt; 
done < "$1"
