#!/bin/bash

path="/home/inavarro/Desktop/keras/images/reduced_dataset/"

while IFS='' read -r line || [[ -n "$line" ]]; do 
	newstr="$path$line"
	echo $newstr
	echo  "$newstr"  >> new_paths.txt; 
done < "$1"
