#!/bin/bash

while IFS='' read -r line || [[ -n "$line" ]]; do  
	echo  "${line##/*/}"  >> wheelchair_names.txt; 
done < "$1"

