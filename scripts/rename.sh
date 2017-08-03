#!/bin/bash
j=1;

echo "Filename: "

read filename

j=1

for i in *.jpg; 
	do
	mv "$i" $filename"$j".jpg; 
	let j=j+1;
done
