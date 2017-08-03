#!/bin/bash

for i in *.jpg; 
	do
	# convert to grayscale
	convert $i -colorspace Gray $i

	# rotate 90 degrees
	# convert $i -rotate 90 $i

	# negative
	# convert $i -function Arcsin -1  $i

	# flip image
	convert -flip $i $i
done
