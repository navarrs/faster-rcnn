#!/bin/bash

for i in *.jpg; do convert -resize 320x320 $i $i; done
