#!/bin/bash

python scalability.py -o npoints1.csv -t points
python scalability.py -o npoints2.csv -t points
python scalability.py -o npoints3.csv -t points
python scalability.py -o npoints4.csv -t points
python scalability.py -o npoints5.csv -t points


python scalability.py -o nfeatures1.csv -t features
python scalability.py -o nfeatures2.csv -t features
python scalability.py -o nfeatures3.csv -t features
python scalability.py -o nfeatures4.csv -t features
python scalability.py -o nfeatures5.csv -t features