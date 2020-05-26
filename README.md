The Makefile has two targets:

1. slicer
2. slicer_no_cairo

The 'slicer' executable has the cairo library as a dependency for visualizing
slices of the model.

The 'slicer_no_cairo' does not have the cairo library as a dependency, but still
performs the slicing process

Running ./demo.sh gives the following output:

'''
loading model: 2.27 seconds
slicing triangles: 11.73 seconds
construcing graph: 0.30 seconds
connecting contours: 0.03 seconds

info: rasterizing images
[======================================================================] 100%
15.64 seconds
'''

Images containing all of the slices can be found in the 'img' folder