The Makefile has two targets:

1. slicer
2. slicer_no_cairo

The 'slicer' executable has the cairo library as a dependency for visualizing
slices of the model.

The 'slicer_no_cairo' does not have the cairo library as a dependency, but still
performs the slicing process