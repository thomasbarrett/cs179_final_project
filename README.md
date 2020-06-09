# Motivation (7.5)

One of the first steps in a 3D printing pipeline is 'slicing'. A DLP 3D printer
works by iteratively projecting thin layers of a model into a resin with an
ultraviolet projector. I have implemented a slicer which, given a 3D model, computes
the contours intersecting with a slicing plane at uniform images and rasterizes it
into a grayscale image.

# High-level overview of algorithms (10)

The basic slicing algorithm works as follows for some slicing plane 'z'
1. Iterate through each triangle in the mesh.
        - Find intersections of triangle with slicing plane.
3. Use mesh connectivity information to connect 'edges' into full contours
4. Rasterize contours using even-odd coloring rule

One of the hardest parts of this algorithm is dealing with degerate cases.
While in most cases, two edges of the triangle will intersect the slicing plane,
other cases exist such as:

1. The triangle is parallel to slicing plane (all three edges are parallel to plane)
2. One edge is parallel to slicing plane
3. Only one vertex intersects slicing plane
4. One vertex and one edge intersects slicing plane

To handle the degenerate cases, I define an 'intersection element' as either
a vertex or an edge. All non-degenerate cases will have exactly two intersection
elements of some combination of vertices and edges. Note that the case of one edge
parallel to the slicing plane can be considered two vertices intersecting the slicing
plane. Care must also be taken to ensure that a contour edge is not double counted.
For instance, in the case that an edge lies flat on the slicing plane, two triangles
will intersect the plane with the shared edge.

# GPU optimizations and specifics (10)

One of the most important parts of implementing a GPU algorithm is the structure
and organizing of data. 

## Mesh Representation

I used a data structure that I called a 'indexed edge mesh' which constists of
a buffer containing a list of all unique triangles in the mesh, unique edges in
the mesh, and unique vertices in the mesh. Each triangle is a list of three
indices into the edge buffer. Each edge is a list of two indices into the vertex
buffer. Thus, it is easy to iterate through each triangle of a mesh and find the
unique edge indices and vertex indices that compose the triangle. Both edge and 
vertex indices may be needed, as explaining above. This is a contiguous memory
alternative to a more complex cpu-oriented data structure such as a half-edge mesh.

## Constant Size Output

Additionally, since a GPU is not well suited to dynamically sized memory such as
a std::vector, we must use the GPU to store two 'intersection element ids' for each
triangle, and use the CPU to iterate through and find all 'non-null' intersection
element ids after the slicing process is complete.

To uniquelly represent an 'intersection element' I give each unique vertex an index
'i_v' and each unique edge an index 'i_e'. The intersection index for vertex 'v' is
simply 'i_v', and the intersection index for edge 'e' is |V| + 'i_e'. Thus, each
vertex and each edge is given a unique index.

A simple solution on the CPU would be to define a map from intersect elements to 
intersection points. For instance, a map from each edge to the intersection of
the edge along the plane. However, this is not well suited to the GPU. However,
since we assign contiguous indices for all edges, we can represent our 'map' as
a single contiguous buffer whose whether the 'ith' point in the buffer represents
the intersection of edge 'i' with the slicing plane. Note also that the intersection
of a vertex with the slicing plane is simply the vertex itself, so there is no
need to make this buffer of size |V| + |E|. 

## Thread usage

In this algorithm, the representation of the input and output data is the most
complex part of porting to the CPU. With the data format defined appopriatelly,
each thread can be assigned to a single triangle. 

# Code structure (7.5) - description of where important files are

Header files are stored in the 'include' directory
Source files are stored in the 'src' directory
The executable is stored in the 'build' directory
The cpu output is stored in the 'img_cpu' directory
The gpu output is stored in the 'img_gpu' directory.

The four main files are:
1. EdgeMesh.h
2. Slicer.cpp
3. Slicer.cu
4. main.cpp

The EdgeMesh.h file contains an OBJ parser as well as useful mesh operations
such as finding the minimum and maximum points. Additionally, it defines two
mesh data structures which are used by the algorithm.

The Slicer.cpp file contains the bulk of the slicing algorithm as well as code
for connecting contours into an adjacency list graph and rasterizing the 
contours using the cairo library. Additionally, it contains a CPU implementation
of the slicing algorithm.

The Slicer.cu file contains the GPU accelerated version of the slicing algorithm
as described above.

The entry point is 'main.cpp'

# Instructions on how to run smaller components of the project (In addition to demo script) (2.5)

The Makefile has two targets:

1. slicer
2. slicer_no_cairo

The 'slicer' executable has the cairo library as a dependency for visualizing
slices of the model.

The 'slicer_no_cairo' does not have the cairo library as a dependency, but still
performs the slicing process

Both executables take two arguments:
1. The path to an OBJ file containing a closed, manifold mesh!
2. The thickness of the slices.

Currently, the only mesh provided is './Nefartiti.obj'.
The slice thickness used in the demo script is 0.25

# Provide code output when run on Titan, especially if it takes a while to run. (2.5)

Running ./demo.sh gives the following output:

'''
loading model: 2.02 seconds

slicing triangles (CPU): 10.87 seconds
construcing graph: 0.24 seconds
connecting contours: 0.03 seconds

info: rasterizing images
[======================================================================] 100%
5.25 seconds

slicing triangles (GPU): 0.96 seconds
construcing graph: 0.24 seconds
connecting contours: 0.02 seconds

info: rasterizing images
[======================================================================] 100%
5.30 seconds
'''

Note the the GPU has an 11.3 times speedup!

Images containing all of the slices can be found in the 'img_cpu' and 'img_gpu'
folder respectively.