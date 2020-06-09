
#ifndef SLICER_H
#define SLICER_H

#include <set>
#include <Progress.h>
#include <map>
#include <EdgeMesh.h>

class Slicer {
public:

    void sliceGeometry(const IndexedEdgeMesh &m, double dz);
    void sliceGeometryGPU(const IndexedEdgeMesh &m, double dz);

private:

   using Point = std::array<double, 2>;

    struct VertexData {
        Point position;
        uint32_t edge_count = 0;
        std::array<uint64_t, 2> edges; 
    };

    using Graph = std::map<uint64_t, VertexData>;
    using Polygon = std::vector<Point>;
    using Polygons = std::vector<Polygon>;

    std::vector<Slicer::Graph> graph_; 
    std::vector<Slicer::Polygons> polygons_;
    int slice_count;

    void sliceTriangles(
        const uint32_t *input_triangles, uint32_t input_triangle_count,
        const uint32_t *input_edges, uint32_t input_edge_count,
        const float *input_vertices, uint32_t input_vertex_count,
        uint32_t *output_edges, float *output_points, double z);
    
    void buildEdgeList(
        const uint32_t *input_triangles, uint32_t input_triangle_count,
        const uint32_t *input_edges, uint32_t input_edge_count,
        const float *input_vertices, uint32_t input_vertex_count,
        uint32_t *output_edges, float *output_points);
    
    void computeContours();
    void exportImages(const std::string &path_prefix);
};

#endif /* SLICER_H */