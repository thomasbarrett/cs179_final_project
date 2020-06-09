#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cassert>
#include <map>
#include <cmath>
#include <set>
#include <algorithm>
#include <stack>
#include <future>

#include <EdgeMesh.h>
#include <Slicer.h>
#include <Progress.h>
#include <Timer.h>

#include <cuda_runtime.h>
#include <transpose_device.cuh>

#if USE_CAIRO
#include <cairo/cairo.h>
#endif

void Slicer::sliceGeometry(const IndexedEdgeMesh &geometry, double dz) {

    auto min_max = geometry.minmax(2);
    float minz = min_max[0];
    float maxz = min_max[1];
    slice_count = (int)((maxz - minz) / dz) + 1;
    
    /*
     * The input to our algorithm is three buffers containing the mesh data.
     * 1. input_triangles: a list of three edge indices for each triangle
     * 2. input_edges: a list of two vertex indices for each edge
     * 3. input_vertices: a list of points in R^3 for each vertex
     */

    const uint32_t *input_triangles = (uint32_t *) geometry.triangles().data();
    uint32_t input_triangle_count = geometry.triangles().size();

    const uint32_t *input_edges = (uint32_t *) geometry.edges().data();
    uint32_t input_edge_count = geometry.edges().size();

    const float *input_vertices = (float *) geometry.vertices().data();
    uint32_t input_vertex_count = geometry.vertices().size();


    /**
     * We perform the following operations for each slicing plane:
     * 1. Intersect triangles with slicing plane
     * 2. Construct edge-list graph for contours
     * 3. Construct list of polygons from edge-list graph
     */

    float elapsed_slice_time = 0.0;
    float elapsed_graph_time = 0.0;

    for (int sidx = 0; sidx < slice_count; sidx++) {

        /*
         * This approach is valid for non-uniform slices, but for simplicity, 
         * we just consider uniform slices.
         */
        double z = sidx * dz + minz;

        /**
         * Our algorithm returns two buffers containing the intersection data
         * 1. output_edges: a list of two intersection indices for each triangle
         * 2. output_points: a list of points in R^3 for each edge.
         * 
         * Since a slicing plane could intersect an edge exactly at one of its
         * end points, we define an intersection to be either an edge or a vertex.
         * Since we assign a unique identifier to each edge and vertex, we can 
         * assign each intersection a unique identifier by offsetting the vertex
         * indices by the number of edges.
         * 
         * If intersection is an edge:
         * uint32_t id = edge_id
         * 
         * If intersection is a vertex:
         * uint32_t id = edge_count + vertex_id
         * 
         * Note that in the degenerate case of a vertex exactly intersecting the
         * slicing plane, its point of intersection would be the point itself.
         * Thus, there is no need for output_points to contain an entry for the
         * vertices.
         */

        uint32_t *output_edges = (uint32_t*) calloc(geometry.triangles().size(), 2 * sizeof(uint32_t));
        float *output_points = (float*) calloc(geometry.edges().size(), 2 * sizeof(float));

        geom::Timer timer_1;

        sliceTriangles(
            input_triangles, input_triangle_count,
            input_edges, input_edge_count,
            input_vertices, input_vertex_count,
            output_edges, output_points, z
        );

        elapsed_slice_time += timer_1.duration();

        geom::Timer timer_2;

        buildEdgeList(
            input_triangles, input_triangle_count,
            input_edges, input_edge_count,
            input_vertices, input_vertex_count,
            output_edges, output_points);

        elapsed_graph_time += timer_2.duration();

        free(output_edges);
        free(output_points);
    }
    printf("slicing triangles (CPU): %.2f seconds\n", elapsed_slice_time);
    printf("construcing graph: %.2f seconds\n", elapsed_graph_time);

    geom::Timer t2;
    computeContours();
    printf("connecting contours: %.2f seconds\n\n", t2.duration());

    #if USE_CAIRO

    geom::Timer t3;
    printf("info: rasterizing images\n");
    exportImages("img_cpu/slice");
    printf("%.2f seconds\n\n", t3.duration()); 

    #endif
}


void Slicer::sliceGeometryGPU(const IndexedEdgeMesh &geometry, double dz) {

    auto min_max = geometry.minmax(2);
    float minz = min_max[0];
    float maxz = min_max[1];
    slice_count = (int)((maxz - minz) / dz) + 1;
    
    /*
     * The input to our algorithm is three buffers containing the mesh data.
     * 1. input_triangles: a list of three edge indices for each triangle
     * 2. input_edges: a list of two vertex indices for each edge
     * 3. input_vertices: a list of points in R^3 for each vertex
     */

    const uint32_t *input_triangles = (uint32_t *) geometry.triangles().data();
    uint32_t input_triangle_count = geometry.triangles().size();

    const uint32_t *input_edges = (uint32_t *) geometry.edges().data();
    uint32_t input_edge_count = geometry.edges().size();

    const float *input_vertices = (float *) geometry.vertices().data();
    uint32_t input_vertex_count = geometry.vertices().size();

    uint32_t *output_edges = (uint32_t*) calloc(geometry.triangles().size(), 2 * sizeof(uint32_t));
    float *output_points = (float*) calloc(geometry.edges().size(), 2 * sizeof(float));

    uint32_t *gpu_input_triangles;
    uint32_t *gpu_input_edges;
    float *gpu_input_vertices;
    uint32_t *gpu_output_edges;
    float *gpu_output_points;


    /**
     * We perform the following operations for each slicing plane:
     * 1. Intersect triangles with slicing plane
     * 2. Construct edge-list graph for contours
     * 3. Construct list of polygons from edge-list graph
     */

    float elapsed_slice_time = 0.0;
    float elapsed_graph_time = 0.0;

    geom::Timer timer_1;

    cudaMalloc((void**)&gpu_input_triangles, sizeof (uint32_t) * 3 * input_triangle_count);
    cudaMemcpy(gpu_input_triangles, input_triangles, sizeof (uint32_t) * 3 * input_triangle_count, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&gpu_input_edges, sizeof (uint32_t) * 2 * input_edge_count);
    cudaMemcpy(gpu_input_edges, input_edges, sizeof (uint32_t) * 2 * input_edge_count, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&gpu_input_vertices, sizeof (float) * 3 * input_vertex_count);
    cudaMemcpy(gpu_input_vertices, input_vertices, sizeof (float) * 3 * input_vertex_count, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&gpu_output_edges, input_triangle_count * 2 * sizeof(uint32_t));
    cudaMalloc((void**)&gpu_output_points, input_edge_count * 2 * sizeof(float));

    elapsed_slice_time += timer_1.duration();

    for (int sidx = 0; sidx < slice_count; sidx++) {

        /*
         * This approach is valid for non-uniform slices, but for simplicity, 
         * we just consider uniform slices.
         */
        double z = sidx * dz + minz;

        /**
         * Our algorithm returns two buffers containing the intersection data
         * 1. output_edges: a list of two intersection indices for each triangle
         * 2. output_points: a list of points in R^3 for each edge.
         * 
         * Since a slicing plane could intersect an edge exactly at one of its
         * end points, we define an intersection to be either an edge or a vertex.
         * Since we assign a unique identifier to each edge and vertex, we can 
         * assign each intersection a unique identifier by offsetting the vertex
         * indices by the number of edges.
         * 
         * If intersection is an edge:
         * uint32_t id = edge_id
         * 
         * If intersection is a vertex:
         * uint32_t id = edge_count + vertex_id
         * 
         * Note that in the degenerate case of a vertex exactly intersecting the
         * slicing plane, its point of intersection would be the point itself.
         * Thus, there is no need for output_points to contain an entry for the
         * vertices.
         */

       

        geom::Timer timer_1;

        CallSliceTrianglesKernel(
            gpu_input_triangles, input_triangle_count,
            gpu_input_edges, input_edge_count,
            gpu_input_vertices, input_vertex_count,
            gpu_output_edges, gpu_output_points, z
        );

        cudaMemcpy(output_edges, gpu_output_edges, input_triangle_count * 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(output_points, gpu_output_points, input_edge_count * 2 * sizeof(float), cudaMemcpyDeviceToHost);
        
        elapsed_slice_time += timer_1.duration();

        geom::Timer timer_2;

        buildEdgeList(
            input_triangles, input_triangle_count,
            input_edges, input_edge_count,
            input_vertices, input_vertex_count,
            output_edges, output_points);

        elapsed_graph_time += timer_2.duration();


    }
    printf("slicing triangles (GPU): %.2f seconds\n", elapsed_slice_time);
    printf("construcing graph: %.2f seconds\n", elapsed_graph_time);

    geom::Timer t2;
    computeContours();
    printf("connecting contours: %.2f seconds\n\n", t2.duration());

    free(output_edges);
    free(output_points);
    
    cudaFree(gpu_input_triangles);
    cudaFree(gpu_input_edges);
    cudaFree(gpu_input_vertices);
    cudaFree(gpu_output_edges);
    cudaFree(gpu_output_points);

    #if USE_CAIRO

    geom::Timer t3;
    printf("info: rasterizing images\n");
    exportImages("img_gpu/slice");
    printf("%.2f seconds\n", t3.duration()); 

    #endif
}

/**
 * This is the primary function that will be translated into CUDA.
 * This code is already written in a style that will be easily translated
 * into cuda. It uses raw buffers and no complex data structures.
 * By assigning each thread to a single triangle, this code should directly
 * translate to cuda. 
 */
void Slicer::sliceTriangles(
    const uint32_t *input_triangles, uint32_t input_triangle_count,
    const uint32_t *input_edges, uint32_t input_edge_count,
    const float *input_vertices, uint32_t input_vertex_count,
    uint32_t *output_edges, float *output_points, double z) {
    
    for (int tidx = 0; tidx < input_triangle_count; tidx += 1) {

        const uint32_t *triangle = &input_triangles[3 * tidx];

        uint32_t intersection_count = 0;
        uint32_t intersections[2] = {(uint32_t)(-1), (uint32_t)(-1)};
        
        for (int eidx = 0; eidx < 3; eidx++) {

            const uint32_t *edge = &input_edges[2 * triangle[eidx]];

            int v1 = edge[0];
            int v2 = edge[1];
            
            const float *p1 = &input_vertices[3 * v1];
            const float *p2 = &input_vertices[3 * v2];

            const float *p_min = p1[2] < p2[2] ? p1: p2;
            const float *p_max = p1[2] < p2[2] ? p2: p1;
        
            if (p_min[2] == z && z == p_max[2]) continue;

            if (p1[2] == z) {

                uint32_t id = input_edge_count + v1;
                if (intersection_count > 0 && intersections[0] == id) continue;
                if (intersection_count > 1 && intersections[1] == id) continue;
                if (intersection_count < 2) {
                    intersections[intersection_count] = id;
                }
                intersection_count++;

            } else if (p2[2] == z) {
                
                uint32_t id = input_edge_count + v2;
                if (intersection_count > 0 && intersections[0] == id) continue;
                if (intersection_count > 1 && intersections[1] == id) continue;
                if (intersection_count < 2) {
                    intersections[intersection_count] = id;
                }
                intersection_count++;

            } else if (p_min[2] < z && z < p_max[2]) {
                
                uint32_t id = triangle[eidx];
                if (intersection_count > 0 && intersections[0] == id) continue;
                if (intersection_count > 1 && intersections[1] == id) continue;
                if (intersection_count < 2) {
                    intersections[intersection_count] = id;
                }
                intersection_count++;

                float s = (z - p_min[2]) / (p_max[2] - p_min[2]);
                output_points[2 * id] = p_min[0] + s * (p_max[0] - p_min[0]);
                output_points[2 * id + 1] = p_min[1] + s * (p_max[1] - p_min[1]);
            }
        }
        
        if (intersection_count == 2) {
            output_edges[2 * tidx] = intersections[0];
            output_edges[2 * tidx + 1] = intersections[1];
        } else {
            output_edges[2 * tidx] = (uint32_t) (-1);
            output_edges[2 * tidx + 1] = (uint32_t) (-1);
        }
                
    }

}


void Slicer::buildEdgeList(
    const uint32_t *input_triangles, uint32_t input_triangle_count,
    const uint32_t *input_edges, uint32_t input_edge_count,
    const float *input_vertices, uint32_t input_vertex_count,
    uint32_t *output_edges, float *output_points) {

    Slicer::Graph graph;

    for (int tidx = 0; tidx < input_triangle_count; tidx++) {
        
        uint32_t i0 = output_edges[2 * tidx];
        uint32_t i1 = output_edges[2 * tidx + 1];

        if (i0 != (uint32_t) (-1) || i1 != (uint32_t) (-1)) {

            const float *p0 = (i0 < input_edge_count) ? &output_points[2 * i0]: &input_vertices[3 * (i0 - input_edge_count)];
            const float *p1 = (i1 < input_edge_count) ? &output_points[2 * i1]: &input_vertices[3 * (i1 - input_edge_count)];

            auto &v0 = graph[i0];
            auto &v1 = graph[i1];

            v0.position[0] = p0[0];
            v0.position[1] = p0[1];
            v0.edges[v0.edge_count++] = i1;

            v1.position[0] = p1[0];
            v1.position[1] = p1[1];
            v1.edges[v1.edge_count++] = i0;
        }
    }
    graph_.push_back(graph);
}

void Slicer::computeContours() {

    polygons_.resize(graph_.size());

    std::vector<std::future<void>> workers;
    
    for (int i = 0; i < slice_count; i++) {
        
        workers.push_back(std::async([&](int i) {

            Slicer::Graph &graph = graph_[i];
            Slicer::Polygons &polygons = polygons_[i];

            std::set<uint64_t> visited;
            std::vector<uint64_t> polygon;
            std::vector<Point> polygon2;

            for (auto &entry: graph) {
                auto &a = entry.first;
                auto &b = entry.second;

                if (visited.find(a) != visited.end()) continue;
                polygon = {a, b.edges[0]};
                polygon2 = {b.position};
                do {
                    auto curr = polygon.back();
                    auto next = graph[curr];

                    polygon2.push_back(next.position);
                    if (next.edges[0] != polygon[polygon.size() - 2]) {
                        polygon.push_back(next.edges[0]);
                    } else {
                        polygon.push_back(next.edges[1]);
                    }
                    visited.insert(curr);
                } while (polygon.back() != polygon.front());
                            
                polygons.push_back(polygon2);
            }
        }, i));
    }

    for (int i = 0; i < workers.size(); i++) {
        workers[i].wait();
    }
}

#if USE_CAIRO
void Slicer::exportImages(const std::string &path_prefix) {

    cairo_surface_t *surface;
    cairo_t *cr;

    surface = cairo_image_surface_create(CAIRO_FORMAT_A8,1920,1080);


    ProgressBar progress;
    for (int i = 0; i < slice_count; i++) {

        cr = cairo_create(surface);

        cairo_set_operator (cr, CAIRO_OPERATOR_SOURCE);
        cairo_set_antialias(cr, CAIRO_ANTIALIAS_NONE);
        cairo_set_fill_rule(cr, CAIRO_FILL_RULE_EVEN_ODD);
        cairo_set_line_width(cr, 0);

        Polygons &polygons = polygons_[i];
        progress.update((float) i / slice_count);

        cairo_set_source_rgba(cr, 0, 0, 0, 0);
        cairo_paint(cr);
        cairo_translate(cr, 960, 540);
        cairo_scale(cr, 20, 20);
        cairo_set_antialias(cr, CAIRO_ANTIALIAS_NONE);
        cairo_set_fill_rule(cr, CAIRO_FILL_RULE_EVEN_ODD);

        for (const auto &polygon: polygons) {
            cairo_set_source_rgba(cr, 0, 0, 0, 1);
            cairo_set_line_width(cr, 0.05);
            cairo_move_to (cr, polygon[0][0], polygon[0][1]);
            for (const auto &point: polygon) {
                cairo_line_to(cr, point[0], point[1]);
            }
        }

        cairo_stroke_preserve(cr);
        cairo_fill(cr);

        std::string path = path_prefix + std::to_string(i) + ".png";
        cairo_surface_write_to_png(surface, path.c_str());
        cairo_destroy(cr);

    }
    progress.finish();

    cairo_surface_destroy(surface);
}
#endif 
