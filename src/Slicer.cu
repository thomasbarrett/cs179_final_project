#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>
#include <Slicer.cuh>

__global__ void sliceTrianglesKernel(
    const uint32_t *input_triangles, uint32_t input_triangle_count,
    const uint32_t *input_edges, uint32_t input_edge_count,
    const float *input_vertices, uint32_t input_vertex_count,
    uint32_t *output_edges, float *output_points, double z) {
    
    uint tidx = blockIdx.x * blockDim.x + threadIdx.x;

    while (tidx < input_triangle_count) {

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
         
        tidx += blockDim.x * gridDim.x;

    }

}

void CallSliceTrianglesKernel(
    const uint32_t *input_triangles, uint32_t input_triangle_count,
    const uint32_t *input_edges, uint32_t input_edge_count,
    const float *input_vertices, uint32_t input_vertex_count,
    uint32_t *output_edges, float *output_points, double z) {

    uint32_t threads_per_block = 64;
    uint32_t blocks = (unsigned int) ceil(input_triangle_count / (float) threads_per_block);
    sliceTrianglesKernel<<<blocks, threads_per_block>>>(
        input_triangles, input_triangle_count,
        input_edges, input_edge_count,
        input_vertices, input_vertex_count,
        output_edges, output_points, z
    );

}
