#ifndef CUDA_TRANSPOSE_CUH
#define CUDA_TRANSPOSE_CUH

void CallSliceTrianglesKernel(
    const uint32_t *input_triangles, uint32_t input_triangle_count,
    const uint32_t *input_edges, uint32_t input_edge_count,
    const float *input_vertices, uint32_t input_vertex_count,
    uint32_t *output_edges, float *output_points, double z);

#endif
