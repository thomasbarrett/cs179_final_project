#ifndef INDEXED_EDGE_MESH_H
#define INDEXED_EDGE_MESH_H

#include <string>
#include <array>
#include <vector>
#include <map>
#include <cmath>
#include <fstream>
#include <sstream>

/**
 * A face-vertex mesh representation is one of the simplest possible mesh
 * data structures, and is a common format for storing mesh data on disk.
 * The face-vertex mesh stores a table of faces and a table of vertices.
 */
class FaceVertexMesh {
public:
    using Triangle = std::array<uint32_t, 3>;
    using Vertex = std::array<float, 3>;
private:
    std::vector<Triangle> triangles_;
    std::vector<Vertex> vertices_;
public:

    /**
     * Construct a face-vertex mesh from a file found at the given path.
     * The file is assumed to be in wavefront OBJ format. Note that most
     * OBJ data, including normals, textures, and materials information
     * is ignored. Only connectivity information is considered.
     */
    FaceVertexMesh(const std::string &path) {
        std::ifstream file;
        file.open(path);
        std::stringstream ss;
        ss << file.rdbuf();
        std::string data = ss.str();
        uint32_t index = 0;
        uint32_t size = data.size();
        uint32_t last_progress = 0;

        auto next_word = [&]() {
            while (data[index] != ' ') {
                if (data[index] == '\0') return false;
                index++;
            }
            while (data[index] == ' ') {
                if (data[index] == '\0') return false;
                index++;
            }
            return true;
        };

        auto next_line = [&]() {
            while (data[index] != '\n' && data[index] != '\0') {
                index++;
            }
            if (data[index] == '\n') {
                index++;
                return true;
            } else {
                return false;
            }
        };

        do {

            if (data.substr(index, 2) == "v ") {
                next_word();
                float x = atof(&data[index]);
                next_word();
                float y = atof(&data[index]);
                next_word();
                float z = atof(&data[index]);
                vertices_.push_back({x, y, z});
            } else if (data.substr(index, 2) == "f ") {
                next_word();
                uint32_t x = atoi(&data[index]);
                next_word();
                uint32_t y = atoi(&data[index]);
                next_word();
                uint32_t z = atoi(&data[index]);
                triangles_.push_back({x - 1, y - 1, z - 1});

            }
        } while (next_line());
    }
    
    const std::vector<Triangle>& triangles() const { return triangles_; }
    const std::vector<Vertex>& vertices() const { return vertices_; }

    std::array<float, 2> minmax(const Triangle& t, int dim) const {
        float min = +INFINITY;
        float max = -INFINITY;
        for (int i = 0; i < 3; i++) {
            float v = vertices_[t[i]][dim];
            if (v < min) min = v;
            if (v > max) max = v;
        }
        return {min, max};
    }

    std::array<float, 2> minmax(int dim) const {
        float min = +INFINITY;
        float max = -INFINITY;
        for (auto &v_: vertices_) {
            float v = v_[dim];
            if (v < min) min = v;
            if (v > max) max = v;
        }
        return {min, max};
    }
};

/**
 * An indexed-edge mesh representation is a simplified winged-edge mesh that
 * stores three tables: triangles, edges, and vertices. Unlike a winged-edge
 * mesh, nearby connectivity information - the 'wings' of a winged edge mesh
 * are not stored explicitly.
 */
class IndexedEdgeMesh {
public:
    using Triangle = std::array<uint32_t, 3>;
    using Edge = std::array<uint32_t, 2>;
    using Vertex = std::array<float, 3>;
private:
    std::vector<Triangle> triangles_;
    std::vector<Edge> edges_;
    std::vector<Vertex> vertices_;
public:

    IndexedEdgeMesh(const FaceVertexMesh &mesh) {
        vertices_ = mesh.vertices();
        std::map<Edge, uint32_t> edge_map;
        for (auto &triangle: mesh.triangles()) {
            std::array<uint32_t, 3> edges;
            for (int i = 0; i < 3; i++) {

                uint32_t v1 = triangle[i];
                uint32_t v2 = triangle[(i + 1) % 3];
                uint32_t vmin = v1 < v2 ? v1: v2;
                uint32_t vmax = v1 < v2 ? v2: v1;

                Edge edge = {vmin, vmax};
                auto it = edge_map.find(edge);
                if (it == edge_map.end()) {
                    edges[i] = edge_map.size();
                    edge_map.emplace(edge, (uint32_t) edge_map.size());
                    edges_.push_back(edge);
                } else {
                    edges[i] = it->second;
                }   
            }
            triangles_.push_back(edges);
        }
        
    }

    const std::vector<Triangle>& triangles() const { return triangles_; }
    const std::vector<Edge>& edges() const { return edges_; }
    const std::vector<Vertex>& vertices() const { return vertices_; }

    std::array<float, 2> minmax(const Triangle& t, int dim) const {
        float min = +INFINITY;
        float max = -INFINITY;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                float v = vertices_[edges_[t[i]][j]][dim];
                if (v < min) min = v;
                if (v > max) max = v;
            }
        }
        return {min, max};
    }

    std::array<float, 2> minmax(int dim) const {
        float min = +INFINITY;
        float max = -INFINITY;
        for (auto &v_: vertices_) {
            float v = v_[dim];
            if (v < min) min = v;
            if (v > max) max = v;
        }
        return {min, max};
    }
};


#endif /* INDEXED_EDGE_MESH_H */