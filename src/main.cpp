#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <map>
#include <algorithm>
#include <EdgeMesh.h>
#include <Timer.h>
#include <Slicer.h>
#include <locale>

bool isFileOBJ(std::string path) {
    std::string fileExtension = path.substr(path.find("."));
    
    if(fileExtension == ".obj") return true;
    if(fileExtension == ".Obj") return true;
    if(fileExtension == ".OBJ") return true;

    return false;
}

int main(int argc, char const *argv[]) {
    if (argc != 3) {
        std::cout << "usage: slicer file.obj dz" << std::endl;
        return 1;
    }

    if (!isFileOBJ(argv[1])) {
        std::cout << "error: input file does not have 'obj' extension" << std::endl;
        return 1;
    }

    double dz = atof(argv[2]);

    bool visualize = atoi(argv[3]);

    geom::Timer timer;
    FaceVertexMesh mesh{argv[1]};
    IndexedEdgeMesh geom{mesh};

    printf("loading model: %.2f seconds\n", timer.duration());

    Slicer().sliceGeometry(geom, dz);

    return 0;
    
}
