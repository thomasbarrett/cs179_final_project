CPP = g++
CPPFLAGS = -Iinclude -std=c++14 -O3

$(shell mkdir -p build)
$(shell mkdir -p img)

all: build/slicer_no_cairo build/slicer 

build/slicer: src/Slicer.cpp src/main.cpp
	$(CPP) $(CPPFLAGS) -DUSE_CAIRO=1 -o $@ src/Slicer.cpp src/main.cpp -lcairo

build/slicer_no_cairo: src/Slicer.cpp src/main.cpp
	$(CPP) $(CPPFLAGS) -DUSE_CAIRO=0 -o $@ src/Slicer.cpp src/main.cpp 

clean:
	rm -r build