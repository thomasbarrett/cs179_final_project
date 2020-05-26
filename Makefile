CPP = g++
CPPFLAGS = -Iinclude -O3 -std=c++14 

$(shell mkdir -p build)
$(shell mkdir -p img)

all: build/slicer_no_cairo build/slicer 

build/slicer: src/Slicer.cpp src/main.cpp
	$(CPP) $(CPPFLAGS) -DUSE_CAIRO=1 -o $@ src/Slicer.cpp src/main.cpp -lcairo -pthread

build/slicer_no_cairo: src/Slicer.cpp src/main.cpp
	$(CPP) $(CPPFLAGS) -DUSE_CAIRO=0 -o $@ src/Slicer.cpp src/main.cpp -pthread

clean:
	rm -r build