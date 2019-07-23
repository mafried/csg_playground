CXX = g++
CXXFLAGS = -msse -DGNU -g
INCLUDE_DIRS = -I.
PROGRAM = main
OBJECTS = Bitmap.o BitmapPrimitiveShape.o Candidate.o Cone.o ConePrimitiveShape.o \
ConePrimitiveShapeConstructor.o Cylinder.o CylinderPrimitiveShape.o CylinderPrimitiveShapeConstructor.o \
LowStretchSphereParametrization.o LowStretchTorusParametrization.o Plane.o PlanePrimitiveShape.o PlanePrimitiveShapeConstructor.o \
PointCloud.o RansacShapeDetector.o SimpleTorusParametrization.o Sphere.o \
SpherePrimitiveShape.o SpherePrimitiveShapeConstructor.o Torus.o TorusPrimitiveShape.o \
TorusPrimitiveShapeConstructor.o main.o solve.o

# implementation
.SUFFIXES: .o .cpp

.cpp.o:
	$(CXX) $(INCLUDE_DIRS) $(CXXFLAGS) -c -o $@ $<

Random.o:
	$(CXX) $(INCLUDE_DIRS) $(CXXFLAGS) -c MiscLib/Random.cpp

RefCount.o:
	$(CXX) $(INCLUDE_DIRS) $(CXXFLAGS) -c MiscLib/RefCount.cpp

all: $(PROGRAM)

$(PROGRAM): $(OBJECTS) Random.o RefCount.o
	$(CXX) -o $(PROGRAM) $(OBJECTS) Random.o RefCount.o

clean:
	rm -f *.o Random.o RefCount.o $(PROGRAM)
