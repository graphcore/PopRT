CXX ?= g++
CXXFLAGS = -std=c++14 -fPIC -g
LDLIBS = -shared -lpopart
ONNX_NAMESPACE = -DONNX_NAMESPACE=onnx

BUILD_DIR = build
SOURCES = ir_serialise_transform.cpp
TARGET = $(BUILD_DIR)/custom_transforms.so

all: create_build_dir ir_serialise_transform

.PHONY: create_build_dir
create_build_dir:
	mkdir -p $(BUILD_DIR)

ir_serialise_transform: ir_serialise_transform.cpp
	$(CXX) $(SOURCES)  $(LDLIBS) $(CXXFLAGS) $(ONNX_NAMESPACE) -o $(TARGET)

.PHONY: clean
clean:
	rm -rf  $(BUILD_DIR)
