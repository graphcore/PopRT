CXX ?= g++
CXXFLAGS = -std=c++14 -fPIC -g
LDLIBS = -shared -lpopart
ONNX_NAMESPACE = -DONNX_NAMESPACE=onnx

BUILD_DIR = build
SOURCES = replace_relu_with_neg_pattern.cpp
TARGET = $(BUILD_DIR)/custom_patterns.so

all: create_build_dir replace_relu_with_neg_pattern

.PHONY: create_build_dir
create_build_dir:
	mkdir -p $(BUILD_DIR)

replace_relu_with_neg_pattern: replace_relu_with_neg_pattern.cpp
	$(CXX) $(SOURCES)  $(LDLIBS) $(CXXFLAGS) $(ONNX_NAMESPACE) -o $(TARGET)

.PHONY: clean
clean:
	rm -rf  $(BUILD_DIR)
