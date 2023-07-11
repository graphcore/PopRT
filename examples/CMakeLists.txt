add_definitions(-DONNX_NAMESPACE=onnx)

add_executable(example_lightrunner example_lightrunner.cpp)
target_link_libraries(
  example_lightrunner PUBLIC model_runtime popart poprt_compiler poprt_runtime
                             spdlog)
target_compile_features(example_lightrunner PUBLIC cxx_std_17)
install(TARGETS example_lightrunner RUNTIME)
