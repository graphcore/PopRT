add_definitions(-DONNX_NAMESPACE=onnx)

add_executable(example_lightrunner example_lightrunner.cpp)
target_link_libraries(
  example_lightrunner PUBLIC model_runtime popart-only poprt_compiler
                             poprt_runtime spdlog)
target_compile_features(example_lightrunner PUBLIC cxx_std_17)

add_executable(model_runner_example model_runner_example.cpp)
target_link_libraries(model_runner_example PUBLIC pvti pthread poprt_runtime)
target_compile_features(model_runner_example PUBLIC cxx_std_17)
