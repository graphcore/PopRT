// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#include <popart/builder.hpp>
#include <popart/datatype.hpp>
#include <popart/tensordebuginfo.hpp>
#include <popart/tensorinfo.hpp>
#include <popef/Model.hpp>
#include <stdint.h>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "poprt/compiler/compiler.hpp"
#include "poprt/executable.hpp"
#include "poprt/logging.hpp"
#include "poprt/runtime/device_manager.hpp"
#include "poprt/runtime/lightrunner.hpp"
#include "poprt/runtime/runtime_commons.hpp"
#include "poprt/runtime/runtime_config.hpp"

std::shared_ptr<popef::Model> createPopefModel() {
  auto builder = popart::Builder::create();
  auto i1      = builder->addInputTensor(
      popart::TensorInfo(popart::DataType::FLOAT, {1, 2, 3, 3}), "input.1");
  auto i2 = builder->addInputTensor(
      popart::TensorInfo(popart::DataType::FLOAT, {1, 2, 3, 3}), "input.2");
  auto o     = builder->aiOnnxOpset10().add({i1, i2}, "output");
  auto proto = builder->getModelProto();

  std::vector<std::string> outputs{o};
  auto exe = poprt::compiler::Compiler::compile(proto, outputs);
  return exe->getPopefModel();
}

int main() {
  using namespace poprt::runtime;

  auto popef = createPopefModel();
  auto conf  = LightRunnerConfig();

  auto dm     = DeviceManager();
  auto device = dm.getDevice(1);
  conf.replica_to_device.emplace(0, device);

  LightRunner modelRunner(popef, conf);
  auto inputs = modelRunner.getExecuteInputs();
  poprt::logging::err("inputs.size() = {}", inputs.size());
  auto outputs = modelRunner.getExecuteOutputs();
  poprt::logging::err("outputs.size() = {}", outputs.size());

  // Create memories by vectors and set the value to inputs
  std::vector<std::vector<float>> memories;
  int32_t dims = 1 * 2 * 3 * 3;
  for (int i = 0; i < 2; i++) {
    float value = float(i + 1);
    memories.push_back(std::vector<float>(dims, float(value)));
  }

  InputMemoryView in;
  OutputMemoryView out;
  in.emplace("input.1",
             ConstTensorMemoryView(&memories[0][0], dims * sizeof(float)));
  in.emplace("input.2",
             ConstTensorMemoryView(&memories[1][0], dims * sizeof(float)));

  TensorMemory out_t(dims * sizeof(float));
  out.emplace("Add:0", out_t.getView());

  float *out_ptr = reinterpret_cast<float *>(out_t.data.get());
  float sum0     = 0.0f;
  for (size_t i = 0; i < dims; i++) {
    sum0 += out_ptr[i];
  }
  poprt::logging::err("uninitialized memory, sum0: {}", sum0);

  // call execute to do the inference
  OutputFutureMemoryView futureMemory = modelRunner.executeAsync(in, out);
  for (const auto &[name, value] : futureMemory)
    value.wait();

  float expected = 3.0f * dims;
  float sum1     = 0.0f;
  for (size_t i = 0; i < dims; i++) {
    sum1 += out_ptr[i];
  }
  poprt::logging::err("sum1: {}, expected: {}", sum1, expected);
}
