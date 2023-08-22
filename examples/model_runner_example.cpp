/**************************************************************************
 * Copyright (c) 2023 Graphcore Ltd. All rights reserved.
 *
 * Example to run PopEF with PopRT runtime C++ API.
 *
 * Compile command:
 * g++ --std=c++17 model_runner_example.cpp -o model_runner_example \
 *     -I/usr/local/lib/python3.8/dist-packages/poprt/include/ \
 *     -L/usr/local/lib/python3.8/dist-packages/poprt/lib \
 *     -lpvti -lpthread -lpoprt_runtime
 **************************************************************************/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <poprt/runtime/model_runner.hpp>
#include <poprt/runtime/runtime_config.hpp>
#include <numeric>
#include <thread>
#include <vector>
#include <pvti/pvti.hpp>

std::vector<std::vector<unsigned char>> memory;

/**
 * This function shows how to create inputs which will be used for inference
 * call. In the example, the memory is allocated according to the `inputDescs`,
 * which describes the names and sizes of the inputs. And in real case, you need
 * to implement this function to convert your application inputs into
 * `poprt::runtime::InputMemoryView`.
 */
poprt::runtime::InputMemoryView
applicationInputs(const std::vector<poprt::runtime::InputDesc> &inputDescs) {
  poprt::runtime::InputMemoryView inputs;
  for (const auto &desc : inputDescs) {
    memory.emplace_back(std::vector<unsigned char>(desc.sizeInBytes));
    inputs.emplace(desc.name,
                   poprt::runtime::ConstTensorMemoryView(memory.back().data(),
                                                         desc.sizeInBytes));
  }
  return inputs;
}

/**
 * This function shows how to create outputs which will be used for inference
 * call.  In the example, the memory is allocated according to the
 * `outputDescs`, which describes the names and sizes of the outputs. And in
 * real case, you need to implement this function to convert your application
 * outputs into `poprt::runtime::OutputMemoryView`.
 */
poprt::runtime::OutputMemoryView
applicationOutputs(const std::vector<poprt::runtime::OutputDesc> &outputDescs) {
  poprt::runtime::OutputMemoryView outputs;
  for (const auto &desc : outputDescs) {
    memory.emplace_back(std::vector<unsigned char>(desc.sizeInBytes));
    outputs.emplace(desc.name,
                    poprt::runtime::TensorMemoryView(memory.back().data(),
                                                     desc.sizeInBytes));
  }
  return outputs;
}

/**
 * This function shows how to run a PopEF and calculate the QPS and latency
 */
void runPopEF(const std::string &popefPath, int iteration, int num_threads) {
  poprt::runtime::RuntimeConfig config;
  config.timeoutNS                = std::chrono::milliseconds(5);
  config.ringBufferSizeMultiplier = 3;
  config.checkPackageHash         = false;
  config.validateIOParams         = false;

  // Create a ModelRunner, which is used to run the model
  poprt::runtime::ModelRunner modelRunner(popefPath, config);

  // Create inputs and outputs which will be used to run the model
  poprt::runtime::InputMemoryView inputs =
      applicationInputs(modelRunner.getExecuteInputs());
  poprt::runtime::OutputMemoryView outputs =
      applicationOutputs(modelRunner.getExecuteOutputs());

  // Make `iteration` be an integer multiple of `num_threads`
  if (iteration % num_threads)
    iteration = (iteration / num_threads + 1) * num_threads;
  std::vector<std::chrono::duration<double, std::milli>> latencies(iteration);

  // Run model with asynchronous API to warm up
  std::cout << "Warm up" << std::endl;
  std::vector<poprt::runtime::OutputFutureMemoryView> futures;
  for (int i = 0; i < 10; i++) {
    auto future = modelRunner.executeAsync(inputs, outputs);
    futures.push_back(future);
  }
  for (auto future : futures) {
    for (const auto &name_future_view : future) {
      auto &&[name, future_memory_view] = name_future_view;
      future_memory_view.wait();
    }
  }
  std::cout << "Warm up completed" << std::endl;

  // Run model with synchronous API in multiple threads
  std::cout << "Execute with sync API in multiple threads." << std::endl;
  std::vector<std::thread> threads;
  static pvti::TraceChannel _channel = {"run_popef_mt"};

  auto threads_begin = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_threads; i++) {
    threads.emplace_back(
        [&](int id) {
          int iteration_per_thread = iteration / num_threads;
          int iter_offset          = id * iteration_per_thread;
          int iter_end             = iter_offset + iteration_per_thread;

          for (int i = iter_offset; i < iter_end; i++) {
            pvti::Tracepoint::begin(&_channel, std::to_string(id));
            auto begin = std::chrono::high_resolution_clock::now();
            modelRunner.execute(inputs, outputs);
            auto end = std::chrono::high_resolution_clock::now();
            pvti::Tracepoint::end(&_channel, std::to_string(id));
            latencies[i] =
                std::chrono::duration<double, std::milli>(end - begin);
          }
        },
        i);
  }

  for (auto &thread : threads) {
    thread.join();
  }
  auto threads_end = std::chrono::high_resolution_clock::now();
  std::cout << "Execute done." << std::endl;

  // Calculate the QPS and latency
  std::sort(latencies.begin(), latencies.end());
  auto max_latency = std::max_element(latencies.begin(), latencies.end());
  auto min_latency = std::min_element(latencies.begin(), latencies.end());
  double average_latency =
      std::accumulate(latencies.begin(),
                      latencies.end(),
                      std::chrono::duration<double, std::milli>::zero())
          .count() /
      latencies.size();
  int p99_index    = std::round(0.99 * latencies.size()) - 1;
  auto p99_latency = latencies[p99_index];

  std::cout << "Max latency: " << std::fixed << std::setprecision(2)
            << max_latency->count() << " ms" << std::endl;
  std::cout << "Min latency: " << std::fixed << std::setprecision(2)
            << min_latency->count() << " ms" << std::endl;
  std::cout << "Average latency: " << std::fixed << std::setprecision(2)
            << average_latency << " ms" << std::endl;
  std::cout << "P99 latency: " << std::fixed << std::setprecision(2)
            << p99_latency.count() << " ms" << std::endl;

  auto ms =
      std::chrono::duration<double, std::milli>(threads_end - threads_begin)
          .count();

  auto inputDescs = modelRunner.getExecuteInputs();
  auto min        = std::min_element(inputDescs.begin(),
                              inputDescs.end(),
                              [](const poprt::runtime::InputDesc &a,
                                 const poprt::runtime::InputDesc &b) {
                                return a.shape.front() < b.shape.front();
                              });

  int64_t batchSize = (*min).shape.front();

  std::cout << "Iteration: " << iteration << std::endl;
  std::cout << "Batch size: " << batchSize << std::endl;
  std::cout << "Total running time in ms: " << std::fixed
            << std::setprecision(2) << ms << std::endl;
  std::cout << "Throughput: " << int(iteration * 1000 * batchSize / ms)
            << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: ./poprt_run $popef_path [$iteration] [num_threads]"
              << std::endl;
    return -1;
  }

  int iteration = 100;
  if (argc >= 3) {
    iteration = std::stoi(argv[2]);
  }

  int num_threads = 3;
  if (argc >= 4) {
    num_threads = std::stoi(argv[3]);
  }

  runPopEF(std::string(argv[1]), iteration, num_threads);
  return 0;
}
