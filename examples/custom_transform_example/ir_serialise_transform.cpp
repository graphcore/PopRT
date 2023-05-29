// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <iostream>
#include <string>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/transforms/transform.hpp>

namespace popart {
class Graph;

class IrSerialise : public Transform {
public:
  static std::size_t id();

  IrSerialise() : Transform() {}
  virtual ~IrSerialise() override {}

  virtual bool apply(Graph &graph) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "IrSerialise"; }
};

std::size_t IrSerialise::id() { return typeid(IrSerialise).hash_code(); }

bool IrSerialise::apply(Graph &graph) const {
  const auto &ir = graph.getIr();
  std::stringstream ss;
  ir.serialise(Ir::SerialiseFormat::JSON, ss);
  const auto modelStr = ss.str();
  std::cout << "SerializedIr : " << std::endl;
  std::cout << modelStr << std::endl;
  return true;
}

namespace {
bool init = Transform::registerTransform(new IrSerialise);
}

} // namespace popart
