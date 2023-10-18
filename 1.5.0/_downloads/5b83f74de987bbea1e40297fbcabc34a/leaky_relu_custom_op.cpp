// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

// This example demonstrates how to create a custom operator for PopART, in this
// case a Leaky ReLU op that returns `x` for any element `x >= 0` and `x *
// alpha` for any element `x < 0`, where `alpha` is provided as a scalar
// attribute to the operator.
#include <popart/operatoridentifier.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/ElementWise.hpp>
#include <popart/popx/opx.hpp>

namespace CustomOperators {
const popart::OperatorIdentifier LeakyReluId = {popart::Domain::ai_graphcore,
                                                "LeakyRelu",
                                                1};
} // namespace CustomOperators

class LeakyReluOp;
class LeakyReluOpx;

class LeakyReluOp : public popart::Op {
public:
  LeakyReluOp(const popart::OperatorIdentifier &_opid,
              float _alpha,
              const popart::Op::Settings &settings_)
      : popart::Op(_opid, settings_), alpha(_alpha) {}

  std::unique_ptr<Op> clone() const final {
    return std::make_unique<LeakyReluOp>(*this);
  }

  void setup() final { outInfo(0) = inInfo(0); }

  void appendAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendAttributes(os);
    os.appendAttribute("alpha", getAlpha());
  }

  void appendOutlineAttributes(popart::OpSerialiserBase &os) const override {
    Op::appendOutlineAttributes(os);
    os.appendAttribute("alpha", getAlpha());
  }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  bool requiresRandomSeed() const override { return false; }

  // Attributes
  float getAlpha() const { return alpha; }

private:
  float alpha;
};

namespace {
using popart::DataType;
using popart::OpDefinition;

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition
    leakyReluOpDef({OpDefinition::Inputs({{"input", T}}),
                    OpDefinition::Outputs({{"output", T}}),
                    OpDefinition::Attributes({{"alpha", {"*"}}})});

static popart::OpCreator<LeakyReluOp> leakyReluOpCreator(
    popart::OpDefinitions({{CustomOperators::LeakyReluId, leakyReluOpDef}}),
    [](const popart::OpCreatorInfo &info) {
      // default alpha is 10**(-2)
      float alpha = info.attributes.getAttribute<popart::Attributes::Float>(
          "alpha", 1e-2f);
      return std::make_unique<LeakyReluOp>(info.opid, alpha, info.settings);
    },
    true);
} // namespace

namespace pe = popops::expr;

class LeakyReluOpx : public popart::popx::Opx {
public:
  LeakyReluOpx(popart::Op *op, popart::popx::Devicex *devicex)
      : popart::popx::Opx(op, devicex) {
    verifyOp<LeakyReluOp>(op, {CustomOperators::LeakyReluId});
  }

  void grow(poplar::program::Sequence &prog) const final {

    auto op = getOp<LeakyReluOp>();

    poplar::Tensor input = getInTensor(0);

    float alpha = op.getAlpha();

    // x < 0.0f ? alpha * x : x
    auto expression = pe::Select(pe::Mul(pe::Const(alpha), pe::_1),
                                 pe::_1,
                                 pe::Lt(pe::_1, pe::Const(0.0f)));

    popops::mapInPlace(graph(),
                       expression,
                       {input},
                       prog,
                       debugContext("LeakyRelu"),
                       poplar::OptionFlags());

    setOutTensor(0, input);
  }
};

static popart::popx::OpxCreator<LeakyReluOpx>
    LeakyReluOpxCreator({CustomOperators::LeakyReluId});
