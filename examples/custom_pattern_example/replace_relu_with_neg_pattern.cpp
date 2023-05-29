// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <string>

#include <popart/graph.hpp>
#include <popart/op/negate.hpp>
#include <popart/op/relu.hpp>
#include <popart/op/sqrt.hpp>
#include <popart/operators.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/patterns/patterns.hpp>

namespace popart {
class IArray;
class Tensor;
} // namespace popart

using namespace popart;

class ReplaceReluWithNeg : public PreAliasPattern {
public:
  bool matches(Op *op) const override { return op->isConvertibleTo<ReluOp>(); }

  std::vector<const Tensor *> touches(Op *) const override { return {}; }

  bool apply(Op *op) const override {
    std::cout << "Custom pattern ReplaceReluWithNeg applied in "
              << op->debugName() << std::endl;

    auto negOp = makeReplacementOpInIr(Onnx::Operators::Neg_6, op);

    auto inputId  = op->inId(ReluOp::getInIndex());
    auto outputId = op->outId(ReluOp::getOutIndex());
    op->disconnectAllInputs();
    op->disconnectAllOutputs();
    op->getGraph().eraseOp(op->id);

    negOp->connectInTensor(NegateOp::getInIndex(), inputId);
    negOp->connectOutTensor(NegateOp::getOutIndex(), outputId);
    negOp->setup();

    return true;
  }
};

namespace {
static PatternCreator<ReplaceReluWithNeg>
    ReplaceReluWithNegPatternCreator("ReplaceReluWithNeg",
                                     /* default enabled = */ false,
                                     /* default mandatory = */ false);
} // namespace
