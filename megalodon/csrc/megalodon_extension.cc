#include <torch/torch.h>

#include "ops/attention.h"
#include "ops/attention_softmax.h"
#include "ops/ema_hidden.h"
#include "ops/ema_parameters.h"
#include "ops/fftconv.h"
#include "ops/sequence_norm.h"
#include "ops/timestep_norm.h"
#include "utils.h"

namespace megalodon {

PYBIND11_MODULE(megalodon_extension, m) {
  m.doc() = "Mega2 Cpp Extensions.";
  py::module m_ops = m.def_submodule("ops", "Submodule for custom ops.");
  ops::DefineAttentionOp(m_ops);
  ops::DefineAttentionSoftmaxOp(m_ops);
  ops::DefineEMAHiddenOp(m_ops);
  ops::DefineEMAParametersOp(m_ops);
  ops::DefineFFTConvOp(m_ops);
  ops::DefineSequenceNormOp(m_ops);
  ops::DefineTimestepNormOp(m_ops);
}

}  // namespace megalodon
