#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// @generated from ../tools/autograd/templates/python_nn_functions.cpp

#include "torch/csrc/Device.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/python_nn_functions.h"
#include "torch/csrc/autograd/generated/python_return_types.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/autograd/utils/python_arg_parsing.h"
#include "torch/csrc/utils/pycfunction_helpers.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/structseq.h"
#include "torch/csrc/utils/tensor_memoryformats.h"

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_conv_depthwise2d.h>
#include <ATen/ops/_pad_circular.h>
#include <ATen/ops/_pad_enum.h>
#include <ATen/ops/_test_ambiguous_defaults.h>
#include <ATen/ops/_test_optional_filled_intlist.h>
#include <ATen/ops/_test_optional_floatlist.h>
#include <ATen/ops/_test_optional_intlist.h>
#include <ATen/ops/_test_string_default.h>
#include <ATen/ops/_test_warn_in_autograd.h>
#include <ATen/ops/_upsample_bicubic2d_aa.h>
#include <ATen/ops/_upsample_bilinear2d_aa.h>
#include <ATen/ops/_upsample_nearest_exact1d.h>
#include <ATen/ops/_upsample_nearest_exact2d.h>
#include <ATen/ops/_upsample_nearest_exact3d.h>
#include <ATen/ops/adaptive_avg_pool2d.h>
#include <ATen/ops/adaptive_avg_pool3d.h>
#include <ATen/ops/adaptive_max_pool2d.h>
#include <ATen/ops/adaptive_max_pool3d.h>
#include <ATen/ops/avg_pool2d.h>
#include <ATen/ops/avg_pool3d.h>
#include <ATen/ops/binary_cross_entropy.h>
#include <ATen/ops/col2im.h>
#include <ATen/ops/conv_depthwise3d.h>
#include <ATen/ops/cross_entropy_loss.h>
#include <ATen/ops/elu.h>
#include <ATen/ops/elu.h>
#include <ATen/ops/flatten_dense_tensors.h>
#include <ATen/ops/fractional_max_pool2d.h>
#include <ATen/ops/fractional_max_pool3d.h>
#include <ATen/ops/gelu.h>
#include <ATen/ops/gelu.h>
#include <ATen/ops/glu.h>
#include <ATen/ops/hardsigmoid.h>
#include <ATen/ops/hardsigmoid.h>
#include <ATen/ops/hardswish.h>
#include <ATen/ops/hardswish.h>
#include <ATen/ops/hardtanh.h>
#include <ATen/ops/hardtanh.h>
#include <ATen/ops/huber_loss.h>
#include <ATen/ops/im2col.h>
#include <ATen/ops/l1_loss.h>
#include <ATen/ops/leaky_relu.h>
#include <ATen/ops/leaky_relu.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/log_sigmoid.h>
#include <ATen/ops/max_pool2d_with_indices.h>
#include <ATen/ops/max_pool3d_with_indices.h>
#include <ATen/ops/max_unpool2d.h>
#include <ATen/ops/max_unpool3d.h>
#include <ATen/ops/mish.h>
#include <ATen/ops/mish.h>
#include <ATen/ops/mkldnn_linear.h>
#include <ATen/ops/mkldnn_reorder_conv2d_weight.h>
#include <ATen/ops/mkldnn_reorder_conv3d_weight.h>
#include <ATen/ops/mse_loss.h>
#include <ATen/ops/multi_margin_loss.h>
#include <ATen/ops/multilabel_margin_loss.h>
#include <ATen/ops/nll_loss.h>
#include <ATen/ops/nll_loss2d.h>
#include <ATen/ops/nll_loss_nd.h>
#include <ATen/ops/one_hot.h>
#include <ATen/ops/pad.h>
#include <ATen/ops/pad_sequence.h>
#include <ATen/ops/reflection_pad1d.h>
#include <ATen/ops/reflection_pad2d.h>
#include <ATen/ops/reflection_pad3d.h>
#include <ATen/ops/relu6.h>
#include <ATen/ops/relu6.h>
#include <ATen/ops/replication_pad1d.h>
#include <ATen/ops/replication_pad2d.h>
#include <ATen/ops/replication_pad3d.h>
#include <ATen/ops/rrelu_with_noise.h>
#include <ATen/ops/rrelu_with_noise.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <ATen/ops/silu.h>
#include <ATen/ops/silu.h>
#include <ATen/ops/slow_conv3d.h>
#include <ATen/ops/slow_conv_dilated2d.h>
#include <ATen/ops/slow_conv_dilated3d.h>
#include <ATen/ops/slow_conv_transpose2d.h>
#include <ATen/ops/slow_conv_transpose3d.h>
#include <ATen/ops/smooth_l1_loss.h>
#include <ATen/ops/soft_margin_loss.h>
#include <ATen/ops/softplus.h>
#include <ATen/ops/softshrink.h>
#include <ATen/ops/thnn_conv2d.h>
#include <ATen/ops/unflatten_dense_tensors.h>
#include <ATen/ops/upsample_bicubic2d.h>
#include <ATen/ops/upsample_bilinear2d.h>
#include <ATen/ops/upsample_linear1d.h>
#include <ATen/ops/upsample_nearest1d.h>
#include <ATen/ops/upsample_nearest2d.h>
#include <ATen/ops/upsample_nearest3d.h>
#include <ATen/ops/upsample_trilinear3d.h>
#endif

using at::Tensor;
using at::Scalar;
using at::MemoryFormat;
using at::Generator;
using at::IntArrayRef;
using at::ArrayRef;

using namespace torch::autograd::utils;

namespace torch::autograd {

static PyObject* THPNNVariableFunctionsModule = nullptr;

static PyObject * THPVariable__parse_to(PyObject* module, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "to(Device device=None, ScalarType dtype=None, bool non_blocking=False, bool copy=False, *, MemoryFormat? memory_format=None)",
    "to(ScalarType dtype, bool non_blocking=False, bool copy=False, *, MemoryFormat? memory_format=None)",
    "to(Tensor tensor, bool non_blocking=False, bool copy=False, *, MemoryFormat? memory_format=None)",
  });
  ParsedArgs<5> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.has_torch_function()) {
    return handle_torch_function(r, args, kwargs, THPNNVariableFunctionsModule, "torch.nn", "_parse_to");
  }
  auto parsed = parse_to_conversion(r, /*allow_copy*/ false); // we don't want copy for nn.Module.to
  auto& device = std::get<0>(parsed);
  auto& scalarType = std::get<1>(parsed);
  auto non_blocking = std::get<2>(parsed);
  auto opt_memory_format = std::get<4>(parsed);
  auto tuple = THPObjectPtr{PyTuple_New(4)};
  if (!tuple) throw python_error();
  if (device) {
    PyTuple_SET_ITEM(tuple.get(), 0, THPDevice_New(*device));
  } else {
    Py_INCREF(Py_None);
    PyTuple_SET_ITEM(tuple.get(), 0, Py_None);
  }
  if (scalarType) {
    PyTuple_SET_ITEM(tuple.get(), 1, Py_NewRef(torch::getTHPDtype(*scalarType)));
  } else {
    Py_INCREF(Py_None);
    PyTuple_SET_ITEM(tuple.get(), 1, Py_None);
  }
  PyTuple_SET_ITEM(tuple.get(), 2, torch::autograd::utils::wrap(non_blocking));
  if (opt_memory_format.has_value()) {
    PyTuple_SET_ITEM(tuple.get(), 3, Py_NewRef(torch::utils::getTHPMemoryFormat(opt_memory_format.value())));
  } else {
    Py_INCREF(Py_None);
    PyTuple_SET_ITEM(tuple.get(), 3, Py_None);
  }
  return tuple.release();
  END_HANDLE_TH_ERRORS
}

// generated forward declarations start here

static PyObject * THPVariable__conv_depthwise2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__pad_circular(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__pad_enum(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__test_ambiguous_defaults(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__test_optional_filled_intlist(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__test_optional_floatlist(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__test_optional_intlist(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__test_string_default(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__test_warn_in_autograd(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__upsample_bicubic2d_aa(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__upsample_bilinear2d_aa(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__upsample_nearest_exact1d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__upsample_nearest_exact2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__upsample_nearest_exact3d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_adaptive_avg_pool2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_adaptive_avg_pool3d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_adaptive_max_pool2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_adaptive_max_pool3d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_avg_pool2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_avg_pool3d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_binary_cross_entropy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_col2im(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_conv_depthwise3d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_cross_entropy_loss(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_elu(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_elu_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_flatten_dense_tensors(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fractional_max_pool2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fractional_max_pool3d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_gelu(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_gelu_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_glu(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_hardsigmoid(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_hardsigmoid_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_hardswish(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_hardswish_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_hardtanh(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_hardtanh_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_huber_loss(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_im2col(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_l1_loss(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_leaky_relu(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_leaky_relu_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linear(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_log_sigmoid(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_max_pool2d_with_indices(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_max_pool3d_with_indices(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_max_unpool2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_max_unpool3d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_mish(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_mish_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_mkldnn_linear(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_mkldnn_reorder_conv2d_weight(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_mkldnn_reorder_conv3d_weight(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_mse_loss(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_multi_margin_loss(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_multilabel_margin_loss(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_nll_loss(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_nll_loss2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_nll_loss_nd(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_one_hot(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_pad(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_pad_sequence(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_reflection_pad1d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_reflection_pad2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_reflection_pad3d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_relu6(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_relu6_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_replication_pad1d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_replication_pad2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_replication_pad3d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_rrelu_with_noise(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_rrelu_with_noise_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_scaled_dot_product_attention(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_silu(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_silu_(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_slow_conv3d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_slow_conv_dilated2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_slow_conv_dilated3d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_slow_conv_transpose2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_slow_conv_transpose3d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_smooth_l1_loss(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_soft_margin_loss(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_softplus(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_softshrink(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_thnn_conv2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_unflatten_dense_tensors(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_upsample_bicubic2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_upsample_bilinear2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_upsample_linear1d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_upsample_nearest1d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_upsample_nearest2d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_upsample_nearest3d(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_upsample_trilinear3d(PyObject* self_, PyObject* args, PyObject* kwargs);

static PyMethodDef nn_functions[] = {
  {"_parse_to", castPyCFunctionWithKeywords(THPVariable__parse_to),
    METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_conv_depthwise2d", castPyCFunctionWithKeywords(THPVariable__conv_depthwise2d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_pad_circular", castPyCFunctionWithKeywords(THPVariable__pad_circular), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_pad_enum", castPyCFunctionWithKeywords(THPVariable__pad_enum), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_test_ambiguous_defaults", castPyCFunctionWithKeywords(THPVariable__test_ambiguous_defaults), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_test_optional_filled_intlist", castPyCFunctionWithKeywords(THPVariable__test_optional_filled_intlist), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_test_optional_floatlist", castPyCFunctionWithKeywords(THPVariable__test_optional_floatlist), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_test_optional_intlist", castPyCFunctionWithKeywords(THPVariable__test_optional_intlist), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_test_string_default", castPyCFunctionWithKeywords(THPVariable__test_string_default), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_test_warn_in_autograd", castPyCFunctionWithKeywords(THPVariable__test_warn_in_autograd), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_upsample_bicubic2d_aa", castPyCFunctionWithKeywords(THPVariable__upsample_bicubic2d_aa), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_upsample_bilinear2d_aa", castPyCFunctionWithKeywords(THPVariable__upsample_bilinear2d_aa), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_upsample_nearest_exact1d", castPyCFunctionWithKeywords(THPVariable__upsample_nearest_exact1d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_upsample_nearest_exact2d", castPyCFunctionWithKeywords(THPVariable__upsample_nearest_exact2d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_upsample_nearest_exact3d", castPyCFunctionWithKeywords(THPVariable__upsample_nearest_exact3d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"adaptive_avg_pool2d", castPyCFunctionWithKeywords(THPVariable_adaptive_avg_pool2d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"adaptive_avg_pool3d", castPyCFunctionWithKeywords(THPVariable_adaptive_avg_pool3d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"adaptive_max_pool2d", castPyCFunctionWithKeywords(THPVariable_adaptive_max_pool2d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"adaptive_max_pool3d", castPyCFunctionWithKeywords(THPVariable_adaptive_max_pool3d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"avg_pool2d", castPyCFunctionWithKeywords(THPVariable_avg_pool2d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"avg_pool3d", castPyCFunctionWithKeywords(THPVariable_avg_pool3d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"binary_cross_entropy", castPyCFunctionWithKeywords(THPVariable_binary_cross_entropy), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"col2im", castPyCFunctionWithKeywords(THPVariable_col2im), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"conv_depthwise3d", castPyCFunctionWithKeywords(THPVariable_conv_depthwise3d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"cross_entropy_loss", castPyCFunctionWithKeywords(THPVariable_cross_entropy_loss), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"elu", castPyCFunctionWithKeywords(THPVariable_elu), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"elu_", castPyCFunctionWithKeywords(THPVariable_elu_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"flatten_dense_tensors", castPyCFunctionWithKeywords(THPVariable_flatten_dense_tensors), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"fractional_max_pool2d", castPyCFunctionWithKeywords(THPVariable_fractional_max_pool2d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"fractional_max_pool3d", castPyCFunctionWithKeywords(THPVariable_fractional_max_pool3d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"gelu", castPyCFunctionWithKeywords(THPVariable_gelu), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"gelu_", castPyCFunctionWithKeywords(THPVariable_gelu_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"glu", castPyCFunctionWithKeywords(THPVariable_glu), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"hardsigmoid", castPyCFunctionWithKeywords(THPVariable_hardsigmoid), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"hardsigmoid_", castPyCFunctionWithKeywords(THPVariable_hardsigmoid_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"hardswish", castPyCFunctionWithKeywords(THPVariable_hardswish), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"hardswish_", castPyCFunctionWithKeywords(THPVariable_hardswish_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"hardtanh", castPyCFunctionWithKeywords(THPVariable_hardtanh), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"hardtanh_", castPyCFunctionWithKeywords(THPVariable_hardtanh_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"huber_loss", castPyCFunctionWithKeywords(THPVariable_huber_loss), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"im2col", castPyCFunctionWithKeywords(THPVariable_im2col), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"l1_loss", castPyCFunctionWithKeywords(THPVariable_l1_loss), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"leaky_relu", castPyCFunctionWithKeywords(THPVariable_leaky_relu), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"leaky_relu_", castPyCFunctionWithKeywords(THPVariable_leaky_relu_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linear", castPyCFunctionWithKeywords(THPVariable_linear), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"log_sigmoid", castPyCFunctionWithKeywords(THPVariable_log_sigmoid), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"max_pool2d_with_indices", castPyCFunctionWithKeywords(THPVariable_max_pool2d_with_indices), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"max_pool3d_with_indices", castPyCFunctionWithKeywords(THPVariable_max_pool3d_with_indices), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"max_unpool2d", castPyCFunctionWithKeywords(THPVariable_max_unpool2d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"max_unpool3d", castPyCFunctionWithKeywords(THPVariable_max_unpool3d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"mish", castPyCFunctionWithKeywords(THPVariable_mish), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"mish_", castPyCFunctionWithKeywords(THPVariable_mish_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"mkldnn_linear", castPyCFunctionWithKeywords(THPVariable_mkldnn_linear), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"mkldnn_reorder_conv2d_weight", castPyCFunctionWithKeywords(THPVariable_mkldnn_reorder_conv2d_weight), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"mkldnn_reorder_conv3d_weight", castPyCFunctionWithKeywords(THPVariable_mkldnn_reorder_conv3d_weight), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"mse_loss", castPyCFunctionWithKeywords(THPVariable_mse_loss), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"multi_margin_loss", castPyCFunctionWithKeywords(THPVariable_multi_margin_loss), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"multilabel_margin_loss", castPyCFunctionWithKeywords(THPVariable_multilabel_margin_loss), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"nll_loss", castPyCFunctionWithKeywords(THPVariable_nll_loss), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"nll_loss2d", castPyCFunctionWithKeywords(THPVariable_nll_loss2d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"nll_loss_nd", castPyCFunctionWithKeywords(THPVariable_nll_loss_nd), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"one_hot", castPyCFunctionWithKeywords(THPVariable_one_hot), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"pad", castPyCFunctionWithKeywords(THPVariable_pad), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"pad_sequence", castPyCFunctionWithKeywords(THPVariable_pad_sequence), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"reflection_pad1d", castPyCFunctionWithKeywords(THPVariable_reflection_pad1d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"reflection_pad2d", castPyCFunctionWithKeywords(THPVariable_reflection_pad2d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"reflection_pad3d", castPyCFunctionWithKeywords(THPVariable_reflection_pad3d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"relu6", castPyCFunctionWithKeywords(THPVariable_relu6), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"relu6_", castPyCFunctionWithKeywords(THPVariable_relu6_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"replication_pad1d", castPyCFunctionWithKeywords(THPVariable_replication_pad1d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"replication_pad2d", castPyCFunctionWithKeywords(THPVariable_replication_pad2d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"replication_pad3d", castPyCFunctionWithKeywords(THPVariable_replication_pad3d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"rrelu_with_noise", castPyCFunctionWithKeywords(THPVariable_rrelu_with_noise), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"rrelu_with_noise_", castPyCFunctionWithKeywords(THPVariable_rrelu_with_noise_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"scaled_dot_product_attention", castPyCFunctionWithKeywords(THPVariable_scaled_dot_product_attention), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"silu", castPyCFunctionWithKeywords(THPVariable_silu), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"silu_", castPyCFunctionWithKeywords(THPVariable_silu_), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"slow_conv3d", castPyCFunctionWithKeywords(THPVariable_slow_conv3d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"slow_conv_dilated2d", castPyCFunctionWithKeywords(THPVariable_slow_conv_dilated2d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"slow_conv_dilated3d", castPyCFunctionWithKeywords(THPVariable_slow_conv_dilated3d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"slow_conv_transpose2d", castPyCFunctionWithKeywords(THPVariable_slow_conv_transpose2d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"slow_conv_transpose3d", castPyCFunctionWithKeywords(THPVariable_slow_conv_transpose3d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"smooth_l1_loss", castPyCFunctionWithKeywords(THPVariable_smooth_l1_loss), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"soft_margin_loss", castPyCFunctionWithKeywords(THPVariable_soft_margin_loss), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"softplus", castPyCFunctionWithKeywords(THPVariable_softplus), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"softshrink", castPyCFunctionWithKeywords(THPVariable_softshrink), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"thnn_conv2d", castPyCFunctionWithKeywords(THPVariable_thnn_conv2d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"unflatten_dense_tensors", castPyCFunctionWithKeywords(THPVariable_unflatten_dense_tensors), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"upsample_bicubic2d", castPyCFunctionWithKeywords(THPVariable_upsample_bicubic2d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"upsample_bilinear2d", castPyCFunctionWithKeywords(THPVariable_upsample_bilinear2d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"upsample_linear1d", castPyCFunctionWithKeywords(THPVariable_upsample_linear1d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"upsample_nearest1d", castPyCFunctionWithKeywords(THPVariable_upsample_nearest1d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"upsample_nearest2d", castPyCFunctionWithKeywords(THPVariable_upsample_nearest2d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"upsample_nearest3d", castPyCFunctionWithKeywords(THPVariable_upsample_nearest3d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"upsample_trilinear3d", castPyCFunctionWithKeywords(THPVariable_upsample_trilinear3d), METH_VARARGS | METH_KEYWORDS, nullptr},
  {nullptr}
};

void initNNFunctions(PyObject* module) {
  static struct PyModuleDef def = {
     PyModuleDef_HEAD_INIT,
     "torch._C._nn",
     nullptr,
     -1,
     nn_functions
  };
  PyObject* nn = PyModule_Create(&def);
  THPNNVariableFunctionsModule = nn;
  if (!nn) {
    throw python_error();
  }
  // steals a reference to nn
  if (PyModule_AddObject(module, "_nn", nn) != 0) {
    throw python_error();
  }
}

// generated methods start here

// _conv_depthwise2d
static PyObject * THPVariable__conv_depthwise2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_conv_depthwise2d(Tensor input, Tensor weight, SymIntArrayRef[2] kernel_size, Tensor? bias, SymIntArrayRef[2] stride, SymIntArrayRef[2] padding, SymIntArrayRef[2] dilation, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(7)) {
    // aten::_conv_depthwise2d(Tensor self, Tensor weight, SymInt[2] kernel_size, Tensor? bias, SymInt[2] stride, SymInt[2] padding, SymInt[2] dilation) -> Tensor
    
    auto dispatch__conv_depthwise2d = [](const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_conv_depthwise2d_symint(self, weight, kernel_size, bias, stride, padding, dilation);
    };
    return wrap(dispatch__conv_depthwise2d(_r.tensor(0), _r.tensor(1), _r.symintlist(2), _r.optionalTensor(3), _r.symintlist(4), _r.symintlist(5), _r.symintlist(6)));
  } else {
    // aten::_conv_depthwise2d.out(Tensor self, Tensor weight, SymInt[2] kernel_size, Tensor? bias, SymInt[2] stride, SymInt[2] padding, SymInt[2] dilation, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch__conv_depthwise2d_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::_conv_depthwise2d_symint_out(out, self, weight, kernel_size, bias, stride, padding, dilation);
    };
    return wrap(dispatch__conv_depthwise2d_out(_r.tensor(7), _r.tensor(0), _r.tensor(1), _r.symintlist(2), _r.optionalTensor(3), _r.symintlist(4), _r.symintlist(5), _r.symintlist(6)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _pad_circular
static PyObject * THPVariable__pad_circular(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_pad_circular(Tensor input, SymIntArrayRef pad)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::_pad_circular(Tensor self, SymInt[] pad) -> Tensor
  
  auto dispatch__pad_circular = [](const at::Tensor & self, c10::SymIntArrayRef pad) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_pad_circular_symint(self, pad);
  };
  return wrap(dispatch__pad_circular(_r.tensor(0), _r.symintlist(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _pad_enum
static PyObject * THPVariable__pad_enum(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_pad_enum(Tensor input, SymIntArrayRef pad, int64_t mode, double? value=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::_pad_enum(Tensor self, SymInt[] pad, int mode, float? value=None) -> Tensor
  
  auto dispatch__pad_enum = [](const at::Tensor & self, c10::SymIntArrayRef pad, int64_t mode, ::std::optional<double> value) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_pad_enum_symint(self, pad, mode, value);
  };
  return wrap(dispatch__pad_enum(_r.tensor(0), _r.symintlist(1), _r.toInt64(2), _r.toDoubleOptional(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _test_ambiguous_defaults
static PyObject * THPVariable__test_ambiguous_defaults(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_test_ambiguous_defaults(Tensor dummy, int64_t a=1, int64_t b=1)",
    "_test_ambiguous_defaults(Tensor dummy, int64_t a=2, c10::string_view b=\"2\")",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_test_ambiguous_defaults.a(Tensor dummy, int a=1, int b=1) -> Tensor
      
      auto dispatch__test_ambiguous_defaults = [](const at::Tensor & dummy, int64_t a, int64_t b) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::_test_ambiguous_defaults(dummy, a, b);
      };
      return wrap(dispatch__test_ambiguous_defaults(_r.tensor(0), _r.toInt64(1), _r.toInt64(2)));
    }
    case 1: {
      // aten::_test_ambiguous_defaults.b(Tensor dummy, int a=2, str b="2") -> Tensor
      
      auto dispatch__test_ambiguous_defaults = [](const at::Tensor & dummy, int64_t a, c10::string_view b) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::_test_ambiguous_defaults(dummy, a, b);
      };
      return wrap(dispatch__test_ambiguous_defaults(_r.tensor(0), _r.toInt64(1), _r.stringView(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _test_optional_filled_intlist
static PyObject * THPVariable__test_optional_filled_intlist(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_test_optional_filled_intlist(Tensor values, IntArrayRef[2]? addends)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::_test_optional_filled_intlist(Tensor values, int[2]? addends) -> Tensor
  
  auto dispatch__test_optional_filled_intlist = [](const at::Tensor & values, at::OptionalIntArrayRef addends) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_test_optional_filled_intlist(values, addends);
  };
  return wrap(dispatch__test_optional_filled_intlist(_r.tensor(0), _r.intlistOptional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _test_optional_floatlist
static PyObject * THPVariable__test_optional_floatlist(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_test_optional_floatlist(Tensor values, ArrayRef<double>? addends)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::_test_optional_floatlist(Tensor values, float[]? addends) -> Tensor
  
  auto dispatch__test_optional_floatlist = [](const at::Tensor & values, ::std::optional<at::ArrayRef<double>> addends) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_test_optional_floatlist(values, addends);
  };
  return wrap(dispatch__test_optional_floatlist(_r.tensor(0), _r.doublelistOptional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _test_optional_intlist
static PyObject * THPVariable__test_optional_intlist(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_test_optional_intlist(Tensor values, IntArrayRef? addends)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::_test_optional_intlist(Tensor values, int[]? addends) -> Tensor
  
  auto dispatch__test_optional_intlist = [](const at::Tensor & values, at::OptionalIntArrayRef addends) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_test_optional_intlist(values, addends);
  };
  return wrap(dispatch__test_optional_intlist(_r.tensor(0), _r.intlistOptional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _test_string_default
static PyObject * THPVariable__test_string_default(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_test_string_default(Tensor dummy, c10::string_view a=\"\\\"'\\\\\", c10::string_view b=\"\\\"'\\\\\")",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::_test_string_default(Tensor dummy, str a="\"'\\", str b='"\'\\') -> Tensor
  
  auto dispatch__test_string_default = [](const at::Tensor & dummy, c10::string_view a, c10::string_view b) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_test_string_default(dummy, a, b);
  };
  return wrap(dispatch__test_string_default(_r.tensor(0), _r.stringView(1), _r.stringView(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _test_warn_in_autograd
static PyObject * THPVariable__test_warn_in_autograd(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_test_warn_in_autograd(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::_test_warn_in_autograd(Tensor self) -> Tensor
  
  auto dispatch__test_warn_in_autograd = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_test_warn_in_autograd(self);
  };
  return wrap(dispatch__test_warn_in_autograd(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _upsample_bicubic2d_aa
static PyObject * THPVariable__upsample_bicubic2d_aa(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_upsample_bicubic2d_aa(Tensor input, SymIntArrayRef? output_size, bool align_corners, ArrayRef<double>? scale_factors)",
    "_upsample_bicubic2d_aa(Tensor input, SymIntArrayRef[2] output_size, bool align_corners, double? scales_h=None, double? scales_w=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_upsample_bicubic2d_aa.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
      
      auto dispatch__upsample_bicubic2d_aa = [](const at::Tensor & input, at::OptionalSymIntArrayRef output_size, bool align_corners, ::std::optional<at::ArrayRef<double>> scale_factors) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::_upsample_bicubic2d_aa_symint(input, output_size, align_corners, scale_factors);
      };
      return wrap(dispatch__upsample_bicubic2d_aa(_r.tensor(0), _r.symintlistOptional(1), _r.toBool(2), _r.doublelistOptional(3)));
    }
    case 1: {
      if (_r.isNone(5)) {
        // aten::_upsample_bicubic2d_aa(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
        
        auto dispatch__upsample_bicubic2d_aa = [](const at::Tensor & self, c10::SymIntArrayRef output_size, bool align_corners, ::std::optional<double> scales_h, ::std::optional<double> scales_w) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::_upsample_bicubic2d_aa_symint(self, output_size, align_corners, scales_h, scales_w);
        };
        return wrap(dispatch__upsample_bicubic2d_aa(_r.tensor(0), _r.symintlist(1), _r.toBool(2), _r.toDoubleOptional(3), _r.toDoubleOptional(4)));
      } else {
        // aten::_upsample_bicubic2d_aa.out(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch__upsample_bicubic2d_aa_out = [](at::Tensor out, const at::Tensor & self, c10::SymIntArrayRef output_size, bool align_corners, ::std::optional<double> scales_h, ::std::optional<double> scales_w) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::_upsample_bicubic2d_aa_symint_out(out, self, output_size, align_corners, scales_h, scales_w);
        };
        return wrap(dispatch__upsample_bicubic2d_aa_out(_r.tensor(5), _r.tensor(0), _r.symintlist(1), _r.toBool(2), _r.toDoubleOptional(3), _r.toDoubleOptional(4)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _upsample_bilinear2d_aa
static PyObject * THPVariable__upsample_bilinear2d_aa(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_upsample_bilinear2d_aa(Tensor input, SymIntArrayRef? output_size, bool align_corners, ArrayRef<double>? scale_factors)",
    "_upsample_bilinear2d_aa(Tensor input, SymIntArrayRef[2] output_size, bool align_corners, double? scales_h=None, double? scales_w=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_upsample_bilinear2d_aa.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
      
      auto dispatch__upsample_bilinear2d_aa = [](const at::Tensor & input, at::OptionalSymIntArrayRef output_size, bool align_corners, ::std::optional<at::ArrayRef<double>> scale_factors) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::_upsample_bilinear2d_aa_symint(input, output_size, align_corners, scale_factors);
      };
      return wrap(dispatch__upsample_bilinear2d_aa(_r.tensor(0), _r.symintlistOptional(1), _r.toBool(2), _r.doublelistOptional(3)));
    }
    case 1: {
      if (_r.isNone(5)) {
        // aten::_upsample_bilinear2d_aa(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
        
        auto dispatch__upsample_bilinear2d_aa = [](const at::Tensor & self, c10::SymIntArrayRef output_size, bool align_corners, ::std::optional<double> scales_h, ::std::optional<double> scales_w) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::_upsample_bilinear2d_aa_symint(self, output_size, align_corners, scales_h, scales_w);
        };
        return wrap(dispatch__upsample_bilinear2d_aa(_r.tensor(0), _r.symintlist(1), _r.toBool(2), _r.toDoubleOptional(3), _r.toDoubleOptional(4)));
      } else {
        // aten::_upsample_bilinear2d_aa.out(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch__upsample_bilinear2d_aa_out = [](at::Tensor out, const at::Tensor & self, c10::SymIntArrayRef output_size, bool align_corners, ::std::optional<double> scales_h, ::std::optional<double> scales_w) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::_upsample_bilinear2d_aa_symint_out(out, self, output_size, align_corners, scales_h, scales_w);
        };
        return wrap(dispatch__upsample_bilinear2d_aa_out(_r.tensor(5), _r.tensor(0), _r.symintlist(1), _r.toBool(2), _r.toDoubleOptional(3), _r.toDoubleOptional(4)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _upsample_nearest_exact1d
static PyObject * THPVariable__upsample_nearest_exact1d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_upsample_nearest_exact1d(Tensor input, SymIntArrayRef? output_size, ArrayRef<double>? scale_factors)",
    "_upsample_nearest_exact1d(Tensor input, SymIntArrayRef[1] output_size, double? scales=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_upsample_nearest_exact1d.vec(Tensor input, SymInt[]? output_size, float[]? scale_factors) -> Tensor
      
      auto dispatch__upsample_nearest_exact1d = [](const at::Tensor & input, at::OptionalSymIntArrayRef output_size, ::std::optional<at::ArrayRef<double>> scale_factors) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::_upsample_nearest_exact1d_symint(input, output_size, scale_factors);
      };
      return wrap(dispatch__upsample_nearest_exact1d(_r.tensor(0), _r.symintlistOptional(1), _r.doublelistOptional(2)));
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::_upsample_nearest_exact1d(Tensor self, SymInt[1] output_size, float? scales=None) -> Tensor
        
        auto dispatch__upsample_nearest_exact1d = [](const at::Tensor & self, c10::SymIntArrayRef output_size, ::std::optional<double> scales) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::_upsample_nearest_exact1d_symint(self, output_size, scales);
        };
        return wrap(dispatch__upsample_nearest_exact1d(_r.tensor(0), _r.symintlist(1), _r.toDoubleOptional(2)));
      } else {
        // aten::_upsample_nearest_exact1d.out(Tensor self, SymInt[1] output_size, float? scales=None, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch__upsample_nearest_exact1d_out = [](at::Tensor out, const at::Tensor & self, c10::SymIntArrayRef output_size, ::std::optional<double> scales) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::_upsample_nearest_exact1d_symint_out(out, self, output_size, scales);
        };
        return wrap(dispatch__upsample_nearest_exact1d_out(_r.tensor(3), _r.tensor(0), _r.symintlist(1), _r.toDoubleOptional(2)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _upsample_nearest_exact2d
static PyObject * THPVariable__upsample_nearest_exact2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_upsample_nearest_exact2d(Tensor input, SymIntArrayRef? output_size, ArrayRef<double>? scale_factors)",
    "_upsample_nearest_exact2d(Tensor input, SymIntArrayRef[2] output_size, double? scales_h=None, double? scales_w=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_upsample_nearest_exact2d.vec(Tensor input, SymInt[]? output_size, float[]? scale_factors) -> Tensor
      
      auto dispatch__upsample_nearest_exact2d = [](const at::Tensor & input, at::OptionalSymIntArrayRef output_size, ::std::optional<at::ArrayRef<double>> scale_factors) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::_upsample_nearest_exact2d_symint(input, output_size, scale_factors);
      };
      return wrap(dispatch__upsample_nearest_exact2d(_r.tensor(0), _r.symintlistOptional(1), _r.doublelistOptional(2)));
    }
    case 1: {
      if (_r.isNone(4)) {
        // aten::_upsample_nearest_exact2d(Tensor self, SymInt[2] output_size, float? scales_h=None, float? scales_w=None) -> Tensor
        
        auto dispatch__upsample_nearest_exact2d = [](const at::Tensor & self, c10::SymIntArrayRef output_size, ::std::optional<double> scales_h, ::std::optional<double> scales_w) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::_upsample_nearest_exact2d_symint(self, output_size, scales_h, scales_w);
        };
        return wrap(dispatch__upsample_nearest_exact2d(_r.tensor(0), _r.symintlist(1), _r.toDoubleOptional(2), _r.toDoubleOptional(3)));
      } else {
        // aten::_upsample_nearest_exact2d.out(Tensor self, SymInt[2] output_size, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch__upsample_nearest_exact2d_out = [](at::Tensor out, const at::Tensor & self, c10::SymIntArrayRef output_size, ::std::optional<double> scales_h, ::std::optional<double> scales_w) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::_upsample_nearest_exact2d_symint_out(out, self, output_size, scales_h, scales_w);
        };
        return wrap(dispatch__upsample_nearest_exact2d_out(_r.tensor(4), _r.tensor(0), _r.symintlist(1), _r.toDoubleOptional(2), _r.toDoubleOptional(3)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _upsample_nearest_exact3d
static PyObject * THPVariable__upsample_nearest_exact3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_upsample_nearest_exact3d(Tensor input, SymIntArrayRef? output_size, ArrayRef<double>? scale_factors)",
    "_upsample_nearest_exact3d(Tensor input, SymIntArrayRef[3] output_size, double? scales_d=None, double? scales_h=None, double? scales_w=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_upsample_nearest_exact3d.vec(Tensor input, SymInt[]? output_size, float[]? scale_factors) -> Tensor
      
      auto dispatch__upsample_nearest_exact3d = [](const at::Tensor & input, at::OptionalSymIntArrayRef output_size, ::std::optional<at::ArrayRef<double>> scale_factors) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::_upsample_nearest_exact3d_symint(input, output_size, scale_factors);
      };
      return wrap(dispatch__upsample_nearest_exact3d(_r.tensor(0), _r.symintlistOptional(1), _r.doublelistOptional(2)));
    }
    case 1: {
      if (_r.isNone(5)) {
        // aten::_upsample_nearest_exact3d(Tensor self, SymInt[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor
        
        auto dispatch__upsample_nearest_exact3d = [](const at::Tensor & self, c10::SymIntArrayRef output_size, ::std::optional<double> scales_d, ::std::optional<double> scales_h, ::std::optional<double> scales_w) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::_upsample_nearest_exact3d_symint(self, output_size, scales_d, scales_h, scales_w);
        };
        return wrap(dispatch__upsample_nearest_exact3d(_r.tensor(0), _r.symintlist(1), _r.toDoubleOptional(2), _r.toDoubleOptional(3), _r.toDoubleOptional(4)));
      } else {
        // aten::_upsample_nearest_exact3d.out(Tensor self, SymInt[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch__upsample_nearest_exact3d_out = [](at::Tensor out, const at::Tensor & self, c10::SymIntArrayRef output_size, ::std::optional<double> scales_d, ::std::optional<double> scales_h, ::std::optional<double> scales_w) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::_upsample_nearest_exact3d_symint_out(out, self, output_size, scales_d, scales_h, scales_w);
        };
        return wrap(dispatch__upsample_nearest_exact3d_out(_r.tensor(5), _r.tensor(0), _r.symintlist(1), _r.toDoubleOptional(2), _r.toDoubleOptional(3), _r.toDoubleOptional(4)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// adaptive_avg_pool2d
static PyObject * THPVariable_adaptive_avg_pool2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "adaptive_avg_pool2d(Tensor input, SymIntArrayRef[2] output_size, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(2)) {
    // aten::adaptive_avg_pool2d(Tensor self, SymInt[2] output_size) -> Tensor
    
    auto dispatch_adaptive_avg_pool2d = [](const at::Tensor & self, c10::SymIntArrayRef output_size) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::adaptive_avg_pool2d_symint(self, output_size);
    };
    return wrap(dispatch_adaptive_avg_pool2d(_r.tensor(0), _r.symintlist(1)));
  } else {
    // aten::adaptive_avg_pool2d.out(Tensor self, SymInt[2] output_size, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_adaptive_avg_pool2d_out = [](at::Tensor out, const at::Tensor & self, c10::SymIntArrayRef output_size) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::adaptive_avg_pool2d_symint_out(out, self, output_size);
    };
    return wrap(dispatch_adaptive_avg_pool2d_out(_r.tensor(2), _r.tensor(0), _r.symintlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// adaptive_avg_pool3d
static PyObject * THPVariable_adaptive_avg_pool3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "adaptive_avg_pool3d(Tensor input, SymIntArrayRef[3] output_size, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(2)) {
    // aten::adaptive_avg_pool3d(Tensor self, SymInt[3] output_size) -> Tensor
    
    auto dispatch_adaptive_avg_pool3d = [](const at::Tensor & self, c10::SymIntArrayRef output_size) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::adaptive_avg_pool3d_symint(self, output_size);
    };
    return wrap(dispatch_adaptive_avg_pool3d(_r.tensor(0), _r.symintlist(1)));
  } else {
    // aten::adaptive_avg_pool3d.out(Tensor self, SymInt[3] output_size, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_adaptive_avg_pool3d_out = [](at::Tensor out, const at::Tensor & self, c10::SymIntArrayRef output_size) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::adaptive_avg_pool3d_symint_out(out, self, output_size);
    };
    return wrap(dispatch_adaptive_avg_pool3d_out(_r.tensor(2), _r.tensor(0), _r.symintlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// adaptive_max_pool2d
static PyObject * THPVariable_adaptive_max_pool2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "adaptive_max_pool2d(Tensor input, IntArrayRef[2] output_size, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(2)) {
    // aten::adaptive_max_pool2d(Tensor self, int[2] output_size) -> (Tensor, Tensor)
    
    auto dispatch_adaptive_max_pool2d = [](const at::Tensor & self, at::IntArrayRef output_size) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::adaptive_max_pool2d(self, output_size);
    };
    return wrap(dispatch_adaptive_max_pool2d(_r.tensor(0), _r.intlist(1)));
  } else {
    // aten::adaptive_max_pool2d.out(Tensor self, int[2] output_size, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
    auto out = _r.tensorlist_n<2>(2);
    auto dispatch_adaptive_max_pool2d_out = [](at::Tensor & out, at::Tensor & indices, const at::Tensor & self, at::IntArrayRef output_size) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::adaptive_max_pool2d_out(out, indices, self, output_size);
    };
    return wrap(dispatch_adaptive_max_pool2d_out(out[0], out[1], _r.tensor(0), _r.intlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// adaptive_max_pool3d
static PyObject * THPVariable_adaptive_max_pool3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "adaptive_max_pool3d(Tensor input, IntArrayRef[3] output_size, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(2)) {
    // aten::adaptive_max_pool3d(Tensor self, int[3] output_size) -> (Tensor, Tensor)
    
    auto dispatch_adaptive_max_pool3d = [](const at::Tensor & self, at::IntArrayRef output_size) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::adaptive_max_pool3d(self, output_size);
    };
    return wrap(dispatch_adaptive_max_pool3d(_r.tensor(0), _r.intlist(1)));
  } else {
    // aten::adaptive_max_pool3d.out(Tensor self, int[3] output_size, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
    auto out = _r.tensorlist_n<2>(2);
    auto dispatch_adaptive_max_pool3d_out = [](at::Tensor & out, at::Tensor & indices, const at::Tensor & self, at::IntArrayRef output_size) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::adaptive_max_pool3d_out(out, indices, self, output_size);
    };
    return wrap(dispatch_adaptive_max_pool3d_out(out[0], out[1], _r.tensor(0), _r.intlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// avg_pool2d
static PyObject * THPVariable_avg_pool2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "avg_pool2d(Tensor input, IntArrayRef[2] kernel_size, IntArrayRef[2] stride=None, IntArrayRef[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int64_t? divisor_override=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(7)) {
    // aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
    
    auto dispatch_avg_pool2d = [](const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, ::std::optional<int64_t> divisor_override) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::avg_pool2d(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    };
    return wrap(dispatch_avg_pool2d(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.toBool(4), _r.toBool(5), _r.toInt64Optional(6)));
  } else {
    // aten::avg_pool2d.out(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_avg_pool2d_out = [](at::Tensor out, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, ::std::optional<int64_t> divisor_override) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::avg_pool2d_out(out, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    };
    return wrap(dispatch_avg_pool2d_out(_r.tensor(7), _r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.toBool(4), _r.toBool(5), _r.toInt64Optional(6)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// avg_pool3d
static PyObject * THPVariable_avg_pool3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "avg_pool3d(Tensor input, IntArrayRef[3] kernel_size, IntArrayRef[3] stride=None, IntArrayRef[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int64_t? divisor_override=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(7)) {
    // aten::avg_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
    
    auto dispatch_avg_pool3d = [](const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, ::std::optional<int64_t> divisor_override) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::avg_pool3d(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    };
    return wrap(dispatch_avg_pool3d(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.toBool(4), _r.toBool(5), _r.toInt64Optional(6)));
  } else {
    // aten::avg_pool3d.out(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_avg_pool3d_out = [](at::Tensor out, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, ::std::optional<int64_t> divisor_override) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::avg_pool3d_out(out, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    };
    return wrap(dispatch_avg_pool3d_out(_r.tensor(7), _r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.toBool(4), _r.toBool(5), _r.toInt64Optional(6)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// binary_cross_entropy
static PyObject * THPVariable_binary_cross_entropy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "binary_cross_entropy(Tensor input, Tensor target, Tensor? weight=None, int64_t reduction=at::Reduction::Mean, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(4)) {
    // aten::binary_cross_entropy(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor
    
    auto dispatch_binary_cross_entropy = [](const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::binary_cross_entropy(self, target, weight, reduction);
    };
    return wrap(dispatch_binary_cross_entropy(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.toInt64(3)));
  } else {
    // aten::binary_cross_entropy.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_binary_cross_entropy_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::binary_cross_entropy_out(out, self, target, weight, reduction);
    };
    return wrap(dispatch_binary_cross_entropy_out(_r.tensor(4), _r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.toInt64(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// col2im
static PyObject * THPVariable_col2im(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "col2im(Tensor input, SymIntArrayRef[2] output_size, IntArrayRef[2] kernel_size, IntArrayRef[2] dilation, IntArrayRef[2] padding, IntArrayRef[2] stride, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(6)) {
    // aten::col2im(Tensor self, SymInt[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor
    
    auto dispatch_col2im = [](const at::Tensor & self, c10::SymIntArrayRef output_size, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::col2im_symint(self, output_size, kernel_size, dilation, padding, stride);
    };
    return wrap(dispatch_col2im(_r.tensor(0), _r.symintlist(1), _r.intlist(2), _r.intlist(3), _r.intlist(4), _r.intlist(5)));
  } else {
    // aten::col2im.out(Tensor self, SymInt[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_col2im_out = [](at::Tensor out, const at::Tensor & self, c10::SymIntArrayRef output_size, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::col2im_symint_out(out, self, output_size, kernel_size, dilation, padding, stride);
    };
    return wrap(dispatch_col2im_out(_r.tensor(6), _r.tensor(0), _r.symintlist(1), _r.intlist(2), _r.intlist(3), _r.intlist(4), _r.intlist(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// conv_depthwise3d
static PyObject * THPVariable_conv_depthwise3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "conv_depthwise3d(Tensor input, Tensor weight, SymIntArrayRef[3] kernel_size, Tensor? bias, SymIntArrayRef[3] stride, SymIntArrayRef[3] padding, SymIntArrayRef[3] dilation)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::conv_depthwise3d(Tensor self, Tensor weight, SymInt[3] kernel_size, Tensor? bias, SymInt[3] stride, SymInt[3] padding, SymInt[3] dilation) -> Tensor
  
  auto dispatch_conv_depthwise3d = [](const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::conv_depthwise3d_symint(self, weight, kernel_size, bias, stride, padding, dilation);
  };
  return wrap(dispatch_conv_depthwise3d(_r.tensor(0), _r.tensor(1), _r.symintlist(2), _r.optionalTensor(3), _r.symintlist(4), _r.symintlist(5), _r.symintlist(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// cross_entropy_loss
static PyObject * THPVariable_cross_entropy_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "cross_entropy_loss(Tensor input, Tensor target, Tensor? weight=None, int64_t reduction=at::Reduction::Mean, SymInt ignore_index=-100, double label_smoothing=0.0)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::cross_entropy_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, SymInt ignore_index=-100, float label_smoothing=0.0) -> Tensor
  
  auto dispatch_cross_entropy_loss = [](const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction, c10::SymInt ignore_index, double label_smoothing) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::cross_entropy_loss_symint(self, target, weight, reduction, ignore_index, label_smoothing);
  };
  return wrap(dispatch_cross_entropy_loss(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.toInt64(3), _r.toSymInt(4), _r.toDouble(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// elu
static PyObject * THPVariable_elu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "elu(Tensor input, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(4)) {
    // aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor
    
    auto dispatch_elu = [](const at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::elu(self, alpha, scale, input_scale);
    };
    return wrap(dispatch_elu(_r.tensor(0), _r.scalar(1), _r.scalar(2), _r.scalar(3)));
  } else {
    // aten::elu.out(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_elu_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::elu_out(out, self, alpha, scale, input_scale);
    };
    return wrap(dispatch_elu_out(_r.tensor(4), _r.tensor(0), _r.scalar(1), _r.scalar(2), _r.scalar(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// elu_
static PyObject * THPVariable_elu_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "elu_(Tensor input, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::elu_(Tensor(a!) self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor(a!)
  
  auto dispatch_elu_ = [](at::Tensor self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::elu_(self, alpha, scale, input_scale);
  };
  return wrap(dispatch_elu_(_r.tensor(0), _r.scalar(1), _r.scalar(2), _r.scalar(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// flatten_dense_tensors
static PyObject * THPVariable_flatten_dense_tensors(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "flatten_dense_tensors(TensorList tensors)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::flatten_dense_tensors(Tensor[] tensors) -> Tensor
  
  auto dispatch_flatten_dense_tensors = [](at::TensorList tensors) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::flatten_dense_tensors(tensors);
  };
  return wrap(dispatch_flatten_dense_tensors(_r.tensorlist(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fractional_max_pool2d
static PyObject * THPVariable_fractional_max_pool2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fractional_max_pool2d(Tensor input, IntArrayRef[2] kernel_size, IntArrayRef[2] output_size, Tensor random_samples, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(4)) {
    // aten::fractional_max_pool2d(Tensor self, int[2] kernel_size, int[2] output_size, Tensor random_samples) -> (Tensor, Tensor)
    
    auto dispatch_fractional_max_pool2d = [](const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & random_samples) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::fractional_max_pool2d(self, kernel_size, output_size, random_samples);
    };
    return wrap(dispatch_fractional_max_pool2d(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.tensor(3)));
  } else {
    // aten::fractional_max_pool2d.output(Tensor self, int[2] kernel_size, int[2] output_size, Tensor random_samples, *, Tensor(a!) output, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
    auto out = _r.tensorlist_n<2>(4);
    auto dispatch_fractional_max_pool2d_out = [](at::Tensor & output, at::Tensor & indices, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & random_samples) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::fractional_max_pool2d_out(output, indices, self, kernel_size, output_size, random_samples);
    };
    return wrap(dispatch_fractional_max_pool2d_out(out[0], out[1], _r.tensor(0), _r.intlist(1), _r.intlist(2), _r.tensor(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fractional_max_pool3d
static PyObject * THPVariable_fractional_max_pool3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fractional_max_pool3d(Tensor input, IntArrayRef[3] kernel_size, IntArrayRef[3] output_size, Tensor random_samples, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(4)) {
    // aten::fractional_max_pool3d(Tensor self, int[3] kernel_size, int[3] output_size, Tensor random_samples) -> (Tensor, Tensor)
    
    auto dispatch_fractional_max_pool3d = [](const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & random_samples) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::fractional_max_pool3d(self, kernel_size, output_size, random_samples);
    };
    return wrap(dispatch_fractional_max_pool3d(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.tensor(3)));
  } else {
    // aten::fractional_max_pool3d.output(Tensor self, int[3] kernel_size, int[3] output_size, Tensor random_samples, *, Tensor(a!) output, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
    auto out = _r.tensorlist_n<2>(4);
    auto dispatch_fractional_max_pool3d_out = [](at::Tensor & output, at::Tensor & indices, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & random_samples) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::fractional_max_pool3d_out(output, indices, self, kernel_size, output_size, random_samples);
    };
    return wrap(dispatch_fractional_max_pool3d_out(out[0], out[1], _r.tensor(0), _r.intlist(1), _r.intlist(2), _r.tensor(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// gelu
static PyObject * THPVariable_gelu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "gelu(Tensor input, *, c10::string_view approximate=\"none\", Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(2)) {
    // aten::gelu(Tensor self, *, str approximate='none') -> Tensor
    
    auto dispatch_gelu = [](const at::Tensor & self, c10::string_view approximate) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::gelu(self, approximate);
    };
    return wrap(dispatch_gelu(_r.tensor(0), _r.stringView(1)));
  } else {
    // aten::gelu.out(Tensor self, *, str approximate='none', Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_gelu_out = [](at::Tensor out, const at::Tensor & self, c10::string_view approximate) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::gelu_out(out, self, approximate);
    };
    return wrap(dispatch_gelu_out(_r.tensor(2), _r.tensor(0), _r.stringView(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// gelu_
static PyObject * THPVariable_gelu_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "gelu_(Tensor input, *, c10::string_view approximate=\"none\")",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::gelu_(Tensor(a!) self, *, str approximate='none') -> Tensor(a!)
  
  auto dispatch_gelu_ = [](at::Tensor self, c10::string_view approximate) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::gelu_(self, approximate);
  };
  return wrap(dispatch_gelu_(_r.tensor(0), _r.stringView(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// glu
static PyObject * THPVariable_glu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "glu(Tensor input, int64_t dim=-1, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(2)) {
    // aten::glu(Tensor self, int dim=-1) -> Tensor
    
    auto dispatch_glu = [](const at::Tensor & self, int64_t dim) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::glu(self, dim);
    };
    return wrap(dispatch_glu(_r.tensor(0), _r.toInt64(1)));
  } else {
    // aten::glu.out(Tensor self, int dim=-1, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_glu_out = [](at::Tensor out, const at::Tensor & self, int64_t dim) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::glu_out(out, self, dim);
    };
    return wrap(dispatch_glu_out(_r.tensor(2), _r.tensor(0), _r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// hardsigmoid
static PyObject * THPVariable_hardsigmoid(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "hardsigmoid(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(1)) {
    // aten::hardsigmoid(Tensor self) -> Tensor
    
    auto dispatch_hardsigmoid = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::hardsigmoid(self);
    };
    return wrap(dispatch_hardsigmoid(_r.tensor(0)));
  } else {
    // aten::hardsigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_hardsigmoid_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::hardsigmoid_out(out, self);
    };
    return wrap(dispatch_hardsigmoid_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// hardsigmoid_
static PyObject * THPVariable_hardsigmoid_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "hardsigmoid_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::hardsigmoid_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_hardsigmoid_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::hardsigmoid_(self);
  };
  return wrap(dispatch_hardsigmoid_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// hardswish
static PyObject * THPVariable_hardswish(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "hardswish(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(1)) {
    // aten::hardswish(Tensor self) -> Tensor
    
    auto dispatch_hardswish = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::hardswish(self);
    };
    return wrap(dispatch_hardswish(_r.tensor(0)));
  } else {
    // aten::hardswish.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_hardswish_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::hardswish_out(out, self);
    };
    return wrap(dispatch_hardswish_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// hardswish_
static PyObject * THPVariable_hardswish_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "hardswish_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::hardswish_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_hardswish_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::hardswish_(self);
  };
  return wrap(dispatch_hardswish_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// hardtanh
static PyObject * THPVariable_hardtanh(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "hardtanh(Tensor input, Scalar min_val=-1, Scalar max_val=1, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(3)) {
    // aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor
    
    auto dispatch_hardtanh = [](const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::hardtanh(self, min_val, max_val);
    };
    return wrap(dispatch_hardtanh(_r.tensor(0), _r.scalar(1), _r.scalar(2)));
  } else {
    // aten::hardtanh.out(Tensor self, Scalar min_val=-1, Scalar max_val=1, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_hardtanh_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::hardtanh_out(out, self, min_val, max_val);
    };
    return wrap(dispatch_hardtanh_out(_r.tensor(3), _r.tensor(0), _r.scalar(1), _r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// hardtanh_
static PyObject * THPVariable_hardtanh_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "hardtanh_(Tensor input, Scalar min_val=-1, Scalar max_val=1)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::hardtanh_(Tensor(a!) self, Scalar min_val=-1, Scalar max_val=1) -> Tensor(a!)
  
  auto dispatch_hardtanh_ = [](at::Tensor self, const at::Scalar & min_val, const at::Scalar & max_val) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::hardtanh_(self, min_val, max_val);
  };
  return wrap(dispatch_hardtanh_(_r.tensor(0), _r.scalar(1), _r.scalar(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// huber_loss
static PyObject * THPVariable_huber_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "huber_loss(Tensor input, Tensor target, int64_t reduction=at::Reduction::Mean, double delta=1.0, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(4)) {
    // aten::huber_loss(Tensor self, Tensor target, int reduction=Mean, float delta=1.0) -> Tensor
    
    auto dispatch_huber_loss = [](const at::Tensor & self, const at::Tensor & target, int64_t reduction, double delta) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::huber_loss(self, target, reduction, delta);
    };
    return wrap(dispatch_huber_loss(_r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.toDouble(3)));
  } else {
    // aten::huber_loss.out(Tensor self, Tensor target, int reduction=Mean, float delta=1.0, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_huber_loss_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double delta) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::huber_loss_out(out, self, target, reduction, delta);
    };
    return wrap(dispatch_huber_loss_out(_r.tensor(4), _r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.toDouble(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// im2col
static PyObject * THPVariable_im2col(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "im2col(Tensor input, IntArrayRef[2] kernel_size, IntArrayRef[2] dilation, IntArrayRef[2] padding, IntArrayRef[2] stride, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(5)) {
    // aten::im2col(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor
    
    auto dispatch_im2col = [](const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::im2col(self, kernel_size, dilation, padding, stride);
    };
    return wrap(dispatch_im2col(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.intlist(4)));
  } else {
    // aten::im2col.out(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_im2col_out = [](at::Tensor out, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::im2col_out(out, self, kernel_size, dilation, padding, stride);
    };
    return wrap(dispatch_im2col_out(_r.tensor(5), _r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.intlist(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// l1_loss
static PyObject * THPVariable_l1_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "l1_loss(Tensor input, Tensor target, int64_t reduction=at::Reduction::Mean)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::l1_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
  
  auto dispatch_l1_loss = [](const at::Tensor & self, const at::Tensor & target, int64_t reduction) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::l1_loss(self, target, reduction);
  };
  return wrap(dispatch_l1_loss(_r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// leaky_relu
static PyObject * THPVariable_leaky_relu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "leaky_relu(Tensor input, Scalar negative_slope=0.01, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(2)) {
    // aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor
    
    auto dispatch_leaky_relu = [](const at::Tensor & self, const at::Scalar & negative_slope) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::leaky_relu(self, negative_slope);
    };
    return wrap(dispatch_leaky_relu(_r.tensor(0), _r.scalar(1)));
  } else {
    // aten::leaky_relu.out(Tensor self, Scalar negative_slope=0.01, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_leaky_relu_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & negative_slope) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::leaky_relu_out(out, self, negative_slope);
    };
    return wrap(dispatch_leaky_relu_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// leaky_relu_
static PyObject * THPVariable_leaky_relu_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "leaky_relu_(Tensor input, Scalar negative_slope=0.01)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::leaky_relu_(Tensor(a!) self, Scalar negative_slope=0.01) -> Tensor(a!)
  
  auto dispatch_leaky_relu_ = [](at::Tensor self, const at::Scalar & negative_slope) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::leaky_relu_(self, negative_slope);
  };
  return wrap(dispatch_leaky_relu_(_r.tensor(0), _r.scalar(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linear
static PyObject * THPVariable_linear(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linear(Tensor input, Tensor weight, Tensor? bias=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(3)) {
    // aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
    
    auto dispatch_linear = [](const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linear(input, weight, bias);
    };
    return wrap(dispatch_linear(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2)));
  } else {
    // aten::linear.out(Tensor input, Tensor weight, Tensor? bias=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linear_out = [](at::Tensor out, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linear_out(out, input, weight, bias);
    };
    return wrap(dispatch_linear_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.optionalTensor(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// log_sigmoid
static PyObject * THPVariable_log_sigmoid(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "log_sigmoid(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(1)) {
    // aten::log_sigmoid(Tensor self) -> Tensor
    
    auto dispatch_log_sigmoid = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::log_sigmoid(self);
    };
    return wrap(dispatch_log_sigmoid(_r.tensor(0)));
  } else {
    // aten::log_sigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_log_sigmoid_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::log_sigmoid_out(out, self);
    };
    return wrap(dispatch_log_sigmoid_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// max_pool2d_with_indices
static PyObject * THPVariable_max_pool2d_with_indices(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "max_pool2d_with_indices(Tensor input, IntArrayRef[2] kernel_size, IntArrayRef[2] stride=None, IntArrayRef[2] padding=0, IntArrayRef[2] dilation=1, bool ceil_mode=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(6)) {
    // aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
    
    auto dispatch_max_pool2d_with_indices = [](const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::max_pool2d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
    };
    return wrap(dispatch_max_pool2d_with_indices(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.intlist(4), _r.toBool(5)));
  } else {
    // aten::max_pool2d_with_indices.out(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
    auto out = _r.tensorlist_n<2>(6);
    auto dispatch_max_pool2d_with_indices_out = [](at::Tensor & out, at::Tensor & indices, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::max_pool2d_with_indices_out(out, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
    };
    return wrap(dispatch_max_pool2d_with_indices_out(out[0], out[1], _r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.intlist(4), _r.toBool(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// max_pool3d_with_indices
static PyObject * THPVariable_max_pool3d_with_indices(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "max_pool3d_with_indices(Tensor input, IntArrayRef[3] kernel_size, IntArrayRef[3] stride=None, IntArrayRef[3] padding=0, IntArrayRef[3] dilation=1, bool ceil_mode=False, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(6)) {
    // aten::max_pool3d_with_indices(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
    
    auto dispatch_max_pool3d_with_indices = [](const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::max_pool3d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
    };
    return wrap(dispatch_max_pool3d_with_indices(_r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.intlist(4), _r.toBool(5)));
  } else {
    // aten::max_pool3d_with_indices.out(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
    auto out = _r.tensorlist_n<2>(6);
    auto dispatch_max_pool3d_with_indices_out = [](at::Tensor & out, at::Tensor & indices, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::max_pool3d_with_indices_out(out, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
    };
    return wrap(dispatch_max_pool3d_with_indices_out(out[0], out[1], _r.tensor(0), _r.intlist(1), _r.intlist(2), _r.intlist(3), _r.intlist(4), _r.toBool(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// max_unpool2d
static PyObject * THPVariable_max_unpool2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "max_unpool2d(Tensor input, Tensor indices, SymIntArrayRef[2] output_size, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(3)) {
    // aten::max_unpool2d(Tensor self, Tensor indices, SymInt[2] output_size) -> Tensor
    
    auto dispatch_max_unpool2d = [](const at::Tensor & self, const at::Tensor & indices, c10::SymIntArrayRef output_size) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::max_unpool2d_symint(self, indices, output_size);
    };
    return wrap(dispatch_max_unpool2d(_r.tensor(0), _r.tensor(1), _r.symintlist(2)));
  } else {
    // aten::max_unpool2d.out(Tensor self, Tensor indices, SymInt[2] output_size, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_max_unpool2d_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & indices, c10::SymIntArrayRef output_size) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::max_unpool2d_symint_out(out, self, indices, output_size);
    };
    return wrap(dispatch_max_unpool2d_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.symintlist(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// max_unpool3d
static PyObject * THPVariable_max_unpool3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "max_unpool3d(Tensor input, Tensor indices, SymIntArrayRef[3] output_size, IntArrayRef[3] stride, IntArrayRef[3] padding, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(5)) {
    // aten::max_unpool3d(Tensor self, Tensor indices, SymInt[3] output_size, int[3] stride, int[3] padding) -> Tensor
    
    auto dispatch_max_unpool3d = [](const at::Tensor & self, const at::Tensor & indices, c10::SymIntArrayRef output_size, at::IntArrayRef stride, at::IntArrayRef padding) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::max_unpool3d_symint(self, indices, output_size, stride, padding);
    };
    return wrap(dispatch_max_unpool3d(_r.tensor(0), _r.tensor(1), _r.symintlist(2), _r.intlist(3), _r.intlist(4)));
  } else {
    // aten::max_unpool3d.out(Tensor self, Tensor indices, SymInt[3] output_size, int[3] stride, int[3] padding, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_max_unpool3d_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & indices, c10::SymIntArrayRef output_size, at::IntArrayRef stride, at::IntArrayRef padding) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::max_unpool3d_symint_out(out, self, indices, output_size, stride, padding);
    };
    return wrap(dispatch_max_unpool3d_out(_r.tensor(5), _r.tensor(0), _r.tensor(1), _r.symintlist(2), _r.intlist(3), _r.intlist(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mish
static PyObject * THPVariable_mish(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mish(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(1)) {
    // aten::mish(Tensor self) -> Tensor
    
    auto dispatch_mish = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::mish(self);
    };
    return wrap(dispatch_mish(_r.tensor(0)));
  } else {
    // aten::mish.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_mish_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::mish_out(out, self);
    };
    return wrap(dispatch_mish_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mish_
static PyObject * THPVariable_mish_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mish_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::mish_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_mish_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::mish_(self);
  };
  return wrap(dispatch_mish_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mkldnn_linear
static PyObject * THPVariable_mkldnn_linear(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mkldnn_linear(Tensor input, Tensor weight, Tensor? bias=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::mkldnn_linear(Tensor self, Tensor weight, Tensor? bias=None) -> Tensor
  
  auto dispatch_mkldnn_linear = [](const at::Tensor & self, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::mkldnn_linear(self, weight, bias);
  };
  return wrap(dispatch_mkldnn_linear(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mkldnn_reorder_conv2d_weight
static PyObject * THPVariable_mkldnn_reorder_conv2d_weight(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mkldnn_reorder_conv2d_weight(Tensor input, SymIntArrayRef[2] padding=0, SymIntArrayRef[2] stride=1, SymIntArrayRef[2] dilation=1, SymInt groups=1, SymIntArrayRef? input_size=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::mkldnn_reorder_conv2d_weight(Tensor self, SymInt[2] padding=0, SymInt[2] stride=1, SymInt[2] dilation=1, SymInt groups=1, SymInt[]? input_size=None) -> Tensor
  
  auto dispatch_mkldnn_reorder_conv2d_weight = [](const at::Tensor & self, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, at::OptionalSymIntArrayRef input_size) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::mkldnn_reorder_conv2d_weight_symint(self, padding, stride, dilation, groups, input_size);
  };
  return wrap(dispatch_mkldnn_reorder_conv2d_weight(_r.tensor(0), _r.symintlist(1), _r.symintlist(2), _r.symintlist(3), _r.toSymInt(4), _r.symintlistOptional(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mkldnn_reorder_conv3d_weight
static PyObject * THPVariable_mkldnn_reorder_conv3d_weight(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mkldnn_reorder_conv3d_weight(Tensor input, SymIntArrayRef[3] padding=0, SymIntArrayRef[3] stride=1, SymIntArrayRef[3] dilation=1, SymInt groups=1, SymIntArrayRef? input_size=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::mkldnn_reorder_conv3d_weight(Tensor self, SymInt[3] padding=0, SymInt[3] stride=1, SymInt[3] dilation=1, SymInt groups=1, SymInt[]? input_size=None) -> Tensor
  
  auto dispatch_mkldnn_reorder_conv3d_weight = [](const at::Tensor & self, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, at::OptionalSymIntArrayRef input_size) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::mkldnn_reorder_conv3d_weight_symint(self, padding, stride, dilation, groups, input_size);
  };
  return wrap(dispatch_mkldnn_reorder_conv3d_weight(_r.tensor(0), _r.symintlist(1), _r.symintlist(2), _r.symintlist(3), _r.toSymInt(4), _r.symintlistOptional(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// mse_loss
static PyObject * THPVariable_mse_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "mse_loss(Tensor input, Tensor target, int64_t reduction=at::Reduction::Mean, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(3)) {
    // aten::mse_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
    
    auto dispatch_mse_loss = [](const at::Tensor & self, const at::Tensor & target, int64_t reduction) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::mse_loss(self, target, reduction);
    };
    return wrap(dispatch_mse_loss(_r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  } else {
    // aten::mse_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_mse_loss_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & target, int64_t reduction) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::mse_loss_out(out, self, target, reduction);
    };
    return wrap(dispatch_mse_loss_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// multi_margin_loss
static PyObject * THPVariable_multi_margin_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "multi_margin_loss(Tensor input, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int64_t reduction=at::Reduction::Mean, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(6)) {
    // aten::multi_margin_loss(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=Mean) -> Tensor
    
    auto dispatch_multi_margin_loss = [](const at::Tensor & self, const at::Tensor & target, const at::Scalar & p, const at::Scalar & margin, const ::std::optional<at::Tensor> & weight, int64_t reduction) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::multi_margin_loss(self, target, p, margin, weight, reduction);
    };
    return wrap(dispatch_multi_margin_loss(_r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3), _r.optionalTensor(4), _r.toInt64(5)));
  } else {
    // aten::multi_margin_loss.out(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_multi_margin_loss_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & target, const at::Scalar & p, const at::Scalar & margin, const ::std::optional<at::Tensor> & weight, int64_t reduction) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::multi_margin_loss_out(out, self, target, p, margin, weight, reduction);
    };
    return wrap(dispatch_multi_margin_loss_out(_r.tensor(6), _r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3), _r.optionalTensor(4), _r.toInt64(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// multilabel_margin_loss
static PyObject * THPVariable_multilabel_margin_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "multilabel_margin_loss(Tensor input, Tensor target, int64_t reduction=at::Reduction::Mean, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(3)) {
    // aten::multilabel_margin_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
    
    auto dispatch_multilabel_margin_loss = [](const at::Tensor & self, const at::Tensor & target, int64_t reduction) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::multilabel_margin_loss(self, target, reduction);
    };
    return wrap(dispatch_multilabel_margin_loss(_r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  } else {
    // aten::multilabel_margin_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_multilabel_margin_loss_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & target, int64_t reduction) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::multilabel_margin_loss_out(out, self, target, reduction);
    };
    return wrap(dispatch_multilabel_margin_loss_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// nll_loss
static PyObject * THPVariable_nll_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "nll_loss(Tensor input, Tensor target, Tensor? weight=None, int64_t reduction=at::Reduction::Mean, SymInt ignore_index=-100, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(5)) {
    // aten::nll_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, SymInt ignore_index=-100) -> Tensor
    
    auto dispatch_nll_loss = [](const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction, c10::SymInt ignore_index) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::nll_loss_symint(self, target, weight, reduction, ignore_index);
    };
    return wrap(dispatch_nll_loss(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.toInt64(3), _r.toSymInt(4)));
  } else {
    // aten::nll_loss.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, SymInt ignore_index=-100, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_nll_loss_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction, c10::SymInt ignore_index) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::nll_loss_symint_out(out, self, target, weight, reduction, ignore_index);
    };
    return wrap(dispatch_nll_loss_out(_r.tensor(5), _r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.toInt64(3), _r.toSymInt(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// nll_loss2d
static PyObject * THPVariable_nll_loss2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "nll_loss2d(Tensor input, Tensor target, Tensor? weight=None, int64_t reduction=at::Reduction::Mean, SymInt ignore_index=-100, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(5)) {
    // aten::nll_loss2d(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, SymInt ignore_index=-100) -> Tensor
    
    auto dispatch_nll_loss2d = [](const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction, c10::SymInt ignore_index) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::nll_loss2d_symint(self, target, weight, reduction, ignore_index);
    };
    return wrap(dispatch_nll_loss2d(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.toInt64(3), _r.toSymInt(4)));
  } else {
    // aten::nll_loss2d.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, SymInt ignore_index=-100, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_nll_loss2d_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction, c10::SymInt ignore_index) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::nll_loss2d_symint_out(out, self, target, weight, reduction, ignore_index);
    };
    return wrap(dispatch_nll_loss2d_out(_r.tensor(5), _r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.toInt64(3), _r.toSymInt(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// nll_loss_nd
static PyObject * THPVariable_nll_loss_nd(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "nll_loss_nd(Tensor input, Tensor target, Tensor? weight=None, int64_t reduction=at::Reduction::Mean, SymInt ignore_index=-100)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::nll_loss_nd(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, SymInt ignore_index=-100) -> Tensor
  
  auto dispatch_nll_loss_nd = [](const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction, c10::SymInt ignore_index) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::nll_loss_nd_symint(self, target, weight, reduction, ignore_index);
  };
  return wrap(dispatch_nll_loss_nd(_r.tensor(0), _r.tensor(1), _r.optionalTensor(2), _r.toInt64(3), _r.toSymInt(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// one_hot
static PyObject * THPVariable_one_hot(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "one_hot(Tensor input, int64_t num_classes=-1)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::one_hot(Tensor self, int num_classes=-1) -> Tensor
  
  auto dispatch_one_hot = [](const at::Tensor & self, int64_t num_classes) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::one_hot(self, num_classes);
  };
  return wrap(dispatch_one_hot(_r.tensor(0), _r.toInt64(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// pad
static PyObject * THPVariable_pad(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "pad(Tensor input, SymIntArrayRef pad, c10::string_view mode=\"constant\", double? value=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::pad(Tensor self, SymInt[] pad, str mode="constant", float? value=None) -> Tensor
  
  auto dispatch_pad = [](const at::Tensor & self, c10::SymIntArrayRef pad, c10::string_view mode, ::std::optional<double> value) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::pad_symint(self, pad, mode, value);
  };
  return wrap(dispatch_pad(_r.tensor(0), _r.symintlist(1), _r.stringView(2), _r.toDoubleOptional(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// pad_sequence
static PyObject * THPVariable_pad_sequence(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "pad_sequence(TensorList sequences, bool batch_first=False, double padding_value=0.0, c10::string_view padding_side=\"right\")",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::pad_sequence(Tensor[] sequences, bool batch_first=False, float padding_value=0.0, str padding_side="right") -> Tensor
  
  auto dispatch_pad_sequence = [](at::TensorList sequences, bool batch_first, double padding_value, c10::string_view padding_side) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::pad_sequence(sequences, batch_first, padding_value, padding_side);
  };
  return wrap(dispatch_pad_sequence(_r.tensorlist(0), _r.toBool(1), _r.toDouble(2), _r.stringView(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// reflection_pad1d
static PyObject * THPVariable_reflection_pad1d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "reflection_pad1d(Tensor input, SymIntArrayRef[2] padding, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(2)) {
    // aten::reflection_pad1d(Tensor self, SymInt[2] padding) -> Tensor
    
    auto dispatch_reflection_pad1d = [](const at::Tensor & self, c10::SymIntArrayRef padding) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::reflection_pad1d_symint(self, padding);
    };
    return wrap(dispatch_reflection_pad1d(_r.tensor(0), _r.symintlist(1)));
  } else {
    // aten::reflection_pad1d.out(Tensor self, SymInt[2] padding, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_reflection_pad1d_out = [](at::Tensor out, const at::Tensor & self, c10::SymIntArrayRef padding) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::reflection_pad1d_symint_out(out, self, padding);
    };
    return wrap(dispatch_reflection_pad1d_out(_r.tensor(2), _r.tensor(0), _r.symintlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// reflection_pad2d
static PyObject * THPVariable_reflection_pad2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "reflection_pad2d(Tensor input, SymIntArrayRef[4] padding, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(2)) {
    // aten::reflection_pad2d(Tensor self, SymInt[4] padding) -> Tensor
    
    auto dispatch_reflection_pad2d = [](const at::Tensor & self, c10::SymIntArrayRef padding) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::reflection_pad2d_symint(self, padding);
    };
    return wrap(dispatch_reflection_pad2d(_r.tensor(0), _r.symintlist(1)));
  } else {
    // aten::reflection_pad2d.out(Tensor self, SymInt[4] padding, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_reflection_pad2d_out = [](at::Tensor out, const at::Tensor & self, c10::SymIntArrayRef padding) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::reflection_pad2d_symint_out(out, self, padding);
    };
    return wrap(dispatch_reflection_pad2d_out(_r.tensor(2), _r.tensor(0), _r.symintlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// reflection_pad3d
static PyObject * THPVariable_reflection_pad3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "reflection_pad3d(Tensor input, SymIntArrayRef[6] padding, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(2)) {
    // aten::reflection_pad3d(Tensor self, SymInt[6] padding) -> Tensor
    
    auto dispatch_reflection_pad3d = [](const at::Tensor & self, c10::SymIntArrayRef padding) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::reflection_pad3d_symint(self, padding);
    };
    return wrap(dispatch_reflection_pad3d(_r.tensor(0), _r.symintlist(1)));
  } else {
    // aten::reflection_pad3d.out(Tensor self, SymInt[6] padding, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_reflection_pad3d_out = [](at::Tensor out, const at::Tensor & self, c10::SymIntArrayRef padding) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::reflection_pad3d_symint_out(out, self, padding);
    };
    return wrap(dispatch_reflection_pad3d_out(_r.tensor(2), _r.tensor(0), _r.symintlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// relu6
static PyObject * THPVariable_relu6(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "relu6(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::relu6(Tensor self) -> Tensor
  
  auto dispatch_relu6 = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::relu6(self);
  };
  return wrap(dispatch_relu6(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// relu6_
static PyObject * THPVariable_relu6_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "relu6_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::relu6_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_relu6_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::relu6_(self);
  };
  return wrap(dispatch_relu6_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// replication_pad1d
static PyObject * THPVariable_replication_pad1d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "replication_pad1d(Tensor input, SymIntArrayRef[2] padding, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(2)) {
    // aten::replication_pad1d(Tensor self, SymInt[2] padding) -> Tensor
    
    auto dispatch_replication_pad1d = [](const at::Tensor & self, c10::SymIntArrayRef padding) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::replication_pad1d_symint(self, padding);
    };
    return wrap(dispatch_replication_pad1d(_r.tensor(0), _r.symintlist(1)));
  } else {
    // aten::replication_pad1d.out(Tensor self, SymInt[2] padding, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_replication_pad1d_out = [](at::Tensor out, const at::Tensor & self, c10::SymIntArrayRef padding) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::replication_pad1d_symint_out(out, self, padding);
    };
    return wrap(dispatch_replication_pad1d_out(_r.tensor(2), _r.tensor(0), _r.symintlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// replication_pad2d
static PyObject * THPVariable_replication_pad2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "replication_pad2d(Tensor input, SymIntArrayRef[4] padding, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(2)) {
    // aten::replication_pad2d(Tensor self, SymInt[4] padding) -> Tensor
    
    auto dispatch_replication_pad2d = [](const at::Tensor & self, c10::SymIntArrayRef padding) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::replication_pad2d_symint(self, padding);
    };
    return wrap(dispatch_replication_pad2d(_r.tensor(0), _r.symintlist(1)));
  } else {
    // aten::replication_pad2d.out(Tensor self, SymInt[4] padding, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_replication_pad2d_out = [](at::Tensor out, const at::Tensor & self, c10::SymIntArrayRef padding) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::replication_pad2d_symint_out(out, self, padding);
    };
    return wrap(dispatch_replication_pad2d_out(_r.tensor(2), _r.tensor(0), _r.symintlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// replication_pad3d
static PyObject * THPVariable_replication_pad3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "replication_pad3d(Tensor input, SymIntArrayRef[6] padding, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(2)) {
    // aten::replication_pad3d(Tensor self, SymInt[6] padding) -> Tensor
    
    auto dispatch_replication_pad3d = [](const at::Tensor & self, c10::SymIntArrayRef padding) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::replication_pad3d_symint(self, padding);
    };
    return wrap(dispatch_replication_pad3d(_r.tensor(0), _r.symintlist(1)));
  } else {
    // aten::replication_pad3d.out(Tensor self, SymInt[6] padding, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_replication_pad3d_out = [](at::Tensor out, const at::Tensor & self, c10::SymIntArrayRef padding) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::replication_pad3d_symint_out(out, self, padding);
    };
    return wrap(dispatch_replication_pad3d_out(_r.tensor(2), _r.tensor(0), _r.symintlist(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rrelu_with_noise
static PyObject * THPVariable_rrelu_with_noise(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rrelu_with_noise(Tensor input, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(6)) {
    // aten::rrelu_with_noise(Tensor self, Tensor(b!) noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor
    
    auto dispatch_rrelu_with_noise = [](const at::Tensor & self, at::Tensor noise, const at::Scalar & lower, const at::Scalar & upper, bool training, ::std::optional<at::Generator> generator) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::rrelu_with_noise(self, noise, lower, upper, training, generator);
    };
    return wrap(dispatch_rrelu_with_noise(_r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3), _r.toBool(4), _r.generator(5)));
  } else {
    // aten::rrelu_with_noise.out(Tensor self, Tensor(b!) noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_rrelu_with_noise_out = [](at::Tensor out, const at::Tensor & self, at::Tensor noise, const at::Scalar & lower, const at::Scalar & upper, bool training, ::std::optional<at::Generator> generator) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::rrelu_with_noise_out(out, self, noise, lower, upper, training, generator);
    };
    return wrap(dispatch_rrelu_with_noise_out(_r.tensor(6), _r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3), _r.toBool(4), _r.generator(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// rrelu_with_noise_
static PyObject * THPVariable_rrelu_with_noise_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "rrelu_with_noise_(Tensor input, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::rrelu_with_noise_(Tensor(a!) self, Tensor(b!) noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor(a!)
  
  auto dispatch_rrelu_with_noise_ = [](at::Tensor self, at::Tensor noise, const at::Scalar & lower, const at::Scalar & upper, bool training, ::std::optional<at::Generator> generator) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::rrelu_with_noise_(self, noise, lower, upper, training, generator);
  };
  return wrap(dispatch_rrelu_with_noise_(_r.tensor(0), _r.tensor(1), _r.scalar(2), _r.scalar(3), _r.toBool(4), _r.generator(5)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// scaled_dot_product_attention
static PyObject * THPVariable_scaled_dot_product_attention(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "scaled_dot_product_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, double dropout_p=0.0, bool is_causal=False, *, double? scale=None, bool enable_gqa=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::scaled_dot_product_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0.0, bool is_causal=False, *, float? scale=None, bool enable_gqa=False) -> Tensor
  
  auto dispatch_scaled_dot_product_attention = [](const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & attn_mask, double dropout_p, bool is_causal, ::std::optional<double> scale, bool enable_gqa) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
  };
  return wrap(dispatch_scaled_dot_product_attention(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.optionalTensor(3), _r.toDouble(4), _r.toBool(5), _r.toDoubleOptional(6), _r.toBool(7)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// silu
static PyObject * THPVariable_silu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "silu(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(1)) {
    // aten::silu(Tensor self) -> Tensor
    
    auto dispatch_silu = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::silu(self);
    };
    return wrap(dispatch_silu(_r.tensor(0)));
  } else {
    // aten::silu.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_silu_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::silu_out(out, self);
    };
    return wrap(dispatch_silu_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// silu_
static PyObject * THPVariable_silu_(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "silu_(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::silu_(Tensor(a!) self) -> Tensor(a!)
  
  auto dispatch_silu_ = [](at::Tensor self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::silu_(self);
  };
  return wrap(dispatch_silu_(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// slow_conv3d
static PyObject * THPVariable_slow_conv3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "slow_conv3d(Tensor input, Tensor weight, SymIntArrayRef[3] kernel_size, Tensor? bias=None, SymIntArrayRef[3] stride=1, SymIntArrayRef[3] padding=0, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(6)) {
    // aten::slow_conv3d(Tensor self, Tensor weight, SymInt[3] kernel_size, Tensor? bias=None, SymInt[3] stride=1, SymInt[3] padding=0) -> Tensor
    
    auto dispatch_slow_conv3d = [](const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::slow_conv3d_symint(self, weight, kernel_size, bias, stride, padding);
    };
    return wrap(dispatch_slow_conv3d(_r.tensor(0), _r.tensor(1), _r.symintlist(2), _r.optionalTensor(3), _r.symintlist(4), _r.symintlist(5)));
  } else {
    // aten::slow_conv3d.out(Tensor self, Tensor weight, SymInt[3] kernel_size, Tensor? bias=None, SymInt[3] stride=1, SymInt[3] padding=0, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_slow_conv3d_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::slow_conv3d_symint_out(out, self, weight, kernel_size, bias, stride, padding);
    };
    return wrap(dispatch_slow_conv3d_out(_r.tensor(6), _r.tensor(0), _r.tensor(1), _r.symintlist(2), _r.optionalTensor(3), _r.symintlist(4), _r.symintlist(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// slow_conv_dilated2d
static PyObject * THPVariable_slow_conv_dilated2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "slow_conv_dilated2d(Tensor input, Tensor weight, SymIntArrayRef[2] kernel_size, Tensor? bias=None, SymIntArrayRef[2] stride=1, SymIntArrayRef[2] padding=0, SymIntArrayRef[2] dilation=1)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::slow_conv_dilated2d(Tensor self, Tensor weight, SymInt[2] kernel_size, Tensor? bias=None, SymInt[2] stride=1, SymInt[2] padding=0, SymInt[2] dilation=1) -> Tensor
  
  auto dispatch_slow_conv_dilated2d = [](const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::slow_conv_dilated2d_symint(self, weight, kernel_size, bias, stride, padding, dilation);
  };
  return wrap(dispatch_slow_conv_dilated2d(_r.tensor(0), _r.tensor(1), _r.symintlist(2), _r.optionalTensor(3), _r.symintlist(4), _r.symintlist(5), _r.symintlist(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// slow_conv_dilated3d
static PyObject * THPVariable_slow_conv_dilated3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "slow_conv_dilated3d(Tensor input, Tensor weight, SymIntArrayRef[3] kernel_size, Tensor? bias=None, SymIntArrayRef[3] stride=1, SymIntArrayRef[3] padding=0, SymIntArrayRef[3] dilation=1)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::slow_conv_dilated3d(Tensor self, Tensor weight, SymInt[3] kernel_size, Tensor? bias=None, SymInt[3] stride=1, SymInt[3] padding=0, SymInt[3] dilation=1) -> Tensor
  
  auto dispatch_slow_conv_dilated3d = [](const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::slow_conv_dilated3d_symint(self, weight, kernel_size, bias, stride, padding, dilation);
  };
  return wrap(dispatch_slow_conv_dilated3d(_r.tensor(0), _r.tensor(1), _r.symintlist(2), _r.optionalTensor(3), _r.symintlist(4), _r.symintlist(5), _r.symintlist(6)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// slow_conv_transpose2d
static PyObject * THPVariable_slow_conv_transpose2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "slow_conv_transpose2d(Tensor input, Tensor weight, SymIntArrayRef[2] kernel_size, Tensor? bias=None, SymIntArrayRef[2] stride=1, SymIntArrayRef[2] padding=0, SymIntArrayRef[2] output_padding=0, SymIntArrayRef[2] dilation=1, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(8)) {
    // aten::slow_conv_transpose2d(Tensor self, Tensor weight, SymInt[2] kernel_size, Tensor? bias=None, SymInt[2] stride=1, SymInt[2] padding=0, SymInt[2] output_padding=0, SymInt[2] dilation=1) -> Tensor
    
    auto dispatch_slow_conv_transpose2d = [](const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef output_padding, c10::SymIntArrayRef dilation) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::slow_conv_transpose2d_symint(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
    };
    return wrap(dispatch_slow_conv_transpose2d(_r.tensor(0), _r.tensor(1), _r.symintlist(2), _r.optionalTensor(3), _r.symintlist(4), _r.symintlist(5), _r.symintlist(6), _r.symintlist(7)));
  } else {
    // aten::slow_conv_transpose2d.out(Tensor self, Tensor weight, SymInt[2] kernel_size, Tensor? bias=None, SymInt[2] stride=1, SymInt[2] padding=0, SymInt[2] output_padding=0, SymInt[2] dilation=1, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_slow_conv_transpose2d_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef output_padding, c10::SymIntArrayRef dilation) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::slow_conv_transpose2d_symint_out(out, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
    };
    return wrap(dispatch_slow_conv_transpose2d_out(_r.tensor(8), _r.tensor(0), _r.tensor(1), _r.symintlist(2), _r.optionalTensor(3), _r.symintlist(4), _r.symintlist(5), _r.symintlist(6), _r.symintlist(7)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// slow_conv_transpose3d
static PyObject * THPVariable_slow_conv_transpose3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "slow_conv_transpose3d(Tensor input, Tensor weight, SymIntArrayRef[3] kernel_size, Tensor? bias=None, SymIntArrayRef[3] stride=1, SymIntArrayRef[3] padding=0, SymIntArrayRef[3] output_padding=0, SymIntArrayRef[3] dilation=1, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<9> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(8)) {
    // aten::slow_conv_transpose3d(Tensor self, Tensor weight, SymInt[3] kernel_size, Tensor? bias=None, SymInt[3] stride=1, SymInt[3] padding=0, SymInt[3] output_padding=0, SymInt[3] dilation=1) -> Tensor
    
    auto dispatch_slow_conv_transpose3d = [](const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef output_padding, c10::SymIntArrayRef dilation) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::slow_conv_transpose3d_symint(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
    };
    return wrap(dispatch_slow_conv_transpose3d(_r.tensor(0), _r.tensor(1), _r.symintlist(2), _r.optionalTensor(3), _r.symintlist(4), _r.symintlist(5), _r.symintlist(6), _r.symintlist(7)));
  } else {
    // aten::slow_conv_transpose3d.out(Tensor self, Tensor weight, SymInt[3] kernel_size, Tensor? bias=None, SymInt[3] stride=1, SymInt[3] padding=0, SymInt[3] output_padding=0, SymInt[3] dilation=1, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_slow_conv_transpose3d_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef output_padding, c10::SymIntArrayRef dilation) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::slow_conv_transpose3d_symint_out(out, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
    };
    return wrap(dispatch_slow_conv_transpose3d_out(_r.tensor(8), _r.tensor(0), _r.tensor(1), _r.symintlist(2), _r.optionalTensor(3), _r.symintlist(4), _r.symintlist(5), _r.symintlist(6), _r.symintlist(7)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// smooth_l1_loss
static PyObject * THPVariable_smooth_l1_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "smooth_l1_loss(Tensor input, Tensor target, int64_t reduction=at::Reduction::Mean, double beta=1.0, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(4)) {
    // aten::smooth_l1_loss(Tensor self, Tensor target, int reduction=Mean, float beta=1.0) -> Tensor
    
    auto dispatch_smooth_l1_loss = [](const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::smooth_l1_loss(self, target, reduction, beta);
    };
    return wrap(dispatch_smooth_l1_loss(_r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.toDouble(3)));
  } else {
    // aten::smooth_l1_loss.out(Tensor self, Tensor target, int reduction=Mean, float beta=1.0, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_smooth_l1_loss_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::smooth_l1_loss_out(out, self, target, reduction, beta);
    };
    return wrap(dispatch_smooth_l1_loss_out(_r.tensor(4), _r.tensor(0), _r.tensor(1), _r.toInt64(2), _r.toDouble(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// soft_margin_loss
static PyObject * THPVariable_soft_margin_loss(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "soft_margin_loss(Tensor input, Tensor target, int64_t reduction=at::Reduction::Mean, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(3)) {
    // aten::soft_margin_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
    
    auto dispatch_soft_margin_loss = [](const at::Tensor & self, const at::Tensor & target, int64_t reduction) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::soft_margin_loss(self, target, reduction);
    };
    return wrap(dispatch_soft_margin_loss(_r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  } else {
    // aten::soft_margin_loss.out(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_soft_margin_loss_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & target, int64_t reduction) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::soft_margin_loss_out(out, self, target, reduction);
    };
    return wrap(dispatch_soft_margin_loss_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// softplus
static PyObject * THPVariable_softplus(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "softplus(Tensor input, Scalar beta=1, Scalar threshold=20, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(3)) {
    // aten::softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> Tensor
    
    auto dispatch_softplus = [](const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::softplus(self, beta, threshold);
    };
    return wrap(dispatch_softplus(_r.tensor(0), _r.scalar(1), _r.scalar(2)));
  } else {
    // aten::softplus.out(Tensor self, Scalar beta=1, Scalar threshold=20, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_softplus_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::softplus_out(out, self, beta, threshold);
    };
    return wrap(dispatch_softplus_out(_r.tensor(3), _r.tensor(0), _r.scalar(1), _r.scalar(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// softshrink
static PyObject * THPVariable_softshrink(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "softshrink(Tensor input, Scalar lambd=0.5, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(2)) {
    // aten::softshrink(Tensor self, Scalar lambd=0.5) -> Tensor
    
    auto dispatch_softshrink = [](const at::Tensor & self, const at::Scalar & lambd) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::softshrink(self, lambd);
    };
    return wrap(dispatch_softshrink(_r.tensor(0), _r.scalar(1)));
  } else {
    // aten::softshrink.out(Tensor self, Scalar lambd=0.5, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_softshrink_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & lambd) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::softshrink_out(out, self, lambd);
    };
    return wrap(dispatch_softshrink_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// thnn_conv2d
static PyObject * THPVariable_thnn_conv2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "thnn_conv2d(Tensor input, Tensor weight, SymIntArrayRef[2] kernel_size, Tensor? bias=None, SymIntArrayRef[2] stride=1, SymIntArrayRef[2] padding=0, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  if (_r.isNone(6)) {
    // aten::thnn_conv2d(Tensor self, Tensor weight, SymInt[2] kernel_size, Tensor? bias=None, SymInt[2] stride=1, SymInt[2] padding=0) -> Tensor
    
    auto dispatch_thnn_conv2d = [](const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::thnn_conv2d_symint(self, weight, kernel_size, bias, stride, padding);
    };
    return wrap(dispatch_thnn_conv2d(_r.tensor(0), _r.tensor(1), _r.symintlist(2), _r.optionalTensor(3), _r.symintlist(4), _r.symintlist(5)));
  } else {
    // aten::thnn_conv2d.out(Tensor self, Tensor weight, SymInt[2] kernel_size, Tensor? bias=None, SymInt[2] stride=1, SymInt[2] padding=0, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_thnn_conv2d_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::thnn_conv2d_symint_out(out, self, weight, kernel_size, bias, stride, padding);
    };
    return wrap(dispatch_thnn_conv2d_out(_r.tensor(6), _r.tensor(0), _r.tensor(1), _r.symintlist(2), _r.optionalTensor(3), _r.symintlist(4), _r.symintlist(5)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// unflatten_dense_tensors
static PyObject * THPVariable_unflatten_dense_tensors(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "unflatten_dense_tensors(Tensor flat, TensorList tensors)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  // aten::unflatten_dense_tensors(Tensor flat, Tensor[] tensors) -> Tensor[]
  
  auto dispatch_unflatten_dense_tensors = [](const at::Tensor & flat, at::TensorList tensors) -> ::std::vector<at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::unflatten_dense_tensors(flat, tensors);
  };
  return wrap(dispatch_unflatten_dense_tensors(_r.tensor(0), _r.tensorlist(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// upsample_bicubic2d
static PyObject * THPVariable_upsample_bicubic2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "upsample_bicubic2d(Tensor input, SymIntArrayRef? output_size, bool align_corners, ArrayRef<double>? scale_factors)",
    "upsample_bicubic2d(Tensor input, SymIntArrayRef[2] output_size, bool align_corners, double? scales_h=None, double? scales_w=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  switch (_r.idx) {
    case 0: {
      // aten::upsample_bicubic2d.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
      
      auto dispatch_upsample_bicubic2d = [](const at::Tensor & input, at::OptionalSymIntArrayRef output_size, bool align_corners, ::std::optional<at::ArrayRef<double>> scale_factors) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::upsample_bicubic2d_symint(input, output_size, align_corners, scale_factors);
      };
      return wrap(dispatch_upsample_bicubic2d(_r.tensor(0), _r.symintlistOptional(1), _r.toBool(2), _r.doublelistOptional(3)));
    }
    case 1: {
      if (_r.isNone(5)) {
        // aten::upsample_bicubic2d(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
        
        auto dispatch_upsample_bicubic2d = [](const at::Tensor & self, c10::SymIntArrayRef output_size, bool align_corners, ::std::optional<double> scales_h, ::std::optional<double> scales_w) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::upsample_bicubic2d_symint(self, output_size, align_corners, scales_h, scales_w);
        };
        return wrap(dispatch_upsample_bicubic2d(_r.tensor(0), _r.symintlist(1), _r.toBool(2), _r.toDoubleOptional(3), _r.toDoubleOptional(4)));
      } else {
        // aten::upsample_bicubic2d.out(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_upsample_bicubic2d_out = [](at::Tensor out, const at::Tensor & self, c10::SymIntArrayRef output_size, bool align_corners, ::std::optional<double> scales_h, ::std::optional<double> scales_w) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::upsample_bicubic2d_symint_out(out, self, output_size, align_corners, scales_h, scales_w);
        };
        return wrap(dispatch_upsample_bicubic2d_out(_r.tensor(5), _r.tensor(0), _r.symintlist(1), _r.toBool(2), _r.toDoubleOptional(3), _r.toDoubleOptional(4)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// upsample_bilinear2d
static PyObject * THPVariable_upsample_bilinear2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "upsample_bilinear2d(Tensor input, SymIntArrayRef? output_size, bool align_corners, ArrayRef<double>? scale_factors)",
    "upsample_bilinear2d(Tensor input, SymIntArrayRef[2] output_size, bool align_corners, double? scales_h=None, double? scales_w=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  switch (_r.idx) {
    case 0: {
      // aten::upsample_bilinear2d.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
      
      auto dispatch_upsample_bilinear2d = [](const at::Tensor & input, at::OptionalSymIntArrayRef output_size, bool align_corners, ::std::optional<at::ArrayRef<double>> scale_factors) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::upsample_bilinear2d_symint(input, output_size, align_corners, scale_factors);
      };
      return wrap(dispatch_upsample_bilinear2d(_r.tensor(0), _r.symintlistOptional(1), _r.toBool(2), _r.doublelistOptional(3)));
    }
    case 1: {
      if (_r.isNone(5)) {
        // aten::upsample_bilinear2d(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
        
        auto dispatch_upsample_bilinear2d = [](const at::Tensor & self, c10::SymIntArrayRef output_size, bool align_corners, ::std::optional<double> scales_h, ::std::optional<double> scales_w) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::upsample_bilinear2d_symint(self, output_size, align_corners, scales_h, scales_w);
        };
        return wrap(dispatch_upsample_bilinear2d(_r.tensor(0), _r.symintlist(1), _r.toBool(2), _r.toDoubleOptional(3), _r.toDoubleOptional(4)));
      } else {
        // aten::upsample_bilinear2d.out(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_upsample_bilinear2d_out = [](at::Tensor out, const at::Tensor & self, c10::SymIntArrayRef output_size, bool align_corners, ::std::optional<double> scales_h, ::std::optional<double> scales_w) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::upsample_bilinear2d_symint_out(out, self, output_size, align_corners, scales_h, scales_w);
        };
        return wrap(dispatch_upsample_bilinear2d_out(_r.tensor(5), _r.tensor(0), _r.symintlist(1), _r.toBool(2), _r.toDoubleOptional(3), _r.toDoubleOptional(4)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// upsample_linear1d
static PyObject * THPVariable_upsample_linear1d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "upsample_linear1d(Tensor input, SymIntArrayRef? output_size, bool align_corners, ArrayRef<double>? scale_factors)",
    "upsample_linear1d(Tensor input, SymIntArrayRef[1] output_size, bool align_corners, double? scales=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  switch (_r.idx) {
    case 0: {
      // aten::upsample_linear1d.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
      
      auto dispatch_upsample_linear1d = [](const at::Tensor & input, at::OptionalSymIntArrayRef output_size, bool align_corners, ::std::optional<at::ArrayRef<double>> scale_factors) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::upsample_linear1d_symint(input, output_size, align_corners, scale_factors);
      };
      return wrap(dispatch_upsample_linear1d(_r.tensor(0), _r.symintlistOptional(1), _r.toBool(2), _r.doublelistOptional(3)));
    }
    case 1: {
      if (_r.isNone(4)) {
        // aten::upsample_linear1d(Tensor self, SymInt[1] output_size, bool align_corners, float? scales=None) -> Tensor
        
        auto dispatch_upsample_linear1d = [](const at::Tensor & self, c10::SymIntArrayRef output_size, bool align_corners, ::std::optional<double> scales) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::upsample_linear1d_symint(self, output_size, align_corners, scales);
        };
        return wrap(dispatch_upsample_linear1d(_r.tensor(0), _r.symintlist(1), _r.toBool(2), _r.toDoubleOptional(3)));
      } else {
        // aten::upsample_linear1d.out(Tensor self, SymInt[1] output_size, bool align_corners, float? scales=None, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_upsample_linear1d_out = [](at::Tensor out, const at::Tensor & self, c10::SymIntArrayRef output_size, bool align_corners, ::std::optional<double> scales) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::upsample_linear1d_symint_out(out, self, output_size, align_corners, scales);
        };
        return wrap(dispatch_upsample_linear1d_out(_r.tensor(4), _r.tensor(0), _r.symintlist(1), _r.toBool(2), _r.toDoubleOptional(3)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// upsample_nearest1d
static PyObject * THPVariable_upsample_nearest1d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "upsample_nearest1d(Tensor input, SymIntArrayRef? output_size, ArrayRef<double>? scale_factors)",
    "upsample_nearest1d(Tensor input, SymIntArrayRef[1] output_size, double? scales=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  switch (_r.idx) {
    case 0: {
      // aten::upsample_nearest1d.vec(Tensor input, SymInt[]? output_size, float[]? scale_factors) -> Tensor
      
      auto dispatch_upsample_nearest1d = [](const at::Tensor & input, at::OptionalSymIntArrayRef output_size, ::std::optional<at::ArrayRef<double>> scale_factors) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::upsample_nearest1d_symint(input, output_size, scale_factors);
      };
      return wrap(dispatch_upsample_nearest1d(_r.tensor(0), _r.symintlistOptional(1), _r.doublelistOptional(2)));
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::upsample_nearest1d(Tensor self, SymInt[1] output_size, float? scales=None) -> Tensor
        
        auto dispatch_upsample_nearest1d = [](const at::Tensor & self, c10::SymIntArrayRef output_size, ::std::optional<double> scales) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::upsample_nearest1d_symint(self, output_size, scales);
        };
        return wrap(dispatch_upsample_nearest1d(_r.tensor(0), _r.symintlist(1), _r.toDoubleOptional(2)));
      } else {
        // aten::upsample_nearest1d.out(Tensor self, SymInt[1] output_size, float? scales=None, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_upsample_nearest1d_out = [](at::Tensor out, const at::Tensor & self, c10::SymIntArrayRef output_size, ::std::optional<double> scales) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::upsample_nearest1d_symint_out(out, self, output_size, scales);
        };
        return wrap(dispatch_upsample_nearest1d_out(_r.tensor(3), _r.tensor(0), _r.symintlist(1), _r.toDoubleOptional(2)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// upsample_nearest2d
static PyObject * THPVariable_upsample_nearest2d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "upsample_nearest2d(Tensor input, SymIntArrayRef? output_size, ArrayRef<double>? scale_factors)",
    "upsample_nearest2d(Tensor input, SymIntArrayRef[2] output_size, double? scales_h=None, double? scales_w=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  switch (_r.idx) {
    case 0: {
      // aten::upsample_nearest2d.vec(Tensor input, SymInt[]? output_size, float[]? scale_factors) -> Tensor
      
      auto dispatch_upsample_nearest2d = [](const at::Tensor & input, at::OptionalSymIntArrayRef output_size, ::std::optional<at::ArrayRef<double>> scale_factors) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::upsample_nearest2d_symint(input, output_size, scale_factors);
      };
      return wrap(dispatch_upsample_nearest2d(_r.tensor(0), _r.symintlistOptional(1), _r.doublelistOptional(2)));
    }
    case 1: {
      if (_r.isNone(4)) {
        // aten::upsample_nearest2d(Tensor self, SymInt[2] output_size, float? scales_h=None, float? scales_w=None) -> Tensor
        
        auto dispatch_upsample_nearest2d = [](const at::Tensor & self, c10::SymIntArrayRef output_size, ::std::optional<double> scales_h, ::std::optional<double> scales_w) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::upsample_nearest2d_symint(self, output_size, scales_h, scales_w);
        };
        return wrap(dispatch_upsample_nearest2d(_r.tensor(0), _r.symintlist(1), _r.toDoubleOptional(2), _r.toDoubleOptional(3)));
      } else {
        // aten::upsample_nearest2d.out(Tensor self, SymInt[2] output_size, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_upsample_nearest2d_out = [](at::Tensor out, const at::Tensor & self, c10::SymIntArrayRef output_size, ::std::optional<double> scales_h, ::std::optional<double> scales_w) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::upsample_nearest2d_symint_out(out, self, output_size, scales_h, scales_w);
        };
        return wrap(dispatch_upsample_nearest2d_out(_r.tensor(4), _r.tensor(0), _r.symintlist(1), _r.toDoubleOptional(2), _r.toDoubleOptional(3)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// upsample_nearest3d
static PyObject * THPVariable_upsample_nearest3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "upsample_nearest3d(Tensor input, SymIntArrayRef? output_size, ArrayRef<double>? scale_factors)",
    "upsample_nearest3d(Tensor input, SymIntArrayRef[3] output_size, double? scales_d=None, double? scales_h=None, double? scales_w=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  switch (_r.idx) {
    case 0: {
      // aten::upsample_nearest3d.vec(Tensor input, SymInt[]? output_size, float[]? scale_factors) -> Tensor
      
      auto dispatch_upsample_nearest3d = [](const at::Tensor & input, at::OptionalSymIntArrayRef output_size, ::std::optional<at::ArrayRef<double>> scale_factors) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::upsample_nearest3d_symint(input, output_size, scale_factors);
      };
      return wrap(dispatch_upsample_nearest3d(_r.tensor(0), _r.symintlistOptional(1), _r.doublelistOptional(2)));
    }
    case 1: {
      if (_r.isNone(5)) {
        // aten::upsample_nearest3d(Tensor self, SymInt[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor
        
        auto dispatch_upsample_nearest3d = [](const at::Tensor & self, c10::SymIntArrayRef output_size, ::std::optional<double> scales_d, ::std::optional<double> scales_h, ::std::optional<double> scales_w) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::upsample_nearest3d_symint(self, output_size, scales_d, scales_h, scales_w);
        };
        return wrap(dispatch_upsample_nearest3d(_r.tensor(0), _r.symintlist(1), _r.toDoubleOptional(2), _r.toDoubleOptional(3), _r.toDoubleOptional(4)));
      } else {
        // aten::upsample_nearest3d.out(Tensor self, SymInt[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_upsample_nearest3d_out = [](at::Tensor out, const at::Tensor & self, c10::SymIntArrayRef output_size, ::std::optional<double> scales_d, ::std::optional<double> scales_h, ::std::optional<double> scales_w) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::upsample_nearest3d_symint_out(out, self, output_size, scales_d, scales_h, scales_w);
        };
        return wrap(dispatch_upsample_nearest3d_out(_r.tensor(5), _r.tensor(0), _r.symintlist(1), _r.toDoubleOptional(2), _r.toDoubleOptional(3), _r.toDoubleOptional(4)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// upsample_trilinear3d
static PyObject * THPVariable_upsample_trilinear3d(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "upsample_trilinear3d(Tensor input, SymIntArrayRef? output_size, bool align_corners, ArrayRef<double>? scale_factors)",
    "upsample_trilinear3d(Tensor input, SymIntArrayRef[3] output_size, bool align_corners, double? scales_d=None, double? scales_h=None, double? scales_w=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<7> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNNVariableFunctionsModule, "torch.nn");
  }
  switch (_r.idx) {
    case 0: {
      // aten::upsample_trilinear3d.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
      
      auto dispatch_upsample_trilinear3d = [](const at::Tensor & input, at::OptionalSymIntArrayRef output_size, bool align_corners, ::std::optional<at::ArrayRef<double>> scale_factors) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::upsample_trilinear3d_symint(input, output_size, align_corners, scale_factors);
      };
      return wrap(dispatch_upsample_trilinear3d(_r.tensor(0), _r.symintlistOptional(1), _r.toBool(2), _r.doublelistOptional(3)));
    }
    case 1: {
      if (_r.isNone(6)) {
        // aten::upsample_trilinear3d(Tensor self, SymInt[3] output_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor
        
        auto dispatch_upsample_trilinear3d = [](const at::Tensor & self, c10::SymIntArrayRef output_size, bool align_corners, ::std::optional<double> scales_d, ::std::optional<double> scales_h, ::std::optional<double> scales_w) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::upsample_trilinear3d_symint(self, output_size, align_corners, scales_d, scales_h, scales_w);
        };
        return wrap(dispatch_upsample_trilinear3d(_r.tensor(0), _r.symintlist(1), _r.toBool(2), _r.toDoubleOptional(3), _r.toDoubleOptional(4), _r.toDoubleOptional(5)));
      } else {
        // aten::upsample_trilinear3d.out(Tensor self, SymInt[3] output_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_upsample_trilinear3d_out = [](at::Tensor out, const at::Tensor & self, c10::SymIntArrayRef output_size, bool align_corners, ::std::optional<double> scales_d, ::std::optional<double> scales_h, ::std::optional<double> scales_w) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::upsample_trilinear3d_symint_out(out, self, output_size, align_corners, scales_d, scales_h, scales_w);
        };
        return wrap(dispatch_upsample_trilinear3d_out(_r.tensor(6), _r.tensor(0), _r.symintlist(1), _r.toBool(2), _r.toDoubleOptional(3), _r.toDoubleOptional(4), _r.toDoubleOptional(5)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

} // namespace torch::autograd
