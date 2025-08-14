#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// @generated from ../tools/autograd/templates/python_fft_functions.cpp

#include "torch/csrc/Device.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/python_fft_functions.h"
#include "torch/csrc/autograd/generated/python_return_types.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/autograd/utils/python_arg_parsing.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/csrc/utils/out_types.h"
#include "torch/csrc/utils/pycfunction_helpers.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/structseq.h"
#include "torch/csrc/utils/device_lazy_init.h"

#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/fft_fft.h>
#include <ATen/ops/fft_fft2.h>
#include <ATen/ops/fft_fftfreq.h>
#include <ATen/ops/fft_fftn.h>
#include <ATen/ops/fft_fftshift.h>
#include <ATen/ops/fft_hfft.h>
#include <ATen/ops/fft_hfft2.h>
#include <ATen/ops/fft_hfftn.h>
#include <ATen/ops/fft_ifft.h>
#include <ATen/ops/fft_ifft2.h>
#include <ATen/ops/fft_ifftn.h>
#include <ATen/ops/fft_ifftshift.h>
#include <ATen/ops/fft_ihfft.h>
#include <ATen/ops/fft_ihfft2.h>
#include <ATen/ops/fft_ihfftn.h>
#include <ATen/ops/fft_irfft.h>
#include <ATen/ops/fft_irfft2.h>
#include <ATen/ops/fft_irfftn.h>
#include <ATen/ops/fft_rfft.h>
#include <ATen/ops/fft_rfft2.h>
#include <ATen/ops/fft_rfftfreq.h>
#include <ATen/ops/fft_rfftn.h>
#endif

using at::Tensor;
using at::Device;
using at::Layout;
using at::Scalar;
using at::ScalarType;
using at::Backend;
using at::OptionalDeviceGuard;
using at::DeviceGuard;
using at::TensorOptions;
using at::IntArrayRef;
using at::Generator;
using at::TensorList;
using at::Dimname;
using at::DimnameList;

using torch::utils::check_out_type_matches;
using namespace torch::autograd::utils;

namespace torch::autograd {

// generated forward declarations start here

static PyObject * THPVariable_fft_fft(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fft_fft2(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fft_fftfreq(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fft_fftn(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fft_fftshift(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fft_hfft(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fft_hfft2(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fft_hfftn(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fft_ifft(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fft_ifft2(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fft_ifftn(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fft_ifftshift(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fft_ihfft(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fft_ihfft2(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fft_ihfftn(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fft_irfft(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fft_irfft2(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fft_irfftn(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fft_rfft(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fft_rfft2(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fft_rfftfreq(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_fft_rfftn(PyObject* self_, PyObject* args, PyObject* kwargs);

static PyMethodDef fft_functions[] = {
  {"fft_fft", castPyCFunctionWithKeywords(THPVariable_fft_fft), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"fft_fft2", castPyCFunctionWithKeywords(THPVariable_fft_fft2), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"fft_fftfreq", castPyCFunctionWithKeywords(THPVariable_fft_fftfreq), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"fft_fftn", castPyCFunctionWithKeywords(THPVariable_fft_fftn), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"fft_fftshift", castPyCFunctionWithKeywords(THPVariable_fft_fftshift), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"fft_hfft", castPyCFunctionWithKeywords(THPVariable_fft_hfft), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"fft_hfft2", castPyCFunctionWithKeywords(THPVariable_fft_hfft2), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"fft_hfftn", castPyCFunctionWithKeywords(THPVariable_fft_hfftn), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"fft_ifft", castPyCFunctionWithKeywords(THPVariable_fft_ifft), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"fft_ifft2", castPyCFunctionWithKeywords(THPVariable_fft_ifft2), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"fft_ifftn", castPyCFunctionWithKeywords(THPVariable_fft_ifftn), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"fft_ifftshift", castPyCFunctionWithKeywords(THPVariable_fft_ifftshift), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"fft_ihfft", castPyCFunctionWithKeywords(THPVariable_fft_ihfft), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"fft_ihfft2", castPyCFunctionWithKeywords(THPVariable_fft_ihfft2), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"fft_ihfftn", castPyCFunctionWithKeywords(THPVariable_fft_ihfftn), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"fft_irfft", castPyCFunctionWithKeywords(THPVariable_fft_irfft), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"fft_irfft2", castPyCFunctionWithKeywords(THPVariable_fft_irfft2), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"fft_irfftn", castPyCFunctionWithKeywords(THPVariable_fft_irfftn), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"fft_rfft", castPyCFunctionWithKeywords(THPVariable_fft_rfft), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"fft_rfft2", castPyCFunctionWithKeywords(THPVariable_fft_rfft2), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"fft_rfftfreq", castPyCFunctionWithKeywords(THPVariable_fft_rfftfreq), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"fft_rfftn", castPyCFunctionWithKeywords(THPVariable_fft_rfftn), METH_VARARGS | METH_KEYWORDS, nullptr},
  {NULL}
};

static PyObject* THPFFTVariableFunctionsModule = NULL;

void initFFTFunctions(PyObject* module) {
  static struct PyModuleDef def = {
     PyModuleDef_HEAD_INIT,
     "torch._C._fft",
     NULL,
     -1,
     fft_functions
  };
  PyObject* fft = PyModule_Create(&def);
  THPFFTVariableFunctionsModule = fft;
  if (!fft) {
    throw python_error();
  }
  // steals a reference to fft
  if (PyModule_AddObject(module, "_fft", fft) != 0) {
    throw python_error();
  }
}

// generated methods start here

// fft_fft
static PyObject * THPVariable_fft_fft(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fft_fft(Tensor input, SymInt? n=None, int64_t dim=-1, c10::string_view? norm=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPFFTVariableFunctionsModule, "torch.fft");
  }
  if (_r.isNone(4)) {
    // aten::fft_fft(Tensor self, SymInt? n=None, int dim=-1, str? norm=None) -> Tensor
    
    auto dispatch_fft_fft = [](const at::Tensor & self, ::std::optional<c10::SymInt> n, int64_t dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_fft_symint(self, n, dim, norm);
    };
    return wrap(dispatch_fft_fft(_r.tensor(0), _r.toSymIntOptional(1), _r.toInt64(2), _r.stringViewOptional(3)));
  } else {
    // aten::fft_fft.out(Tensor self, SymInt? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_fft_fft_out = [](at::Tensor out, const at::Tensor & self, ::std::optional<c10::SymInt> n, int64_t dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_fft_symint_out(out, self, n, dim, norm);
    };
    return wrap(dispatch_fft_fft_out(_r.tensor(4), _r.tensor(0), _r.toSymIntOptional(1), _r.toInt64(2), _r.stringViewOptional(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fft_fft2
static PyObject * THPVariable_fft_fft2(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fft_fft2(Tensor input, SymIntArrayRef[1]? s=None, IntArrayRef[1] dim={-2,-1}, c10::string_view? norm=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPFFTVariableFunctionsModule, "torch.fft");
  }
  if (_r.isNone(4)) {
    // aten::fft_fft2(Tensor self, SymInt[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor
    
    auto dispatch_fft_fft2 = [](const at::Tensor & self, at::OptionalSymIntArrayRef s, at::IntArrayRef dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_fft2_symint(self, s, dim, norm);
    };
    return wrap(dispatch_fft_fft2(_r.tensor(0), _r.symintlistOptional(1), _r.intlist(2), _r.stringViewOptional(3)));
  } else {
    // aten::fft_fft2.out(Tensor self, SymInt[1]? s=None, int[1] dim=[-2,-1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_fft_fft2_out = [](at::Tensor out, const at::Tensor & self, at::OptionalSymIntArrayRef s, at::IntArrayRef dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_fft2_symint_out(out, self, s, dim, norm);
    };
    return wrap(dispatch_fft_fft2_out(_r.tensor(4), _r.tensor(0), _r.symintlistOptional(1), _r.intlist(2), _r.stringViewOptional(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fft_fftfreq
static PyObject * THPVariable_fft_fftfreq(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fft_fftfreq(int64_t n, double d=1.0, *, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPFFTVariableFunctionsModule, "torch.fft");
  }
  if (_r.isNone(2)) {
    // aten::fft_fftfreq(int n, float d=1.0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
    const auto options = TensorOptions()
        .dtype(_r.scalartypeOptional(3))
        .device(_r.deviceWithDefault(5, torch::tensors::get_default_device()))
        .layout(_r.layoutOptional(4))
        .requires_grad(_r.toBool(7))
        .pinned_memory(_r.toBool(6));
    torch::utils::maybe_initialize_device(options);
    
    auto dispatch_fft_fftfreq = [](int64_t n, double d, at::TensorOptions options) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return torch::fft_fftfreq(n, d, options);
    };
    return wrap(dispatch_fft_fftfreq(_r.toInt64(0), _r.toDouble(1), options));
  } else {
    // aten::fft_fftfreq.out(int n, float d=1.0, *, Tensor(a!) out) -> Tensor(a!)
    check_out_type_matches(_r.tensor(2), _r.scalartypeOptional(3),
                           _r.isNone(3), _r.layoutOptional(4),
                           _r.deviceWithDefault(5, torch::tensors::get_default_device()), _r.isNone(5));
    
    auto dispatch_fft_fftfreq_out = [](at::Tensor out, int64_t n, double d) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_fftfreq_out(out, n, d);
    };
    return wrap(dispatch_fft_fftfreq_out(_r.tensor(2), _r.toInt64(0), _r.toDouble(1)).set_requires_grad(_r.toBool(7)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fft_fftn
static PyObject * THPVariable_fft_fftn(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fft_fftn(Tensor input, SymIntArrayRef[1]? s=None, IntArrayRef[1]? dim=None, c10::string_view? norm=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPFFTVariableFunctionsModule, "torch.fft");
  }
  if (_r.isNone(4)) {
    // aten::fft_fftn(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor
    
    auto dispatch_fft_fftn = [](const at::Tensor & self, at::OptionalSymIntArrayRef s, at::OptionalIntArrayRef dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_fftn_symint(self, s, dim, norm);
    };
    return wrap(dispatch_fft_fftn(_r.tensor(0), _r.symintlistOptional(1), _r.intlistOptional(2), _r.stringViewOptional(3)));
  } else {
    // aten::fft_fftn.out(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_fft_fftn_out = [](at::Tensor out, const at::Tensor & self, at::OptionalSymIntArrayRef s, at::OptionalIntArrayRef dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_fftn_symint_out(out, self, s, dim, norm);
    };
    return wrap(dispatch_fft_fftn_out(_r.tensor(4), _r.tensor(0), _r.symintlistOptional(1), _r.intlistOptional(2), _r.stringViewOptional(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fft_fftshift
static PyObject * THPVariable_fft_fftshift(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fft_fftshift(Tensor input, IntArrayRef[1]? dim=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPFFTVariableFunctionsModule, "torch.fft");
  }
  // aten::fft_fftshift(Tensor self, int[1]? dim=None) -> Tensor
  
  auto dispatch_fft_fftshift = [](const at::Tensor & self, at::OptionalIntArrayRef dim) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::fft_fftshift(self, dim);
  };
  return wrap(dispatch_fft_fftshift(_r.tensor(0), _r.intlistOptional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fft_hfft
static PyObject * THPVariable_fft_hfft(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fft_hfft(Tensor input, SymInt? n=None, int64_t dim=-1, c10::string_view? norm=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPFFTVariableFunctionsModule, "torch.fft");
  }
  if (_r.isNone(4)) {
    // aten::fft_hfft(Tensor self, SymInt? n=None, int dim=-1, str? norm=None) -> Tensor
    
    auto dispatch_fft_hfft = [](const at::Tensor & self, ::std::optional<c10::SymInt> n, int64_t dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_hfft_symint(self, n, dim, norm);
    };
    return wrap(dispatch_fft_hfft(_r.tensor(0), _r.toSymIntOptional(1), _r.toInt64(2), _r.stringViewOptional(3)));
  } else {
    // aten::fft_hfft.out(Tensor self, SymInt? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_fft_hfft_out = [](at::Tensor out, const at::Tensor & self, ::std::optional<c10::SymInt> n, int64_t dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_hfft_symint_out(out, self, n, dim, norm);
    };
    return wrap(dispatch_fft_hfft_out(_r.tensor(4), _r.tensor(0), _r.toSymIntOptional(1), _r.toInt64(2), _r.stringViewOptional(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fft_hfft2
static PyObject * THPVariable_fft_hfft2(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fft_hfft2(Tensor input, SymIntArrayRef[1]? s=None, IntArrayRef[1] dim={-2,-1}, c10::string_view? norm=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPFFTVariableFunctionsModule, "torch.fft");
  }
  if (_r.isNone(4)) {
    // aten::fft_hfft2(Tensor self, SymInt[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor
    
    auto dispatch_fft_hfft2 = [](const at::Tensor & self, at::OptionalSymIntArrayRef s, at::IntArrayRef dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_hfft2_symint(self, s, dim, norm);
    };
    return wrap(dispatch_fft_hfft2(_r.tensor(0), _r.symintlistOptional(1), _r.intlist(2), _r.stringViewOptional(3)));
  } else {
    // aten::fft_hfft2.out(Tensor self, SymInt[1]? s=None, int[1] dim=[-2,-1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_fft_hfft2_out = [](at::Tensor out, const at::Tensor & self, at::OptionalSymIntArrayRef s, at::IntArrayRef dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_hfft2_symint_out(out, self, s, dim, norm);
    };
    return wrap(dispatch_fft_hfft2_out(_r.tensor(4), _r.tensor(0), _r.symintlistOptional(1), _r.intlist(2), _r.stringViewOptional(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fft_hfftn
static PyObject * THPVariable_fft_hfftn(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fft_hfftn(Tensor input, SymIntArrayRef[1]? s=None, IntArrayRef[1]? dim=None, c10::string_view? norm=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPFFTVariableFunctionsModule, "torch.fft");
  }
  if (_r.isNone(4)) {
    // aten::fft_hfftn(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor
    
    auto dispatch_fft_hfftn = [](const at::Tensor & self, at::OptionalSymIntArrayRef s, at::OptionalIntArrayRef dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_hfftn_symint(self, s, dim, norm);
    };
    return wrap(dispatch_fft_hfftn(_r.tensor(0), _r.symintlistOptional(1), _r.intlistOptional(2), _r.stringViewOptional(3)));
  } else {
    // aten::fft_hfftn.out(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_fft_hfftn_out = [](at::Tensor out, const at::Tensor & self, at::OptionalSymIntArrayRef s, at::OptionalIntArrayRef dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_hfftn_symint_out(out, self, s, dim, norm);
    };
    return wrap(dispatch_fft_hfftn_out(_r.tensor(4), _r.tensor(0), _r.symintlistOptional(1), _r.intlistOptional(2), _r.stringViewOptional(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fft_ifft
static PyObject * THPVariable_fft_ifft(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fft_ifft(Tensor input, SymInt? n=None, int64_t dim=-1, c10::string_view? norm=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPFFTVariableFunctionsModule, "torch.fft");
  }
  if (_r.isNone(4)) {
    // aten::fft_ifft(Tensor self, SymInt? n=None, int dim=-1, str? norm=None) -> Tensor
    
    auto dispatch_fft_ifft = [](const at::Tensor & self, ::std::optional<c10::SymInt> n, int64_t dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_ifft_symint(self, n, dim, norm);
    };
    return wrap(dispatch_fft_ifft(_r.tensor(0), _r.toSymIntOptional(1), _r.toInt64(2), _r.stringViewOptional(3)));
  } else {
    // aten::fft_ifft.out(Tensor self, SymInt? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_fft_ifft_out = [](at::Tensor out, const at::Tensor & self, ::std::optional<c10::SymInt> n, int64_t dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_ifft_symint_out(out, self, n, dim, norm);
    };
    return wrap(dispatch_fft_ifft_out(_r.tensor(4), _r.tensor(0), _r.toSymIntOptional(1), _r.toInt64(2), _r.stringViewOptional(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fft_ifft2
static PyObject * THPVariable_fft_ifft2(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fft_ifft2(Tensor input, SymIntArrayRef[1]? s=None, IntArrayRef[1] dim={-2,-1}, c10::string_view? norm=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPFFTVariableFunctionsModule, "torch.fft");
  }
  if (_r.isNone(4)) {
    // aten::fft_ifft2(Tensor self, SymInt[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor
    
    auto dispatch_fft_ifft2 = [](const at::Tensor & self, at::OptionalSymIntArrayRef s, at::IntArrayRef dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_ifft2_symint(self, s, dim, norm);
    };
    return wrap(dispatch_fft_ifft2(_r.tensor(0), _r.symintlistOptional(1), _r.intlist(2), _r.stringViewOptional(3)));
  } else {
    // aten::fft_ifft2.out(Tensor self, SymInt[1]? s=None, int[1] dim=[-2,-1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_fft_ifft2_out = [](at::Tensor out, const at::Tensor & self, at::OptionalSymIntArrayRef s, at::IntArrayRef dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_ifft2_symint_out(out, self, s, dim, norm);
    };
    return wrap(dispatch_fft_ifft2_out(_r.tensor(4), _r.tensor(0), _r.symintlistOptional(1), _r.intlist(2), _r.stringViewOptional(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fft_ifftn
static PyObject * THPVariable_fft_ifftn(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fft_ifftn(Tensor input, SymIntArrayRef[1]? s=None, IntArrayRef[1]? dim=None, c10::string_view? norm=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPFFTVariableFunctionsModule, "torch.fft");
  }
  if (_r.isNone(4)) {
    // aten::fft_ifftn(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor
    
    auto dispatch_fft_ifftn = [](const at::Tensor & self, at::OptionalSymIntArrayRef s, at::OptionalIntArrayRef dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_ifftn_symint(self, s, dim, norm);
    };
    return wrap(dispatch_fft_ifftn(_r.tensor(0), _r.symintlistOptional(1), _r.intlistOptional(2), _r.stringViewOptional(3)));
  } else {
    // aten::fft_ifftn.out(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_fft_ifftn_out = [](at::Tensor out, const at::Tensor & self, at::OptionalSymIntArrayRef s, at::OptionalIntArrayRef dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_ifftn_symint_out(out, self, s, dim, norm);
    };
    return wrap(dispatch_fft_ifftn_out(_r.tensor(4), _r.tensor(0), _r.symintlistOptional(1), _r.intlistOptional(2), _r.stringViewOptional(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fft_ifftshift
static PyObject * THPVariable_fft_ifftshift(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fft_ifftshift(Tensor input, IntArrayRef[1]? dim=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPFFTVariableFunctionsModule, "torch.fft");
  }
  // aten::fft_ifftshift(Tensor self, int[1]? dim=None) -> Tensor
  
  auto dispatch_fft_ifftshift = [](const at::Tensor & self, at::OptionalIntArrayRef dim) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::fft_ifftshift(self, dim);
  };
  return wrap(dispatch_fft_ifftshift(_r.tensor(0), _r.intlistOptional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fft_ihfft
static PyObject * THPVariable_fft_ihfft(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fft_ihfft(Tensor input, SymInt? n=None, int64_t dim=-1, c10::string_view? norm=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPFFTVariableFunctionsModule, "torch.fft");
  }
  if (_r.isNone(4)) {
    // aten::fft_ihfft(Tensor self, SymInt? n=None, int dim=-1, str? norm=None) -> Tensor
    
    auto dispatch_fft_ihfft = [](const at::Tensor & self, ::std::optional<c10::SymInt> n, int64_t dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_ihfft_symint(self, n, dim, norm);
    };
    return wrap(dispatch_fft_ihfft(_r.tensor(0), _r.toSymIntOptional(1), _r.toInt64(2), _r.stringViewOptional(3)));
  } else {
    // aten::fft_ihfft.out(Tensor self, SymInt? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_fft_ihfft_out = [](at::Tensor out, const at::Tensor & self, ::std::optional<c10::SymInt> n, int64_t dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_ihfft_symint_out(out, self, n, dim, norm);
    };
    return wrap(dispatch_fft_ihfft_out(_r.tensor(4), _r.tensor(0), _r.toSymIntOptional(1), _r.toInt64(2), _r.stringViewOptional(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fft_ihfft2
static PyObject * THPVariable_fft_ihfft2(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fft_ihfft2(Tensor input, SymIntArrayRef[1]? s=None, IntArrayRef[1] dim={-2,-1}, c10::string_view? norm=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPFFTVariableFunctionsModule, "torch.fft");
  }
  if (_r.isNone(4)) {
    // aten::fft_ihfft2(Tensor self, SymInt[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor
    
    auto dispatch_fft_ihfft2 = [](const at::Tensor & self, at::OptionalSymIntArrayRef s, at::IntArrayRef dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_ihfft2_symint(self, s, dim, norm);
    };
    return wrap(dispatch_fft_ihfft2(_r.tensor(0), _r.symintlistOptional(1), _r.intlist(2), _r.stringViewOptional(3)));
  } else {
    // aten::fft_ihfft2.out(Tensor self, SymInt[1]? s=None, int[1] dim=[-2,-1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_fft_ihfft2_out = [](at::Tensor out, const at::Tensor & self, at::OptionalSymIntArrayRef s, at::IntArrayRef dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_ihfft2_symint_out(out, self, s, dim, norm);
    };
    return wrap(dispatch_fft_ihfft2_out(_r.tensor(4), _r.tensor(0), _r.symintlistOptional(1), _r.intlist(2), _r.stringViewOptional(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fft_ihfftn
static PyObject * THPVariable_fft_ihfftn(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fft_ihfftn(Tensor input, SymIntArrayRef[1]? s=None, IntArrayRef[1]? dim=None, c10::string_view? norm=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPFFTVariableFunctionsModule, "torch.fft");
  }
  if (_r.isNone(4)) {
    // aten::fft_ihfftn(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor
    
    auto dispatch_fft_ihfftn = [](const at::Tensor & self, at::OptionalSymIntArrayRef s, at::OptionalIntArrayRef dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_ihfftn_symint(self, s, dim, norm);
    };
    return wrap(dispatch_fft_ihfftn(_r.tensor(0), _r.symintlistOptional(1), _r.intlistOptional(2), _r.stringViewOptional(3)));
  } else {
    // aten::fft_ihfftn.out(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_fft_ihfftn_out = [](at::Tensor out, const at::Tensor & self, at::OptionalSymIntArrayRef s, at::OptionalIntArrayRef dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_ihfftn_symint_out(out, self, s, dim, norm);
    };
    return wrap(dispatch_fft_ihfftn_out(_r.tensor(4), _r.tensor(0), _r.symintlistOptional(1), _r.intlistOptional(2), _r.stringViewOptional(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fft_irfft
static PyObject * THPVariable_fft_irfft(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fft_irfft(Tensor input, SymInt? n=None, int64_t dim=-1, c10::string_view? norm=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPFFTVariableFunctionsModule, "torch.fft");
  }
  if (_r.isNone(4)) {
    // aten::fft_irfft(Tensor self, SymInt? n=None, int dim=-1, str? norm=None) -> Tensor
    
    auto dispatch_fft_irfft = [](const at::Tensor & self, ::std::optional<c10::SymInt> n, int64_t dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_irfft_symint(self, n, dim, norm);
    };
    return wrap(dispatch_fft_irfft(_r.tensor(0), _r.toSymIntOptional(1), _r.toInt64(2), _r.stringViewOptional(3)));
  } else {
    // aten::fft_irfft.out(Tensor self, SymInt? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_fft_irfft_out = [](at::Tensor out, const at::Tensor & self, ::std::optional<c10::SymInt> n, int64_t dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_irfft_symint_out(out, self, n, dim, norm);
    };
    return wrap(dispatch_fft_irfft_out(_r.tensor(4), _r.tensor(0), _r.toSymIntOptional(1), _r.toInt64(2), _r.stringViewOptional(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fft_irfft2
static PyObject * THPVariable_fft_irfft2(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fft_irfft2(Tensor input, SymIntArrayRef[1]? s=None, IntArrayRef[1] dim={-2,-1}, c10::string_view? norm=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPFFTVariableFunctionsModule, "torch.fft");
  }
  if (_r.isNone(4)) {
    // aten::fft_irfft2(Tensor self, SymInt[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor
    
    auto dispatch_fft_irfft2 = [](const at::Tensor & self, at::OptionalSymIntArrayRef s, at::IntArrayRef dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_irfft2_symint(self, s, dim, norm);
    };
    return wrap(dispatch_fft_irfft2(_r.tensor(0), _r.symintlistOptional(1), _r.intlist(2), _r.stringViewOptional(3)));
  } else {
    // aten::fft_irfft2.out(Tensor self, SymInt[1]? s=None, int[1] dim=[-2,-1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_fft_irfft2_out = [](at::Tensor out, const at::Tensor & self, at::OptionalSymIntArrayRef s, at::IntArrayRef dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_irfft2_symint_out(out, self, s, dim, norm);
    };
    return wrap(dispatch_fft_irfft2_out(_r.tensor(4), _r.tensor(0), _r.symintlistOptional(1), _r.intlist(2), _r.stringViewOptional(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fft_irfftn
static PyObject * THPVariable_fft_irfftn(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fft_irfftn(Tensor input, SymIntArrayRef[1]? s=None, IntArrayRef[1]? dim=None, c10::string_view? norm=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPFFTVariableFunctionsModule, "torch.fft");
  }
  if (_r.isNone(4)) {
    // aten::fft_irfftn(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor
    
    auto dispatch_fft_irfftn = [](const at::Tensor & self, at::OptionalSymIntArrayRef s, at::OptionalIntArrayRef dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_irfftn_symint(self, s, dim, norm);
    };
    return wrap(dispatch_fft_irfftn(_r.tensor(0), _r.symintlistOptional(1), _r.intlistOptional(2), _r.stringViewOptional(3)));
  } else {
    // aten::fft_irfftn.out(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_fft_irfftn_out = [](at::Tensor out, const at::Tensor & self, at::OptionalSymIntArrayRef s, at::OptionalIntArrayRef dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_irfftn_symint_out(out, self, s, dim, norm);
    };
    return wrap(dispatch_fft_irfftn_out(_r.tensor(4), _r.tensor(0), _r.symintlistOptional(1), _r.intlistOptional(2), _r.stringViewOptional(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fft_rfft
static PyObject * THPVariable_fft_rfft(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fft_rfft(Tensor input, SymInt? n=None, int64_t dim=-1, c10::string_view? norm=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPFFTVariableFunctionsModule, "torch.fft");
  }
  if (_r.isNone(4)) {
    // aten::fft_rfft(Tensor self, SymInt? n=None, int dim=-1, str? norm=None) -> Tensor
    
    auto dispatch_fft_rfft = [](const at::Tensor & self, ::std::optional<c10::SymInt> n, int64_t dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_rfft_symint(self, n, dim, norm);
    };
    return wrap(dispatch_fft_rfft(_r.tensor(0), _r.toSymIntOptional(1), _r.toInt64(2), _r.stringViewOptional(3)));
  } else {
    // aten::fft_rfft.out(Tensor self, SymInt? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_fft_rfft_out = [](at::Tensor out, const at::Tensor & self, ::std::optional<c10::SymInt> n, int64_t dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_rfft_symint_out(out, self, n, dim, norm);
    };
    return wrap(dispatch_fft_rfft_out(_r.tensor(4), _r.tensor(0), _r.toSymIntOptional(1), _r.toInt64(2), _r.stringViewOptional(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fft_rfft2
static PyObject * THPVariable_fft_rfft2(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fft_rfft2(Tensor input, SymIntArrayRef[1]? s=None, IntArrayRef[1] dim={-2,-1}, c10::string_view? norm=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPFFTVariableFunctionsModule, "torch.fft");
  }
  if (_r.isNone(4)) {
    // aten::fft_rfft2(Tensor self, SymInt[1]? s=None, int[1] dim=[-2,-1], str? norm=None) -> Tensor
    
    auto dispatch_fft_rfft2 = [](const at::Tensor & self, at::OptionalSymIntArrayRef s, at::IntArrayRef dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_rfft2_symint(self, s, dim, norm);
    };
    return wrap(dispatch_fft_rfft2(_r.tensor(0), _r.symintlistOptional(1), _r.intlist(2), _r.stringViewOptional(3)));
  } else {
    // aten::fft_rfft2.out(Tensor self, SymInt[1]? s=None, int[1] dim=[-2,-1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_fft_rfft2_out = [](at::Tensor out, const at::Tensor & self, at::OptionalSymIntArrayRef s, at::IntArrayRef dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_rfft2_symint_out(out, self, s, dim, norm);
    };
    return wrap(dispatch_fft_rfft2_out(_r.tensor(4), _r.tensor(0), _r.symintlistOptional(1), _r.intlist(2), _r.stringViewOptional(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fft_rfftfreq
static PyObject * THPVariable_fft_rfftfreq(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fft_rfftfreq(int64_t n, double d=1.0, *, Tensor out=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? requires_grad=False)",
  }, /*traceable=*/true);

  ParsedArgs<8> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPFFTVariableFunctionsModule, "torch.fft");
  }
  if (_r.isNone(2)) {
    // aten::fft_rfftfreq(int n, float d=1.0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
    const auto options = TensorOptions()
        .dtype(_r.scalartypeOptional(3))
        .device(_r.deviceWithDefault(5, torch::tensors::get_default_device()))
        .layout(_r.layoutOptional(4))
        .requires_grad(_r.toBool(7))
        .pinned_memory(_r.toBool(6));
    torch::utils::maybe_initialize_device(options);
    
    auto dispatch_fft_rfftfreq = [](int64_t n, double d, at::TensorOptions options) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return torch::fft_rfftfreq(n, d, options);
    };
    return wrap(dispatch_fft_rfftfreq(_r.toInt64(0), _r.toDouble(1), options));
  } else {
    // aten::fft_rfftfreq.out(int n, float d=1.0, *, Tensor(a!) out) -> Tensor(a!)
    check_out_type_matches(_r.tensor(2), _r.scalartypeOptional(3),
                           _r.isNone(3), _r.layoutOptional(4),
                           _r.deviceWithDefault(5, torch::tensors::get_default_device()), _r.isNone(5));
    
    auto dispatch_fft_rfftfreq_out = [](at::Tensor out, int64_t n, double d) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_rfftfreq_out(out, n, d);
    };
    return wrap(dispatch_fft_rfftfreq_out(_r.tensor(2), _r.toInt64(0), _r.toDouble(1)).set_requires_grad(_r.toBool(7)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// fft_rfftn
static PyObject * THPVariable_fft_rfftn(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "fft_rfftn(Tensor input, SymIntArrayRef[1]? s=None, IntArrayRef[1]? dim=None, c10::string_view? norm=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPFFTVariableFunctionsModule, "torch.fft");
  }
  if (_r.isNone(4)) {
    // aten::fft_rfftn(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor
    
    auto dispatch_fft_rfftn = [](const at::Tensor & self, at::OptionalSymIntArrayRef s, at::OptionalIntArrayRef dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_rfftn_symint(self, s, dim, norm);
    };
    return wrap(dispatch_fft_rfftn(_r.tensor(0), _r.symintlistOptional(1), _r.intlistOptional(2), _r.stringViewOptional(3)));
  } else {
    // aten::fft_rfftn.out(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_fft_rfftn_out = [](at::Tensor out, const at::Tensor & self, at::OptionalSymIntArrayRef s, at::OptionalIntArrayRef dim, ::std::optional<c10::string_view> norm) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::fft_rfftn_symint_out(out, self, s, dim, norm);
    };
    return wrap(dispatch_fft_rfftn_out(_r.tensor(4), _r.tensor(0), _r.symintlistOptional(1), _r.intlistOptional(2), _r.stringViewOptional(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

} // namespace torch::autograd
