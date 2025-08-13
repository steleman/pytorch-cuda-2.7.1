#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// @generated from ../tools/autograd/templates/python_special_functions.cpp

#include "torch/csrc/Device.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/python_special_functions.h"
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

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/special_airy_ai.h>
#include <ATen/ops/special_bessel_j0.h>
#include <ATen/ops/special_bessel_j1.h>
#include <ATen/ops/special_bessel_y0.h>
#include <ATen/ops/special_bessel_y1.h>
#include <ATen/ops/special_chebyshev_polynomial_t.h>
#include <ATen/ops/special_chebyshev_polynomial_u.h>
#include <ATen/ops/special_chebyshev_polynomial_v.h>
#include <ATen/ops/special_chebyshev_polynomial_w.h>
#include <ATen/ops/special_digamma.h>
#include <ATen/ops/special_entr.h>
#include <ATen/ops/special_erf.h>
#include <ATen/ops/special_erfc.h>
#include <ATen/ops/special_erfcx.h>
#include <ATen/ops/special_erfinv.h>
#include <ATen/ops/special_exp2.h>
#include <ATen/ops/special_expit.h>
#include <ATen/ops/special_expm1.h>
#include <ATen/ops/special_gammainc.h>
#include <ATen/ops/special_gammaincc.h>
#include <ATen/ops/special_gammaln.h>
#include <ATen/ops/special_hermite_polynomial_h.h>
#include <ATen/ops/special_hermite_polynomial_he.h>
#include <ATen/ops/special_i0.h>
#include <ATen/ops/special_i0e.h>
#include <ATen/ops/special_i1.h>
#include <ATen/ops/special_i1e.h>
#include <ATen/ops/special_laguerre_polynomial_l.h>
#include <ATen/ops/special_legendre_polynomial_p.h>
#include <ATen/ops/special_log1p.h>
#include <ATen/ops/special_log_ndtr.h>
#include <ATen/ops/special_log_softmax.h>
#include <ATen/ops/special_logit.h>
#include <ATen/ops/special_logsumexp.h>
#include <ATen/ops/special_modified_bessel_i0.h>
#include <ATen/ops/special_modified_bessel_i1.h>
#include <ATen/ops/special_modified_bessel_k0.h>
#include <ATen/ops/special_modified_bessel_k1.h>
#include <ATen/ops/special_multigammaln.h>
#include <ATen/ops/special_ndtr.h>
#include <ATen/ops/special_ndtri.h>
#include <ATen/ops/special_polygamma.h>
#include <ATen/ops/special_psi.h>
#include <ATen/ops/special_round.h>
#include <ATen/ops/special_scaled_modified_bessel_k0.h>
#include <ATen/ops/special_scaled_modified_bessel_k1.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_t.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_u.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_v.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_w.h>
#include <ATen/ops/special_sinc.h>
#include <ATen/ops/special_softmax.h>
#include <ATen/ops/special_spherical_bessel_j0.h>
#include <ATen/ops/special_xlog1py.h>
#include <ATen/ops/special_xlogy.h>
#include <ATen/ops/special_zeta.h>
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

static PyObject * THPVariable_special_airy_ai(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_bessel_j0(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_bessel_j1(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_bessel_y0(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_bessel_y1(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_chebyshev_polynomial_t(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_chebyshev_polynomial_u(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_chebyshev_polynomial_v(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_chebyshev_polynomial_w(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_digamma(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_entr(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_erf(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_erfc(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_erfcx(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_erfinv(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_exp2(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_expit(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_expm1(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_gammainc(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_gammaincc(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_gammaln(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_hermite_polynomial_h(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_hermite_polynomial_he(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_i0(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_i0e(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_i1(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_i1e(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_laguerre_polynomial_l(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_legendre_polynomial_p(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_log1p(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_log_ndtr(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_log_softmax(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_logit(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_logsumexp(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_modified_bessel_i0(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_modified_bessel_i1(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_modified_bessel_k0(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_modified_bessel_k1(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_multigammaln(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_ndtr(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_ndtri(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_polygamma(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_psi(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_round(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_scaled_modified_bessel_k0(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_scaled_modified_bessel_k1(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_shifted_chebyshev_polynomial_t(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_shifted_chebyshev_polynomial_u(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_shifted_chebyshev_polynomial_v(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_shifted_chebyshev_polynomial_w(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_sinc(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_softmax(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_spherical_bessel_j0(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_xlog1py(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_xlogy(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_zeta(PyObject* self_, PyObject* args, PyObject* kwargs);

static PyMethodDef special_functions[] = {
  {"special_airy_ai", castPyCFunctionWithKeywords(THPVariable_special_airy_ai), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_bessel_j0", castPyCFunctionWithKeywords(THPVariable_special_bessel_j0), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_bessel_j1", castPyCFunctionWithKeywords(THPVariable_special_bessel_j1), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_bessel_y0", castPyCFunctionWithKeywords(THPVariable_special_bessel_y0), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_bessel_y1", castPyCFunctionWithKeywords(THPVariable_special_bessel_y1), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_chebyshev_polynomial_t", castPyCFunctionWithKeywords(THPVariable_special_chebyshev_polynomial_t), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_chebyshev_polynomial_u", castPyCFunctionWithKeywords(THPVariable_special_chebyshev_polynomial_u), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_chebyshev_polynomial_v", castPyCFunctionWithKeywords(THPVariable_special_chebyshev_polynomial_v), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_chebyshev_polynomial_w", castPyCFunctionWithKeywords(THPVariable_special_chebyshev_polynomial_w), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_digamma", castPyCFunctionWithKeywords(THPVariable_special_digamma), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_entr", castPyCFunctionWithKeywords(THPVariable_special_entr), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_erf", castPyCFunctionWithKeywords(THPVariable_special_erf), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_erfc", castPyCFunctionWithKeywords(THPVariable_special_erfc), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_erfcx", castPyCFunctionWithKeywords(THPVariable_special_erfcx), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_erfinv", castPyCFunctionWithKeywords(THPVariable_special_erfinv), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_exp2", castPyCFunctionWithKeywords(THPVariable_special_exp2), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_expit", castPyCFunctionWithKeywords(THPVariable_special_expit), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_expm1", castPyCFunctionWithKeywords(THPVariable_special_expm1), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_gammainc", castPyCFunctionWithKeywords(THPVariable_special_gammainc), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_gammaincc", castPyCFunctionWithKeywords(THPVariable_special_gammaincc), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_gammaln", castPyCFunctionWithKeywords(THPVariable_special_gammaln), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_hermite_polynomial_h", castPyCFunctionWithKeywords(THPVariable_special_hermite_polynomial_h), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_hermite_polynomial_he", castPyCFunctionWithKeywords(THPVariable_special_hermite_polynomial_he), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_i0", castPyCFunctionWithKeywords(THPVariable_special_i0), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_i0e", castPyCFunctionWithKeywords(THPVariable_special_i0e), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_i1", castPyCFunctionWithKeywords(THPVariable_special_i1), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_i1e", castPyCFunctionWithKeywords(THPVariable_special_i1e), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_laguerre_polynomial_l", castPyCFunctionWithKeywords(THPVariable_special_laguerre_polynomial_l), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_legendre_polynomial_p", castPyCFunctionWithKeywords(THPVariable_special_legendre_polynomial_p), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_log1p", castPyCFunctionWithKeywords(THPVariable_special_log1p), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_log_ndtr", castPyCFunctionWithKeywords(THPVariable_special_log_ndtr), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_log_softmax", castPyCFunctionWithKeywords(THPVariable_special_log_softmax), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_logit", castPyCFunctionWithKeywords(THPVariable_special_logit), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_logsumexp", castPyCFunctionWithKeywords(THPVariable_special_logsumexp), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_modified_bessel_i0", castPyCFunctionWithKeywords(THPVariable_special_modified_bessel_i0), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_modified_bessel_i1", castPyCFunctionWithKeywords(THPVariable_special_modified_bessel_i1), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_modified_bessel_k0", castPyCFunctionWithKeywords(THPVariable_special_modified_bessel_k0), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_modified_bessel_k1", castPyCFunctionWithKeywords(THPVariable_special_modified_bessel_k1), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_multigammaln", castPyCFunctionWithKeywords(THPVariable_special_multigammaln), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_ndtr", castPyCFunctionWithKeywords(THPVariable_special_ndtr), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_ndtri", castPyCFunctionWithKeywords(THPVariable_special_ndtri), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_polygamma", castPyCFunctionWithKeywords(THPVariable_special_polygamma), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_psi", castPyCFunctionWithKeywords(THPVariable_special_psi), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_round", castPyCFunctionWithKeywords(THPVariable_special_round), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_scaled_modified_bessel_k0", castPyCFunctionWithKeywords(THPVariable_special_scaled_modified_bessel_k0), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_scaled_modified_bessel_k1", castPyCFunctionWithKeywords(THPVariable_special_scaled_modified_bessel_k1), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_shifted_chebyshev_polynomial_t", castPyCFunctionWithKeywords(THPVariable_special_shifted_chebyshev_polynomial_t), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_shifted_chebyshev_polynomial_u", castPyCFunctionWithKeywords(THPVariable_special_shifted_chebyshev_polynomial_u), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_shifted_chebyshev_polynomial_v", castPyCFunctionWithKeywords(THPVariable_special_shifted_chebyshev_polynomial_v), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_shifted_chebyshev_polynomial_w", castPyCFunctionWithKeywords(THPVariable_special_shifted_chebyshev_polynomial_w), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_sinc", castPyCFunctionWithKeywords(THPVariable_special_sinc), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_softmax", castPyCFunctionWithKeywords(THPVariable_special_softmax), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_spherical_bessel_j0", castPyCFunctionWithKeywords(THPVariable_special_spherical_bessel_j0), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_xlog1py", castPyCFunctionWithKeywords(THPVariable_special_xlog1py), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_xlogy", castPyCFunctionWithKeywords(THPVariable_special_xlogy), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"special_zeta", castPyCFunctionWithKeywords(THPVariable_special_zeta), METH_VARARGS | METH_KEYWORDS, nullptr},
  {NULL}
};

static PyObject* THPSpecialVariableFunctionsModule = NULL;

void initSpecialFunctions(PyObject* module) {
  static struct PyModuleDef def = {
     PyModuleDef_HEAD_INIT,
     "torch._C._special",
     NULL,
     -1,
     special_functions
  };
  PyObject* special = PyModule_Create(&def);
  THPSpecialVariableFunctionsModule = special;
  if (!special) {
    throw python_error();
  }
  // steals a reference to special
  if (PyModule_AddObject(module, "_special", special) != 0) {
    throw python_error();
  }
}

// generated methods start here

// special_airy_ai
static PyObject * THPVariable_special_airy_ai(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_airy_ai(Tensor x, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_airy_ai(Tensor x) -> Tensor
    
    auto dispatch_special_airy_ai = [](const at::Tensor & x) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_airy_ai(x);
    };
    return wrap(dispatch_special_airy_ai(_r.tensor(0)));
  } else {
    // aten::special_airy_ai.out(Tensor x, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_airy_ai_out = [](at::Tensor out, const at::Tensor & x) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_airy_ai_out(out, x);
    };
    return wrap(dispatch_special_airy_ai_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_bessel_j0
static PyObject * THPVariable_special_bessel_j0(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_bessel_j0(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_bessel_j0(Tensor self) -> Tensor
    
    auto dispatch_special_bessel_j0 = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_bessel_j0(self);
    };
    return wrap(dispatch_special_bessel_j0(_r.tensor(0)));
  } else {
    // aten::special_bessel_j0.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_bessel_j0_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_bessel_j0_out(out, self);
    };
    return wrap(dispatch_special_bessel_j0_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_bessel_j1
static PyObject * THPVariable_special_bessel_j1(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_bessel_j1(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_bessel_j1(Tensor self) -> Tensor
    
    auto dispatch_special_bessel_j1 = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_bessel_j1(self);
    };
    return wrap(dispatch_special_bessel_j1(_r.tensor(0)));
  } else {
    // aten::special_bessel_j1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_bessel_j1_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_bessel_j1_out(out, self);
    };
    return wrap(dispatch_special_bessel_j1_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_bessel_y0
static PyObject * THPVariable_special_bessel_y0(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_bessel_y0(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_bessel_y0(Tensor self) -> Tensor
    
    auto dispatch_special_bessel_y0 = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_bessel_y0(self);
    };
    return wrap(dispatch_special_bessel_y0(_r.tensor(0)));
  } else {
    // aten::special_bessel_y0.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_bessel_y0_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_bessel_y0_out(out, self);
    };
    return wrap(dispatch_special_bessel_y0_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_bessel_y1
static PyObject * THPVariable_special_bessel_y1(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_bessel_y1(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_bessel_y1(Tensor self) -> Tensor
    
    auto dispatch_special_bessel_y1 = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_bessel_y1(self);
    };
    return wrap(dispatch_special_bessel_y1(_r.tensor(0)));
  } else {
    // aten::special_bessel_y1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_bessel_y1_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_bessel_y1_out(out, self);
    };
    return wrap(dispatch_special_bessel_y1_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// special_chebyshev_polynomial_t
static PyObject * THPVariable_special_chebyshev_polynomial_t(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_chebyshev_polynomial_t(Tensor x, Tensor n, *, Tensor out=None)",
    "special_chebyshev_polynomial_t(Scalar x, Tensor n, *, Tensor out=None)",
    "special_chebyshev_polynomial_t(Tensor x, Scalar n, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::special_chebyshev_polynomial_t(Tensor x, Tensor n) -> Tensor
        
        auto dispatch_special_chebyshev_polynomial_t = [](const at::Tensor & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_chebyshev_polynomial_t(x, n);
        };
        return wrap(dispatch_special_chebyshev_polynomial_t(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::special_chebyshev_polynomial_t.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_chebyshev_polynomial_t_out = [](at::Tensor out, const at::Tensor & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_chebyshev_polynomial_t_out(out, x, n);
        };
        return wrap(dispatch_special_chebyshev_polynomial_t_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::special_chebyshev_polynomial_t.x_scalar(Scalar x, Tensor n) -> Tensor
        
        auto dispatch_special_chebyshev_polynomial_t = [](const at::Scalar & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_chebyshev_polynomial_t(x, n);
        };
        return wrap(dispatch_special_chebyshev_polynomial_t(_r.scalar(0), _r.tensor(1)));
      } else {
        // aten::special_chebyshev_polynomial_t.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_chebyshev_polynomial_t_out = [](at::Tensor out, const at::Scalar & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_chebyshev_polynomial_t_out(out, x, n);
        };
        return wrap(dispatch_special_chebyshev_polynomial_t_out(_r.tensor(2), _r.scalar(0), _r.tensor(1)));
      }
    }
    case 2: {
      if (_r.isNone(2)) {
        // aten::special_chebyshev_polynomial_t.n_scalar(Tensor x, Scalar n) -> Tensor
        
        auto dispatch_special_chebyshev_polynomial_t = [](const at::Tensor & x, const at::Scalar & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_chebyshev_polynomial_t(x, n);
        };
        return wrap(dispatch_special_chebyshev_polynomial_t(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::special_chebyshev_polynomial_t.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_chebyshev_polynomial_t_out = [](at::Tensor out, const at::Tensor & x, const at::Scalar & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_chebyshev_polynomial_t_out(out, x, n);
        };
        return wrap(dispatch_special_chebyshev_polynomial_t_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// special_chebyshev_polynomial_u
static PyObject * THPVariable_special_chebyshev_polynomial_u(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_chebyshev_polynomial_u(Tensor x, Tensor n, *, Tensor out=None)",
    "special_chebyshev_polynomial_u(Scalar x, Tensor n, *, Tensor out=None)",
    "special_chebyshev_polynomial_u(Tensor x, Scalar n, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::special_chebyshev_polynomial_u(Tensor x, Tensor n) -> Tensor
        
        auto dispatch_special_chebyshev_polynomial_u = [](const at::Tensor & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_chebyshev_polynomial_u(x, n);
        };
        return wrap(dispatch_special_chebyshev_polynomial_u(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::special_chebyshev_polynomial_u.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_chebyshev_polynomial_u_out = [](at::Tensor out, const at::Tensor & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_chebyshev_polynomial_u_out(out, x, n);
        };
        return wrap(dispatch_special_chebyshev_polynomial_u_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::special_chebyshev_polynomial_u.x_scalar(Scalar x, Tensor n) -> Tensor
        
        auto dispatch_special_chebyshev_polynomial_u = [](const at::Scalar & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_chebyshev_polynomial_u(x, n);
        };
        return wrap(dispatch_special_chebyshev_polynomial_u(_r.scalar(0), _r.tensor(1)));
      } else {
        // aten::special_chebyshev_polynomial_u.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_chebyshev_polynomial_u_out = [](at::Tensor out, const at::Scalar & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_chebyshev_polynomial_u_out(out, x, n);
        };
        return wrap(dispatch_special_chebyshev_polynomial_u_out(_r.tensor(2), _r.scalar(0), _r.tensor(1)));
      }
    }
    case 2: {
      if (_r.isNone(2)) {
        // aten::special_chebyshev_polynomial_u.n_scalar(Tensor x, Scalar n) -> Tensor
        
        auto dispatch_special_chebyshev_polynomial_u = [](const at::Tensor & x, const at::Scalar & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_chebyshev_polynomial_u(x, n);
        };
        return wrap(dispatch_special_chebyshev_polynomial_u(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::special_chebyshev_polynomial_u.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_chebyshev_polynomial_u_out = [](at::Tensor out, const at::Tensor & x, const at::Scalar & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_chebyshev_polynomial_u_out(out, x, n);
        };
        return wrap(dispatch_special_chebyshev_polynomial_u_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// special_chebyshev_polynomial_v
static PyObject * THPVariable_special_chebyshev_polynomial_v(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_chebyshev_polynomial_v(Tensor x, Tensor n, *, Tensor out=None)",
    "special_chebyshev_polynomial_v(Scalar x, Tensor n, *, Tensor out=None)",
    "special_chebyshev_polynomial_v(Tensor x, Scalar n, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::special_chebyshev_polynomial_v(Tensor x, Tensor n) -> Tensor
        
        auto dispatch_special_chebyshev_polynomial_v = [](const at::Tensor & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_chebyshev_polynomial_v(x, n);
        };
        return wrap(dispatch_special_chebyshev_polynomial_v(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::special_chebyshev_polynomial_v.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_chebyshev_polynomial_v_out = [](at::Tensor out, const at::Tensor & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_chebyshev_polynomial_v_out(out, x, n);
        };
        return wrap(dispatch_special_chebyshev_polynomial_v_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::special_chebyshev_polynomial_v.x_scalar(Scalar x, Tensor n) -> Tensor
        
        auto dispatch_special_chebyshev_polynomial_v = [](const at::Scalar & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_chebyshev_polynomial_v(x, n);
        };
        return wrap(dispatch_special_chebyshev_polynomial_v(_r.scalar(0), _r.tensor(1)));
      } else {
        // aten::special_chebyshev_polynomial_v.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_chebyshev_polynomial_v_out = [](at::Tensor out, const at::Scalar & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_chebyshev_polynomial_v_out(out, x, n);
        };
        return wrap(dispatch_special_chebyshev_polynomial_v_out(_r.tensor(2), _r.scalar(0), _r.tensor(1)));
      }
    }
    case 2: {
      if (_r.isNone(2)) {
        // aten::special_chebyshev_polynomial_v.n_scalar(Tensor x, Scalar n) -> Tensor
        
        auto dispatch_special_chebyshev_polynomial_v = [](const at::Tensor & x, const at::Scalar & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_chebyshev_polynomial_v(x, n);
        };
        return wrap(dispatch_special_chebyshev_polynomial_v(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::special_chebyshev_polynomial_v.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_chebyshev_polynomial_v_out = [](at::Tensor out, const at::Tensor & x, const at::Scalar & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_chebyshev_polynomial_v_out(out, x, n);
        };
        return wrap(dispatch_special_chebyshev_polynomial_v_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// special_chebyshev_polynomial_w
static PyObject * THPVariable_special_chebyshev_polynomial_w(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_chebyshev_polynomial_w(Tensor x, Tensor n, *, Tensor out=None)",
    "special_chebyshev_polynomial_w(Scalar x, Tensor n, *, Tensor out=None)",
    "special_chebyshev_polynomial_w(Tensor x, Scalar n, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::special_chebyshev_polynomial_w(Tensor x, Tensor n) -> Tensor
        
        auto dispatch_special_chebyshev_polynomial_w = [](const at::Tensor & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_chebyshev_polynomial_w(x, n);
        };
        return wrap(dispatch_special_chebyshev_polynomial_w(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::special_chebyshev_polynomial_w.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_chebyshev_polynomial_w_out = [](at::Tensor out, const at::Tensor & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_chebyshev_polynomial_w_out(out, x, n);
        };
        return wrap(dispatch_special_chebyshev_polynomial_w_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::special_chebyshev_polynomial_w.x_scalar(Scalar x, Tensor n) -> Tensor
        
        auto dispatch_special_chebyshev_polynomial_w = [](const at::Scalar & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_chebyshev_polynomial_w(x, n);
        };
        return wrap(dispatch_special_chebyshev_polynomial_w(_r.scalar(0), _r.tensor(1)));
      } else {
        // aten::special_chebyshev_polynomial_w.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_chebyshev_polynomial_w_out = [](at::Tensor out, const at::Scalar & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_chebyshev_polynomial_w_out(out, x, n);
        };
        return wrap(dispatch_special_chebyshev_polynomial_w_out(_r.tensor(2), _r.scalar(0), _r.tensor(1)));
      }
    }
    case 2: {
      if (_r.isNone(2)) {
        // aten::special_chebyshev_polynomial_w.n_scalar(Tensor x, Scalar n) -> Tensor
        
        auto dispatch_special_chebyshev_polynomial_w = [](const at::Tensor & x, const at::Scalar & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_chebyshev_polynomial_w(x, n);
        };
        return wrap(dispatch_special_chebyshev_polynomial_w(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::special_chebyshev_polynomial_w.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_chebyshev_polynomial_w_out = [](at::Tensor out, const at::Tensor & x, const at::Scalar & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_chebyshev_polynomial_w_out(out, x, n);
        };
        return wrap(dispatch_special_chebyshev_polynomial_w_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_digamma
static PyObject * THPVariable_special_digamma(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_digamma(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_digamma(Tensor self) -> Tensor
    
    auto dispatch_special_digamma = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_digamma(self);
    };
    return wrap(dispatch_special_digamma(_r.tensor(0)));
  } else {
    // aten::special_digamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_digamma_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_digamma_out(out, self);
    };
    return wrap(dispatch_special_digamma_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_entr
static PyObject * THPVariable_special_entr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_entr(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_entr(Tensor self) -> Tensor
    
    auto dispatch_special_entr = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_entr(self);
    };
    return wrap(dispatch_special_entr(_r.tensor(0)));
  } else {
    // aten::special_entr.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_entr_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_entr_out(out, self);
    };
    return wrap(dispatch_special_entr_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_erf
static PyObject * THPVariable_special_erf(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_erf(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_erf(Tensor self) -> Tensor
    
    auto dispatch_special_erf = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_erf(self);
    };
    return wrap(dispatch_special_erf(_r.tensor(0)));
  } else {
    // aten::special_erf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_erf_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_erf_out(out, self);
    };
    return wrap(dispatch_special_erf_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_erfc
static PyObject * THPVariable_special_erfc(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_erfc(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_erfc(Tensor self) -> Tensor
    
    auto dispatch_special_erfc = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_erfc(self);
    };
    return wrap(dispatch_special_erfc(_r.tensor(0)));
  } else {
    // aten::special_erfc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_erfc_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_erfc_out(out, self);
    };
    return wrap(dispatch_special_erfc_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_erfcx
static PyObject * THPVariable_special_erfcx(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_erfcx(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_erfcx(Tensor self) -> Tensor
    
    auto dispatch_special_erfcx = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_erfcx(self);
    };
    return wrap(dispatch_special_erfcx(_r.tensor(0)));
  } else {
    // aten::special_erfcx.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_erfcx_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_erfcx_out(out, self);
    };
    return wrap(dispatch_special_erfcx_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_erfinv
static PyObject * THPVariable_special_erfinv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_erfinv(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_erfinv(Tensor self) -> Tensor
    
    auto dispatch_special_erfinv = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_erfinv(self);
    };
    return wrap(dispatch_special_erfinv(_r.tensor(0)));
  } else {
    // aten::special_erfinv.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_erfinv_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_erfinv_out(out, self);
    };
    return wrap(dispatch_special_erfinv_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_exp2
static PyObject * THPVariable_special_exp2(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_exp2(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_exp2(Tensor self) -> Tensor
    
    auto dispatch_special_exp2 = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_exp2(self);
    };
    return wrap(dispatch_special_exp2(_r.tensor(0)));
  } else {
    // aten::special_exp2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_exp2_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_exp2_out(out, self);
    };
    return wrap(dispatch_special_exp2_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_expit
static PyObject * THPVariable_special_expit(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_expit(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_expit(Tensor self) -> Tensor
    
    auto dispatch_special_expit = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_expit(self);
    };
    return wrap(dispatch_special_expit(_r.tensor(0)));
  } else {
    // aten::special_expit.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_expit_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_expit_out(out, self);
    };
    return wrap(dispatch_special_expit_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_expm1
static PyObject * THPVariable_special_expm1(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_expm1(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_expm1(Tensor self) -> Tensor
    
    auto dispatch_special_expm1 = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_expm1(self);
    };
    return wrap(dispatch_special_expm1(_r.tensor(0)));
  } else {
    // aten::special_expm1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_expm1_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_expm1_out(out, self);
    };
    return wrap(dispatch_special_expm1_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_gammainc
static PyObject * THPVariable_special_gammainc(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_gammainc(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(2)) {
    // aten::special_gammainc(Tensor self, Tensor other) -> Tensor
    
    auto dispatch_special_gammainc = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_gammainc(self, other);
    };
    return wrap(dispatch_special_gammainc(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::special_gammainc.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_gammainc_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_gammainc_out(out, self, other);
    };
    return wrap(dispatch_special_gammainc_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_gammaincc
static PyObject * THPVariable_special_gammaincc(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_gammaincc(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(2)) {
    // aten::special_gammaincc(Tensor self, Tensor other) -> Tensor
    
    auto dispatch_special_gammaincc = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_gammaincc(self, other);
    };
    return wrap(dispatch_special_gammaincc(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::special_gammaincc.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_gammaincc_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_gammaincc_out(out, self, other);
    };
    return wrap(dispatch_special_gammaincc_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_gammaln
static PyObject * THPVariable_special_gammaln(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_gammaln(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_gammaln(Tensor self) -> Tensor
    
    auto dispatch_special_gammaln = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_gammaln(self);
    };
    return wrap(dispatch_special_gammaln(_r.tensor(0)));
  } else {
    // aten::special_gammaln.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_gammaln_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_gammaln_out(out, self);
    };
    return wrap(dispatch_special_gammaln_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// special_hermite_polynomial_h
static PyObject * THPVariable_special_hermite_polynomial_h(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_hermite_polynomial_h(Tensor x, Tensor n, *, Tensor out=None)",
    "special_hermite_polynomial_h(Scalar x, Tensor n, *, Tensor out=None)",
    "special_hermite_polynomial_h(Tensor x, Scalar n, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::special_hermite_polynomial_h(Tensor x, Tensor n) -> Tensor
        
        auto dispatch_special_hermite_polynomial_h = [](const at::Tensor & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_hermite_polynomial_h(x, n);
        };
        return wrap(dispatch_special_hermite_polynomial_h(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::special_hermite_polynomial_h.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_hermite_polynomial_h_out = [](at::Tensor out, const at::Tensor & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_hermite_polynomial_h_out(out, x, n);
        };
        return wrap(dispatch_special_hermite_polynomial_h_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::special_hermite_polynomial_h.x_scalar(Scalar x, Tensor n) -> Tensor
        
        auto dispatch_special_hermite_polynomial_h = [](const at::Scalar & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_hermite_polynomial_h(x, n);
        };
        return wrap(dispatch_special_hermite_polynomial_h(_r.scalar(0), _r.tensor(1)));
      } else {
        // aten::special_hermite_polynomial_h.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_hermite_polynomial_h_out = [](at::Tensor out, const at::Scalar & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_hermite_polynomial_h_out(out, x, n);
        };
        return wrap(dispatch_special_hermite_polynomial_h_out(_r.tensor(2), _r.scalar(0), _r.tensor(1)));
      }
    }
    case 2: {
      if (_r.isNone(2)) {
        // aten::special_hermite_polynomial_h.n_scalar(Tensor x, Scalar n) -> Tensor
        
        auto dispatch_special_hermite_polynomial_h = [](const at::Tensor & x, const at::Scalar & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_hermite_polynomial_h(x, n);
        };
        return wrap(dispatch_special_hermite_polynomial_h(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::special_hermite_polynomial_h.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_hermite_polynomial_h_out = [](at::Tensor out, const at::Tensor & x, const at::Scalar & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_hermite_polynomial_h_out(out, x, n);
        };
        return wrap(dispatch_special_hermite_polynomial_h_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// special_hermite_polynomial_he
static PyObject * THPVariable_special_hermite_polynomial_he(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_hermite_polynomial_he(Tensor x, Tensor n, *, Tensor out=None)",
    "special_hermite_polynomial_he(Scalar x, Tensor n, *, Tensor out=None)",
    "special_hermite_polynomial_he(Tensor x, Scalar n, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::special_hermite_polynomial_he(Tensor x, Tensor n) -> Tensor
        
        auto dispatch_special_hermite_polynomial_he = [](const at::Tensor & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_hermite_polynomial_he(x, n);
        };
        return wrap(dispatch_special_hermite_polynomial_he(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::special_hermite_polynomial_he.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_hermite_polynomial_he_out = [](at::Tensor out, const at::Tensor & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_hermite_polynomial_he_out(out, x, n);
        };
        return wrap(dispatch_special_hermite_polynomial_he_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::special_hermite_polynomial_he.x_scalar(Scalar x, Tensor n) -> Tensor
        
        auto dispatch_special_hermite_polynomial_he = [](const at::Scalar & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_hermite_polynomial_he(x, n);
        };
        return wrap(dispatch_special_hermite_polynomial_he(_r.scalar(0), _r.tensor(1)));
      } else {
        // aten::special_hermite_polynomial_he.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_hermite_polynomial_he_out = [](at::Tensor out, const at::Scalar & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_hermite_polynomial_he_out(out, x, n);
        };
        return wrap(dispatch_special_hermite_polynomial_he_out(_r.tensor(2), _r.scalar(0), _r.tensor(1)));
      }
    }
    case 2: {
      if (_r.isNone(2)) {
        // aten::special_hermite_polynomial_he.n_scalar(Tensor x, Scalar n) -> Tensor
        
        auto dispatch_special_hermite_polynomial_he = [](const at::Tensor & x, const at::Scalar & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_hermite_polynomial_he(x, n);
        };
        return wrap(dispatch_special_hermite_polynomial_he(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::special_hermite_polynomial_he.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_hermite_polynomial_he_out = [](at::Tensor out, const at::Tensor & x, const at::Scalar & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_hermite_polynomial_he_out(out, x, n);
        };
        return wrap(dispatch_special_hermite_polynomial_he_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_i0
static PyObject * THPVariable_special_i0(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_i0(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_i0(Tensor self) -> Tensor
    
    auto dispatch_special_i0 = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_i0(self);
    };
    return wrap(dispatch_special_i0(_r.tensor(0)));
  } else {
    // aten::special_i0.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_i0_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_i0_out(out, self);
    };
    return wrap(dispatch_special_i0_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_i0e
static PyObject * THPVariable_special_i0e(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_i0e(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_i0e(Tensor self) -> Tensor
    
    auto dispatch_special_i0e = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_i0e(self);
    };
    return wrap(dispatch_special_i0e(_r.tensor(0)));
  } else {
    // aten::special_i0e.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_i0e_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_i0e_out(out, self);
    };
    return wrap(dispatch_special_i0e_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_i1
static PyObject * THPVariable_special_i1(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_i1(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_i1(Tensor self) -> Tensor
    
    auto dispatch_special_i1 = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_i1(self);
    };
    return wrap(dispatch_special_i1(_r.tensor(0)));
  } else {
    // aten::special_i1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_i1_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_i1_out(out, self);
    };
    return wrap(dispatch_special_i1_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_i1e
static PyObject * THPVariable_special_i1e(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_i1e(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_i1e(Tensor self) -> Tensor
    
    auto dispatch_special_i1e = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_i1e(self);
    };
    return wrap(dispatch_special_i1e(_r.tensor(0)));
  } else {
    // aten::special_i1e.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_i1e_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_i1e_out(out, self);
    };
    return wrap(dispatch_special_i1e_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// special_laguerre_polynomial_l
static PyObject * THPVariable_special_laguerre_polynomial_l(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_laguerre_polynomial_l(Tensor x, Tensor n, *, Tensor out=None)",
    "special_laguerre_polynomial_l(Scalar x, Tensor n, *, Tensor out=None)",
    "special_laguerre_polynomial_l(Tensor x, Scalar n, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::special_laguerre_polynomial_l(Tensor x, Tensor n) -> Tensor
        
        auto dispatch_special_laguerre_polynomial_l = [](const at::Tensor & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_laguerre_polynomial_l(x, n);
        };
        return wrap(dispatch_special_laguerre_polynomial_l(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::special_laguerre_polynomial_l.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_laguerre_polynomial_l_out = [](at::Tensor out, const at::Tensor & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_laguerre_polynomial_l_out(out, x, n);
        };
        return wrap(dispatch_special_laguerre_polynomial_l_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::special_laguerre_polynomial_l.x_scalar(Scalar x, Tensor n) -> Tensor
        
        auto dispatch_special_laguerre_polynomial_l = [](const at::Scalar & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_laguerre_polynomial_l(x, n);
        };
        return wrap(dispatch_special_laguerre_polynomial_l(_r.scalar(0), _r.tensor(1)));
      } else {
        // aten::special_laguerre_polynomial_l.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_laguerre_polynomial_l_out = [](at::Tensor out, const at::Scalar & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_laguerre_polynomial_l_out(out, x, n);
        };
        return wrap(dispatch_special_laguerre_polynomial_l_out(_r.tensor(2), _r.scalar(0), _r.tensor(1)));
      }
    }
    case 2: {
      if (_r.isNone(2)) {
        // aten::special_laguerre_polynomial_l.n_scalar(Tensor x, Scalar n) -> Tensor
        
        auto dispatch_special_laguerre_polynomial_l = [](const at::Tensor & x, const at::Scalar & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_laguerre_polynomial_l(x, n);
        };
        return wrap(dispatch_special_laguerre_polynomial_l(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::special_laguerre_polynomial_l.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_laguerre_polynomial_l_out = [](at::Tensor out, const at::Tensor & x, const at::Scalar & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_laguerre_polynomial_l_out(out, x, n);
        };
        return wrap(dispatch_special_laguerre_polynomial_l_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// special_legendre_polynomial_p
static PyObject * THPVariable_special_legendre_polynomial_p(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_legendre_polynomial_p(Tensor x, Tensor n, *, Tensor out=None)",
    "special_legendre_polynomial_p(Scalar x, Tensor n, *, Tensor out=None)",
    "special_legendre_polynomial_p(Tensor x, Scalar n, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::special_legendre_polynomial_p(Tensor x, Tensor n) -> Tensor
        
        auto dispatch_special_legendre_polynomial_p = [](const at::Tensor & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_legendre_polynomial_p(x, n);
        };
        return wrap(dispatch_special_legendre_polynomial_p(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::special_legendre_polynomial_p.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_legendre_polynomial_p_out = [](at::Tensor out, const at::Tensor & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_legendre_polynomial_p_out(out, x, n);
        };
        return wrap(dispatch_special_legendre_polynomial_p_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::special_legendre_polynomial_p.x_scalar(Scalar x, Tensor n) -> Tensor
        
        auto dispatch_special_legendre_polynomial_p = [](const at::Scalar & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_legendre_polynomial_p(x, n);
        };
        return wrap(dispatch_special_legendre_polynomial_p(_r.scalar(0), _r.tensor(1)));
      } else {
        // aten::special_legendre_polynomial_p.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_legendre_polynomial_p_out = [](at::Tensor out, const at::Scalar & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_legendre_polynomial_p_out(out, x, n);
        };
        return wrap(dispatch_special_legendre_polynomial_p_out(_r.tensor(2), _r.scalar(0), _r.tensor(1)));
      }
    }
    case 2: {
      if (_r.isNone(2)) {
        // aten::special_legendre_polynomial_p.n_scalar(Tensor x, Scalar n) -> Tensor
        
        auto dispatch_special_legendre_polynomial_p = [](const at::Tensor & x, const at::Scalar & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_legendre_polynomial_p(x, n);
        };
        return wrap(dispatch_special_legendre_polynomial_p(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::special_legendre_polynomial_p.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_legendre_polynomial_p_out = [](at::Tensor out, const at::Tensor & x, const at::Scalar & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_legendre_polynomial_p_out(out, x, n);
        };
        return wrap(dispatch_special_legendre_polynomial_p_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_log1p
static PyObject * THPVariable_special_log1p(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_log1p(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_log1p(Tensor self) -> Tensor
    
    auto dispatch_special_log1p = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_log1p(self);
    };
    return wrap(dispatch_special_log1p(_r.tensor(0)));
  } else {
    // aten::special_log1p.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_log1p_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_log1p_out(out, self);
    };
    return wrap(dispatch_special_log1p_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_log_ndtr
static PyObject * THPVariable_special_log_ndtr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_log_ndtr(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_log_ndtr(Tensor self) -> Tensor
    
    auto dispatch_special_log_ndtr = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_log_ndtr(self);
    };
    return wrap(dispatch_special_log_ndtr(_r.tensor(0)));
  } else {
    // aten::special_log_ndtr.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_log_ndtr_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_log_ndtr_out(out, self);
    };
    return wrap(dispatch_special_log_ndtr_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_log_softmax
static PyObject * THPVariable_special_log_softmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_log_softmax(Tensor input, int64_t dim, *, ScalarType? dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  // aten::special_log_softmax(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
  
  auto dispatch_special_log_softmax = [](const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::special_log_softmax(self, dim, dtype);
  };
  return wrap(dispatch_special_log_softmax(_r.tensor(0), _r.toInt64(1), _r.scalartypeOptional(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_logit
static PyObject * THPVariable_special_logit(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_logit(Tensor input, double? eps=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(2)) {
    // aten::special_logit(Tensor self, float? eps=None) -> Tensor
    
    auto dispatch_special_logit = [](const at::Tensor & self, ::std::optional<double> eps) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_logit(self, eps);
    };
    return wrap(dispatch_special_logit(_r.tensor(0), _r.toDoubleOptional(1)));
  } else {
    // aten::special_logit.out(Tensor self, float? eps=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_logit_out = [](at::Tensor out, const at::Tensor & self, ::std::optional<double> eps) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_logit_out(out, self, eps);
    };
    return wrap(dispatch_special_logit_out(_r.tensor(2), _r.tensor(0), _r.toDoubleOptional(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_logsumexp
static PyObject * THPVariable_special_logsumexp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_logsumexp(Tensor input, IntArrayRef[1] dim, bool keepdim=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(3)) {
    // aten::special_logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
    
    auto dispatch_special_logsumexp = [](const at::Tensor & self, at::IntArrayRef dim, bool keepdim) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_logsumexp(self, dim, keepdim);
    };
    return wrap(dispatch_special_logsumexp(_r.tensor(0), _r.intlist(1), _r.toBool(2)));
  } else {
    // aten::special_logsumexp.out(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_logsumexp_out = [](at::Tensor out, const at::Tensor & self, at::IntArrayRef dim, bool keepdim) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_logsumexp_out(out, self, dim, keepdim);
    };
    return wrap(dispatch_special_logsumexp_out(_r.tensor(3), _r.tensor(0), _r.intlist(1), _r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_modified_bessel_i0
static PyObject * THPVariable_special_modified_bessel_i0(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_modified_bessel_i0(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_modified_bessel_i0(Tensor self) -> Tensor
    
    auto dispatch_special_modified_bessel_i0 = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_modified_bessel_i0(self);
    };
    return wrap(dispatch_special_modified_bessel_i0(_r.tensor(0)));
  } else {
    // aten::special_modified_bessel_i0.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_modified_bessel_i0_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_modified_bessel_i0_out(out, self);
    };
    return wrap(dispatch_special_modified_bessel_i0_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_modified_bessel_i1
static PyObject * THPVariable_special_modified_bessel_i1(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_modified_bessel_i1(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_modified_bessel_i1(Tensor self) -> Tensor
    
    auto dispatch_special_modified_bessel_i1 = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_modified_bessel_i1(self);
    };
    return wrap(dispatch_special_modified_bessel_i1(_r.tensor(0)));
  } else {
    // aten::special_modified_bessel_i1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_modified_bessel_i1_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_modified_bessel_i1_out(out, self);
    };
    return wrap(dispatch_special_modified_bessel_i1_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_modified_bessel_k0
static PyObject * THPVariable_special_modified_bessel_k0(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_modified_bessel_k0(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_modified_bessel_k0(Tensor self) -> Tensor
    
    auto dispatch_special_modified_bessel_k0 = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_modified_bessel_k0(self);
    };
    return wrap(dispatch_special_modified_bessel_k0(_r.tensor(0)));
  } else {
    // aten::special_modified_bessel_k0.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_modified_bessel_k0_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_modified_bessel_k0_out(out, self);
    };
    return wrap(dispatch_special_modified_bessel_k0_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_modified_bessel_k1
static PyObject * THPVariable_special_modified_bessel_k1(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_modified_bessel_k1(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_modified_bessel_k1(Tensor self) -> Tensor
    
    auto dispatch_special_modified_bessel_k1 = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_modified_bessel_k1(self);
    };
    return wrap(dispatch_special_modified_bessel_k1(_r.tensor(0)));
  } else {
    // aten::special_modified_bessel_k1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_modified_bessel_k1_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_modified_bessel_k1_out(out, self);
    };
    return wrap(dispatch_special_modified_bessel_k1_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_multigammaln
static PyObject * THPVariable_special_multigammaln(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_multigammaln(Tensor input, int64_t p, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(2)) {
    // aten::special_multigammaln(Tensor self, int p) -> Tensor
    
    auto dispatch_special_multigammaln = [](const at::Tensor & self, int64_t p) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_multigammaln(self, p);
    };
    return wrap(dispatch_special_multigammaln(_r.tensor(0), _r.toInt64(1)));
  } else {
    // aten::special_multigammaln.out(Tensor self, int p, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_multigammaln_out = [](at::Tensor out, const at::Tensor & self, int64_t p) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_multigammaln_out(out, self, p);
    };
    return wrap(dispatch_special_multigammaln_out(_r.tensor(2), _r.tensor(0), _r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_ndtr
static PyObject * THPVariable_special_ndtr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_ndtr(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_ndtr(Tensor self) -> Tensor
    
    auto dispatch_special_ndtr = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_ndtr(self);
    };
    return wrap(dispatch_special_ndtr(_r.tensor(0)));
  } else {
    // aten::special_ndtr.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_ndtr_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_ndtr_out(out, self);
    };
    return wrap(dispatch_special_ndtr_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_ndtri
static PyObject * THPVariable_special_ndtri(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_ndtri(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_ndtri(Tensor self) -> Tensor
    
    auto dispatch_special_ndtri = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_ndtri(self);
    };
    return wrap(dispatch_special_ndtri(_r.tensor(0)));
  } else {
    // aten::special_ndtri.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_ndtri_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_ndtri_out(out, self);
    };
    return wrap(dispatch_special_ndtri_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_polygamma
static PyObject * THPVariable_special_polygamma(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_polygamma(int64_t n, Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(2)) {
    // aten::special_polygamma(int n, Tensor self) -> Tensor
    
    auto dispatch_special_polygamma = [](int64_t n, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_polygamma(n, self);
    };
    return wrap(dispatch_special_polygamma(_r.toInt64(0), _r.tensor(1)));
  } else {
    // aten::special_polygamma.out(int n, Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_polygamma_out = [](at::Tensor out, int64_t n, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_polygamma_out(out, n, self);
    };
    return wrap(dispatch_special_polygamma_out(_r.tensor(2), _r.toInt64(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_psi
static PyObject * THPVariable_special_psi(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_psi(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_psi(Tensor self) -> Tensor
    
    auto dispatch_special_psi = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_psi(self);
    };
    return wrap(dispatch_special_psi(_r.tensor(0)));
  } else {
    // aten::special_psi.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_psi_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_psi_out(out, self);
    };
    return wrap(dispatch_special_psi_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_round
static PyObject * THPVariable_special_round(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_round(Tensor input, *, int64_t decimals=0, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(2)) {
    // aten::special_round(Tensor self, *, int decimals=0) -> Tensor
    
    auto dispatch_special_round = [](const at::Tensor & self, int64_t decimals) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_round(self, decimals);
    };
    return wrap(dispatch_special_round(_r.tensor(0), _r.toInt64(1)));
  } else {
    // aten::special_round.out(Tensor self, *, int decimals=0, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_round_out = [](at::Tensor out, const at::Tensor & self, int64_t decimals) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_round_out(out, self, decimals);
    };
    return wrap(dispatch_special_round_out(_r.tensor(2), _r.tensor(0), _r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_scaled_modified_bessel_k0
static PyObject * THPVariable_special_scaled_modified_bessel_k0(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_scaled_modified_bessel_k0(Tensor x, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_scaled_modified_bessel_k0(Tensor x) -> Tensor
    
    auto dispatch_special_scaled_modified_bessel_k0 = [](const at::Tensor & x) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_scaled_modified_bessel_k0(x);
    };
    return wrap(dispatch_special_scaled_modified_bessel_k0(_r.tensor(0)));
  } else {
    // aten::special_scaled_modified_bessel_k0.out(Tensor x, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_scaled_modified_bessel_k0_out = [](at::Tensor out, const at::Tensor & x) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_scaled_modified_bessel_k0_out(out, x);
    };
    return wrap(dispatch_special_scaled_modified_bessel_k0_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_scaled_modified_bessel_k1
static PyObject * THPVariable_special_scaled_modified_bessel_k1(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_scaled_modified_bessel_k1(Tensor x, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_scaled_modified_bessel_k1(Tensor x) -> Tensor
    
    auto dispatch_special_scaled_modified_bessel_k1 = [](const at::Tensor & x) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_scaled_modified_bessel_k1(x);
    };
    return wrap(dispatch_special_scaled_modified_bessel_k1(_r.tensor(0)));
  } else {
    // aten::special_scaled_modified_bessel_k1.out(Tensor x, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_scaled_modified_bessel_k1_out = [](at::Tensor out, const at::Tensor & x) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_scaled_modified_bessel_k1_out(out, x);
    };
    return wrap(dispatch_special_scaled_modified_bessel_k1_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// special_shifted_chebyshev_polynomial_t
static PyObject * THPVariable_special_shifted_chebyshev_polynomial_t(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_shifted_chebyshev_polynomial_t(Tensor x, Tensor n, *, Tensor out=None)",
    "special_shifted_chebyshev_polynomial_t(Scalar x, Tensor n, *, Tensor out=None)",
    "special_shifted_chebyshev_polynomial_t(Tensor x, Scalar n, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::special_shifted_chebyshev_polynomial_t(Tensor x, Tensor n) -> Tensor
        
        auto dispatch_special_shifted_chebyshev_polynomial_t = [](const at::Tensor & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_shifted_chebyshev_polynomial_t(x, n);
        };
        return wrap(dispatch_special_shifted_chebyshev_polynomial_t(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::special_shifted_chebyshev_polynomial_t.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_shifted_chebyshev_polynomial_t_out = [](at::Tensor out, const at::Tensor & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_shifted_chebyshev_polynomial_t_out(out, x, n);
        };
        return wrap(dispatch_special_shifted_chebyshev_polynomial_t_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::special_shifted_chebyshev_polynomial_t.x_scalar(Scalar x, Tensor n) -> Tensor
        
        auto dispatch_special_shifted_chebyshev_polynomial_t = [](const at::Scalar & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_shifted_chebyshev_polynomial_t(x, n);
        };
        return wrap(dispatch_special_shifted_chebyshev_polynomial_t(_r.scalar(0), _r.tensor(1)));
      } else {
        // aten::special_shifted_chebyshev_polynomial_t.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_shifted_chebyshev_polynomial_t_out = [](at::Tensor out, const at::Scalar & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_shifted_chebyshev_polynomial_t_out(out, x, n);
        };
        return wrap(dispatch_special_shifted_chebyshev_polynomial_t_out(_r.tensor(2), _r.scalar(0), _r.tensor(1)));
      }
    }
    case 2: {
      if (_r.isNone(2)) {
        // aten::special_shifted_chebyshev_polynomial_t.n_scalar(Tensor x, Scalar n) -> Tensor
        
        auto dispatch_special_shifted_chebyshev_polynomial_t = [](const at::Tensor & x, const at::Scalar & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_shifted_chebyshev_polynomial_t(x, n);
        };
        return wrap(dispatch_special_shifted_chebyshev_polynomial_t(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::special_shifted_chebyshev_polynomial_t.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_shifted_chebyshev_polynomial_t_out = [](at::Tensor out, const at::Tensor & x, const at::Scalar & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_shifted_chebyshev_polynomial_t_out(out, x, n);
        };
        return wrap(dispatch_special_shifted_chebyshev_polynomial_t_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// special_shifted_chebyshev_polynomial_u
static PyObject * THPVariable_special_shifted_chebyshev_polynomial_u(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_shifted_chebyshev_polynomial_u(Tensor x, Tensor n, *, Tensor out=None)",
    "special_shifted_chebyshev_polynomial_u(Scalar x, Tensor n, *, Tensor out=None)",
    "special_shifted_chebyshev_polynomial_u(Tensor x, Scalar n, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::special_shifted_chebyshev_polynomial_u(Tensor x, Tensor n) -> Tensor
        
        auto dispatch_special_shifted_chebyshev_polynomial_u = [](const at::Tensor & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_shifted_chebyshev_polynomial_u(x, n);
        };
        return wrap(dispatch_special_shifted_chebyshev_polynomial_u(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::special_shifted_chebyshev_polynomial_u.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_shifted_chebyshev_polynomial_u_out = [](at::Tensor out, const at::Tensor & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_shifted_chebyshev_polynomial_u_out(out, x, n);
        };
        return wrap(dispatch_special_shifted_chebyshev_polynomial_u_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::special_shifted_chebyshev_polynomial_u.x_scalar(Scalar x, Tensor n) -> Tensor
        
        auto dispatch_special_shifted_chebyshev_polynomial_u = [](const at::Scalar & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_shifted_chebyshev_polynomial_u(x, n);
        };
        return wrap(dispatch_special_shifted_chebyshev_polynomial_u(_r.scalar(0), _r.tensor(1)));
      } else {
        // aten::special_shifted_chebyshev_polynomial_u.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_shifted_chebyshev_polynomial_u_out = [](at::Tensor out, const at::Scalar & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_shifted_chebyshev_polynomial_u_out(out, x, n);
        };
        return wrap(dispatch_special_shifted_chebyshev_polynomial_u_out(_r.tensor(2), _r.scalar(0), _r.tensor(1)));
      }
    }
    case 2: {
      if (_r.isNone(2)) {
        // aten::special_shifted_chebyshev_polynomial_u.n_scalar(Tensor x, Scalar n) -> Tensor
        
        auto dispatch_special_shifted_chebyshev_polynomial_u = [](const at::Tensor & x, const at::Scalar & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_shifted_chebyshev_polynomial_u(x, n);
        };
        return wrap(dispatch_special_shifted_chebyshev_polynomial_u(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::special_shifted_chebyshev_polynomial_u.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_shifted_chebyshev_polynomial_u_out = [](at::Tensor out, const at::Tensor & x, const at::Scalar & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_shifted_chebyshev_polynomial_u_out(out, x, n);
        };
        return wrap(dispatch_special_shifted_chebyshev_polynomial_u_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// special_shifted_chebyshev_polynomial_v
static PyObject * THPVariable_special_shifted_chebyshev_polynomial_v(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_shifted_chebyshev_polynomial_v(Tensor x, Tensor n, *, Tensor out=None)",
    "special_shifted_chebyshev_polynomial_v(Scalar x, Tensor n, *, Tensor out=None)",
    "special_shifted_chebyshev_polynomial_v(Tensor x, Scalar n, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::special_shifted_chebyshev_polynomial_v(Tensor x, Tensor n) -> Tensor
        
        auto dispatch_special_shifted_chebyshev_polynomial_v = [](const at::Tensor & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_shifted_chebyshev_polynomial_v(x, n);
        };
        return wrap(dispatch_special_shifted_chebyshev_polynomial_v(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::special_shifted_chebyshev_polynomial_v.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_shifted_chebyshev_polynomial_v_out = [](at::Tensor out, const at::Tensor & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_shifted_chebyshev_polynomial_v_out(out, x, n);
        };
        return wrap(dispatch_special_shifted_chebyshev_polynomial_v_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::special_shifted_chebyshev_polynomial_v.x_scalar(Scalar x, Tensor n) -> Tensor
        
        auto dispatch_special_shifted_chebyshev_polynomial_v = [](const at::Scalar & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_shifted_chebyshev_polynomial_v(x, n);
        };
        return wrap(dispatch_special_shifted_chebyshev_polynomial_v(_r.scalar(0), _r.tensor(1)));
      } else {
        // aten::special_shifted_chebyshev_polynomial_v.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_shifted_chebyshev_polynomial_v_out = [](at::Tensor out, const at::Scalar & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_shifted_chebyshev_polynomial_v_out(out, x, n);
        };
        return wrap(dispatch_special_shifted_chebyshev_polynomial_v_out(_r.tensor(2), _r.scalar(0), _r.tensor(1)));
      }
    }
    case 2: {
      if (_r.isNone(2)) {
        // aten::special_shifted_chebyshev_polynomial_v.n_scalar(Tensor x, Scalar n) -> Tensor
        
        auto dispatch_special_shifted_chebyshev_polynomial_v = [](const at::Tensor & x, const at::Scalar & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_shifted_chebyshev_polynomial_v(x, n);
        };
        return wrap(dispatch_special_shifted_chebyshev_polynomial_v(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::special_shifted_chebyshev_polynomial_v.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_shifted_chebyshev_polynomial_v_out = [](at::Tensor out, const at::Tensor & x, const at::Scalar & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_shifted_chebyshev_polynomial_v_out(out, x, n);
        };
        return wrap(dispatch_special_shifted_chebyshev_polynomial_v_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// special_shifted_chebyshev_polynomial_w
static PyObject * THPVariable_special_shifted_chebyshev_polynomial_w(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_shifted_chebyshev_polynomial_w(Tensor x, Tensor n, *, Tensor out=None)",
    "special_shifted_chebyshev_polynomial_w(Scalar x, Tensor n, *, Tensor out=None)",
    "special_shifted_chebyshev_polynomial_w(Tensor x, Scalar n, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::special_shifted_chebyshev_polynomial_w(Tensor x, Tensor n) -> Tensor
        
        auto dispatch_special_shifted_chebyshev_polynomial_w = [](const at::Tensor & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_shifted_chebyshev_polynomial_w(x, n);
        };
        return wrap(dispatch_special_shifted_chebyshev_polynomial_w(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::special_shifted_chebyshev_polynomial_w.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_shifted_chebyshev_polynomial_w_out = [](at::Tensor out, const at::Tensor & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_shifted_chebyshev_polynomial_w_out(out, x, n);
        };
        return wrap(dispatch_special_shifted_chebyshev_polynomial_w_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::special_shifted_chebyshev_polynomial_w.x_scalar(Scalar x, Tensor n) -> Tensor
        
        auto dispatch_special_shifted_chebyshev_polynomial_w = [](const at::Scalar & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_shifted_chebyshev_polynomial_w(x, n);
        };
        return wrap(dispatch_special_shifted_chebyshev_polynomial_w(_r.scalar(0), _r.tensor(1)));
      } else {
        // aten::special_shifted_chebyshev_polynomial_w.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_shifted_chebyshev_polynomial_w_out = [](at::Tensor out, const at::Scalar & x, const at::Tensor & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_shifted_chebyshev_polynomial_w_out(out, x, n);
        };
        return wrap(dispatch_special_shifted_chebyshev_polynomial_w_out(_r.tensor(2), _r.scalar(0), _r.tensor(1)));
      }
    }
    case 2: {
      if (_r.isNone(2)) {
        // aten::special_shifted_chebyshev_polynomial_w.n_scalar(Tensor x, Scalar n) -> Tensor
        
        auto dispatch_special_shifted_chebyshev_polynomial_w = [](const at::Tensor & x, const at::Scalar & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_shifted_chebyshev_polynomial_w(x, n);
        };
        return wrap(dispatch_special_shifted_chebyshev_polynomial_w(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::special_shifted_chebyshev_polynomial_w.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_shifted_chebyshev_polynomial_w_out = [](at::Tensor out, const at::Tensor & x, const at::Scalar & n) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_shifted_chebyshev_polynomial_w_out(out, x, n);
        };
        return wrap(dispatch_special_shifted_chebyshev_polynomial_w_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_sinc
static PyObject * THPVariable_special_sinc(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_sinc(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_sinc(Tensor self) -> Tensor
    
    auto dispatch_special_sinc = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_sinc(self);
    };
    return wrap(dispatch_special_sinc(_r.tensor(0)));
  } else {
    // aten::special_sinc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_sinc_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_sinc_out(out, self);
    };
    return wrap(dispatch_special_sinc_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_softmax
static PyObject * THPVariable_special_softmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_softmax(Tensor input, int64_t dim, ScalarType? dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  // aten::special_softmax(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
  
  auto dispatch_special_softmax = [](const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::special_softmax(self, dim, dtype);
  };
  return wrap(dispatch_special_softmax(_r.tensor(0), _r.toInt64(1), _r.scalartypeOptional(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_spherical_bessel_j0
static PyObject * THPVariable_special_spherical_bessel_j0(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_spherical_bessel_j0(Tensor x, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_spherical_bessel_j0(Tensor x) -> Tensor
    
    auto dispatch_special_spherical_bessel_j0 = [](const at::Tensor & x) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_spherical_bessel_j0(x);
    };
    return wrap(dispatch_special_spherical_bessel_j0(_r.tensor(0)));
  } else {
    // aten::special_spherical_bessel_j0.out(Tensor x, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_spherical_bessel_j0_out = [](at::Tensor out, const at::Tensor & x) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_spherical_bessel_j0_out(out, x);
    };
    return wrap(dispatch_special_spherical_bessel_j0_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// special_xlog1py
static PyObject * THPVariable_special_xlog1py(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_xlog1py(Tensor input, Tensor other, *, Tensor out=None)",
    "special_xlog1py(Scalar self, Tensor other, *, Tensor out=None)",
    "special_xlog1py(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::special_xlog1py(Tensor self, Tensor other) -> Tensor
        
        auto dispatch_special_xlog1py = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_xlog1py(self, other);
        };
        return wrap(dispatch_special_xlog1py(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::special_xlog1py.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_xlog1py_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_xlog1py_out(out, self, other);
        };
        return wrap(dispatch_special_xlog1py_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::special_xlog1py.self_scalar(Scalar self, Tensor other) -> Tensor
        
        auto dispatch_special_xlog1py = [](const at::Scalar & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_xlog1py(self, other);
        };
        return wrap(dispatch_special_xlog1py(_r.scalar(0), _r.tensor(1)));
      } else {
        // aten::special_xlog1py.self_scalar_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_xlog1py_out = [](at::Tensor out, const at::Scalar & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_xlog1py_out(out, self, other);
        };
        return wrap(dispatch_special_xlog1py_out(_r.tensor(2), _r.scalar(0), _r.tensor(1)));
      }
    }
    case 2: {
      if (_r.isNone(2)) {
        // aten::special_xlog1py.other_scalar(Tensor self, Scalar other) -> Tensor
        
        auto dispatch_special_xlog1py = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_xlog1py(self, other);
        };
        return wrap(dispatch_special_xlog1py(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::special_xlog1py.other_scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_xlog1py_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_xlog1py_out(out, self, other);
        };
        return wrap(dispatch_special_xlog1py_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// special_xlogy
static PyObject * THPVariable_special_xlogy(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_xlogy(Tensor input, Tensor other, *, Tensor out=None)",
    "special_xlogy(Scalar self, Tensor other, *, Tensor out=None)",
    "special_xlogy(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::special_xlogy(Tensor self, Tensor other) -> Tensor
        
        auto dispatch_special_xlogy = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_xlogy(self, other);
        };
        return wrap(dispatch_special_xlogy(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::special_xlogy.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_xlogy_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_xlogy_out(out, self, other);
        };
        return wrap(dispatch_special_xlogy_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::special_xlogy.self_scalar(Scalar self, Tensor other) -> Tensor
        
        auto dispatch_special_xlogy = [](const at::Scalar & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_xlogy(self, other);
        };
        return wrap(dispatch_special_xlogy(_r.scalar(0), _r.tensor(1)));
      } else {
        // aten::special_xlogy.self_scalar_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_xlogy_out = [](at::Tensor out, const at::Scalar & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_xlogy_out(out, self, other);
        };
        return wrap(dispatch_special_xlogy_out(_r.tensor(2), _r.scalar(0), _r.tensor(1)));
      }
    }
    case 2: {
      if (_r.isNone(2)) {
        // aten::special_xlogy.other_scalar(Tensor self, Scalar other) -> Tensor
        
        auto dispatch_special_xlogy = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_xlogy(self, other);
        };
        return wrap(dispatch_special_xlogy(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::special_xlogy.other_scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_xlogy_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_xlogy_out(out, self, other);
        };
        return wrap(dispatch_special_xlogy_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// special_zeta
static PyObject * THPVariable_special_zeta(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_zeta(Tensor input, Tensor other, *, Tensor out=None)",
    "special_zeta(Scalar self, Tensor other, *, Tensor out=None)",
    "special_zeta(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::special_zeta(Tensor self, Tensor other) -> Tensor
        
        auto dispatch_special_zeta = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_zeta(self, other);
        };
        return wrap(dispatch_special_zeta(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::special_zeta.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_zeta_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_zeta_out(out, self, other);
        };
        return wrap(dispatch_special_zeta_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::special_zeta.self_scalar(Scalar self, Tensor other) -> Tensor
        
        auto dispatch_special_zeta = [](const at::Scalar & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_zeta(self, other);
        };
        return wrap(dispatch_special_zeta(_r.scalar(0), _r.tensor(1)));
      } else {
        // aten::special_zeta.self_scalar_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_zeta_out = [](at::Tensor out, const at::Scalar & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_zeta_out(out, self, other);
        };
        return wrap(dispatch_special_zeta_out(_r.tensor(2), _r.scalar(0), _r.tensor(1)));
      }
    }
    case 2: {
      if (_r.isNone(2)) {
        // aten::special_zeta.other_scalar(Tensor self, Scalar other) -> Tensor
        
        auto dispatch_special_zeta = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_zeta(self, other);
        };
        return wrap(dispatch_special_zeta(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::special_zeta.other_scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_zeta_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_zeta_out(out, self, other);
        };
        return wrap(dispatch_special_zeta_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

} // namespace torch::autograd
