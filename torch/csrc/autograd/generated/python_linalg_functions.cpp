#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// @generated from ../tools/autograd/templates/python_linalg_functions.cpp

#include "torch/csrc/Device.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/python_linalg_functions.h"
#include "torch/csrc/autograd/generated/python_return_types.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/autograd/utils/python_arg_parsing.h"
#include "torch/csrc/utils/pycfunction_helpers.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/structseq.h"

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_linalg_eigvals.h>
#include <ATen/ops/linalg_cholesky.h>
#include <ATen/ops/linalg_cholesky_ex.h>
#include <ATen/ops/linalg_cond.h>
#include <ATen/ops/linalg_cross.h>
#include <ATen/ops/linalg_det.h>
#include <ATen/ops/linalg_diagonal.h>
#include <ATen/ops/linalg_eig.h>
#include <ATen/ops/linalg_eigh.h>
#include <ATen/ops/linalg_eigvals.h>
#include <ATen/ops/linalg_eigvalsh.h>
#include <ATen/ops/linalg_householder_product.h>
#include <ATen/ops/linalg_inv.h>
#include <ATen/ops/linalg_inv_ex.h>
#include <ATen/ops/linalg_ldl_factor.h>
#include <ATen/ops/linalg_ldl_factor_ex.h>
#include <ATen/ops/linalg_ldl_solve.h>
#include <ATen/ops/linalg_lstsq.h>
#include <ATen/ops/linalg_lu.h>
#include <ATen/ops/linalg_lu_factor.h>
#include <ATen/ops/linalg_lu_factor_ex.h>
#include <ATen/ops/linalg_lu_solve.h>
#include <ATen/ops/linalg_matmul.h>
#include <ATen/ops/linalg_matrix_exp.h>
#include <ATen/ops/linalg_matrix_norm.h>
#include <ATen/ops/linalg_matrix_power.h>
#include <ATen/ops/linalg_matrix_rank.h>
#include <ATen/ops/linalg_multi_dot.h>
#include <ATen/ops/linalg_norm.h>
#include <ATen/ops/linalg_pinv.h>
#include <ATen/ops/linalg_qr.h>
#include <ATen/ops/linalg_slogdet.h>
#include <ATen/ops/linalg_solve.h>
#include <ATen/ops/linalg_solve_ex.h>
#include <ATen/ops/linalg_solve_triangular.h>
#include <ATen/ops/linalg_svd.h>
#include <ATen/ops/linalg_svdvals.h>
#include <ATen/ops/linalg_tensorinv.h>
#include <ATen/ops/linalg_tensorsolve.h>
#include <ATen/ops/linalg_vander.h>
#include <ATen/ops/linalg_vecdot.h>
#include <ATen/ops/linalg_vector_norm.h>
#endif

using at::Tensor;
using at::Scalar;
using at::ScalarType;
using at::MemoryFormat;
using at::Generator;
using at::IntArrayRef;
using at::TensorList;

using namespace torch::autograd::utils;

namespace torch::autograd {

// generated forward declarations start here

static PyObject * THPVariable__linalg_eigvals(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_cholesky(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_cholesky_ex(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_cond(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_cross(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_det(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_diagonal(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_eig(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_eigh(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_eigvals(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_eigvalsh(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_householder_product(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_inv(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_inv_ex(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_ldl_factor(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_ldl_factor_ex(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_ldl_solve(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_lstsq(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_lu(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_lu_factor(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_lu_factor_ex(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_lu_solve(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_matmul(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_matrix_exp(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_matrix_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_matrix_power(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_matrix_rank(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_multi_dot(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_pinv(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_qr(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_slogdet(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_solve(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_solve_ex(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_solve_triangular(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_svd(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_svdvals(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_tensorinv(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_tensorsolve(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_vander(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_vecdot(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_vector_norm(PyObject* self_, PyObject* args, PyObject* kwargs);

static PyMethodDef linalg_functions[] = {
  {"_linalg_eigvals", castPyCFunctionWithKeywords(THPVariable__linalg_eigvals), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_cholesky", castPyCFunctionWithKeywords(THPVariable_linalg_cholesky), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_cholesky_ex", castPyCFunctionWithKeywords(THPVariable_linalg_cholesky_ex), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_cond", castPyCFunctionWithKeywords(THPVariable_linalg_cond), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_cross", castPyCFunctionWithKeywords(THPVariable_linalg_cross), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_det", castPyCFunctionWithKeywords(THPVariable_linalg_det), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_diagonal", castPyCFunctionWithKeywords(THPVariable_linalg_diagonal), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_eig", castPyCFunctionWithKeywords(THPVariable_linalg_eig), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_eigh", castPyCFunctionWithKeywords(THPVariable_linalg_eigh), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_eigvals", castPyCFunctionWithKeywords(THPVariable_linalg_eigvals), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_eigvalsh", castPyCFunctionWithKeywords(THPVariable_linalg_eigvalsh), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_householder_product", castPyCFunctionWithKeywords(THPVariable_linalg_householder_product), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_inv", castPyCFunctionWithKeywords(THPVariable_linalg_inv), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_inv_ex", castPyCFunctionWithKeywords(THPVariable_linalg_inv_ex), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_ldl_factor", castPyCFunctionWithKeywords(THPVariable_linalg_ldl_factor), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_ldl_factor_ex", castPyCFunctionWithKeywords(THPVariable_linalg_ldl_factor_ex), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_ldl_solve", castPyCFunctionWithKeywords(THPVariable_linalg_ldl_solve), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_lstsq", castPyCFunctionWithKeywords(THPVariable_linalg_lstsq), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_lu", castPyCFunctionWithKeywords(THPVariable_linalg_lu), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_lu_factor", castPyCFunctionWithKeywords(THPVariable_linalg_lu_factor), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_lu_factor_ex", castPyCFunctionWithKeywords(THPVariable_linalg_lu_factor_ex), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_lu_solve", castPyCFunctionWithKeywords(THPVariable_linalg_lu_solve), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_matmul", castPyCFunctionWithKeywords(THPVariable_linalg_matmul), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_matrix_exp", castPyCFunctionWithKeywords(THPVariable_linalg_matrix_exp), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_matrix_norm", castPyCFunctionWithKeywords(THPVariable_linalg_matrix_norm), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_matrix_power", castPyCFunctionWithKeywords(THPVariable_linalg_matrix_power), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_matrix_rank", castPyCFunctionWithKeywords(THPVariable_linalg_matrix_rank), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_multi_dot", castPyCFunctionWithKeywords(THPVariable_linalg_multi_dot), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_norm", castPyCFunctionWithKeywords(THPVariable_linalg_norm), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_pinv", castPyCFunctionWithKeywords(THPVariable_linalg_pinv), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_qr", castPyCFunctionWithKeywords(THPVariable_linalg_qr), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_slogdet", castPyCFunctionWithKeywords(THPVariable_linalg_slogdet), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_solve", castPyCFunctionWithKeywords(THPVariable_linalg_solve), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_solve_ex", castPyCFunctionWithKeywords(THPVariable_linalg_solve_ex), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_solve_triangular", castPyCFunctionWithKeywords(THPVariable_linalg_solve_triangular), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_svd", castPyCFunctionWithKeywords(THPVariable_linalg_svd), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_svdvals", castPyCFunctionWithKeywords(THPVariable_linalg_svdvals), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_tensorinv", castPyCFunctionWithKeywords(THPVariable_linalg_tensorinv), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_tensorsolve", castPyCFunctionWithKeywords(THPVariable_linalg_tensorsolve), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_vander", castPyCFunctionWithKeywords(THPVariable_linalg_vander), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_vecdot", castPyCFunctionWithKeywords(THPVariable_linalg_vecdot), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"linalg_vector_norm", castPyCFunctionWithKeywords(THPVariable_linalg_vector_norm), METH_VARARGS | METH_KEYWORDS, nullptr},
  {NULL}
};

static PyObject* THPLinalgVariableFunctionsModule = NULL;

void initLinalgFunctions(PyObject* module) {
  static struct PyModuleDef def = {
     PyModuleDef_HEAD_INIT,
     "torch._C._linalg",
     NULL,
     -1,
     linalg_functions
  };
  PyObject* linalg = PyModule_Create(&def);
  THPLinalgVariableFunctionsModule = linalg;
  if (!linalg) {
    throw python_error();
  }
  // steals a reference to linalg
  if (PyModule_AddObject(module, "_linalg", linalg) != 0) {
    throw python_error();
  }
}

// generated methods start here

// _linalg_eigvals
static PyObject * THPVariable__linalg_eigvals(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_linalg_eigvals(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  // aten::_linalg_eigvals(Tensor self) -> Tensor
  
  auto dispatch__linalg_eigvals = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_linalg_eigvals(self);
  };
  return wrap(dispatch__linalg_eigvals(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_cholesky
static PyObject * THPVariable_linalg_cholesky(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_cholesky(Tensor input, *, bool upper=False, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(2)) {
    // aten::linalg_cholesky(Tensor self, *, bool upper=False) -> Tensor
    
    auto dispatch_linalg_cholesky = [](const at::Tensor & self, bool upper) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_cholesky(self, upper);
    };
    return wrap(dispatch_linalg_cholesky(_r.tensor(0), _r.toBool(1)));
  } else {
    // aten::linalg_cholesky.out(Tensor self, *, bool upper=False, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_cholesky_out = [](at::Tensor out, const at::Tensor & self, bool upper) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_cholesky_out(out, self, upper);
    };
    return wrap(dispatch_linalg_cholesky_out(_r.tensor(2), _r.tensor(0), _r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_cholesky_ex
static PyObject * THPVariable_linalg_cholesky_ex(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_linalg_cholesky_ex_structseq();
  static PyTypeObject* NamedTuple1 = generated::get_linalg_cholesky_ex_out_structseq();
  static PythonArgParser parser({
    "linalg_cholesky_ex(Tensor input, *, bool upper=False, bool check_errors=False, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(3)) {
    // aten::linalg_cholesky_ex(Tensor self, *, bool upper=False, bool check_errors=False) -> (Tensor L, Tensor info)
    
    auto dispatch_linalg_cholesky_ex = [](const at::Tensor & self, bool upper, bool check_errors) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_cholesky_ex(self, upper, check_errors);
    };
    return wrap(NamedTuple, dispatch_linalg_cholesky_ex(_r.tensor(0), _r.toBool(1), _r.toBool(2)));
  } else {
    // aten::linalg_cholesky_ex.L(Tensor self, *, bool upper=False, bool check_errors=False, Tensor(a!) L, Tensor(b!) info) -> (Tensor(a!) L, Tensor(b!) info)
    auto out = _r.tensorlist_n<2>(3);
    auto dispatch_linalg_cholesky_ex_out = [](at::Tensor & L, at::Tensor & info, const at::Tensor & self, bool upper, bool check_errors) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_cholesky_ex_out(L, info, self, upper, check_errors);
    };
    return wrap(NamedTuple1, dispatch_linalg_cholesky_ex_out(out[0], out[1], _r.tensor(0), _r.toBool(1), _r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// linalg_cond
static PyObject * THPVariable_linalg_cond(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_cond(Tensor input, Scalar? p=None, *, Tensor out=None)",
    "linalg_cond(Tensor input, c10::string_view p, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::linalg_cond(Tensor self, Scalar? p=None) -> Tensor
        
        auto dispatch_linalg_cond = [](const at::Tensor & self, const ::std::optional<at::Scalar> & p) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_cond(self, p);
        };
        return wrap(dispatch_linalg_cond(_r.tensor(0), _r.scalarOptional(1)));
      } else {
        // aten::linalg_cond.out(Tensor self, Scalar? p=None, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_linalg_cond_out = [](at::Tensor out, const at::Tensor & self, const ::std::optional<at::Scalar> & p) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_cond_out(out, self, p);
        };
        return wrap(dispatch_linalg_cond_out(_r.tensor(2), _r.tensor(0), _r.scalarOptional(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::linalg_cond.p_str(Tensor self, str p) -> Tensor
        
        auto dispatch_linalg_cond = [](const at::Tensor & self, c10::string_view p) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_cond(self, p);
        };
        return wrap(dispatch_linalg_cond(_r.tensor(0), _r.stringView(1)));
      } else {
        // aten::linalg_cond.p_str_out(Tensor self, str p, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_linalg_cond_out = [](at::Tensor out, const at::Tensor & self, c10::string_view p) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_cond_out(out, self, p);
        };
        return wrap(dispatch_linalg_cond_out(_r.tensor(2), _r.tensor(0), _r.stringView(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_cross
static PyObject * THPVariable_linalg_cross(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_cross(Tensor input, Tensor other, *, int64_t dim=-1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(3)) {
    // aten::linalg_cross(Tensor self, Tensor other, *, int dim=-1) -> Tensor
    
    auto dispatch_linalg_cross = [](const at::Tensor & self, const at::Tensor & other, int64_t dim) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_cross(self, other, dim);
    };
    return wrap(dispatch_linalg_cross(_r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  } else {
    // aten::linalg_cross.out(Tensor self, Tensor other, *, int dim=-1, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_cross_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other, int64_t dim) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_cross_out(out, self, other, dim);
    };
    return wrap(dispatch_linalg_cross_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_det
static PyObject * THPVariable_linalg_det(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_det(Tensor A, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(1)) {
    // aten::linalg_det(Tensor A) -> Tensor
    
    auto dispatch_linalg_det = [](const at::Tensor & A) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_det(A);
    };
    return wrap(dispatch_linalg_det(_r.tensor(0)));
  } else {
    // aten::linalg_det.out(Tensor A, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_det_out = [](at::Tensor out, const at::Tensor & A) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_det_out(out, A);
    };
    return wrap(dispatch_linalg_det_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_diagonal
static PyObject * THPVariable_linalg_diagonal(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_diagonal(Tensor A, *, int64_t offset=0, int64_t dim1=-2, int64_t dim2=-1)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  // aten::linalg_diagonal(Tensor(a) A, *, int offset=0, int dim1=-2, int dim2=-1) -> Tensor(a)
  
  auto dispatch_linalg_diagonal = [](const at::Tensor & A, int64_t offset, int64_t dim1, int64_t dim2) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::linalg_diagonal(A, offset, dim1, dim2);
  };
  return wrap(dispatch_linalg_diagonal(_r.tensor(0), _r.toInt64(1), _r.toInt64(2), _r.toInt64(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_eig
static PyObject * THPVariable_linalg_eig(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_linalg_eig_structseq();
  static PyTypeObject* NamedTuple1 = generated::get_linalg_eig_out_structseq();
  static PythonArgParser parser({
    "linalg_eig(Tensor input, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(1)) {
    // aten::linalg_eig(Tensor self) -> (Tensor eigenvalues, Tensor eigenvectors)
    
    auto dispatch_linalg_eig = [](const at::Tensor & self) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_eig(self);
    };
    return wrap(NamedTuple, dispatch_linalg_eig(_r.tensor(0)));
  } else {
    // aten::linalg_eig.out(Tensor self, *, Tensor(a!) eigenvalues, Tensor(b!) eigenvectors) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)
    auto out = _r.tensorlist_n<2>(1);
    auto dispatch_linalg_eig_out = [](at::Tensor & eigenvalues, at::Tensor & eigenvectors, const at::Tensor & self) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_eig_out(eigenvalues, eigenvectors, self);
    };
    return wrap(NamedTuple1, dispatch_linalg_eig_out(out[0], out[1], _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_eigh
static PyObject * THPVariable_linalg_eigh(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_linalg_eigh_structseq();
  static PyTypeObject* NamedTuple1 = generated::get_linalg_eigh_out_structseq();
  static PythonArgParser parser({
    "linalg_eigh(Tensor input, c10::string_view UPLO=\"L\", *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(2)) {
    // aten::linalg_eigh(Tensor self, str UPLO="L") -> (Tensor eigenvalues, Tensor eigenvectors)
    
    auto dispatch_linalg_eigh = [](const at::Tensor & self, c10::string_view UPLO) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_eigh(self, UPLO);
    };
    return wrap(NamedTuple, dispatch_linalg_eigh(_r.tensor(0), _r.stringView(1)));
  } else {
    // aten::linalg_eigh.eigvals(Tensor self, str UPLO="L", *, Tensor(a!) eigvals, Tensor(b!) eigvecs) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)
    auto out = _r.tensorlist_n<2>(2);
    auto dispatch_linalg_eigh_out = [](at::Tensor & eigvals, at::Tensor & eigvecs, const at::Tensor & self, c10::string_view UPLO) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_eigh_out(eigvals, eigvecs, self, UPLO);
    };
    return wrap(NamedTuple1, dispatch_linalg_eigh_out(out[0], out[1], _r.tensor(0), _r.stringView(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_eigvals
static PyObject * THPVariable_linalg_eigvals(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_eigvals(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(1)) {
    // aten::linalg_eigvals(Tensor self) -> Tensor
    
    auto dispatch_linalg_eigvals = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_eigvals(self);
    };
    return wrap(dispatch_linalg_eigvals(_r.tensor(0)));
  } else {
    // aten::linalg_eigvals.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_eigvals_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_eigvals_out(out, self);
    };
    return wrap(dispatch_linalg_eigvals_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_eigvalsh
static PyObject * THPVariable_linalg_eigvalsh(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_eigvalsh(Tensor input, c10::string_view UPLO=\"L\", *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(2)) {
    // aten::linalg_eigvalsh(Tensor self, str UPLO="L") -> Tensor
    
    auto dispatch_linalg_eigvalsh = [](const at::Tensor & self, c10::string_view UPLO) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_eigvalsh(self, UPLO);
    };
    return wrap(dispatch_linalg_eigvalsh(_r.tensor(0), _r.stringView(1)));
  } else {
    // aten::linalg_eigvalsh.out(Tensor self, str UPLO="L", *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_eigvalsh_out = [](at::Tensor out, const at::Tensor & self, c10::string_view UPLO) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_eigvalsh_out(out, self, UPLO);
    };
    return wrap(dispatch_linalg_eigvalsh_out(_r.tensor(2), _r.tensor(0), _r.stringView(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_householder_product
static PyObject * THPVariable_linalg_householder_product(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_householder_product(Tensor input, Tensor tau, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(2)) {
    // aten::linalg_householder_product(Tensor input, Tensor tau) -> Tensor
    
    auto dispatch_linalg_householder_product = [](const at::Tensor & input, const at::Tensor & tau) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_householder_product(input, tau);
    };
    return wrap(dispatch_linalg_householder_product(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::linalg_householder_product.out(Tensor input, Tensor tau, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_householder_product_out = [](at::Tensor out, const at::Tensor & input, const at::Tensor & tau) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_householder_product_out(out, input, tau);
    };
    return wrap(dispatch_linalg_householder_product_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_inv
static PyObject * THPVariable_linalg_inv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_inv(Tensor A, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(1)) {
    // aten::linalg_inv(Tensor A) -> Tensor
    
    auto dispatch_linalg_inv = [](const at::Tensor & A) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_inv(A);
    };
    return wrap(dispatch_linalg_inv(_r.tensor(0)));
  } else {
    // aten::linalg_inv.out(Tensor A, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_inv_out = [](at::Tensor out, const at::Tensor & A) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_inv_out(out, A);
    };
    return wrap(dispatch_linalg_inv_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_inv_ex
static PyObject * THPVariable_linalg_inv_ex(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_linalg_inv_ex_structseq();
  static PyTypeObject* NamedTuple1 = generated::get_linalg_inv_ex_out_structseq();
  static PythonArgParser parser({
    "linalg_inv_ex(Tensor A, *, bool check_errors=False, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(2)) {
    // aten::linalg_inv_ex(Tensor A, *, bool check_errors=False) -> (Tensor inverse, Tensor info)
    
    auto dispatch_linalg_inv_ex = [](const at::Tensor & A, bool check_errors) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_inv_ex(A, check_errors);
    };
    return wrap(NamedTuple, dispatch_linalg_inv_ex(_r.tensor(0), _r.toBool(1)));
  } else {
    // aten::linalg_inv_ex.inverse(Tensor A, *, bool check_errors=False, Tensor(a!) inverse, Tensor(b!) info) -> (Tensor(a!) inverse, Tensor(b!) info)
    auto out = _r.tensorlist_n<2>(2);
    auto dispatch_linalg_inv_ex_out = [](at::Tensor & inverse, at::Tensor & info, const at::Tensor & A, bool check_errors) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_inv_ex_out(inverse, info, A, check_errors);
    };
    return wrap(NamedTuple1, dispatch_linalg_inv_ex_out(out[0], out[1], _r.tensor(0), _r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_ldl_factor
static PyObject * THPVariable_linalg_ldl_factor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_linalg_ldl_factor_structseq();
  static PyTypeObject* NamedTuple1 = generated::get_linalg_ldl_factor_out_structseq();
  static PythonArgParser parser({
    "linalg_ldl_factor(Tensor input, *, bool hermitian=False, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(2)) {
    // aten::linalg_ldl_factor(Tensor self, *, bool hermitian=False) -> (Tensor LD, Tensor pivots)
    
    auto dispatch_linalg_ldl_factor = [](const at::Tensor & self, bool hermitian) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_ldl_factor(self, hermitian);
    };
    return wrap(NamedTuple, dispatch_linalg_ldl_factor(_r.tensor(0), _r.toBool(1)));
  } else {
    // aten::linalg_ldl_factor.out(Tensor self, *, bool hermitian=False, Tensor(a!) LD, Tensor(b!) pivots) -> (Tensor(a!) LD, Tensor(b!) pivots)
    auto out = _r.tensorlist_n<2>(2);
    auto dispatch_linalg_ldl_factor_out = [](at::Tensor & LD, at::Tensor & pivots, const at::Tensor & self, bool hermitian) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_ldl_factor_out(LD, pivots, self, hermitian);
    };
    return wrap(NamedTuple1, dispatch_linalg_ldl_factor_out(out[0], out[1], _r.tensor(0), _r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_ldl_factor_ex
static PyObject * THPVariable_linalg_ldl_factor_ex(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_linalg_ldl_factor_ex_structseq();
  static PyTypeObject* NamedTuple1 = generated::get_linalg_ldl_factor_ex_out_structseq();
  static PythonArgParser parser({
    "linalg_ldl_factor_ex(Tensor input, *, bool hermitian=False, bool check_errors=False, TensorList[3] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(3)) {
    // aten::linalg_ldl_factor_ex(Tensor self, *, bool hermitian=False, bool check_errors=False) -> (Tensor LD, Tensor pivots, Tensor info)
    
    auto dispatch_linalg_ldl_factor_ex = [](const at::Tensor & self, bool hermitian, bool check_errors) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_ldl_factor_ex(self, hermitian, check_errors);
    };
    return wrap(NamedTuple, dispatch_linalg_ldl_factor_ex(_r.tensor(0), _r.toBool(1), _r.toBool(2)));
  } else {
    // aten::linalg_ldl_factor_ex.out(Tensor self, *, bool hermitian=False, bool check_errors=False, Tensor(a!) LD, Tensor(b!) pivots, Tensor(c!) info) -> (Tensor(a!) LD, Tensor(b!) pivots, Tensor(c!) info)
    auto out = _r.tensorlist_n<3>(3);
    auto dispatch_linalg_ldl_factor_ex_out = [](at::Tensor & LD, at::Tensor & pivots, at::Tensor & info, const at::Tensor & self, bool hermitian, bool check_errors) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_ldl_factor_ex_out(LD, pivots, info, self, hermitian, check_errors);
    };
    return wrap(NamedTuple1, dispatch_linalg_ldl_factor_ex_out(out[0], out[1], out[2], _r.tensor(0), _r.toBool(1), _r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_ldl_solve
static PyObject * THPVariable_linalg_ldl_solve(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_ldl_solve(Tensor LD, Tensor pivots, Tensor B, *, bool hermitian=False, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(4)) {
    // aten::linalg_ldl_solve(Tensor LD, Tensor pivots, Tensor B, *, bool hermitian=False) -> Tensor
    
    auto dispatch_linalg_ldl_solve = [](const at::Tensor & LD, const at::Tensor & pivots, const at::Tensor & B, bool hermitian) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_ldl_solve(LD, pivots, B, hermitian);
    };
    return wrap(dispatch_linalg_ldl_solve(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toBool(3)));
  } else {
    // aten::linalg_ldl_solve.out(Tensor LD, Tensor pivots, Tensor B, *, bool hermitian=False, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_ldl_solve_out = [](at::Tensor out, const at::Tensor & LD, const at::Tensor & pivots, const at::Tensor & B, bool hermitian) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_ldl_solve_out(out, LD, pivots, B, hermitian);
    };
    return wrap(dispatch_linalg_ldl_solve_out(_r.tensor(4), _r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toBool(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_lstsq
static PyObject * THPVariable_linalg_lstsq(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_linalg_lstsq_structseq();
  static PyTypeObject* NamedTuple1 = generated::get_linalg_lstsq_out_structseq();
  static PythonArgParser parser({
    "linalg_lstsq(Tensor input, Tensor b, double? rcond=None, *, c10::string_view? driver=None, TensorList[4] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(4)) {
    // aten::linalg_lstsq(Tensor self, Tensor b, float? rcond=None, *, str? driver=None) -> (Tensor solution, Tensor residuals, Tensor rank, Tensor singular_values)
    
    auto dispatch_linalg_lstsq = [](const at::Tensor & self, const at::Tensor & b, ::std::optional<double> rcond, ::std::optional<c10::string_view> driver) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_lstsq(self, b, rcond, driver);
    };
    return wrap(NamedTuple, dispatch_linalg_lstsq(_r.tensor(0), _r.tensor(1), _r.toDoubleOptional(2), _r.stringViewOptional(3)));
  } else {
    // aten::linalg_lstsq.out(Tensor self, Tensor b, float? rcond=None, *, str? driver=None, Tensor(a!) solution, Tensor(b!) residuals, Tensor(c!) rank, Tensor(d!) singular_values) -> (Tensor(a!) solution, Tensor(b!) residuals, Tensor(c!) rank, Tensor(d!) singular_values)
    auto out = _r.tensorlist_n<4>(4);
    auto dispatch_linalg_lstsq_out = [](at::Tensor & solution, at::Tensor & residuals, at::Tensor & rank, at::Tensor & singular_values, const at::Tensor & self, const at::Tensor & b, ::std::optional<double> rcond, ::std::optional<c10::string_view> driver) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_lstsq_out(solution, residuals, rank, singular_values, self, b, rcond, driver);
    };
    return wrap(NamedTuple1, dispatch_linalg_lstsq_out(out[0], out[1], out[2], out[3], _r.tensor(0), _r.tensor(1), _r.toDoubleOptional(2), _r.stringViewOptional(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_lu
static PyObject * THPVariable_linalg_lu(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_linalg_lu_structseq();
  static PyTypeObject* NamedTuple1 = generated::get_linalg_lu_out_structseq();
  static PythonArgParser parser({
    "linalg_lu(Tensor A, *, bool pivot=True, TensorList[3] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(2)) {
    // aten::linalg_lu(Tensor A, *, bool pivot=True) -> (Tensor P, Tensor L, Tensor U)
    
    auto dispatch_linalg_lu = [](const at::Tensor & A, bool pivot) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_lu(A, pivot);
    };
    return wrap(NamedTuple, dispatch_linalg_lu(_r.tensor(0), _r.toBool(1)));
  } else {
    // aten::linalg_lu.out(Tensor A, *, bool pivot=True, Tensor(a!) P, Tensor(b!) L, Tensor(c!) U) -> (Tensor(a!) P, Tensor(b!) L, Tensor(c!) U)
    auto out = _r.tensorlist_n<3>(2);
    auto dispatch_linalg_lu_out = [](at::Tensor & P, at::Tensor & L, at::Tensor & U, const at::Tensor & A, bool pivot) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_lu_out(P, L, U, A, pivot);
    };
    return wrap(NamedTuple1, dispatch_linalg_lu_out(out[0], out[1], out[2], _r.tensor(0), _r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_lu_factor
static PyObject * THPVariable_linalg_lu_factor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_linalg_lu_factor_structseq();
  static PyTypeObject* NamedTuple1 = generated::get_linalg_lu_factor_out_structseq();
  static PythonArgParser parser({
    "linalg_lu_factor(Tensor A, *, bool pivot=True, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(2)) {
    // aten::linalg_lu_factor(Tensor A, *, bool pivot=True) -> (Tensor LU, Tensor pivots)
    
    auto dispatch_linalg_lu_factor = [](const at::Tensor & A, bool pivot) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_lu_factor(A, pivot);
    };
    return wrap(NamedTuple, dispatch_linalg_lu_factor(_r.tensor(0), _r.toBool(1)));
  } else {
    // aten::linalg_lu_factor.out(Tensor A, *, bool pivot=True, Tensor(a!) LU, Tensor(b!) pivots) -> (Tensor(a!) LU, Tensor(b!) pivots)
    auto out = _r.tensorlist_n<2>(2);
    auto dispatch_linalg_lu_factor_out = [](at::Tensor & LU, at::Tensor & pivots, const at::Tensor & A, bool pivot) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_lu_factor_out(LU, pivots, A, pivot);
    };
    return wrap(NamedTuple1, dispatch_linalg_lu_factor_out(out[0], out[1], _r.tensor(0), _r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_lu_factor_ex
static PyObject * THPVariable_linalg_lu_factor_ex(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_linalg_lu_factor_ex_structseq();
  static PyTypeObject* NamedTuple1 = generated::get_linalg_lu_factor_ex_out_structseq();
  static PythonArgParser parser({
    "linalg_lu_factor_ex(Tensor A, *, bool pivot=True, bool check_errors=False, TensorList[3] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(3)) {
    // aten::linalg_lu_factor_ex(Tensor A, *, bool pivot=True, bool check_errors=False) -> (Tensor LU, Tensor pivots, Tensor info)
    
    auto dispatch_linalg_lu_factor_ex = [](const at::Tensor & A, bool pivot, bool check_errors) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_lu_factor_ex(A, pivot, check_errors);
    };
    return wrap(NamedTuple, dispatch_linalg_lu_factor_ex(_r.tensor(0), _r.toBool(1), _r.toBool(2)));
  } else {
    // aten::linalg_lu_factor_ex.out(Tensor A, *, bool pivot=True, bool check_errors=False, Tensor(a!) LU, Tensor(b!) pivots, Tensor(c!) info) -> (Tensor(a!) LU, Tensor(b!) pivots, Tensor(c!) info)
    auto out = _r.tensorlist_n<3>(3);
    auto dispatch_linalg_lu_factor_ex_out = [](at::Tensor & LU, at::Tensor & pivots, at::Tensor & info, const at::Tensor & A, bool pivot, bool check_errors) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_lu_factor_ex_out(LU, pivots, info, A, pivot, check_errors);
    };
    return wrap(NamedTuple1, dispatch_linalg_lu_factor_ex_out(out[0], out[1], out[2], _r.tensor(0), _r.toBool(1), _r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_lu_solve
static PyObject * THPVariable_linalg_lu_solve(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_lu_solve(Tensor LU, Tensor pivots, Tensor B, *, bool left=True, bool adjoint=False, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(5)) {
    // aten::linalg_lu_solve(Tensor LU, Tensor pivots, Tensor B, *, bool left=True, bool adjoint=False) -> Tensor
    
    auto dispatch_linalg_lu_solve = [](const at::Tensor & LU, const at::Tensor & pivots, const at::Tensor & B, bool left, bool adjoint) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_lu_solve(LU, pivots, B, left, adjoint);
    };
    return wrap(dispatch_linalg_lu_solve(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toBool(3), _r.toBool(4)));
  } else {
    // aten::linalg_lu_solve.out(Tensor LU, Tensor pivots, Tensor B, *, bool left=True, bool adjoint=False, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_lu_solve_out = [](at::Tensor out, const at::Tensor & LU, const at::Tensor & pivots, const at::Tensor & B, bool left, bool adjoint) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_lu_solve_out(out, LU, pivots, B, left, adjoint);
    };
    return wrap(dispatch_linalg_lu_solve_out(_r.tensor(5), _r.tensor(0), _r.tensor(1), _r.tensor(2), _r.toBool(3), _r.toBool(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_matmul
static PyObject * THPVariable_linalg_matmul(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_matmul(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(2)) {
    // aten::linalg_matmul(Tensor self, Tensor other) -> Tensor
    
    auto dispatch_linalg_matmul = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_matmul(self, other);
    };
    return wrap(dispatch_linalg_matmul(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::linalg_matmul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_matmul_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_matmul_out(out, self, other);
    };
    return wrap(dispatch_linalg_matmul_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_matrix_exp
static PyObject * THPVariable_linalg_matrix_exp(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_matrix_exp(Tensor input)",
  }, /*traceable=*/true);

  ParsedArgs<1> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  // aten::linalg_matrix_exp(Tensor self) -> Tensor
  
  auto dispatch_linalg_matrix_exp = [](const at::Tensor & self) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::linalg_matrix_exp(self);
  };
  return wrap(dispatch_linalg_matrix_exp(_r.tensor(0)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// linalg_matrix_norm
static PyObject * THPVariable_linalg_matrix_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_matrix_norm(Tensor input, Scalar ord, IntArrayRef dim={-2,-1}, bool keepdim=False, *, ScalarType? dtype=None, Tensor out=None)",
    "linalg_matrix_norm(Tensor input, c10::string_view ord=\"fro\", IntArrayRef dim={-2,-1}, bool keepdim=False, *, ScalarType? dtype=None, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(5)) {
        // aten::linalg_matrix_norm(Tensor self, Scalar ord, int[] dim=[-2,-1], bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
        
        auto dispatch_linalg_matrix_norm = [](const at::Tensor & self, const at::Scalar & ord, at::IntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_matrix_norm(self, ord, dim, keepdim, dtype);
        };
        return wrap(dispatch_linalg_matrix_norm(_r.tensor(0), _r.scalar(1), _r.intlist(2), _r.toBool(3), _r.scalartypeOptional(4)));
      } else {
        // aten::linalg_matrix_norm.out(Tensor self, Scalar ord, int[] dim=[-2,-1], bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_linalg_matrix_norm_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & ord, at::IntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_matrix_norm_out(out, self, ord, dim, keepdim, dtype);
        };
        return wrap(dispatch_linalg_matrix_norm_out(_r.tensor(5), _r.tensor(0), _r.scalar(1), _r.intlist(2), _r.toBool(3), _r.scalartypeOptional(4)));
      }
    }
    case 1: {
      if (_r.isNone(5)) {
        // aten::linalg_matrix_norm.str_ord(Tensor self, str ord='fro', int[] dim=[-2,-1], bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
        
        auto dispatch_linalg_matrix_norm = [](const at::Tensor & self, c10::string_view ord, at::IntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_matrix_norm(self, ord, dim, keepdim, dtype);
        };
        return wrap(dispatch_linalg_matrix_norm(_r.tensor(0), _r.stringView(1), _r.intlist(2), _r.toBool(3), _r.scalartypeOptional(4)));
      } else {
        // aten::linalg_matrix_norm.str_ord_out(Tensor self, str ord='fro', int[] dim=[-2,-1], bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_linalg_matrix_norm_out = [](at::Tensor out, const at::Tensor & self, c10::string_view ord, at::IntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_matrix_norm_out(out, self, ord, dim, keepdim, dtype);
        };
        return wrap(dispatch_linalg_matrix_norm_out(_r.tensor(5), _r.tensor(0), _r.stringView(1), _r.intlist(2), _r.toBool(3), _r.scalartypeOptional(4)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_matrix_power
static PyObject * THPVariable_linalg_matrix_power(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_matrix_power(Tensor input, int64_t n, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(2)) {
    // aten::linalg_matrix_power(Tensor self, int n) -> Tensor
    
    auto dispatch_linalg_matrix_power = [](const at::Tensor & self, int64_t n) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_matrix_power(self, n);
    };
    return wrap(dispatch_linalg_matrix_power(_r.tensor(0), _r.toInt64(1)));
  } else {
    // aten::linalg_matrix_power.out(Tensor self, int n, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_matrix_power_out = [](at::Tensor out, const at::Tensor & self, int64_t n) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_matrix_power_out(out, self, n);
    };
    return wrap(dispatch_linalg_matrix_power_out(_r.tensor(2), _r.tensor(0), _r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// linalg_matrix_rank
static PyObject * THPVariable_linalg_matrix_rank(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_matrix_rank(Tensor input, *, Tensor? atol=None, Tensor? rtol=None, bool hermitian=False, Tensor out=None)",
    "linalg_matrix_rank(Tensor input, *, double? atol=None, double? rtol=None, bool hermitian=False, Tensor out=None)",
    "linalg_matrix_rank(Tensor input, Tensor tol, bool hermitian=False, *, Tensor out=None)",
    "linalg_matrix_rank(Tensor input, double tol, bool hermitian=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(4)) {
        // aten::linalg_matrix_rank.atol_rtol_tensor(Tensor input, *, Tensor? atol=None, Tensor? rtol=None, bool hermitian=False) -> Tensor
        
        auto dispatch_linalg_matrix_rank = [](const at::Tensor & input, const ::std::optional<at::Tensor> & atol, const ::std::optional<at::Tensor> & rtol, bool hermitian) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_matrix_rank(input, atol, rtol, hermitian);
        };
        return wrap(dispatch_linalg_matrix_rank(_r.tensor(0), _r.optionalTensor(1), _r.optionalTensor(2), _r.toBool(3)));
      } else {
        // aten::linalg_matrix_rank.atol_rtol_tensor_out(Tensor input, *, Tensor? atol=None, Tensor? rtol=None, bool hermitian=False, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_linalg_matrix_rank_out = [](at::Tensor out, const at::Tensor & input, const ::std::optional<at::Tensor> & atol, const ::std::optional<at::Tensor> & rtol, bool hermitian) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_matrix_rank_out(out, input, atol, rtol, hermitian);
        };
        return wrap(dispatch_linalg_matrix_rank_out(_r.tensor(4), _r.tensor(0), _r.optionalTensor(1), _r.optionalTensor(2), _r.toBool(3)));
      }
    }
    case 1: {
      if (_r.isNone(4)) {
        // aten::linalg_matrix_rank.atol_rtol_float(Tensor self, *, float? atol=None, float? rtol=None, bool hermitian=False) -> Tensor
        
        auto dispatch_linalg_matrix_rank = [](const at::Tensor & self, ::std::optional<double> atol, ::std::optional<double> rtol, bool hermitian) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_matrix_rank(self, atol, rtol, hermitian);
        };
        return wrap(dispatch_linalg_matrix_rank(_r.tensor(0), _r.toDoubleOptional(1), _r.toDoubleOptional(2), _r.toBool(3)));
      } else {
        // aten::linalg_matrix_rank.atol_rtol_float_out(Tensor self, *, float? atol=None, float? rtol=None, bool hermitian=False, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_linalg_matrix_rank_out = [](at::Tensor out, const at::Tensor & self, ::std::optional<double> atol, ::std::optional<double> rtol, bool hermitian) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_matrix_rank_out(out, self, atol, rtol, hermitian);
        };
        return wrap(dispatch_linalg_matrix_rank_out(_r.tensor(4), _r.tensor(0), _r.toDoubleOptional(1), _r.toDoubleOptional(2), _r.toBool(3)));
      }
    }
    case 2: {
      if (_r.isNone(3)) {
        // aten::linalg_matrix_rank.tol_tensor(Tensor input, Tensor tol, bool hermitian=False) -> Tensor
        
        auto dispatch_linalg_matrix_rank = [](const at::Tensor & input, const at::Tensor & tol, bool hermitian) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_matrix_rank(input, tol, hermitian);
        };
        return wrap(dispatch_linalg_matrix_rank(_r.tensor(0), _r.tensor(1), _r.toBool(2)));
      } else {
        // aten::linalg_matrix_rank.out_tol_tensor(Tensor input, Tensor tol, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_linalg_matrix_rank_out = [](at::Tensor out, const at::Tensor & input, const at::Tensor & tol, bool hermitian) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_matrix_rank_out(out, input, tol, hermitian);
        };
        return wrap(dispatch_linalg_matrix_rank_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.toBool(2)));
      }
    }
    case 3: {
      if (_r.isNone(3)) {
        // aten::linalg_matrix_rank(Tensor self, float tol, bool hermitian=False) -> Tensor
        
        auto dispatch_linalg_matrix_rank = [](const at::Tensor & self, double tol, bool hermitian) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_matrix_rank(self, tol, hermitian);
        };
        return wrap(dispatch_linalg_matrix_rank(_r.tensor(0), _r.toDouble(1), _r.toBool(2)));
      } else {
        // aten::linalg_matrix_rank.out(Tensor self, float tol, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_linalg_matrix_rank_out = [](at::Tensor out, const at::Tensor & self, double tol, bool hermitian) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_matrix_rank_out(out, self, tol, hermitian);
        };
        return wrap(dispatch_linalg_matrix_rank_out(_r.tensor(3), _r.tensor(0), _r.toDouble(1), _r.toBool(2)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_multi_dot
static PyObject * THPVariable_linalg_multi_dot(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_multi_dot(TensorList tensors, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(1)) {
    // aten::linalg_multi_dot(Tensor[] tensors) -> Tensor
    
    auto dispatch_linalg_multi_dot = [](at::TensorList tensors) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_multi_dot(tensors);
    };
    return wrap(dispatch_linalg_multi_dot(_r.tensorlist(0)));
  } else {
    // aten::linalg_multi_dot.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_multi_dot_out = [](at::Tensor out, at::TensorList tensors) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_multi_dot_out(out, tensors);
    };
    return wrap(dispatch_linalg_multi_dot_out(_r.tensor(1), _r.tensorlist(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// linalg_norm
static PyObject * THPVariable_linalg_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_norm(Tensor input, Scalar? ord=None, IntArrayRef[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor out=None)",
    "linalg_norm(Tensor input, c10::string_view ord, IntArrayRef[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(5)) {
        // aten::linalg_norm(Tensor self, Scalar? ord=None, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
        
        auto dispatch_linalg_norm = [](const at::Tensor & self, const ::std::optional<at::Scalar> & ord, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_norm(self, ord, dim, keepdim, dtype);
        };
        return wrap(dispatch_linalg_norm(_r.tensor(0), _r.scalarOptional(1), _r.intlistOptional(2), _r.toBool(3), _r.scalartypeOptional(4)));
      } else {
        // aten::linalg_norm.out(Tensor self, Scalar? ord=None, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_linalg_norm_out = [](at::Tensor out, const at::Tensor & self, const ::std::optional<at::Scalar> & ord, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_norm_out(out, self, ord, dim, keepdim, dtype);
        };
        return wrap(dispatch_linalg_norm_out(_r.tensor(5), _r.tensor(0), _r.scalarOptional(1), _r.intlistOptional(2), _r.toBool(3), _r.scalartypeOptional(4)));
      }
    }
    case 1: {
      if (_r.isNone(5)) {
        // aten::linalg_norm.ord_str(Tensor self, str ord, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
        
        auto dispatch_linalg_norm = [](const at::Tensor & self, c10::string_view ord, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_norm(self, ord, dim, keepdim, dtype);
        };
        return wrap(dispatch_linalg_norm(_r.tensor(0), _r.stringView(1), _r.intlistOptional(2), _r.toBool(3), _r.scalartypeOptional(4)));
      } else {
        // aten::linalg_norm.ord_str_out(Tensor self, str ord, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_linalg_norm_out = [](at::Tensor out, const at::Tensor & self, c10::string_view ord, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_norm_out(out, self, ord, dim, keepdim, dtype);
        };
        return wrap(dispatch_linalg_norm_out(_r.tensor(5), _r.tensor(0), _r.stringView(1), _r.intlistOptional(2), _r.toBool(3), _r.scalartypeOptional(4)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// linalg_pinv
static PyObject * THPVariable_linalg_pinv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_pinv(Tensor input, *, Tensor? atol=None, Tensor? rtol=None, bool hermitian=False, Tensor out=None)",
    "linalg_pinv(Tensor input, *, double? atol=None, double? rtol=None, bool hermitian=False, Tensor out=None)",
    "linalg_pinv(Tensor input, Tensor rcond, bool hermitian=False, *, Tensor out=None)",
    "linalg_pinv(Tensor input, double rcond, bool hermitian=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(4)) {
        // aten::linalg_pinv.atol_rtol_tensor(Tensor self, *, Tensor? atol=None, Tensor? rtol=None, bool hermitian=False) -> Tensor
        
        auto dispatch_linalg_pinv = [](const at::Tensor & self, const ::std::optional<at::Tensor> & atol, const ::std::optional<at::Tensor> & rtol, bool hermitian) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_pinv(self, atol, rtol, hermitian);
        };
        return wrap(dispatch_linalg_pinv(_r.tensor(0), _r.optionalTensor(1), _r.optionalTensor(2), _r.toBool(3)));
      } else {
        // aten::linalg_pinv.atol_rtol_tensor_out(Tensor self, *, Tensor? atol=None, Tensor? rtol=None, bool hermitian=False, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_linalg_pinv_out = [](at::Tensor out, const at::Tensor & self, const ::std::optional<at::Tensor> & atol, const ::std::optional<at::Tensor> & rtol, bool hermitian) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_pinv_out(out, self, atol, rtol, hermitian);
        };
        return wrap(dispatch_linalg_pinv_out(_r.tensor(4), _r.tensor(0), _r.optionalTensor(1), _r.optionalTensor(2), _r.toBool(3)));
      }
    }
    case 1: {
      if (_r.isNone(4)) {
        // aten::linalg_pinv.atol_rtol_float(Tensor self, *, float? atol=None, float? rtol=None, bool hermitian=False) -> Tensor
        
        auto dispatch_linalg_pinv = [](const at::Tensor & self, ::std::optional<double> atol, ::std::optional<double> rtol, bool hermitian) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_pinv(self, atol, rtol, hermitian);
        };
        return wrap(dispatch_linalg_pinv(_r.tensor(0), _r.toDoubleOptional(1), _r.toDoubleOptional(2), _r.toBool(3)));
      } else {
        // aten::linalg_pinv.atol_rtol_float_out(Tensor self, *, float? atol=None, float? rtol=None, bool hermitian=False, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_linalg_pinv_out = [](at::Tensor out, const at::Tensor & self, ::std::optional<double> atol, ::std::optional<double> rtol, bool hermitian) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_pinv_out(out, self, atol, rtol, hermitian);
        };
        return wrap(dispatch_linalg_pinv_out(_r.tensor(4), _r.tensor(0), _r.toDoubleOptional(1), _r.toDoubleOptional(2), _r.toBool(3)));
      }
    }
    case 2: {
      if (_r.isNone(3)) {
        // aten::linalg_pinv.rcond_tensor(Tensor self, Tensor rcond, bool hermitian=False) -> Tensor
        
        auto dispatch_linalg_pinv = [](const at::Tensor & self, const at::Tensor & rcond, bool hermitian) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_pinv(self, rcond, hermitian);
        };
        return wrap(dispatch_linalg_pinv(_r.tensor(0), _r.tensor(1), _r.toBool(2)));
      } else {
        // aten::linalg_pinv.out_rcond_tensor(Tensor self, Tensor rcond, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_linalg_pinv_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & rcond, bool hermitian) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_pinv_out(out, self, rcond, hermitian);
        };
        return wrap(dispatch_linalg_pinv_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.toBool(2)));
      }
    }
    case 3: {
      if (_r.isNone(3)) {
        // aten::linalg_pinv(Tensor self, float rcond, bool hermitian=False) -> Tensor
        
        auto dispatch_linalg_pinv = [](const at::Tensor & self, double rcond, bool hermitian) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_pinv(self, rcond, hermitian);
        };
        return wrap(dispatch_linalg_pinv(_r.tensor(0), _r.toDouble(1), _r.toBool(2)));
      } else {
        // aten::linalg_pinv.out(Tensor self, float rcond, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_linalg_pinv_out = [](at::Tensor out, const at::Tensor & self, double rcond, bool hermitian) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_pinv_out(out, self, rcond, hermitian);
        };
        return wrap(dispatch_linalg_pinv_out(_r.tensor(3), _r.tensor(0), _r.toDouble(1), _r.toBool(2)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_qr
static PyObject * THPVariable_linalg_qr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_linalg_qr_structseq();
  static PyTypeObject* NamedTuple1 = generated::get_linalg_qr_out_structseq();
  static PythonArgParser parser({
    "linalg_qr(Tensor A, c10::string_view mode=\"reduced\", *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(2)) {
    // aten::linalg_qr(Tensor A, str mode='reduced') -> (Tensor Q, Tensor R)
    
    auto dispatch_linalg_qr = [](const at::Tensor & A, c10::string_view mode) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_qr(A, mode);
    };
    return wrap(NamedTuple, dispatch_linalg_qr(_r.tensor(0), _r.stringView(1)));
  } else {
    // aten::linalg_qr.out(Tensor A, str mode='reduced', *, Tensor(a!) Q, Tensor(b!) R) -> (Tensor(a!) Q, Tensor(b!) R)
    auto out = _r.tensorlist_n<2>(2);
    auto dispatch_linalg_qr_out = [](at::Tensor & Q, at::Tensor & R, const at::Tensor & A, c10::string_view mode) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_qr_out(Q, R, A, mode);
    };
    return wrap(NamedTuple1, dispatch_linalg_qr_out(out[0], out[1], _r.tensor(0), _r.stringView(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_slogdet
static PyObject * THPVariable_linalg_slogdet(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_linalg_slogdet_structseq();
  static PyTypeObject* NamedTuple1 = generated::get_linalg_slogdet_out_structseq();
  static PythonArgParser parser({
    "linalg_slogdet(Tensor A, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(1)) {
    // aten::linalg_slogdet(Tensor A) -> (Tensor sign, Tensor logabsdet)
    
    auto dispatch_linalg_slogdet = [](const at::Tensor & A) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_slogdet(A);
    };
    return wrap(NamedTuple, dispatch_linalg_slogdet(_r.tensor(0)));
  } else {
    // aten::linalg_slogdet.out(Tensor A, *, Tensor(a!) sign, Tensor(b!) logabsdet) -> (Tensor(a!) sign, Tensor(b!) logabsdet)
    auto out = _r.tensorlist_n<2>(1);
    auto dispatch_linalg_slogdet_out = [](at::Tensor & sign, at::Tensor & logabsdet, const at::Tensor & A) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_slogdet_out(sign, logabsdet, A);
    };
    return wrap(NamedTuple1, dispatch_linalg_slogdet_out(out[0], out[1], _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_solve
static PyObject * THPVariable_linalg_solve(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_solve(Tensor A, Tensor B, *, bool left=True, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(3)) {
    // aten::linalg_solve(Tensor A, Tensor B, *, bool left=True) -> Tensor
    
    auto dispatch_linalg_solve = [](const at::Tensor & A, const at::Tensor & B, bool left) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_solve(A, B, left);
    };
    return wrap(dispatch_linalg_solve(_r.tensor(0), _r.tensor(1), _r.toBool(2)));
  } else {
    // aten::linalg_solve.out(Tensor A, Tensor B, *, bool left=True, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_solve_out = [](at::Tensor out, const at::Tensor & A, const at::Tensor & B, bool left) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_solve_out(out, A, B, left);
    };
    return wrap(dispatch_linalg_solve_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.toBool(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_solve_ex
static PyObject * THPVariable_linalg_solve_ex(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_linalg_solve_ex_structseq();
  static PyTypeObject* NamedTuple1 = generated::get_linalg_solve_ex_out_structseq();
  static PythonArgParser parser({
    "linalg_solve_ex(Tensor A, Tensor B, *, bool left=True, bool check_errors=False, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(4)) {
    // aten::linalg_solve_ex(Tensor A, Tensor B, *, bool left=True, bool check_errors=False) -> (Tensor result, Tensor info)
    
    auto dispatch_linalg_solve_ex = [](const at::Tensor & A, const at::Tensor & B, bool left, bool check_errors) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_solve_ex(A, B, left, check_errors);
    };
    return wrap(NamedTuple, dispatch_linalg_solve_ex(_r.tensor(0), _r.tensor(1), _r.toBool(2), _r.toBool(3)));
  } else {
    // aten::linalg_solve_ex.out(Tensor A, Tensor B, *, bool left=True, bool check_errors=False, Tensor(a!) result, Tensor(b!) info) -> (Tensor(a!) result, Tensor(b!) info)
    auto out = _r.tensorlist_n<2>(4);
    auto dispatch_linalg_solve_ex_out = [](at::Tensor & result, at::Tensor & info, const at::Tensor & A, const at::Tensor & B, bool left, bool check_errors) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_solve_ex_out(result, info, A, B, left, check_errors);
    };
    return wrap(NamedTuple1, dispatch_linalg_solve_ex_out(out[0], out[1], _r.tensor(0), _r.tensor(1), _r.toBool(2), _r.toBool(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_solve_triangular
static PyObject * THPVariable_linalg_solve_triangular(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_solve_triangular(Tensor input, Tensor B, *, bool upper, bool left=True, bool unitriangular=False, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(5)) {
    // aten::linalg_solve_triangular(Tensor self, Tensor B, *, bool upper, bool left=True, bool unitriangular=False) -> Tensor
    
    auto dispatch_linalg_solve_triangular = [](const at::Tensor & self, const at::Tensor & B, bool upper, bool left, bool unitriangular) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_solve_triangular(self, B, upper, left, unitriangular);
    };
    return wrap(dispatch_linalg_solve_triangular(_r.tensor(0), _r.tensor(1), _r.toBool(2), _r.toBool(3), _r.toBool(4)));
  } else {
    // aten::linalg_solve_triangular.out(Tensor self, Tensor B, *, bool upper, bool left=True, bool unitriangular=False, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_solve_triangular_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & B, bool upper, bool left, bool unitriangular) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_solve_triangular_out(out, self, B, upper, left, unitriangular);
    };
    return wrap(dispatch_linalg_solve_triangular_out(_r.tensor(5), _r.tensor(0), _r.tensor(1), _r.toBool(2), _r.toBool(3), _r.toBool(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_svd
static PyObject * THPVariable_linalg_svd(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyTypeObject* NamedTuple = generated::get_linalg_svd_structseq();
  static PyTypeObject* NamedTuple1 = generated::get_linalg_svd_out_structseq();
  static PythonArgParser parser({
    "linalg_svd(Tensor A, bool full_matrices=True, *, c10::string_view? driver=None, TensorList[3] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(3)) {
    // aten::linalg_svd(Tensor A, bool full_matrices=True, *, str? driver=None) -> (Tensor U, Tensor S, Tensor Vh)
    
    auto dispatch_linalg_svd = [](const at::Tensor & A, bool full_matrices, ::std::optional<c10::string_view> driver) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_svd(A, full_matrices, driver);
    };
    return wrap(NamedTuple, dispatch_linalg_svd(_r.tensor(0), _r.toBool(1), _r.stringViewOptional(2)));
  } else {
    // aten::linalg_svd.U(Tensor A, bool full_matrices=True, *, str? driver=None, Tensor(a!) U, Tensor(b!) S, Tensor(c!) Vh) -> (Tensor(a!) U, Tensor(b!) S, Tensor(c!) Vh)
    auto out = _r.tensorlist_n<3>(3);
    auto dispatch_linalg_svd_out = [](at::Tensor & U, at::Tensor & S, at::Tensor & Vh, const at::Tensor & A, bool full_matrices, ::std::optional<c10::string_view> driver) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_svd_out(U, S, Vh, A, full_matrices, driver);
    };
    return wrap(NamedTuple1, dispatch_linalg_svd_out(out[0], out[1], out[2], _r.tensor(0), _r.toBool(1), _r.stringViewOptional(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_svdvals
static PyObject * THPVariable_linalg_svdvals(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_svdvals(Tensor A, *, c10::string_view? driver=None, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(2)) {
    // aten::linalg_svdvals(Tensor A, *, str? driver=None) -> Tensor
    
    auto dispatch_linalg_svdvals = [](const at::Tensor & A, ::std::optional<c10::string_view> driver) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_svdvals(A, driver);
    };
    return wrap(dispatch_linalg_svdvals(_r.tensor(0), _r.stringViewOptional(1)));
  } else {
    // aten::linalg_svdvals.out(Tensor A, *, str? driver=None, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_svdvals_out = [](at::Tensor out, const at::Tensor & A, ::std::optional<c10::string_view> driver) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_svdvals_out(out, A, driver);
    };
    return wrap(dispatch_linalg_svdvals_out(_r.tensor(2), _r.tensor(0), _r.stringViewOptional(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_tensorinv
static PyObject * THPVariable_linalg_tensorinv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_tensorinv(Tensor input, int64_t ind=2, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(2)) {
    // aten::linalg_tensorinv(Tensor self, int ind=2) -> Tensor
    
    auto dispatch_linalg_tensorinv = [](const at::Tensor & self, int64_t ind) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_tensorinv(self, ind);
    };
    return wrap(dispatch_linalg_tensorinv(_r.tensor(0), _r.toInt64(1)));
  } else {
    // aten::linalg_tensorinv.out(Tensor self, int ind=2, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_tensorinv_out = [](at::Tensor out, const at::Tensor & self, int64_t ind) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_tensorinv_out(out, self, ind);
    };
    return wrap(dispatch_linalg_tensorinv_out(_r.tensor(2), _r.tensor(0), _r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_tensorsolve
static PyObject * THPVariable_linalg_tensorsolve(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_tensorsolve(Tensor input, Tensor other, IntArrayRef? dims=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(3)) {
    // aten::linalg_tensorsolve(Tensor self, Tensor other, int[]? dims=None) -> Tensor
    
    auto dispatch_linalg_tensorsolve = [](const at::Tensor & self, const at::Tensor & other, at::OptionalIntArrayRef dims) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_tensorsolve(self, other, dims);
    };
    return wrap(dispatch_linalg_tensorsolve(_r.tensor(0), _r.tensor(1), _r.intlistOptional(2)));
  } else {
    // aten::linalg_tensorsolve.out(Tensor self, Tensor other, int[]? dims=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_tensorsolve_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other, at::OptionalIntArrayRef dims) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_tensorsolve_out(out, self, other, dims);
    };
    return wrap(dispatch_linalg_tensorsolve_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.intlistOptional(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_vander
static PyObject * THPVariable_linalg_vander(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_vander(Tensor x, *, SymInt? N=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  // aten::linalg_vander(Tensor x, *, SymInt? N=None) -> Tensor
  
  auto dispatch_linalg_vander = [](const at::Tensor & x, ::std::optional<c10::SymInt> N) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::linalg_vander_symint(x, N);
  };
  return wrap(dispatch_linalg_vander(_r.tensor(0), _r.toSymIntOptional(1)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_vecdot
static PyObject * THPVariable_linalg_vecdot(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_vecdot(Tensor x, Tensor y, *, int64_t dim=-1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(3)) {
    // aten::linalg_vecdot(Tensor x, Tensor y, *, int dim=-1) -> Tensor
    
    auto dispatch_linalg_vecdot = [](const at::Tensor & x, const at::Tensor & y, int64_t dim) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_vecdot(x, y, dim);
    };
    return wrap(dispatch_linalg_vecdot(_r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  } else {
    // aten::linalg_vecdot.out(Tensor x, Tensor y, *, int dim=-1, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_vecdot_out = [](at::Tensor out, const at::Tensor & x, const at::Tensor & y, int64_t dim) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_vecdot_out(out, x, y, dim);
    };
    return wrap(dispatch_linalg_vecdot_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.toInt64(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_vector_norm
static PyObject * THPVariable_linalg_vector_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_vector_norm(Tensor input, Scalar ord=2, IntArrayRef[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(5)) {
    // aten::linalg_vector_norm(Tensor self, Scalar ord=2, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    
    auto dispatch_linalg_vector_norm = [](const at::Tensor & self, const at::Scalar & ord, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_vector_norm(self, ord, dim, keepdim, dtype);
    };
    return wrap(dispatch_linalg_vector_norm(_r.tensor(0), _r.scalar(1), _r.intlistOptional(2), _r.toBool(3), _r.scalartypeOptional(4)));
  } else {
    // aten::linalg_vector_norm.out(Tensor self, Scalar ord=2, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_vector_norm_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & ord, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_vector_norm_out(out, self, ord, dim, keepdim, dtype);
    };
    return wrap(dispatch_linalg_vector_norm_out(_r.tensor(5), _r.tensor(0), _r.scalar(1), _r.intlistOptional(2), _r.toBool(3), _r.scalartypeOptional(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

} // namespace torch::autograd
