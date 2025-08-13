#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// @generated from ../tools/autograd/templates/python_sparse_functions.cpp

#include "torch/csrc/Device.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/python_sparse_functions.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/autograd/utils/python_arg_parsing.h"
#include "torch/csrc/utils/pycfunction_helpers.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/structseq.h"

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_sparse_addmm.h>
#include <ATen/ops/_sparse_log_softmax.h>
#include <ATen/ops/_sparse_mm.h>
#include <ATen/ops/_sparse_mm_reduce_impl.h>
#include <ATen/ops/_sparse_softmax.h>
#include <ATen/ops/_spdiags.h>
#include <ATen/ops/_spsolve.h>
#include <ATen/ops/sparse_sampled_addmm.h>
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

static PyObject * THPVariable__sparse_addmm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__sparse_log_softmax(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__sparse_mm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__sparse_mm_reduce_impl(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__sparse_softmax(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__spdiags(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable__spsolve(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_sparse_sampled_addmm(PyObject* self_, PyObject* args, PyObject* kwargs);

static PyMethodDef sparse_functions[] = {
  {"_sparse_addmm", castPyCFunctionWithKeywords(THPVariable__sparse_addmm), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_sparse_log_softmax", castPyCFunctionWithKeywords(THPVariable__sparse_log_softmax), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_sparse_mm", castPyCFunctionWithKeywords(THPVariable__sparse_mm), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_sparse_mm_reduce_impl", castPyCFunctionWithKeywords(THPVariable__sparse_mm_reduce_impl), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_sparse_softmax", castPyCFunctionWithKeywords(THPVariable__sparse_softmax), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_spdiags", castPyCFunctionWithKeywords(THPVariable__spdiags), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"_spsolve", castPyCFunctionWithKeywords(THPVariable__spsolve), METH_VARARGS | METH_KEYWORDS, nullptr},
  {"sparse_sampled_addmm", castPyCFunctionWithKeywords(THPVariable_sparse_sampled_addmm), METH_VARARGS | METH_KEYWORDS, nullptr},
  {NULL}
};

static PyObject* THPSparseVariableFunctionsModule = NULL;

void initSparseFunctions(PyObject* module) {
  static struct PyModuleDef def = {
     PyModuleDef_HEAD_INIT,
     "torch._C._sparse",
     NULL,
     -1,
     sparse_functions
  };
  PyObject* sparse = PyModule_Create(&def);
  THPSparseVariableFunctionsModule = sparse;
  if (!sparse) {
    throw python_error();
  }
  // steals a reference to sparse
  if (PyModule_AddObject(module, "_sparse", sparse) != 0) {
    throw python_error();
  }
}

// generated methods start here

// _sparse_addmm
static PyObject * THPVariable__sparse_addmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sparse_addmm(Tensor input, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSparseVariableFunctionsModule, "torch.sparse");
  }
  // aten::_sparse_addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
  
  auto dispatch__sparse_addmm = [](const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_sparse_addmm(self, mat1, mat2, beta, alpha);
  };
  return wrap(dispatch__sparse_addmm(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _sparse_log_softmax
static PyObject * THPVariable__sparse_log_softmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sparse_log_softmax(Tensor input, int64_t dim, ScalarType? dtype=None)",
    "_sparse_log_softmax(Tensor input, int64_t dim, bool half_to_float)",
    "_sparse_log_softmax(Tensor input, Dimname dim, *, ScalarType? dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSparseVariableFunctionsModule, "torch.sparse");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_sparse_log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
      
      auto dispatch__sparse_log_softmax = [](const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::_sparse_log_softmax(self, dim, dtype);
      };
      return wrap(dispatch__sparse_log_softmax(_r.tensor(0), _r.toInt64(1), _r.scalartypeOptional(2)));
    }
    case 1: {
      // aten::_sparse_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
      
      auto dispatch__sparse_log_softmax = [](const at::Tensor & self, int64_t dim, bool half_to_float) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::_sparse_log_softmax(self, dim, half_to_float);
      };
      return wrap(dispatch__sparse_log_softmax(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
    }
    case 2: {
      // aten::_sparse_log_softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
      
      auto dispatch__sparse_log_softmax = [](const at::Tensor & self, at::Dimname dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::_sparse_log_softmax(self, dim, dtype);
      };
      return wrap(dispatch__sparse_log_softmax(_r.tensor(0), _r.dimname(1), _r.scalartypeOptional(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _sparse_mm
static PyObject * THPVariable__sparse_mm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sparse_mm(Tensor sparse, Tensor dense)",
    "_sparse_mm(Tensor sparse, Tensor dense, c10::string_view reduce)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSparseVariableFunctionsModule, "torch.sparse");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_sparse_mm(Tensor sparse, Tensor dense) -> Tensor
      
      auto dispatch__sparse_mm = [](const at::Tensor & sparse, const at::Tensor & dense) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::_sparse_mm(sparse, dense);
      };
      return wrap(dispatch__sparse_mm(_r.tensor(0), _r.tensor(1)));
    }
    case 1: {
      // aten::_sparse_mm.reduce(Tensor sparse, Tensor dense, str reduce) -> Tensor
      
      auto dispatch__sparse_mm = [](const at::Tensor & sparse, const at::Tensor & dense, c10::string_view reduce) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::_sparse_mm(sparse, dense, reduce);
      };
      return wrap(dispatch__sparse_mm(_r.tensor(0), _r.tensor(1), _r.stringView(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _sparse_mm_reduce_impl
static PyObject * THPVariable__sparse_mm_reduce_impl(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sparse_mm_reduce_impl(Tensor input, Tensor other, c10::string_view reduce)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSparseVariableFunctionsModule, "torch.sparse");
  }
  // aten::_sparse_mm_reduce_impl(Tensor self, Tensor other, str reduce) -> (Tensor, Tensor)
  
  auto dispatch__sparse_mm_reduce_impl = [](const at::Tensor & self, const at::Tensor & other, c10::string_view reduce) -> ::std::tuple<at::Tensor,at::Tensor> {
    pybind11::gil_scoped_release no_gil;
    return at::_sparse_mm_reduce_impl(self, other, reduce);
  };
  return wrap(dispatch__sparse_mm_reduce_impl(_r.tensor(0), _r.tensor(1), _r.stringView(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// _sparse_softmax
static PyObject * THPVariable__sparse_softmax(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_sparse_softmax(Tensor input, int64_t dim, ScalarType? dtype=None)",
    "_sparse_softmax(Tensor input, int64_t dim, bool half_to_float)",
    "_sparse_softmax(Tensor input, Dimname dim, *, ScalarType? dtype=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSparseVariableFunctionsModule, "torch.sparse");
  }
  switch (_r.idx) {
    case 0: {
      // aten::_sparse_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
      
      auto dispatch__sparse_softmax = [](const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::_sparse_softmax(self, dim, dtype);
      };
      return wrap(dispatch__sparse_softmax(_r.tensor(0), _r.toInt64(1), _r.scalartypeOptional(2)));
    }
    case 1: {
      // aten::_sparse_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
      
      auto dispatch__sparse_softmax = [](const at::Tensor & self, int64_t dim, bool half_to_float) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::_sparse_softmax(self, dim, half_to_float);
      };
      return wrap(dispatch__sparse_softmax(_r.tensor(0), _r.toInt64(1), _r.toBool(2)));
    }
    case 2: {
      // aten::_sparse_softmax.Dimname(Tensor self, Dimname dim, *, ScalarType? dtype=None) -> Tensor
      
      auto dispatch__sparse_softmax = [](const at::Tensor & self, at::Dimname dim, ::std::optional<at::ScalarType> dtype) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return at::_sparse_softmax(self, dim, dtype);
      };
      return wrap(dispatch__sparse_softmax(_r.tensor(0), _r.dimname(1), _r.scalartypeOptional(2)));
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _spdiags
static PyObject * THPVariable__spdiags(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_spdiags(Tensor diagonals, Tensor offsets, IntArrayRef shape, Layout? layout=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSparseVariableFunctionsModule, "torch.sparse");
  }
  // aten::_spdiags(Tensor diagonals, Tensor offsets, int[] shape, Layout? layout=None) -> Tensor
  
  auto dispatch__spdiags = [](const at::Tensor & diagonals, const at::Tensor & offsets, at::IntArrayRef shape, ::std::optional<at::Layout> layout) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_spdiags(diagonals, offsets, shape, layout);
  };
  return wrap(dispatch__spdiags(_r.tensor(0), _r.tensor(1), _r.intlist(2), _r.layoutOptional(3)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// _spsolve
static PyObject * THPVariable__spsolve(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "_spsolve(Tensor A, Tensor B, *, bool left=True)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSparseVariableFunctionsModule, "torch.sparse");
  }
  // aten::_spsolve(Tensor A, Tensor B, *, bool left=True) -> Tensor
  
  auto dispatch__spsolve = [](const at::Tensor & A, const at::Tensor & B, bool left) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::_spsolve(A, B, left);
  };
  return wrap(dispatch__spsolve(_r.tensor(0), _r.tensor(1), _r.toBool(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// sparse_sampled_addmm
static PyObject * THPVariable_sparse_sampled_addmm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "sparse_sampled_addmm(Tensor input, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSparseVariableFunctionsModule, "torch.sparse");
  }
  if (_r.isNone(5)) {
    // aten::sparse_sampled_addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
    
    auto dispatch_sparse_sampled_addmm = [](const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::sparse_sampled_addmm(self, mat1, mat2, beta, alpha);
    };
    return wrap(dispatch_sparse_sampled_addmm(_r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
  } else {
    // aten::sparse_sampled_addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_sparse_sampled_addmm_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::sparse_sampled_addmm_out(out, self, mat1, mat2, beta, alpha);
    };
    return wrap(dispatch_sparse_sampled_addmm_out(_r.tensor(5), _r.tensor(0), _r.tensor(1), _r.tensor(2), _r.scalar(3), _r.scalar(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

} // namespace torch::autograd
