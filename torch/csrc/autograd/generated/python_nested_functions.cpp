#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// @generated from ../tools/autograd/templates/python_nested_functions.cpp

#include "torch/csrc/Device.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/python_nested_functions.h"
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
#include <ATen/ops/nested_to_padded_tensor.h>
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
using at::OptionalIntArrayRef;
using at::Generator;
using at::TensorList;
using at::Dimname;
using at::DimnameList;

using namespace torch::autograd::utils;

namespace torch::autograd {

// generated forward declarations start here

static PyObject * THPVariable_nested_to_padded_tensor(PyObject* self_, PyObject* args, PyObject* kwargs);

static PyMethodDef nested_functions[] = {
  {NULL, NULL, 0, NULL},
  {"nested_to_padded_tensor", castPyCFunctionWithKeywords(THPVariable_nested_to_padded_tensor), METH_VARARGS | METH_KEYWORDS, nullptr},
  {NULL}
};

static PyObject* THPNestedVariableFunctionsModule = NULL;

void initNestedFunctions(PyObject* module) {
  nested_functions[0] = get_nested_functions_manual()[0];
  static struct PyModuleDef def = {
     PyModuleDef_HEAD_INIT,
     "torch._C._nested",
     NULL,
     -1,
     nested_functions
  };
  PyObject* nested = PyModule_Create(&def);
  THPNestedVariableFunctionsModule = nested;
  if (!nested) {
    throw python_error();
  }
  // steals a reference to nested
  if (PyModule_AddObject(module, "_nested", nested) != 0) {
    throw python_error();
  }
}

// generated methods start here

// nested_to_padded_tensor
static PyObject * THPVariable_nested_to_padded_tensor(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "nested_to_padded_tensor(Tensor input, double padding, IntArrayRef? output_size=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPNestedVariableFunctionsModule, "torch.nested");
  }
  // aten::nested_to_padded_tensor(Tensor self, float padding, int[]? output_size=None) -> Tensor
  
  auto dispatch_nested_to_padded_tensor = [](const at::Tensor & self, double padding, at::OptionalIntArrayRef output_size) -> at::Tensor {
    pybind11::gil_scoped_release no_gil;
    return at::nested_to_padded_tensor(self, padding, output_size);
  };
  return wrap(dispatch_nested_to_padded_tensor(_r.tensor(0), _r.toDouble(1), _r.intlistOptional(2)));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

} // namespace torch::autograd
