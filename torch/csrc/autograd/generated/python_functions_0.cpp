#include <torch/csrc/autograd/generated/python_functions.h>

// @generated from ../tools/autograd/templates/python_functions.cpp

#include <Python.h>
#include <ATen/ATen.h>

#include <c10/core/SymNodeImpl.h>
#include "torch/csrc/autograd/generated/Functions.h"
#include "torch/csrc/autograd/python_cpp_function.h"
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>

// NOTE: See [Sharded File] comment in VariableType

namespace torch::autograd::generated {

template<typename C>
static void addClass(PyObject* module, PyTypeObject& type, const char* name,
  PyGetSetDef* function_properties=NULL, PyMethodDef* function_methods=NULL)
{
  _initFunctionPyTypeObject(type, name, function_properties, function_methods);
  Py_INCREF(&type);
  PyModule_AddObject(module, name, (PyObject*)&type);
  registerCppFunction(typeid(C), &type);
}

PyObject* THPAddcmulBackward0_tensor1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddcmulBackward0*>(self->cdata.get())->tensor1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddcmulBackward0_tensor1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddcmulBackward0*>(self->cdata.get())->tensor1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddcmulBackward0_tensor2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddcmulBackward0*>(self->cdata.get())->tensor2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddcmulBackward0_tensor2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddcmulBackward0*>(self->cdata.get())->tensor2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddcmulBackward0_value_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddcmulBackward0*>(self->cdata.get())->value;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AddcmulBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_tensor1", (getter)THPAddcmulBackward0_tensor1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_tensor1", (getter)THPAddcmulBackward0_tensor1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_tensor2", (getter)THPAddcmulBackward0_tensor2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_tensor2", (getter)THPAddcmulBackward0_tensor2_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_value", (getter)THPAddcmulBackward0_value_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAddmmBackward0_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddmmBackward0*>(self->cdata.get())->alpha;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddmmBackward0_beta_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddmmBackward0*>(self->cdata.get())->beta;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddmmBackward0_mat1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddmmBackward0*>(self->cdata.get())->mat1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddmmBackward0_mat1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddmmBackward0*>(self->cdata.get())->mat1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddmmBackward0_mat1_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddmmBackward0*>(self->cdata.get())->mat1_sym_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddmmBackward0_mat1_sym_strides_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddmmBackward0*>(self->cdata.get())->mat1_sym_strides;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddmmBackward0_mat2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddmmBackward0*>(self->cdata.get())->mat2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddmmBackward0_mat2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddmmBackward0*>(self->cdata.get())->mat2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddmmBackward0_mat2_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddmmBackward0*>(self->cdata.get())->mat2_sym_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddmmBackward0_mat2_sym_strides_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddmmBackward0*>(self->cdata.get())->mat2_sym_strides;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AddmmBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_alpha", (getter)THPAddmmBackward0_alpha_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_beta", (getter)THPAddmmBackward0_beta_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mat1", (getter)THPAddmmBackward0_mat1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_mat1", (getter)THPAddmmBackward0_mat1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mat1_sym_sizes", (getter)THPAddmmBackward0_mat1_sym_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mat1_sym_strides", (getter)THPAddmmBackward0_mat1_sym_strides_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mat2", (getter)THPAddmmBackward0_mat2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_mat2", (getter)THPAddmmBackward0_mat2_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mat2_sym_sizes", (getter)THPAddmmBackward0_mat2_sym_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mat2_sym_strides", (getter)THPAddmmBackward0_mat2_sym_strides_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSparseAddmmBackward0_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SparseAddmmBackward0*>(self->cdata.get())->alpha;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseAddmmBackward0_beta_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SparseAddmmBackward0*>(self->cdata.get())->beta;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseAddmmBackward0_mat1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseAddmmBackward0*>(self->cdata.get())->mat1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseAddmmBackward0_mat1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseAddmmBackward0*>(self->cdata.get())->mat1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseAddmmBackward0_mat2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseAddmmBackward0*>(self->cdata.get())->mat2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseAddmmBackward0_mat2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseAddmmBackward0*>(self->cdata.get())->mat2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseAddmmBackward0_mat2_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SparseAddmmBackward0*>(self->cdata.get())->mat2_sym_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseAddmmBackward0_mat2_sym_strides_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SparseAddmmBackward0*>(self->cdata.get())->mat2_sym_strides;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SparseAddmmBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_alpha", (getter)THPSparseAddmmBackward0_alpha_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_beta", (getter)THPSparseAddmmBackward0_beta_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mat1", (getter)THPSparseAddmmBackward0_mat1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_mat1", (getter)THPSparseAddmmBackward0_mat1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mat2", (getter)THPSparseAddmmBackward0_mat2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_mat2", (getter)THPSparseAddmmBackward0_mat2_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mat2_sym_sizes", (getter)THPSparseAddmmBackward0_mat2_sym_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mat2_sym_strides", (getter)THPSparseAddmmBackward0_mat2_sym_strides_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAcoshBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AcoshBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAcoshBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AcoshBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AcoshBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPAcoshBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPAcoshBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef AtanhBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPAsStridedBackward0_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AsStridedBackward0*>(self->cdata.get())->size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAsStridedBackward0_storage_offset_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<AsStridedBackward0*>(self->cdata.get())->storage_offset;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAsStridedBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AsStridedBackward0*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AsStridedBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_size", (getter)THPAsStridedBackward0_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_storage_offset", (getter)THPAsStridedBackward0_storage_offset_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPAsStridedBackward0_stride_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCatBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CatBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CatBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPCatBackward0_dim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCholeskyBackward0_upper_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CholeskyBackward0*>(self->cdata.get())->upper;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCholeskyBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CholeskyBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCholeskyBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CholeskyBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CholeskyBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_upper", (getter)THPCholeskyBackward0_upper_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPCholeskyBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPCholeskyBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef ToCopyBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef ConjBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef NegViewBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef ConjPhysicalBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPCosBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CosBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCosBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CosBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CosBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPCosBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPCosBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCumsumBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CumsumBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CumsumBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPCumsumBackward0_dim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCtcLossBackward0_blank_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CtcLossBackward0*>(self->cdata.get())->blank;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCtcLossBackward0_input_lengths_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CtcLossBackward0*>(self->cdata.get())->input_lengths;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCtcLossBackward0_log_probs_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CtcLossBackward0*>(self->cdata.get())->log_probs_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCtcLossBackward0_log_probs_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CtcLossBackward0*>(self->cdata.get())->log_probs_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCtcLossBackward0_target_lengths_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CtcLossBackward0*>(self->cdata.get())->target_lengths;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCtcLossBackward0_targets_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CtcLossBackward0*>(self->cdata.get())->targets_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCtcLossBackward0_targets_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CtcLossBackward0*>(self->cdata.get())->targets_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCtcLossBackward0_zero_infinity_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CtcLossBackward0*>(self->cdata.get())->zero_infinity;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCtcLossBackward0_result0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CtcLossBackward0*>(self->cdata.get())->result0_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCtcLossBackward0_result0_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CtcLossBackward0*>(self->cdata.get())->result0_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCtcLossBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CtcLossBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCtcLossBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CtcLossBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CtcLossBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_blank", (getter)THPCtcLossBackward0_blank_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input_lengths", (getter)THPCtcLossBackward0_input_lengths_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_log_probs", (getter)THPCtcLossBackward0_log_probs_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_log_probs", (getter)THPCtcLossBackward0_log_probs_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target_lengths", (getter)THPCtcLossBackward0_target_lengths_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_targets", (getter)THPCtcLossBackward0_targets_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_targets", (getter)THPCtcLossBackward0_targets_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_zero_infinity", (getter)THPCtcLossBackward0_zero_infinity_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result0", (getter)THPCtcLossBackward0_result0_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result0", (getter)THPCtcLossBackward0_result0_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPCtcLossBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPCtcLossBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCtcLossBackward1_blank_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CtcLossBackward1*>(self->cdata.get())->blank;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCtcLossBackward1_input_lengths_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CtcLossBackward1*>(self->cdata.get())->input_lengths_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCtcLossBackward1_input_lengths_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CtcLossBackward1*>(self->cdata.get())->input_lengths_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCtcLossBackward1_log_probs_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CtcLossBackward1*>(self->cdata.get())->log_probs_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCtcLossBackward1_log_probs_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CtcLossBackward1*>(self->cdata.get())->log_probs_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCtcLossBackward1_target_lengths_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CtcLossBackward1*>(self->cdata.get())->target_lengths_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCtcLossBackward1_target_lengths_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CtcLossBackward1*>(self->cdata.get())->target_lengths_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCtcLossBackward1_targets_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CtcLossBackward1*>(self->cdata.get())->targets_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCtcLossBackward1_targets_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CtcLossBackward1*>(self->cdata.get())->targets_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCtcLossBackward1_zero_infinity_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CtcLossBackward1*>(self->cdata.get())->zero_infinity;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCtcLossBackward1_result0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CtcLossBackward1*>(self->cdata.get())->result0_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCtcLossBackward1_result0_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CtcLossBackward1*>(self->cdata.get())->result0_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCtcLossBackward1_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CtcLossBackward1*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCtcLossBackward1_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CtcLossBackward1*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CtcLossBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_blank", (getter)THPCtcLossBackward1_blank_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input_lengths", (getter)THPCtcLossBackward1_input_lengths_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input_lengths", (getter)THPCtcLossBackward1_input_lengths_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_log_probs", (getter)THPCtcLossBackward1_log_probs_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_log_probs", (getter)THPCtcLossBackward1_log_probs_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target_lengths", (getter)THPCtcLossBackward1_target_lengths_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_target_lengths", (getter)THPCtcLossBackward1_target_lengths_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_targets", (getter)THPCtcLossBackward1_targets_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_targets", (getter)THPCtcLossBackward1_targets_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_zero_infinity", (getter)THPCtcLossBackward1_zero_infinity_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result0", (getter)THPCtcLossBackward1_result0_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result0", (getter)THPCtcLossBackward1_result0_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPCtcLossBackward1_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPCtcLossBackward1_result1_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPVdotBackward0_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<VdotBackward0*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPVdotBackward0_other_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<VdotBackward0*>(self->cdata.get())->other_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPVdotBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<VdotBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPVdotBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<VdotBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef VdotBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPVdotBackward0_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_other", (getter)THPVdotBackward0_other_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPVdotBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPVdotBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPFusedDropoutBackward0_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FusedDropoutBackward0*>(self->cdata.get())->p;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPFusedDropoutBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FusedDropoutBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFusedDropoutBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FusedDropoutBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FusedDropoutBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_p", (getter)THPFusedDropoutBackward0_p_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPFusedDropoutBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPFusedDropoutBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPExpm1Backward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<Expm1Backward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPExpm1Backward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<Expm1Backward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef Expm1Backward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_result", (getter)THPExpm1Backward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPExpm1Backward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef ExponentialBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPFakeQuantizePerChannelAffineCachemaskBackward0_mask_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FakeQuantizePerChannelAffineCachemaskBackward0*>(self->cdata.get())->mask_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFakeQuantizePerChannelAffineCachemaskBackward0_mask_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FakeQuantizePerChannelAffineCachemaskBackward0*>(self->cdata.get())->mask_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FakeQuantizePerChannelAffineCachemaskBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_mask", (getter)THPFakeQuantizePerChannelAffineCachemaskBackward0_mask_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_mask", (getter)THPFakeQuantizePerChannelAffineCachemaskBackward0_mask_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPFusedMovingAvgObsFqHelperBackward0_mask_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FusedMovingAvgObsFqHelperBackward0*>(self->cdata.get())->mask_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFusedMovingAvgObsFqHelperBackward0_mask_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FusedMovingAvgObsFqHelperBackward0*>(self->cdata.get())->mask_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FusedMovingAvgObsFqHelperBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_mask", (getter)THPFusedMovingAvgObsFqHelperBackward0_mask_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_mask", (getter)THPFusedMovingAvgObsFqHelperBackward0_mask_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef FillBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef FillBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef GeometricBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPSpecialI1EBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialI1EBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialI1EBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialI1EBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialI1EBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialI1EBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialI1EBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialI1EBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SpecialI1EBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSpecialI1EBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSpecialI1EBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPSpecialI1EBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPSpecialI1EBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPIgammaBackward0_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IgammaBackward0*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPIgammaBackward0_other_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IgammaBackward0*>(self->cdata.get())->other_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPIgammaBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IgammaBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPIgammaBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IgammaBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef IgammaBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPIgammaBackward0_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_other", (getter)THPIgammaBackward0_other_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPIgammaBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPIgammaBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPIndexCopyBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<IndexCopyBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexCopyBackward0_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IndexCopyBackward0*>(self->cdata.get())->index_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexCopyBackward0_index_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IndexCopyBackward0*>(self->cdata.get())->index_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexCopyBackward0_source_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IndexCopyBackward0*>(self->cdata.get())->source_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexCopyBackward0_source_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IndexCopyBackward0*>(self->cdata.get())->source_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexCopyBackward0_source_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<IndexCopyBackward0*>(self->cdata.get())->source_dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef IndexCopyBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPIndexCopyBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_index", (getter)THPIndexCopyBackward0_index_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_index", (getter)THPIndexCopyBackward0_index_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_source", (getter)THPIndexCopyBackward0_source_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_source", (getter)THPIndexCopyBackward0_source_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_source_dim", (getter)THPIndexCopyBackward0_source_dim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUnsafeIndexPutBackward0_accumulate_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnsafeIndexPutBackward0*>(self->cdata.get())->accumulate;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPUnsafeIndexPutBackward0_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<UnsafeIndexPutBackward0*>(self->cdata.get());
  const auto& prop = node->indices_;
  if (node->indices_released_) {
    PyErr_SetString(PyExc_RuntimeError, ERR_BACKWARD_TWICE);
    return nullptr;
  }
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i: c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, THPVariable_Wrap(prop[i].unpack(self->cdata)));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUnsafeIndexPutBackward0_indices_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<UnsafeIndexPutBackward0*>(self->cdata.get());
  const auto& prop = node->indices_;
  if (node->indices_released_) {
    PyErr_SetString(PyExc_RuntimeError, ERR_BACKWARD_TWICE);
    return nullptr;
  }
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    pybind11::object obj = pybind11::cast(prop[i], pybind11::return_value_policy::reference);
    PyTuple_SetItem(tup, (Py_ssize_t) i, obj.release().ptr());
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UnsafeIndexPutBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_accumulate", (getter)THPUnsafeIndexPutBackward0_accumulate_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_indices", (getter)THPUnsafeIndexPutBackward0_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_indices", (getter)THPUnsafeIndexPutBackward0_indices_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPIndexSelectBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<IndexSelectBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexSelectBackward0_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IndexSelectBackward0*>(self->cdata.get())->index_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexSelectBackward0_index_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IndexSelectBackward0*>(self->cdata.get())->index_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexSelectBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<IndexSelectBackward0*>(self->cdata.get())->self_sym_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef IndexSelectBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPIndexSelectBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_index", (getter)THPIndexSelectBackward0_index_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_index", (getter)THPIndexSelectBackward0_index_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPIndexSelectBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef LeBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef LeBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPLgammaBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LgammaBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLgammaBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LgammaBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LgammaBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPLgammaBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPLgammaBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLog10Backward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<Log10Backward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLog10Backward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<Log10Backward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef Log10Backward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPLog10Backward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPLog10Backward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLog1PBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<Log1PBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLog1PBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<Log1PBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef Log1PBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPLog1PBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPLog1PBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLogaddexp2Backward0_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<Logaddexp2Backward0*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogaddexp2Backward0_other_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<Logaddexp2Backward0*>(self->cdata.get())->other_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogaddexp2Backward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<Logaddexp2Backward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogaddexp2Backward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<Logaddexp2Backward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef Logaddexp2Backward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPLogaddexp2Backward0_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_other", (getter)THPLogaddexp2Backward0_other_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPLogaddexp2Backward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPLogaddexp2Backward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLogsumexpBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LogsumexpBackward0*>(self->cdata.get())->dim;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogsumexpBackward0_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LogsumexpBackward0*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogsumexpBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogsumexpBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogsumexpBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogsumexpBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogsumexpBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogsumexpBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogsumexpBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogsumexpBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LogsumexpBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPLogsumexpBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPLogsumexpBackward0_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPLogsumexpBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPLogsumexpBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPLogsumexpBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPLogsumexpBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLinalgLstsqBackward0_b_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgLstsqBackward0*>(self->cdata.get())->b_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgLstsqBackward0_b_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgLstsqBackward0*>(self->cdata.get())->b_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgLstsqBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgLstsqBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgLstsqBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgLstsqBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgLstsqBackward0_solution_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgLstsqBackward0*>(self->cdata.get())->solution_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgLstsqBackward0_solution_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgLstsqBackward0*>(self->cdata.get())->solution_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LinalgLstsqBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_b", (getter)THPLinalgLstsqBackward0_b_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_b", (getter)THPLinalgLstsqBackward0_b_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPLinalgLstsqBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPLinalgLstsqBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_solution", (getter)THPLinalgLstsqBackward0_solution_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_solution", (getter)THPLinalgLstsqBackward0_solution_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLinalgLuBackward0_pivot_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LinalgLuBackward0*>(self->cdata.get())->pivot;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgLuBackward0_L_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgLuBackward0*>(self->cdata.get())->L_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgLuBackward0_L_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgLuBackward0*>(self->cdata.get())->L_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgLuBackward0_P_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgLuBackward0*>(self->cdata.get())->P_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgLuBackward0_P_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgLuBackward0*>(self->cdata.get())->P_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgLuBackward0_U_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgLuBackward0*>(self->cdata.get())->U_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgLuBackward0_U_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgLuBackward0*>(self->cdata.get())->U_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LinalgLuBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_pivot", (getter)THPLinalgLuBackward0_pivot_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_L", (getter)THPLinalgLuBackward0_L_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_L", (getter)THPLinalgLuBackward0_L_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_P", (getter)THPLinalgLuBackward0_P_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_P", (getter)THPLinalgLuBackward0_P_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_U", (getter)THPLinalgLuBackward0_U_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_U", (getter)THPLinalgLuBackward0_U_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLuUnpackBackward0_LU_data_sym_argsize_minus_1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LuUnpackBackward0*>(self->cdata.get())->LU_data_sym_argsize_minus_1;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPLuUnpackBackward0_LU_data_sym_argsize_minus_2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LuUnpackBackward0*>(self->cdata.get())->LU_data_sym_argsize_minus_2;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LuUnpackBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_LU_data_sym_argsize_minus_1", (getter)THPLuUnpackBackward0_LU_data_sym_argsize_minus_1_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_LU_data_sym_argsize_minus_2", (getter)THPLuUnpackBackward0_LU_data_sym_argsize_minus_2_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLinalgMatrixExpBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgMatrixExpBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgMatrixExpBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgMatrixExpBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LinalgMatrixExpBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPLinalgMatrixExpBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPLinalgMatrixExpBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMaxBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxBackward0_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxBackward0*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxBackward0*>(self->cdata.get())->self_sym_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxBackward0_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxBackward0*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxBackward0_indices_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxBackward0*>(self->cdata.get())->indices_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MaxBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPMaxBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPMaxBackward0_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPMaxBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_indices", (getter)THPMaxBackward0_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_indices", (getter)THPMaxBackward0_indices_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMaxBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxBackward1*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxBackward1_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxBackward1*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxBackward1_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxBackward1*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxBackward1_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxBackward1*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MaxBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMaxBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPMaxBackward1_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPMaxBackward1_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPMaxBackward1_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPModeBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ModeBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPModeBackward0_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ModeBackward0*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPModeBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ModeBackward0*>(self->cdata.get())->self_sym_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModeBackward0_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ModeBackward0*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPModeBackward0_indices_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ModeBackward0*>(self->cdata.get())->indices_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ModeBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPModeBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPModeBackward0_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPModeBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_indices", (getter)THPModeBackward0_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_indices", (getter)THPModeBackward0_indices_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMulBackward0_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MulBackward0*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMulBackward0_other_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MulBackward0*>(self->cdata.get())->other_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMulBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MulBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMulBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MulBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MulBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPMulBackward0_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_other", (getter)THPMulBackward0_other_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPMulBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPMulBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMulBackward1_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MulBackward1*>(self->cdata.get())->other;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MulBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPMulBackward1_other_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMvlgammaBackward0_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MvlgammaBackward0*>(self->cdata.get())->p;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMvlgammaBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MvlgammaBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMvlgammaBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MvlgammaBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MvlgammaBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_p", (getter)THPMvlgammaBackward0_p_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPMvlgammaBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPMvlgammaBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef NegBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef NormalBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPPowBackward0_exponent_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<PowBackward0*>(self->cdata.get())->exponent;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPPowBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PowBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPPowBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PowBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef PowBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_exponent", (getter)THPPowBackward0_exponent_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPPowBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPPowBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPPowBackward1_exponent_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PowBackward1*>(self->cdata.get())->exponent_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPPowBackward1_exponent_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PowBackward1*>(self->cdata.get())->exponent_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPPowBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PowBackward1*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPPowBackward1_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PowBackward1*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPPowBackward1_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PowBackward1*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPPowBackward1_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PowBackward1*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef PowBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_exponent", (getter)THPPowBackward1_exponent_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_exponent", (getter)THPPowBackward1_exponent_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPPowBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPPowBackward1_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPPowBackward1_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPPowBackward1_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPPowBackward2_exponent_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PowBackward2*>(self->cdata.get())->exponent_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPPowBackward2_exponent_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PowBackward2*>(self->cdata.get())->exponent_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPPowBackward2_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<PowBackward2*>(self->cdata.get())->self;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPPowBackward2_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PowBackward2*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPPowBackward2_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PowBackward2*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef PowBackward2_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_exponent", (getter)THPPowBackward2_exponent_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_exponent", (getter)THPPowBackward2_exponent_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPPowBackward2_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPPowBackward2_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPPowBackward2_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPReciprocalBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ReciprocalBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPReciprocalBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ReciprocalBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ReciprocalBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_result", (getter)THPReciprocalBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPReciprocalBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSpecialEntrBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialEntrBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialEntrBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialEntrBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SpecialEntrBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSpecialEntrBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSpecialEntrBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSpecialNdtriBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialNdtriBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialNdtriBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialNdtriBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SpecialNdtriBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_result", (getter)THPSpecialNdtriBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPSpecialNdtriBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSgnBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SgnBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSgnBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SgnBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSgnBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SgnBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSgnBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SgnBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SgnBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSgnBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSgnBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPSgnBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPSgnBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSinBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SinBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSinBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SinBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SinBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSinBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSinBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPDiagonalScatterBackward0_dim1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<DiagonalScatterBackward0*>(self->cdata.get())->dim1;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPDiagonalScatterBackward0_dim2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<DiagonalScatterBackward0*>(self->cdata.get())->dim2;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPDiagonalScatterBackward0_offset_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<DiagonalScatterBackward0*>(self->cdata.get())->offset;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef DiagonalScatterBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim1", (getter)THPDiagonalScatterBackward0_dim1_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim2", (getter)THPDiagonalScatterBackward0_dim2_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_offset", (getter)THPDiagonalScatterBackward0_offset_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAsStridedScatterBackward0_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AsStridedScatterBackward0*>(self->cdata.get())->size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAsStridedScatterBackward0_storage_offset_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<AsStridedScatterBackward0*>(self->cdata.get())->storage_offset;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAsStridedScatterBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AsStridedScatterBackward0*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AsStridedScatterBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_size", (getter)THPAsStridedScatterBackward0_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_storage_offset", (getter)THPAsStridedScatterBackward0_storage_offset_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPAsStridedScatterBackward0_stride_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSplitBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SplitBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSplitBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SplitBackward0*>(self->cdata.get())->self_sym_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSplitBackward0_split_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SplitBackward0*>(self->cdata.get())->split_size;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SplitBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSplitBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPSplitBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_split_size", (getter)THPSplitBackward0_split_size_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSqueezeBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SqueezeBackward0*>(self->cdata.get())->self_sym_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SqueezeBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sym_sizes", (getter)THPSqueezeBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSqueezeBackward1_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SqueezeBackward1*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSqueezeBackward1_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SqueezeBackward1*>(self->cdata.get())->self_sym_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SqueezeBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSqueezeBackward1_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPSqueezeBackward1_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSqueezeBackwardAutogradNestedTensor0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SqueezeBackwardAutogradNestedTensor0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SqueezeBackwardAutogradNestedTensor0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSqueezeBackwardAutogradNestedTensor0_dim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSqueezeBackward2_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SqueezeBackward2*>(self->cdata.get())->dim;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSqueezeBackward2_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SqueezeBackward2*>(self->cdata.get())->self_sym_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SqueezeBackward2_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSqueezeBackward2_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPSqueezeBackward2_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSqueezeBackwardAutogradNestedTensor1_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SqueezeBackwardAutogradNestedTensor1*>(self->cdata.get())->dim;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSqueezeBackwardAutogradNestedTensor1_self_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SqueezeBackwardAutogradNestedTensor1*>(self->cdata.get())->self_dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SqueezeBackwardAutogradNestedTensor1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSqueezeBackwardAutogradNestedTensor1_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_dim", (getter)THPSqueezeBackwardAutogradNestedTensor1_self_dim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLinalgEighBackward0_eigenvalues_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgEighBackward0*>(self->cdata.get())->eigenvalues_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgEighBackward0_eigenvalues_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgEighBackward0*>(self->cdata.get())->eigenvalues_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgEighBackward0_eigenvectors_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgEighBackward0*>(self->cdata.get())->eigenvectors_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgEighBackward0_eigenvectors_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgEighBackward0*>(self->cdata.get())->eigenvectors_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LinalgEighBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_eigenvalues", (getter)THPLinalgEighBackward0_eigenvalues_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_eigenvalues", (getter)THPLinalgEighBackward0_eigenvalues_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_eigenvectors", (getter)THPLinalgEighBackward0_eigenvectors_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_eigenvectors", (getter)THPLinalgEighBackward0_eigenvectors_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPToMkldnnBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ToMkldnnBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPToMkldnnBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ToMkldnnBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ToMkldnnBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPToMkldnnBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPToMkldnnBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef UniqueBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef LiftBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef LiftFreshBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPWhereBackward0_condition_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<WhereBackward0*>(self->cdata.get())->condition_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPWhereBackward0_condition_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<WhereBackward0*>(self->cdata.get())->condition_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef WhereBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_condition", (getter)THPWhereBackward0_condition_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_condition", (getter)THPWhereBackward0_condition_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPWeightNormInterfaceBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<WeightNormInterfaceBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPWeightNormInterfaceBackward0_g_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<WeightNormInterfaceBackward0*>(self->cdata.get())->g_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPWeightNormInterfaceBackward0_g_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<WeightNormInterfaceBackward0*>(self->cdata.get())->g_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPWeightNormInterfaceBackward0_v_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<WeightNormInterfaceBackward0*>(self->cdata.get())->v_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPWeightNormInterfaceBackward0_v_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<WeightNormInterfaceBackward0*>(self->cdata.get())->v_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPWeightNormInterfaceBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<WeightNormInterfaceBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPWeightNormInterfaceBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<WeightNormInterfaceBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef WeightNormInterfaceBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPWeightNormInterfaceBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_g", (getter)THPWeightNormInterfaceBackward0_g_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_g", (getter)THPWeightNormInterfaceBackward0_g_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_v", (getter)THPWeightNormInterfaceBackward0_v_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_v", (getter)THPWeightNormInterfaceBackward0_v_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPWeightNormInterfaceBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPWeightNormInterfaceBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef ZeroBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPSparseMaskBackward0_mask_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseMaskBackward0*>(self->cdata.get())->mask_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseMaskBackward0_mask_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseMaskBackward0*>(self->cdata.get())->mask_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SparseMaskBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_mask", (getter)THPSparseMaskBackward0_mask_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_mask", (getter)THPSparseMaskBackward0_mask_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSparseSumBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SparseSumBackward0*>(self->cdata.get())->dim;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseSumBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseSumBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseSumBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseSumBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SparseSumBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSparseSumBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSparseSumBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSparseSumBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPStandardGammaBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<StandardGammaBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPStandardGammaBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<StandardGammaBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPStandardGammaBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<StandardGammaBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPStandardGammaBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<StandardGammaBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef StandardGammaBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPStandardGammaBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPStandardGammaBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPStandardGammaBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPStandardGammaBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPValuesBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ValuesBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPValuesBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ValuesBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ValuesBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPValuesBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPValuesBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPValuesBackwardAutogradNestedTensor0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ValuesBackwardAutogradNestedTensor0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPValuesBackwardAutogradNestedTensor0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ValuesBackwardAutogradNestedTensor0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ValuesBackwardAutogradNestedTensor0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPValuesBackwardAutogradNestedTensor0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPValuesBackwardAutogradNestedTensor0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPEmbeddingDenseBackwardBackward0_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EmbeddingDenseBackwardBackward0*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingDenseBackwardBackward0_indices_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EmbeddingDenseBackwardBackward0*>(self->cdata.get())->indices_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingDenseBackwardBackward0_padding_idx_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EmbeddingDenseBackwardBackward0*>(self->cdata.get())->padding_idx;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef EmbeddingDenseBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_indices", (getter)THPEmbeddingDenseBackwardBackward0_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_indices", (getter)THPEmbeddingDenseBackwardBackward0_indices_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding_idx", (getter)THPEmbeddingDenseBackwardBackward0_padding_idx_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPEmbeddingBagBackward0_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EmbeddingBagBackward0*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBagBackward0_indices_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EmbeddingBagBackward0*>(self->cdata.get())->indices_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBagBackward0_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EmbeddingBagBackward0*>(self->cdata.get())->mode;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBagBackward0_offsets_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EmbeddingBagBackward0*>(self->cdata.get())->offsets_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBagBackward0_offsets_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EmbeddingBagBackward0*>(self->cdata.get())->offsets_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBagBackward0_padding_idx_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EmbeddingBagBackward0*>(self->cdata.get())->padding_idx;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBagBackward0_per_sample_weights_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EmbeddingBagBackward0*>(self->cdata.get())->per_sample_weights_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBagBackward0_per_sample_weights_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EmbeddingBagBackward0*>(self->cdata.get())->per_sample_weights_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBagBackward0_scale_grad_by_freq_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EmbeddingBagBackward0*>(self->cdata.get())->scale_grad_by_freq;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBagBackward0_sparse_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EmbeddingBagBackward0*>(self->cdata.get())->sparse;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBagBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EmbeddingBagBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBagBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EmbeddingBagBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBagBackward0_weight_sym_argsize_0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EmbeddingBagBackward0*>(self->cdata.get())->weight_sym_argsize_0;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBagBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EmbeddingBagBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBagBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EmbeddingBagBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBagBackward0_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EmbeddingBagBackward0*>(self->cdata.get())->result2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBagBackward0_result2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EmbeddingBagBackward0*>(self->cdata.get())->result2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBagBackward0_result3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EmbeddingBagBackward0*>(self->cdata.get())->result3_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBagBackward0_result3_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EmbeddingBagBackward0*>(self->cdata.get())->result3_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef EmbeddingBagBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_indices", (getter)THPEmbeddingBagBackward0_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_indices", (getter)THPEmbeddingBagBackward0_indices_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mode", (getter)THPEmbeddingBagBackward0_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_offsets", (getter)THPEmbeddingBagBackward0_offsets_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_offsets", (getter)THPEmbeddingBagBackward0_offsets_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding_idx", (getter)THPEmbeddingBagBackward0_padding_idx_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_per_sample_weights", (getter)THPEmbeddingBagBackward0_per_sample_weights_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_per_sample_weights", (getter)THPEmbeddingBagBackward0_per_sample_weights_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale_grad_by_freq", (getter)THPEmbeddingBagBackward0_scale_grad_by_freq_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_sparse", (getter)THPEmbeddingBagBackward0_sparse_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPEmbeddingBagBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPEmbeddingBagBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight_sym_argsize_0", (getter)THPEmbeddingBagBackward0_weight_sym_argsize_0_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPEmbeddingBagBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPEmbeddingBagBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPEmbeddingBagBackward0_result2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result2", (getter)THPEmbeddingBagBackward0_result2_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result3", (getter)THPEmbeddingBagBackward0_result3_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result3", (getter)THPEmbeddingBagBackward0_result3_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPGeluBackwardBackward0_approximate_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<GeluBackwardBackward0*>(self->cdata.get())->approximate;
  return PyUnicode_FromStringAndSize(prop.data(), prop.size());
  END_HANDLE_TH_ERRORS
}

PyObject* THPGeluBackwardBackward0_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GeluBackwardBackward0*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPGeluBackwardBackward0_grad_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GeluBackwardBackward0*>(self->cdata.get())->grad_output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPGeluBackwardBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GeluBackwardBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPGeluBackwardBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GeluBackwardBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef GeluBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_approximate", (getter)THPGeluBackwardBackward0_approximate_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output", (getter)THPGeluBackwardBackward0_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_grad_output", (getter)THPGeluBackwardBackward0_grad_output_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPGeluBackwardBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPGeluBackwardBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLogSoftmaxBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LogSoftmaxBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogSoftmaxBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogSoftmaxBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogSoftmaxBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogSoftmaxBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LogSoftmaxBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPLogSoftmaxBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPLogSoftmaxBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPLogSoftmaxBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPPreluKernelBackwardBackward0_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PreluKernelBackwardBackward0*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPPreluKernelBackwardBackward0_grad_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PreluKernelBackwardBackward0*>(self->cdata.get())->grad_output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPPreluKernelBackwardBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PreluKernelBackwardBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPPreluKernelBackwardBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PreluKernelBackwardBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPPreluKernelBackwardBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PreluKernelBackwardBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPPreluKernelBackwardBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PreluKernelBackwardBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef PreluKernelBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_output", (getter)THPPreluKernelBackwardBackward0_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_grad_output", (getter)THPPreluKernelBackwardBackward0_grad_output_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPPreluKernelBackwardBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPPreluKernelBackwardBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPPreluKernelBackwardBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPPreluKernelBackwardBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPReflectionPad1DBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ReflectionPad1DBackward0*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPReflectionPad1DBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ReflectionPad1DBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPReflectionPad1DBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ReflectionPad1DBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ReflectionPad1DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_padding", (getter)THPReflectionPad1DBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPReflectionPad1DBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPReflectionPad1DBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleBicubic2DAaBackward0_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBicubic2DAaBackward0*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBicubic2DAaBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBicubic2DAaBackward0*>(self->cdata.get())->output_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBicubic2DAaBackward0_scales_h_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleBicubic2DAaBackward0*>(self->cdata.get())->scales_h;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBicubic2DAaBackward0_scales_w_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleBicubic2DAaBackward0*>(self->cdata.get())->scales_w;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBicubic2DAaBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBicubic2DAaBackward0*>(self->cdata.get())->self_sym_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleBicubic2DAaBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_align_corners", (getter)THPUpsampleBicubic2DAaBackward0_align_corners_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPUpsampleBicubic2DAaBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_h", (getter)THPUpsampleBicubic2DAaBackward0_scales_h_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_w", (getter)THPUpsampleBicubic2DAaBackward0_scales_w_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPUpsampleBicubic2DAaBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleTrilinear3DBackward0_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleTrilinear3DBackward0*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleTrilinear3DBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleTrilinear3DBackward0*>(self->cdata.get())->output_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleTrilinear3DBackward0_scales_d_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleTrilinear3DBackward0*>(self->cdata.get())->scales_d;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleTrilinear3DBackward0_scales_h_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleTrilinear3DBackward0*>(self->cdata.get())->scales_h;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleTrilinear3DBackward0_scales_w_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleTrilinear3DBackward0*>(self->cdata.get())->scales_w;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleTrilinear3DBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleTrilinear3DBackward0*>(self->cdata.get())->self_sym_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleTrilinear3DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_align_corners", (getter)THPUpsampleTrilinear3DBackward0_align_corners_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPUpsampleTrilinear3DBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_d", (getter)THPUpsampleTrilinear3DBackward0_scales_d_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_h", (getter)THPUpsampleTrilinear3DBackward0_scales_h_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_w", (getter)THPUpsampleTrilinear3DBackward0_scales_w_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPUpsampleTrilinear3DBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleNearest3DBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleNearest3DBackward0*>(self->cdata.get())->output_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest3DBackward0_scales_d_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest3DBackward0*>(self->cdata.get())->scales_d;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest3DBackward0_scales_h_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest3DBackward0*>(self->cdata.get())->scales_h;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest3DBackward0_scales_w_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest3DBackward0*>(self->cdata.get())->scales_w;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest3DBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleNearest3DBackward0*>(self->cdata.get())->self_sym_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleNearest3DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_output_size", (getter)THPUpsampleNearest3DBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_d", (getter)THPUpsampleNearest3DBackward0_scales_d_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_h", (getter)THPUpsampleNearest3DBackward0_scales_h_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_w", (getter)THPUpsampleNearest3DBackward0_scales_w_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPUpsampleNearest3DBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPPixelUnshuffleBackward0_downscale_factor_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<PixelUnshuffleBackward0*>(self->cdata.get())->downscale_factor;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef PixelUnshuffleBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_downscale_factor", (getter)THPPixelUnshuffleBackward0_downscale_factor_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLinearBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinearBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinearBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinearBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinearBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinearBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinearBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinearBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LinearBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_input", (getter)THPLinearBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPLinearBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPLinearBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPLinearBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMaxPool2DWithIndicesBackward0_ceil_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxPool2DWithIndicesBackward0*>(self->cdata.get())->ceil_mode;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool2DWithIndicesBackward0_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxPool2DWithIndicesBackward0*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool2DWithIndicesBackward0_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxPool2DWithIndicesBackward0*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool2DWithIndicesBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxPool2DWithIndicesBackward0*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool2DWithIndicesBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxPool2DWithIndicesBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool2DWithIndicesBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxPool2DWithIndicesBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool2DWithIndicesBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxPool2DWithIndicesBackward0*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool2DWithIndicesBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxPool2DWithIndicesBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool2DWithIndicesBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxPool2DWithIndicesBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MaxPool2DWithIndicesBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_ceil_mode", (getter)THPMaxPool2DWithIndicesBackward0_ceil_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPMaxPool2DWithIndicesBackward0_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_kernel_size", (getter)THPMaxPool2DWithIndicesBackward0_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPMaxPool2DWithIndicesBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPMaxPool2DWithIndicesBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPMaxPool2DWithIndicesBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPMaxPool2DWithIndicesBackward0_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPMaxPool2DWithIndicesBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPMaxPool2DWithIndicesBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPConvolutionBackward0_bias_sym_sizes_opt_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<ConvolutionBackward0*>(self->cdata.get())->bias_sym_sizes_opt;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackward0_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionBackward0*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackward0_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionBackward0*>(self->cdata.get())->groups;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvolutionBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvolutionBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackward0_output_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionBackward0*>(self->cdata.get())->output_padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionBackward0*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionBackward0*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackward0_transposed_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionBackward0*>(self->cdata.get())->transposed;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvolutionBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvolutionBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ConvolutionBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_bias_sym_sizes_opt", (getter)THPConvolutionBackward0_bias_sym_sizes_opt_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPConvolutionBackward0_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_groups", (getter)THPConvolutionBackward0_groups_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPConvolutionBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPConvolutionBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_padding", (getter)THPConvolutionBackward0_output_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPConvolutionBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPConvolutionBackward0_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_transposed", (getter)THPConvolutionBackward0_transposed_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPConvolutionBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPConvolutionBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPConvolutionBackward1_bias_sym_sizes_opt_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<ConvolutionBackward1*>(self->cdata.get())->bias_sym_sizes_opt;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackward1_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionBackward1*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackward1_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionBackward1*>(self->cdata.get())->groups;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackward1_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvolutionBackward1*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackward1_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvolutionBackward1*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackward1_output_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionBackward1*>(self->cdata.get())->output_padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackward1_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionBackward1*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackward1_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionBackward1*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackward1_transposed_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionBackward1*>(self->cdata.get())->transposed;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackward1_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvolutionBackward1*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackward1_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvolutionBackward1*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ConvolutionBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_bias_sym_sizes_opt", (getter)THPConvolutionBackward1_bias_sym_sizes_opt_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPConvolutionBackward1_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_groups", (getter)THPConvolutionBackward1_groups_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPConvolutionBackward1_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPConvolutionBackward1_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_padding", (getter)THPConvolutionBackward1_output_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPConvolutionBackward1_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPConvolutionBackward1_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_transposed", (getter)THPConvolutionBackward1_transposed_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPConvolutionBackward1_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPConvolutionBackward1_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPConvolutionBackwardOverrideableBackward0_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionBackwardOverrideableBackward0*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackwardOverrideableBackward0_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvolutionBackwardOverrideableBackward0*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackwardOverrideableBackward0_grad_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvolutionBackwardOverrideableBackward0*>(self->cdata.get())->grad_output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackwardOverrideableBackward0_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionBackwardOverrideableBackward0*>(self->cdata.get())->groups;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackwardOverrideableBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvolutionBackwardOverrideableBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackwardOverrideableBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvolutionBackwardOverrideableBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackwardOverrideableBackward0_output_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionBackwardOverrideableBackward0*>(self->cdata.get())->output_padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackwardOverrideableBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionBackwardOverrideableBackward0*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackwardOverrideableBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionBackwardOverrideableBackward0*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackwardOverrideableBackward0_transposed_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionBackwardOverrideableBackward0*>(self->cdata.get())->transposed;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackwardOverrideableBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvolutionBackwardOverrideableBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackwardOverrideableBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvolutionBackwardOverrideableBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ConvolutionBackwardOverrideableBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dilation", (getter)THPConvolutionBackwardOverrideableBackward0_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output", (getter)THPConvolutionBackwardOverrideableBackward0_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_grad_output", (getter)THPConvolutionBackwardOverrideableBackward0_grad_output_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_groups", (getter)THPConvolutionBackwardOverrideableBackward0_groups_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPConvolutionBackwardOverrideableBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPConvolutionBackwardOverrideableBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_padding", (getter)THPConvolutionBackwardOverrideableBackward0_output_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPConvolutionBackwardOverrideableBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPConvolutionBackwardOverrideableBackward0_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_transposed", (getter)THPConvolutionBackwardOverrideableBackward0_transposed_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPConvolutionBackwardOverrideableBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPConvolutionBackwardOverrideableBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSlowConv2DBackward0_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConv2DBackward0*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConv2DBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConv2DBackward0*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConv2DBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConv2DBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConv2DBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConv2DBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConv2DBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConv2DBackward0*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConv2DBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConv2DBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConv2DBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConv2DBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SlowConv2DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_kernel_size", (getter)THPSlowConv2DBackward0_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPSlowConv2DBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSlowConv2DBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSlowConv2DBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPSlowConv2DBackward0_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPSlowConv2DBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPSlowConv2DBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPEluBackwardBackward0_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EluBackwardBackward0*>(self->cdata.get())->alpha;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPEluBackwardBackward0_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EluBackwardBackward0*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEluBackwardBackward0_grad_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EluBackwardBackward0*>(self->cdata.get())->grad_output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPEluBackwardBackward0_input_scale_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EluBackwardBackward0*>(self->cdata.get())->input_scale;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPEluBackwardBackward0_is_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EluBackwardBackward0*>(self->cdata.get())->is_result;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPEluBackwardBackward0_scale_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EluBackwardBackward0*>(self->cdata.get())->scale;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPEluBackwardBackward0_self_or_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EluBackwardBackward0*>(self->cdata.get())->self_or_result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEluBackwardBackward0_self_or_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EluBackwardBackward0*>(self->cdata.get())->self_or_result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef EluBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_alpha", (getter)THPEluBackwardBackward0_alpha_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output", (getter)THPEluBackwardBackward0_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_grad_output", (getter)THPEluBackwardBackward0_grad_output_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input_scale", (getter)THPEluBackwardBackward0_input_scale_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_is_result", (getter)THPEluBackwardBackward0_is_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale", (getter)THPEluBackwardBackward0_scale_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_or_result", (getter)THPEluBackwardBackward0_self_or_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self_or_result", (getter)THPEluBackwardBackward0_self_or_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPFractionalMaxPool3DBackwardBackward0_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FractionalMaxPool3DBackwardBackward0*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFractionalMaxPool3DBackwardBackward0_indices_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FractionalMaxPool3DBackwardBackward0*>(self->cdata.get())->indices_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FractionalMaxPool3DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_indices", (getter)THPFractionalMaxPool3DBackwardBackward0_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_indices", (getter)THPFractionalMaxPool3DBackwardBackward0_indices_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMseLossBackwardBackward0_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MseLossBackwardBackward0*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMseLossBackwardBackward0_grad_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MseLossBackwardBackward0*>(self->cdata.get())->grad_output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMseLossBackwardBackward0_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MseLossBackwardBackward0*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMseLossBackwardBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MseLossBackwardBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMseLossBackwardBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MseLossBackwardBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMseLossBackwardBackward0_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MseLossBackwardBackward0*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMseLossBackwardBackward0_target_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MseLossBackwardBackward0*>(self->cdata.get())->target_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MseLossBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_output", (getter)THPMseLossBackwardBackward0_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_grad_output", (getter)THPMseLossBackwardBackward0_grad_output_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPMseLossBackwardBackward0_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPMseLossBackwardBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPMseLossBackwardBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPMseLossBackwardBackward0_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_target", (getter)THPMseLossBackwardBackward0_target_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPRreluWithNoiseBackwardBackward0_lower_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RreluWithNoiseBackwardBackward0*>(self->cdata.get())->lower;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPRreluWithNoiseBackwardBackward0_noise_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RreluWithNoiseBackwardBackward0*>(self->cdata.get())->noise_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPRreluWithNoiseBackwardBackward0_noise_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RreluWithNoiseBackwardBackward0*>(self->cdata.get())->noise_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPRreluWithNoiseBackwardBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RreluWithNoiseBackwardBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPRreluWithNoiseBackwardBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RreluWithNoiseBackwardBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPRreluWithNoiseBackwardBackward0_training_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RreluWithNoiseBackwardBackward0*>(self->cdata.get())->training;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPRreluWithNoiseBackwardBackward0_upper_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RreluWithNoiseBackwardBackward0*>(self->cdata.get())->upper;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef RreluWithNoiseBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_lower", (getter)THPRreluWithNoiseBackwardBackward0_lower_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_noise", (getter)THPRreluWithNoiseBackwardBackward0_noise_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_noise", (getter)THPRreluWithNoiseBackwardBackward0_noise_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPRreluWithNoiseBackwardBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPRreluWithNoiseBackwardBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_training", (getter)THPRreluWithNoiseBackwardBackward0_training_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_upper", (getter)THPRreluWithNoiseBackwardBackward0_upper_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSoftplusBackwardBackward0_beta_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SoftplusBackwardBackward0*>(self->cdata.get())->beta;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftplusBackwardBackward0_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftplusBackwardBackward0*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftplusBackwardBackward0_grad_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftplusBackwardBackward0*>(self->cdata.get())->grad_output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftplusBackwardBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftplusBackwardBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftplusBackwardBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftplusBackwardBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftplusBackwardBackward0_threshold_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SoftplusBackwardBackward0*>(self->cdata.get())->threshold;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SoftplusBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_beta", (getter)THPSoftplusBackwardBackward0_beta_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output", (getter)THPSoftplusBackwardBackward0_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_grad_output", (getter)THPSoftplusBackwardBackward0_grad_output_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSoftplusBackwardBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSoftplusBackwardBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_threshold", (getter)THPSoftplusBackwardBackward0_threshold_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPThresholdBackwardBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThresholdBackwardBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThresholdBackwardBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThresholdBackwardBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPThresholdBackwardBackward0_threshold_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ThresholdBackwardBackward0*>(self->cdata.get())->threshold;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ThresholdBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPThresholdBackwardBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPThresholdBackwardBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_threshold", (getter)THPThresholdBackwardBackward0_threshold_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleBilinear2DAaBackwardBackward0_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBilinear2DAaBackwardBackward0*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBilinear2DAaBackwardBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBilinear2DAaBackwardBackward0*>(self->cdata.get())->output_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBilinear2DAaBackwardBackward0_scales_h_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleBilinear2DAaBackwardBackward0*>(self->cdata.get())->scales_h;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBilinear2DAaBackwardBackward0_scales_w_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleBilinear2DAaBackwardBackward0*>(self->cdata.get())->scales_w;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleBilinear2DAaBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_align_corners", (getter)THPUpsampleBilinear2DAaBackwardBackward0_align_corners_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPUpsampleBilinear2DAaBackwardBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_h", (getter)THPUpsampleBilinear2DAaBackwardBackward0_scales_h_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_w", (getter)THPUpsampleBilinear2DAaBackwardBackward0_scales_w_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSigmoidBackwardBackward0_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SigmoidBackwardBackward0*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSigmoidBackwardBackward0_grad_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SigmoidBackwardBackward0*>(self->cdata.get())->grad_output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSigmoidBackwardBackward0_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SigmoidBackwardBackward0*>(self->cdata.get())->output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSigmoidBackwardBackward0_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SigmoidBackwardBackward0*>(self->cdata.get())->output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SigmoidBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_output", (getter)THPSigmoidBackwardBackward0_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_grad_output", (getter)THPSigmoidBackwardBackward0_grad_output_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output", (getter)THPSigmoidBackwardBackward0_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_output", (getter)THPSigmoidBackwardBackward0_output_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPTanhBackwardBackward0_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TanhBackwardBackward0*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPTanhBackwardBackward0_grad_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TanhBackwardBackward0*>(self->cdata.get())->grad_output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPTanhBackwardBackward0_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TanhBackwardBackward0*>(self->cdata.get())->output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPTanhBackwardBackward0_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TanhBackwardBackward0*>(self->cdata.get())->output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef TanhBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_output", (getter)THPTanhBackwardBackward0_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_grad_output", (getter)THPTanhBackwardBackward0_grad_output_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output", (getter)THPTanhBackwardBackward0_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_output", (getter)THPTanhBackwardBackward0_output_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCudnnCtcLossBackward0_zero_infinity_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnCtcLossBackward0*>(self->cdata.get())->zero_infinity;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnCtcLossBackward0_result0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnCtcLossBackward0*>(self->cdata.get())->result0_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnCtcLossBackward0_result0_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnCtcLossBackward0*>(self->cdata.get())->result0_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnCtcLossBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnCtcLossBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnCtcLossBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnCtcLossBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CudnnCtcLossBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_zero_infinity", (getter)THPCudnnCtcLossBackward0_zero_infinity_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result0", (getter)THPCudnnCtcLossBackward0_result0_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result0", (getter)THPCudnnCtcLossBackward0_result0_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPCudnnCtcLossBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPCudnnCtcLossBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCudnnCtcLossBackward1_zero_infinity_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnCtcLossBackward1*>(self->cdata.get())->zero_infinity;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnCtcLossBackward1_result0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnCtcLossBackward1*>(self->cdata.get())->result0_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnCtcLossBackward1_result0_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnCtcLossBackward1*>(self->cdata.get())->result0_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnCtcLossBackward1_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnCtcLossBackward1*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnCtcLossBackward1_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnCtcLossBackward1*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CudnnCtcLossBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_zero_infinity", (getter)THPCudnnCtcLossBackward1_zero_infinity_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result0", (getter)THPCudnnCtcLossBackward1_result0_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result0", (getter)THPCudnnCtcLossBackward1_result0_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPCudnnCtcLossBackward1_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPCudnnCtcLossBackward1_result1_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCudnnConvolutionTransposeBackward0_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionTransposeBackward0*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionTransposeBackward0_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionTransposeBackward0*>(self->cdata.get())->groups;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionTransposeBackward0_output_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionTransposeBackward0*>(self->cdata.get())->output_padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionTransposeBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionTransposeBackward0*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionTransposeBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnConvolutionTransposeBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionTransposeBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnConvolutionTransposeBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionTransposeBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionTransposeBackward0*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionTransposeBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnConvolutionTransposeBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionTransposeBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnConvolutionTransposeBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CudnnConvolutionTransposeBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dilation", (getter)THPCudnnConvolutionTransposeBackward0_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_groups", (getter)THPCudnnConvolutionTransposeBackward0_groups_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_padding", (getter)THPCudnnConvolutionTransposeBackward0_output_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPCudnnConvolutionTransposeBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPCudnnConvolutionTransposeBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPCudnnConvolutionTransposeBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPCudnnConvolutionTransposeBackward0_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPCudnnConvolutionTransposeBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPCudnnConvolutionTransposeBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCudnnAffineGridGeneratorBackward0_C_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnAffineGridGeneratorBackward0*>(self->cdata.get())->C;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnAffineGridGeneratorBackward0_H_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnAffineGridGeneratorBackward0*>(self->cdata.get())->H;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnAffineGridGeneratorBackward0_N_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnAffineGridGeneratorBackward0*>(self->cdata.get())->N;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnAffineGridGeneratorBackward0_W_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnAffineGridGeneratorBackward0*>(self->cdata.get())->W;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CudnnAffineGridGeneratorBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_C", (getter)THPCudnnAffineGridGeneratorBackward0_C_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_H", (getter)THPCudnnAffineGridGeneratorBackward0_H_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_N", (getter)THPCudnnAffineGridGeneratorBackward0_N_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_W", (getter)THPCudnnAffineGridGeneratorBackward0_W_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCudnnBatchNormBackwardBackward0_epsilon_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnBatchNormBackwardBackward0*>(self->cdata.get())->epsilon;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackwardBackward0_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackwardBackward0*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackwardBackward0_grad_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackwardBackward0*>(self->cdata.get())->grad_output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackwardBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackwardBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackwardBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackwardBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackwardBackward0_reserveSpace_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackwardBackward0*>(self->cdata.get())->reserveSpace_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackwardBackward0_reserveSpace_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackwardBackward0*>(self->cdata.get())->reserveSpace_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackwardBackward0_running_mean_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackwardBackward0*>(self->cdata.get())->running_mean_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackwardBackward0_running_mean_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackwardBackward0*>(self->cdata.get())->running_mean_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackwardBackward0_running_var_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackwardBackward0*>(self->cdata.get())->running_var_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackwardBackward0_running_var_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackwardBackward0*>(self->cdata.get())->running_var_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackwardBackward0_save_mean_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackwardBackward0*>(self->cdata.get())->save_mean_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackwardBackward0_save_mean_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackwardBackward0*>(self->cdata.get())->save_mean_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackwardBackward0_save_var_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackwardBackward0*>(self->cdata.get())->save_var_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackwardBackward0_save_var_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackwardBackward0*>(self->cdata.get())->save_var_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackwardBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackwardBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackwardBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackwardBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CudnnBatchNormBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_epsilon", (getter)THPCudnnBatchNormBackwardBackward0_epsilon_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output", (getter)THPCudnnBatchNormBackwardBackward0_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_grad_output", (getter)THPCudnnBatchNormBackwardBackward0_grad_output_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPCudnnBatchNormBackwardBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPCudnnBatchNormBackwardBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reserveSpace", (getter)THPCudnnBatchNormBackwardBackward0_reserveSpace_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_reserveSpace", (getter)THPCudnnBatchNormBackwardBackward0_reserveSpace_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_mean", (getter)THPCudnnBatchNormBackwardBackward0_running_mean_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_running_mean", (getter)THPCudnnBatchNormBackwardBackward0_running_mean_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_var", (getter)THPCudnnBatchNormBackwardBackward0_running_var_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_running_var", (getter)THPCudnnBatchNormBackwardBackward0_running_var_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_save_mean", (getter)THPCudnnBatchNormBackwardBackward0_save_mean_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_save_mean", (getter)THPCudnnBatchNormBackwardBackward0_save_mean_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_save_var", (getter)THPCudnnBatchNormBackwardBackward0_save_var_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_save_var", (getter)THPCudnnBatchNormBackwardBackward0_save_var_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPCudnnBatchNormBackwardBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPCudnnBatchNormBackwardBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCudnnRnnBackward0_batch_first_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnRnnBackward0*>(self->cdata.get())->batch_first;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward0_batch_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnRnnBackward0*>(self->cdata.get())->batch_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward0_bidirectional_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnRnnBackward0*>(self->cdata.get())->bidirectional;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward0_cx_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnRnnBackward0*>(self->cdata.get())->cx_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward0_cx_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnRnnBackward0*>(self->cdata.get())->cx_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward0_dropout_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnRnnBackward0*>(self->cdata.get())->dropout;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward0_dropout_state_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnRnnBackward0*>(self->cdata.get())->dropout_state_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward0_dropout_state_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnRnnBackward0*>(self->cdata.get())->dropout_state_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward0_hidden_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnRnnBackward0*>(self->cdata.get())->hidden_size;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward0_hx_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnRnnBackward0*>(self->cdata.get())->hx_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward0_hx_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnRnnBackward0*>(self->cdata.get())->hx_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnRnnBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnRnnBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward0_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnRnnBackward0*>(self->cdata.get())->mode;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward0_num_layers_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnRnnBackward0*>(self->cdata.get())->num_layers;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward0_proj_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnRnnBackward0*>(self->cdata.get())->proj_size;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward0_train_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnRnnBackward0*>(self->cdata.get())->train;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<CudnnRnnBackward0*>(self->cdata.get());
  const auto& prop = node->weight_;
  if (node->weight_released_) {
    PyErr_SetString(PyExc_RuntimeError, ERR_BACKWARD_TWICE);
    return nullptr;
  }
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i: c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, THPVariable_Wrap(prop[i].unpack(self->cdata)));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<CudnnRnnBackward0*>(self->cdata.get());
  const auto& prop = node->weight_;
  if (node->weight_released_) {
    PyErr_SetString(PyExc_RuntimeError, ERR_BACKWARD_TWICE);
    return nullptr;
  }
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    pybind11::object obj = pybind11::cast(prop[i], pybind11::return_value_policy::reference);
    PyTuple_SetItem(tup, (Py_ssize_t) i, obj.release().ptr());
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward0_weight_stride0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnRnnBackward0*>(self->cdata.get())->weight_stride0;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward0_result0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnRnnBackward0*>(self->cdata.get())->result0_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward0_result0_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnRnnBackward0*>(self->cdata.get())->result0_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward0_result3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnRnnBackward0*>(self->cdata.get())->result3_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward0_result3_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnRnnBackward0*>(self->cdata.get())->result3_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward0_result4_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnRnnBackward0*>(self->cdata.get())->result4_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward0_result4_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnRnnBackward0*>(self->cdata.get())->result4_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CudnnRnnBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_batch_first", (getter)THPCudnnRnnBackward0_batch_first_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_batch_sizes", (getter)THPCudnnRnnBackward0_batch_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_bidirectional", (getter)THPCudnnRnnBackward0_bidirectional_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_cx", (getter)THPCudnnRnnBackward0_cx_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_cx", (getter)THPCudnnRnnBackward0_cx_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dropout", (getter)THPCudnnRnnBackward0_dropout_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dropout_state", (getter)THPCudnnRnnBackward0_dropout_state_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_dropout_state", (getter)THPCudnnRnnBackward0_dropout_state_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_hidden_size", (getter)THPCudnnRnnBackward0_hidden_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_hx", (getter)THPCudnnRnnBackward0_hx_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_hx", (getter)THPCudnnRnnBackward0_hx_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPCudnnRnnBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPCudnnRnnBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mode", (getter)THPCudnnRnnBackward0_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_num_layers", (getter)THPCudnnRnnBackward0_num_layers_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_proj_size", (getter)THPCudnnRnnBackward0_proj_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_train", (getter)THPCudnnRnnBackward0_train_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPCudnnRnnBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPCudnnRnnBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight_stride0", (getter)THPCudnnRnnBackward0_weight_stride0_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result0", (getter)THPCudnnRnnBackward0_result0_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result0", (getter)THPCudnnRnnBackward0_result0_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result3", (getter)THPCudnnRnnBackward0_result3_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result3", (getter)THPCudnnRnnBackward0_result3_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result4", (getter)THPCudnnRnnBackward0_result4_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result4", (getter)THPCudnnRnnBackward0_result4_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMiopenBatchNormBackwardBackward0_epsilon_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenBatchNormBackwardBackward0*>(self->cdata.get())->epsilon;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackwardBackward0_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackwardBackward0*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackwardBackward0_grad_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackwardBackward0*>(self->cdata.get())->grad_output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackwardBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackwardBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackwardBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackwardBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackwardBackward0_running_mean_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackwardBackward0*>(self->cdata.get())->running_mean_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackwardBackward0_running_mean_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackwardBackward0*>(self->cdata.get())->running_mean_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackwardBackward0_running_var_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackwardBackward0*>(self->cdata.get())->running_var_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackwardBackward0_running_var_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackwardBackward0*>(self->cdata.get())->running_var_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackwardBackward0_save_mean_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackwardBackward0*>(self->cdata.get())->save_mean_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackwardBackward0_save_mean_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackwardBackward0*>(self->cdata.get())->save_mean_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackwardBackward0_save_var_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackwardBackward0*>(self->cdata.get())->save_var_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackwardBackward0_save_var_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackwardBackward0*>(self->cdata.get())->save_var_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackwardBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackwardBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackwardBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackwardBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MiopenBatchNormBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_epsilon", (getter)THPMiopenBatchNormBackwardBackward0_epsilon_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output", (getter)THPMiopenBatchNormBackwardBackward0_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_grad_output", (getter)THPMiopenBatchNormBackwardBackward0_grad_output_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPMiopenBatchNormBackwardBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPMiopenBatchNormBackwardBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_mean", (getter)THPMiopenBatchNormBackwardBackward0_running_mean_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_running_mean", (getter)THPMiopenBatchNormBackwardBackward0_running_mean_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_var", (getter)THPMiopenBatchNormBackwardBackward0_running_var_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_running_var", (getter)THPMiopenBatchNormBackwardBackward0_running_var_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_save_mean", (getter)THPMiopenBatchNormBackwardBackward0_save_mean_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_save_mean", (getter)THPMiopenBatchNormBackwardBackward0_save_mean_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_save_var", (getter)THPMiopenBatchNormBackwardBackward0_save_var_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_save_var", (getter)THPMiopenBatchNormBackwardBackward0_save_var_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPMiopenBatchNormBackwardBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPMiopenBatchNormBackwardBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMkldnnConvolutionBackward0_bias_sym_sizes_opt_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<MkldnnConvolutionBackward0*>(self->cdata.get())->bias_sym_sizes_opt;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnConvolutionBackward0_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnConvolutionBackward0*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnConvolutionBackward0_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnConvolutionBackward0*>(self->cdata.get())->groups;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnConvolutionBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnConvolutionBackward0*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnConvolutionBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnConvolutionBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnConvolutionBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnConvolutionBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnConvolutionBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnConvolutionBackward0*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnConvolutionBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnConvolutionBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnConvolutionBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnConvolutionBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MkldnnConvolutionBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_bias_sym_sizes_opt", (getter)THPMkldnnConvolutionBackward0_bias_sym_sizes_opt_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPMkldnnConvolutionBackward0_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_groups", (getter)THPMkldnnConvolutionBackward0_groups_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPMkldnnConvolutionBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPMkldnnConvolutionBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPMkldnnConvolutionBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPMkldnnConvolutionBackward0_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPMkldnnConvolutionBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPMkldnnConvolutionBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMkldnnLinearBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnLinearBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnLinearBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnLinearBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnLinearBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnLinearBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnLinearBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnLinearBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MkldnnLinearBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMkldnnLinearBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPMkldnnLinearBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPMkldnnLinearBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPMkldnnLinearBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPToPaddedTensorBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ToPaddedTensorBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPToPaddedTensorBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ToPaddedTensorBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ToPaddedTensorBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPToPaddedTensorBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPToPaddedTensorBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPFlashAttentionBackward0_cum_seq_k_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FlashAttentionBackward0*>(self->cdata.get())->cum_seq_k_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFlashAttentionBackward0_cum_seq_k_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FlashAttentionBackward0*>(self->cdata.get())->cum_seq_k_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPFlashAttentionBackward0_cum_seq_q_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FlashAttentionBackward0*>(self->cdata.get())->cum_seq_q_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFlashAttentionBackward0_cum_seq_q_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FlashAttentionBackward0*>(self->cdata.get())->cum_seq_q_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPFlashAttentionBackward0_dropout_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FlashAttentionBackward0*>(self->cdata.get())->dropout_p;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPFlashAttentionBackward0_is_causal_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FlashAttentionBackward0*>(self->cdata.get())->is_causal;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPFlashAttentionBackward0_key_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FlashAttentionBackward0*>(self->cdata.get())->key_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFlashAttentionBackward0_key_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FlashAttentionBackward0*>(self->cdata.get())->key_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPFlashAttentionBackward0_max_k_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FlashAttentionBackward0*>(self->cdata.get())->max_k;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPFlashAttentionBackward0_max_q_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FlashAttentionBackward0*>(self->cdata.get())->max_q;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPFlashAttentionBackward0_query_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FlashAttentionBackward0*>(self->cdata.get())->query_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFlashAttentionBackward0_query_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FlashAttentionBackward0*>(self->cdata.get())->query_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPFlashAttentionBackward0_scale_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<FlashAttentionBackward0*>(self->cdata.get())->scale;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPFlashAttentionBackward0_value_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FlashAttentionBackward0*>(self->cdata.get())->value_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFlashAttentionBackward0_value_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FlashAttentionBackward0*>(self->cdata.get())->value_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPFlashAttentionBackward0_window_size_left_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<FlashAttentionBackward0*>(self->cdata.get())->window_size_left;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPFlashAttentionBackward0_window_size_right_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<FlashAttentionBackward0*>(self->cdata.get())->window_size_right;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPFlashAttentionBackward0_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FlashAttentionBackward0*>(self->cdata.get())->output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFlashAttentionBackward0_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FlashAttentionBackward0*>(self->cdata.get())->output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPFlashAttentionBackward0_rng_state_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FlashAttentionBackward0*>(self->cdata.get())->rng_state_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFlashAttentionBackward0_rng_state_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FlashAttentionBackward0*>(self->cdata.get())->rng_state_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPFlashAttentionBackward0_softmax_logsumexp_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FlashAttentionBackward0*>(self->cdata.get())->softmax_logsumexp_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFlashAttentionBackward0_softmax_logsumexp_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FlashAttentionBackward0*>(self->cdata.get())->softmax_logsumexp_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPFlashAttentionBackward0_unused_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FlashAttentionBackward0*>(self->cdata.get())->unused_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFlashAttentionBackward0_unused_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FlashAttentionBackward0*>(self->cdata.get())->unused_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FlashAttentionBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_cum_seq_k", (getter)THPFlashAttentionBackward0_cum_seq_k_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_cum_seq_k", (getter)THPFlashAttentionBackward0_cum_seq_k_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_cum_seq_q", (getter)THPFlashAttentionBackward0_cum_seq_q_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_cum_seq_q", (getter)THPFlashAttentionBackward0_cum_seq_q_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dropout_p", (getter)THPFlashAttentionBackward0_dropout_p_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_is_causal", (getter)THPFlashAttentionBackward0_is_causal_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_key", (getter)THPFlashAttentionBackward0_key_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_key", (getter)THPFlashAttentionBackward0_key_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_max_k", (getter)THPFlashAttentionBackward0_max_k_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_max_q", (getter)THPFlashAttentionBackward0_max_q_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_query", (getter)THPFlashAttentionBackward0_query_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_query", (getter)THPFlashAttentionBackward0_query_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale", (getter)THPFlashAttentionBackward0_scale_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_value", (getter)THPFlashAttentionBackward0_value_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_value", (getter)THPFlashAttentionBackward0_value_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_window_size_left", (getter)THPFlashAttentionBackward0_window_size_left_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_window_size_right", (getter)THPFlashAttentionBackward0_window_size_right_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output", (getter)THPFlashAttentionBackward0_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_output", (getter)THPFlashAttentionBackward0_output_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_rng_state", (getter)THPFlashAttentionBackward0_rng_state_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_rng_state", (getter)THPFlashAttentionBackward0_rng_state_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_softmax_logsumexp", (getter)THPFlashAttentionBackward0_softmax_logsumexp_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_softmax_logsumexp", (getter)THPFlashAttentionBackward0_softmax_logsumexp_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_unused", (getter)THPFlashAttentionBackward0_unused_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_unused", (getter)THPFlashAttentionBackward0_unused_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUnbindBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnbindBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UnbindBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPUnbindBackward0_dim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUnbindBackwardAutogradNestedTensor0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnbindBackwardAutogradNestedTensor0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUnbindBackwardAutogradNestedTensor0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<UnbindBackwardAutogradNestedTensor0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPUnbindBackwardAutogradNestedTensor0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<UnbindBackwardAutogradNestedTensor0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UnbindBackwardAutogradNestedTensor0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPUnbindBackwardAutogradNestedTensor0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPUnbindBackwardAutogradNestedTensor0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPUnbindBackwardAutogradNestedTensor0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSegmentReduceBackward0_axis_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SegmentReduceBackward0*>(self->cdata.get())->axis;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSegmentReduceBackward0_data_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SegmentReduceBackward0*>(self->cdata.get())->data_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSegmentReduceBackward0_data_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SegmentReduceBackward0*>(self->cdata.get())->data_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSegmentReduceBackward0_initial_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<SegmentReduceBackward0*>(self->cdata.get())->initial;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPSegmentReduceBackward0_lengths_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SegmentReduceBackward0*>(self->cdata.get())->lengths_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSegmentReduceBackward0_lengths_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SegmentReduceBackward0*>(self->cdata.get())->lengths_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSegmentReduceBackward0_offsets_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SegmentReduceBackward0*>(self->cdata.get())->offsets_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSegmentReduceBackward0_offsets_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SegmentReduceBackward0*>(self->cdata.get())->offsets_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSegmentReduceBackward0_reduce_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SegmentReduceBackward0*>(self->cdata.get())->reduce;
  return PyUnicode_FromStringAndSize(prop.data(), prop.size());
  END_HANDLE_TH_ERRORS
}

PyObject* THPSegmentReduceBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SegmentReduceBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSegmentReduceBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SegmentReduceBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SegmentReduceBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_axis", (getter)THPSegmentReduceBackward0_axis_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_data", (getter)THPSegmentReduceBackward0_data_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_data", (getter)THPSegmentReduceBackward0_data_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_initial", (getter)THPSegmentReduceBackward0_initial_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_lengths", (getter)THPSegmentReduceBackward0_lengths_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_lengths", (getter)THPSegmentReduceBackward0_lengths_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_offsets", (getter)THPSegmentReduceBackward0_offsets_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_offsets", (getter)THPSegmentReduceBackward0_offsets_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduce", (getter)THPSegmentReduceBackward0_reduce_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPSegmentReduceBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPSegmentReduceBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef PinMemoryBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPTestAutogradMultipleDispatchBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TestAutogradMultipleDispatchBackward0*>(self->cdata.get())->self_sym_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef TestAutogradMultipleDispatchBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sym_sizes", (getter)THPTestAutogradMultipleDispatchBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef TestAutogradMultipleDispatchBackwardAutogradNestedTensor0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPTestAutogradMultipleDispatchBackwardAutogradCUDA0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TestAutogradMultipleDispatchBackwardAutogradCUDA0*>(self->cdata.get())->self_sym_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef TestAutogradMultipleDispatchBackwardAutogradCUDA0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sym_sizes", (getter)THPTestAutogradMultipleDispatchBackwardAutogradCUDA0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef TestAutogradMultipleDispatchBackwardAutogradNestedTensor1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPForeachMinimumBackward0_scalar_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ForeachMinimumBackward0*>(self->cdata.get())->scalar;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPForeachMinimumBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<ForeachMinimumBackward0*>(self->cdata.get());
  const auto& prop = node->self_;
  if (node->self_released_) {
    PyErr_SetString(PyExc_RuntimeError, ERR_BACKWARD_TWICE);
    return nullptr;
  }
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i: c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, THPVariable_Wrap(prop[i].unpack(self->cdata)));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPForeachMinimumBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<ForeachMinimumBackward0*>(self->cdata.get());
  const auto& prop = node->self_;
  if (node->self_released_) {
    PyErr_SetString(PyExc_RuntimeError, ERR_BACKWARD_TWICE);
    return nullptr;
  }
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    pybind11::object obj = pybind11::cast(prop[i], pybind11::return_value_policy::reference);
    PyTuple_SetItem(tup, (Py_ssize_t) i, obj.release().ptr());
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ForeachMinimumBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_scalar", (getter)THPForeachMinimumBackward0_scalar_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPForeachMinimumBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPForeachMinimumBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPForeachMinimumBackward1_scalars_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<ForeachMinimumBackward1*>(self->cdata.get());
  const auto& prop = node->scalars;
  if (node->scalars_released_) {
    PyErr_SetString(PyExc_RuntimeError, ERR_BACKWARD_TWICE);
    return nullptr;
  }
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i: c10::irange(prop.size())) {
    if (prop[i].isComplex()) {
      auto cprop = prop[i].to<c10::complex<double>>();
      PyTuple_SetItem(tup, (Py_ssize_t) i, PyComplex_FromDoubles(cprop.real(), cprop.imag()));
    } else if (prop[i].isFloatingPoint()) {
      auto double_prop = prop[i].to<double>();
      PyTuple_SetItem(tup, (Py_ssize_t) i, PyFloat_FromDouble(double_prop));
    } else if (prop[i].isIntegral(/*includeBool=*/false)) {
      auto long_prop = prop[i].to<int64_t>();
      PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromLong(long_prop));
    } else if (prop[i].isBoolean()) {
      if (prop[i].to<bool>()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, Py_True);
      } else {
        PyTuple_SetItem(tup, (Py_ssize_t) i, Py_False);
      }
    } else {
      PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
      return nullptr;
    }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}
                            
PyObject* THPForeachMinimumBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<ForeachMinimumBackward1*>(self->cdata.get());
  const auto& prop = node->self_;
  if (node->self_released_) {
    PyErr_SetString(PyExc_RuntimeError, ERR_BACKWARD_TWICE);
    return nullptr;
  }
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i: c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, THPVariable_Wrap(prop[i].unpack(self->cdata)));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPForeachMinimumBackward1_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<ForeachMinimumBackward1*>(self->cdata.get());
  const auto& prop = node->self_;
  if (node->self_released_) {
    PyErr_SetString(PyExc_RuntimeError, ERR_BACKWARD_TWICE);
    return nullptr;
  }
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    pybind11::object obj = pybind11::cast(prop[i], pybind11::return_value_policy::reference);
    PyTuple_SetItem(tup, (Py_ssize_t) i, obj.release().ptr());
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ForeachMinimumBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_scalars", (getter)THPForeachMinimumBackward1_scalars_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPForeachMinimumBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPForeachMinimumBackward1_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAsStridedBackward0_copy_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AsStridedBackward0_copy*>(self->cdata.get())->size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAsStridedBackward0_copy_storage_offset_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<AsStridedBackward0_copy*>(self->cdata.get())->storage_offset;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAsStridedBackward0_copy_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AsStridedBackward0_copy*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
      auto si = prop[i];
      if (auto m = si.maybe_as_int()) {
        PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong(*m));
      } else {
        auto py_symint = py::cast(si).release().ptr();
        PyTuple_SetItem(tup, (Py_ssize_t) i, py_symint);
      }
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AsStridedBackward0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_size", (getter)THPAsStridedBackward0_copy_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_storage_offset", (getter)THPAsStridedBackward0_copy_storage_offset_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPAsStridedBackward0_copy_stride_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef ConjBackward0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef NegViewBackward0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef ViewAsComplexBackward0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef NestedViewFromBufferBackward0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

void initialize_autogenerated_functions_0(PyObject* module) {
  static PyTypeObject AddcmulBackward0Class;
  addClass<AddcmulBackward0>(module, AddcmulBackward0Class, "AddcmulBackward0", AddcmulBackward0_properties);
  static PyTypeObject AddmmBackward0Class;
  addClass<AddmmBackward0>(module, AddmmBackward0Class, "AddmmBackward0", AddmmBackward0_properties);
  static PyTypeObject SparseAddmmBackward0Class;
  addClass<SparseAddmmBackward0>(module, SparseAddmmBackward0Class, "SparseAddmmBackward0", SparseAddmmBackward0_properties);
  static PyTypeObject AcoshBackward0Class;
  addClass<AcoshBackward0>(module, AcoshBackward0Class, "AcoshBackward0", AcoshBackward0_properties);
  static PyTypeObject AtanhBackward1Class;
  addClass<AtanhBackward1>(module, AtanhBackward1Class, "AtanhBackward1", AtanhBackward1_properties);
  static PyTypeObject AsStridedBackward0Class;
  addClass<AsStridedBackward0>(module, AsStridedBackward0Class, "AsStridedBackward0", AsStridedBackward0_properties);
  static PyTypeObject CatBackward0Class;
  addClass<CatBackward0>(module, CatBackward0Class, "CatBackward0", CatBackward0_properties);
  static PyTypeObject CholeskyBackward0Class;
  addClass<CholeskyBackward0>(module, CholeskyBackward0Class, "CholeskyBackward0", CholeskyBackward0_properties);
  static PyTypeObject ToCopyBackward0Class;
  addClass<ToCopyBackward0>(module, ToCopyBackward0Class, "ToCopyBackward0", ToCopyBackward0_properties);
  static PyTypeObject ConjBackward0Class;
  addClass<ConjBackward0>(module, ConjBackward0Class, "ConjBackward0", ConjBackward0_properties);
  static PyTypeObject NegViewBackward0Class;
  addClass<NegViewBackward0>(module, NegViewBackward0Class, "NegViewBackward0", NegViewBackward0_properties);
  static PyTypeObject ConjPhysicalBackward0Class;
  addClass<ConjPhysicalBackward0>(module, ConjPhysicalBackward0Class, "ConjPhysicalBackward0", ConjPhysicalBackward0_properties);
  static PyTypeObject CosBackward0Class;
  addClass<CosBackward0>(module, CosBackward0Class, "CosBackward0", CosBackward0_properties);
  static PyTypeObject CumsumBackward0Class;
  addClass<CumsumBackward0>(module, CumsumBackward0Class, "CumsumBackward0", CumsumBackward0_properties);
  static PyTypeObject CtcLossBackward0Class;
  addClass<CtcLossBackward0>(module, CtcLossBackward0Class, "CtcLossBackward0", CtcLossBackward0_properties);
  static PyTypeObject CtcLossBackward1Class;
  addClass<CtcLossBackward1>(module, CtcLossBackward1Class, "CtcLossBackward1", CtcLossBackward1_properties);
  static PyTypeObject VdotBackward0Class;
  addClass<VdotBackward0>(module, VdotBackward0Class, "VdotBackward0", VdotBackward0_properties);
  static PyTypeObject FusedDropoutBackward0Class;
  addClass<FusedDropoutBackward0>(module, FusedDropoutBackward0Class, "FusedDropoutBackward0", FusedDropoutBackward0_properties);
  static PyTypeObject Expm1Backward0Class;
  addClass<Expm1Backward0>(module, Expm1Backward0Class, "Expm1Backward0", Expm1Backward0_properties);
  static PyTypeObject ExponentialBackward0Class;
  addClass<ExponentialBackward0>(module, ExponentialBackward0Class, "ExponentialBackward0", ExponentialBackward0_properties);
  static PyTypeObject FakeQuantizePerChannelAffineCachemaskBackward0Class;
  addClass<FakeQuantizePerChannelAffineCachemaskBackward0>(module, FakeQuantizePerChannelAffineCachemaskBackward0Class, "FakeQuantizePerChannelAffineCachemaskBackward0", FakeQuantizePerChannelAffineCachemaskBackward0_properties);
  static PyTypeObject FusedMovingAvgObsFqHelperBackward0Class;
  addClass<FusedMovingAvgObsFqHelperBackward0>(module, FusedMovingAvgObsFqHelperBackward0Class, "FusedMovingAvgObsFqHelperBackward0", FusedMovingAvgObsFqHelperBackward0_properties);
  static PyTypeObject FillBackward0Class;
  addClass<FillBackward0>(module, FillBackward0Class, "FillBackward0", FillBackward0_properties);
  static PyTypeObject FillBackward1Class;
  addClass<FillBackward1>(module, FillBackward1Class, "FillBackward1", FillBackward1_properties);
  static PyTypeObject GeometricBackward0Class;
  addClass<GeometricBackward0>(module, GeometricBackward0Class, "GeometricBackward0", GeometricBackward0_properties);
  static PyTypeObject SpecialI1EBackward0Class;
  addClass<SpecialI1EBackward0>(module, SpecialI1EBackward0Class, "SpecialI1EBackward0", SpecialI1EBackward0_properties);
  static PyTypeObject IgammaBackward0Class;
  addClass<IgammaBackward0>(module, IgammaBackward0Class, "IgammaBackward0", IgammaBackward0_properties);
  static PyTypeObject IndexCopyBackward0Class;
  addClass<IndexCopyBackward0>(module, IndexCopyBackward0Class, "IndexCopyBackward0", IndexCopyBackward0_properties);
  static PyTypeObject UnsafeIndexPutBackward0Class;
  addClass<UnsafeIndexPutBackward0>(module, UnsafeIndexPutBackward0Class, "UnsafeIndexPutBackward0", UnsafeIndexPutBackward0_properties);
  static PyTypeObject IndexSelectBackward0Class;
  addClass<IndexSelectBackward0>(module, IndexSelectBackward0Class, "IndexSelectBackward0", IndexSelectBackward0_properties);
  static PyTypeObject LeBackward0Class;
  addClass<LeBackward0>(module, LeBackward0Class, "LeBackward0", LeBackward0_properties);
  static PyTypeObject LeBackward1Class;
  addClass<LeBackward1>(module, LeBackward1Class, "LeBackward1", LeBackward1_properties);
  static PyTypeObject LgammaBackward0Class;
  addClass<LgammaBackward0>(module, LgammaBackward0Class, "LgammaBackward0", LgammaBackward0_properties);
  static PyTypeObject Log10Backward0Class;
  addClass<Log10Backward0>(module, Log10Backward0Class, "Log10Backward0", Log10Backward0_properties);
  static PyTypeObject Log1PBackward0Class;
  addClass<Log1PBackward0>(module, Log1PBackward0Class, "Log1PBackward0", Log1PBackward0_properties);
  static PyTypeObject Logaddexp2Backward0Class;
  addClass<Logaddexp2Backward0>(module, Logaddexp2Backward0Class, "Logaddexp2Backward0", Logaddexp2Backward0_properties);
  static PyTypeObject LogsumexpBackward0Class;
  addClass<LogsumexpBackward0>(module, LogsumexpBackward0Class, "LogsumexpBackward0", LogsumexpBackward0_properties);
  static PyTypeObject LinalgLstsqBackward0Class;
  addClass<LinalgLstsqBackward0>(module, LinalgLstsqBackward0Class, "LinalgLstsqBackward0", LinalgLstsqBackward0_properties);
  static PyTypeObject LinalgLuBackward0Class;
  addClass<LinalgLuBackward0>(module, LinalgLuBackward0Class, "LinalgLuBackward0", LinalgLuBackward0_properties);
  static PyTypeObject LuUnpackBackward0Class;
  addClass<LuUnpackBackward0>(module, LuUnpackBackward0Class, "LuUnpackBackward0", LuUnpackBackward0_properties);
  static PyTypeObject LinalgMatrixExpBackward0Class;
  addClass<LinalgMatrixExpBackward0>(module, LinalgMatrixExpBackward0Class, "LinalgMatrixExpBackward0", LinalgMatrixExpBackward0_properties);
  static PyTypeObject MaxBackward0Class;
  addClass<MaxBackward0>(module, MaxBackward0Class, "MaxBackward0", MaxBackward0_properties);
  static PyTypeObject MaxBackward1Class;
  addClass<MaxBackward1>(module, MaxBackward1Class, "MaxBackward1", MaxBackward1_properties);
  static PyTypeObject ModeBackward0Class;
  addClass<ModeBackward0>(module, ModeBackward0Class, "ModeBackward0", ModeBackward0_properties);
  static PyTypeObject MulBackward0Class;
  addClass<MulBackward0>(module, MulBackward0Class, "MulBackward0", MulBackward0_properties);
  static PyTypeObject MulBackward1Class;
  addClass<MulBackward1>(module, MulBackward1Class, "MulBackward1", MulBackward1_properties);
  static PyTypeObject MvlgammaBackward0Class;
  addClass<MvlgammaBackward0>(module, MvlgammaBackward0Class, "MvlgammaBackward0", MvlgammaBackward0_properties);
  static PyTypeObject NegBackward0Class;
  addClass<NegBackward0>(module, NegBackward0Class, "NegBackward0", NegBackward0_properties);
  static PyTypeObject NormalBackward0Class;
  addClass<NormalBackward0>(module, NormalBackward0Class, "NormalBackward0", NormalBackward0_properties);
  static PyTypeObject PowBackward0Class;
  addClass<PowBackward0>(module, PowBackward0Class, "PowBackward0", PowBackward0_properties);
  static PyTypeObject PowBackward1Class;
  addClass<PowBackward1>(module, PowBackward1Class, "PowBackward1", PowBackward1_properties);
  static PyTypeObject PowBackward2Class;
  addClass<PowBackward2>(module, PowBackward2Class, "PowBackward2", PowBackward2_properties);
  static PyTypeObject ReciprocalBackward0Class;
  addClass<ReciprocalBackward0>(module, ReciprocalBackward0Class, "ReciprocalBackward0", ReciprocalBackward0_properties);
  static PyTypeObject SpecialEntrBackward0Class;
  addClass<SpecialEntrBackward0>(module, SpecialEntrBackward0Class, "SpecialEntrBackward0", SpecialEntrBackward0_properties);
  static PyTypeObject SpecialNdtriBackward0Class;
  addClass<SpecialNdtriBackward0>(module, SpecialNdtriBackward0Class, "SpecialNdtriBackward0", SpecialNdtriBackward0_properties);
  static PyTypeObject SgnBackward0Class;
  addClass<SgnBackward0>(module, SgnBackward0Class, "SgnBackward0", SgnBackward0_properties);
  static PyTypeObject SinBackward0Class;
  addClass<SinBackward0>(module, SinBackward0Class, "SinBackward0", SinBackward0_properties);
  static PyTypeObject DiagonalScatterBackward0Class;
  addClass<DiagonalScatterBackward0>(module, DiagonalScatterBackward0Class, "DiagonalScatterBackward0", DiagonalScatterBackward0_properties);
  static PyTypeObject AsStridedScatterBackward0Class;
  addClass<AsStridedScatterBackward0>(module, AsStridedScatterBackward0Class, "AsStridedScatterBackward0", AsStridedScatterBackward0_properties);
  static PyTypeObject SplitBackward0Class;
  addClass<SplitBackward0>(module, SplitBackward0Class, "SplitBackward0", SplitBackward0_properties);
  static PyTypeObject SqueezeBackward0Class;
  addClass<SqueezeBackward0>(module, SqueezeBackward0Class, "SqueezeBackward0", SqueezeBackward0_properties);
  static PyTypeObject SqueezeBackward1Class;
  addClass<SqueezeBackward1>(module, SqueezeBackward1Class, "SqueezeBackward1", SqueezeBackward1_properties);
  static PyTypeObject SqueezeBackwardAutogradNestedTensor0Class;
  addClass<SqueezeBackwardAutogradNestedTensor0>(module, SqueezeBackwardAutogradNestedTensor0Class, "SqueezeBackwardAutogradNestedTensor0", SqueezeBackwardAutogradNestedTensor0_properties);
  static PyTypeObject SqueezeBackward2Class;
  addClass<SqueezeBackward2>(module, SqueezeBackward2Class, "SqueezeBackward2", SqueezeBackward2_properties);
  static PyTypeObject SqueezeBackwardAutogradNestedTensor1Class;
  addClass<SqueezeBackwardAutogradNestedTensor1>(module, SqueezeBackwardAutogradNestedTensor1Class, "SqueezeBackwardAutogradNestedTensor1", SqueezeBackwardAutogradNestedTensor1_properties);
  static PyTypeObject LinalgEighBackward0Class;
  addClass<LinalgEighBackward0>(module, LinalgEighBackward0Class, "LinalgEighBackward0", LinalgEighBackward0_properties);
  static PyTypeObject ToMkldnnBackward0Class;
  addClass<ToMkldnnBackward0>(module, ToMkldnnBackward0Class, "ToMkldnnBackward0", ToMkldnnBackward0_properties);
  static PyTypeObject UniqueBackward0Class;
  addClass<UniqueBackward0>(module, UniqueBackward0Class, "UniqueBackward0", UniqueBackward0_properties);
  static PyTypeObject LiftBackward0Class;
  addClass<LiftBackward0>(module, LiftBackward0Class, "LiftBackward0", LiftBackward0_properties);
  static PyTypeObject LiftFreshBackward0Class;
  addClass<LiftFreshBackward0>(module, LiftFreshBackward0Class, "LiftFreshBackward0", LiftFreshBackward0_properties);
  static PyTypeObject WhereBackward0Class;
  addClass<WhereBackward0>(module, WhereBackward0Class, "WhereBackward0", WhereBackward0_properties);
  static PyTypeObject WeightNormInterfaceBackward0Class;
  addClass<WeightNormInterfaceBackward0>(module, WeightNormInterfaceBackward0Class, "WeightNormInterfaceBackward0", WeightNormInterfaceBackward0_properties);
  static PyTypeObject ZeroBackward0Class;
  addClass<ZeroBackward0>(module, ZeroBackward0Class, "ZeroBackward0", ZeroBackward0_properties);
  static PyTypeObject SparseMaskBackward0Class;
  addClass<SparseMaskBackward0>(module, SparseMaskBackward0Class, "SparseMaskBackward0", SparseMaskBackward0_properties);
  static PyTypeObject SparseSumBackward0Class;
  addClass<SparseSumBackward0>(module, SparseSumBackward0Class, "SparseSumBackward0", SparseSumBackward0_properties);
  static PyTypeObject StandardGammaBackward0Class;
  addClass<StandardGammaBackward0>(module, StandardGammaBackward0Class, "StandardGammaBackward0", StandardGammaBackward0_properties);
  static PyTypeObject ValuesBackward0Class;
  addClass<ValuesBackward0>(module, ValuesBackward0Class, "ValuesBackward0", ValuesBackward0_properties);
  static PyTypeObject ValuesBackwardAutogradNestedTensor0Class;
  addClass<ValuesBackwardAutogradNestedTensor0>(module, ValuesBackwardAutogradNestedTensor0Class, "ValuesBackwardAutogradNestedTensor0", ValuesBackwardAutogradNestedTensor0_properties);
  static PyTypeObject EmbeddingDenseBackwardBackward0Class;
  addClass<EmbeddingDenseBackwardBackward0>(module, EmbeddingDenseBackwardBackward0Class, "EmbeddingDenseBackwardBackward0", EmbeddingDenseBackwardBackward0_properties);
  static PyTypeObject EmbeddingBagBackward0Class;
  addClass<EmbeddingBagBackward0>(module, EmbeddingBagBackward0Class, "EmbeddingBagBackward0", EmbeddingBagBackward0_properties);
  static PyTypeObject GeluBackwardBackward0Class;
  addClass<GeluBackwardBackward0>(module, GeluBackwardBackward0Class, "GeluBackwardBackward0", GeluBackwardBackward0_properties);
  static PyTypeObject LogSoftmaxBackward0Class;
  addClass<LogSoftmaxBackward0>(module, LogSoftmaxBackward0Class, "LogSoftmaxBackward0", LogSoftmaxBackward0_properties);
  static PyTypeObject PreluKernelBackwardBackward0Class;
  addClass<PreluKernelBackwardBackward0>(module, PreluKernelBackwardBackward0Class, "PreluKernelBackwardBackward0", PreluKernelBackwardBackward0_properties);
  static PyTypeObject ReflectionPad1DBackward0Class;
  addClass<ReflectionPad1DBackward0>(module, ReflectionPad1DBackward0Class, "ReflectionPad1DBackward0", ReflectionPad1DBackward0_properties);
  static PyTypeObject UpsampleBicubic2DAaBackward0Class;
  addClass<UpsampleBicubic2DAaBackward0>(module, UpsampleBicubic2DAaBackward0Class, "UpsampleBicubic2DAaBackward0", UpsampleBicubic2DAaBackward0_properties);
  static PyTypeObject UpsampleTrilinear3DBackward0Class;
  addClass<UpsampleTrilinear3DBackward0>(module, UpsampleTrilinear3DBackward0Class, "UpsampleTrilinear3DBackward0", UpsampleTrilinear3DBackward0_properties);
  static PyTypeObject UpsampleNearest3DBackward0Class;
  addClass<UpsampleNearest3DBackward0>(module, UpsampleNearest3DBackward0Class, "UpsampleNearest3DBackward0", UpsampleNearest3DBackward0_properties);
  static PyTypeObject PixelUnshuffleBackward0Class;
  addClass<PixelUnshuffleBackward0>(module, PixelUnshuffleBackward0Class, "PixelUnshuffleBackward0", PixelUnshuffleBackward0_properties);
  static PyTypeObject LinearBackward0Class;
  addClass<LinearBackward0>(module, LinearBackward0Class, "LinearBackward0", LinearBackward0_properties);
  static PyTypeObject MaxPool2DWithIndicesBackward0Class;
  addClass<MaxPool2DWithIndicesBackward0>(module, MaxPool2DWithIndicesBackward0Class, "MaxPool2DWithIndicesBackward0", MaxPool2DWithIndicesBackward0_properties);
  static PyTypeObject ConvolutionBackward0Class;
  addClass<ConvolutionBackward0>(module, ConvolutionBackward0Class, "ConvolutionBackward0", ConvolutionBackward0_properties);
  static PyTypeObject ConvolutionBackward1Class;
  addClass<ConvolutionBackward1>(module, ConvolutionBackward1Class, "ConvolutionBackward1", ConvolutionBackward1_properties);
  static PyTypeObject ConvolutionBackwardOverrideableBackward0Class;
  addClass<ConvolutionBackwardOverrideableBackward0>(module, ConvolutionBackwardOverrideableBackward0Class, "ConvolutionBackwardOverrideableBackward0", ConvolutionBackwardOverrideableBackward0_properties);
  static PyTypeObject SlowConv2DBackward0Class;
  addClass<SlowConv2DBackward0>(module, SlowConv2DBackward0Class, "SlowConv2DBackward0", SlowConv2DBackward0_properties);
  static PyTypeObject EluBackwardBackward0Class;
  addClass<EluBackwardBackward0>(module, EluBackwardBackward0Class, "EluBackwardBackward0", EluBackwardBackward0_properties);
  static PyTypeObject FractionalMaxPool3DBackwardBackward0Class;
  addClass<FractionalMaxPool3DBackwardBackward0>(module, FractionalMaxPool3DBackwardBackward0Class, "FractionalMaxPool3DBackwardBackward0", FractionalMaxPool3DBackwardBackward0_properties);
  static PyTypeObject MseLossBackwardBackward0Class;
  addClass<MseLossBackwardBackward0>(module, MseLossBackwardBackward0Class, "MseLossBackwardBackward0", MseLossBackwardBackward0_properties);
  static PyTypeObject RreluWithNoiseBackwardBackward0Class;
  addClass<RreluWithNoiseBackwardBackward0>(module, RreluWithNoiseBackwardBackward0Class, "RreluWithNoiseBackwardBackward0", RreluWithNoiseBackwardBackward0_properties);
  static PyTypeObject SoftplusBackwardBackward0Class;
  addClass<SoftplusBackwardBackward0>(module, SoftplusBackwardBackward0Class, "SoftplusBackwardBackward0", SoftplusBackwardBackward0_properties);
  static PyTypeObject ThresholdBackwardBackward0Class;
  addClass<ThresholdBackwardBackward0>(module, ThresholdBackwardBackward0Class, "ThresholdBackwardBackward0", ThresholdBackwardBackward0_properties);
  static PyTypeObject UpsampleBilinear2DAaBackwardBackward0Class;
  addClass<UpsampleBilinear2DAaBackwardBackward0>(module, UpsampleBilinear2DAaBackwardBackward0Class, "UpsampleBilinear2DAaBackwardBackward0", UpsampleBilinear2DAaBackwardBackward0_properties);
  static PyTypeObject SigmoidBackwardBackward0Class;
  addClass<SigmoidBackwardBackward0>(module, SigmoidBackwardBackward0Class, "SigmoidBackwardBackward0", SigmoidBackwardBackward0_properties);
  static PyTypeObject TanhBackwardBackward0Class;
  addClass<TanhBackwardBackward0>(module, TanhBackwardBackward0Class, "TanhBackwardBackward0", TanhBackwardBackward0_properties);
  static PyTypeObject CudnnCtcLossBackward0Class;
  addClass<CudnnCtcLossBackward0>(module, CudnnCtcLossBackward0Class, "CudnnCtcLossBackward0", CudnnCtcLossBackward0_properties);
  static PyTypeObject CudnnCtcLossBackward1Class;
  addClass<CudnnCtcLossBackward1>(module, CudnnCtcLossBackward1Class, "CudnnCtcLossBackward1", CudnnCtcLossBackward1_properties);
  static PyTypeObject CudnnConvolutionTransposeBackward0Class;
  addClass<CudnnConvolutionTransposeBackward0>(module, CudnnConvolutionTransposeBackward0Class, "CudnnConvolutionTransposeBackward0", CudnnConvolutionTransposeBackward0_properties);
  static PyTypeObject CudnnAffineGridGeneratorBackward0Class;
  addClass<CudnnAffineGridGeneratorBackward0>(module, CudnnAffineGridGeneratorBackward0Class, "CudnnAffineGridGeneratorBackward0", CudnnAffineGridGeneratorBackward0_properties);
  static PyTypeObject CudnnBatchNormBackwardBackward0Class;
  addClass<CudnnBatchNormBackwardBackward0>(module, CudnnBatchNormBackwardBackward0Class, "CudnnBatchNormBackwardBackward0", CudnnBatchNormBackwardBackward0_properties);
  static PyTypeObject CudnnRnnBackward0Class;
  addClass<CudnnRnnBackward0>(module, CudnnRnnBackward0Class, "CudnnRnnBackward0", CudnnRnnBackward0_properties);
  static PyTypeObject MiopenBatchNormBackwardBackward0Class;
  addClass<MiopenBatchNormBackwardBackward0>(module, MiopenBatchNormBackwardBackward0Class, "MiopenBatchNormBackwardBackward0", MiopenBatchNormBackwardBackward0_properties);
  static PyTypeObject MkldnnConvolutionBackward0Class;
  addClass<MkldnnConvolutionBackward0>(module, MkldnnConvolutionBackward0Class, "MkldnnConvolutionBackward0", MkldnnConvolutionBackward0_properties);
  static PyTypeObject MkldnnLinearBackward0Class;
  addClass<MkldnnLinearBackward0>(module, MkldnnLinearBackward0Class, "MkldnnLinearBackward0", MkldnnLinearBackward0_properties);
  static PyTypeObject ToPaddedTensorBackward0Class;
  addClass<ToPaddedTensorBackward0>(module, ToPaddedTensorBackward0Class, "ToPaddedTensorBackward0", ToPaddedTensorBackward0_properties);
  static PyTypeObject FlashAttentionBackward0Class;
  addClass<FlashAttentionBackward0>(module, FlashAttentionBackward0Class, "FlashAttentionBackward0", FlashAttentionBackward0_properties);
  static PyTypeObject UnbindBackward0Class;
  addClass<UnbindBackward0>(module, UnbindBackward0Class, "UnbindBackward0", UnbindBackward0_properties);
  static PyTypeObject UnbindBackwardAutogradNestedTensor0Class;
  addClass<UnbindBackwardAutogradNestedTensor0>(module, UnbindBackwardAutogradNestedTensor0Class, "UnbindBackwardAutogradNestedTensor0", UnbindBackwardAutogradNestedTensor0_properties);
  static PyTypeObject SegmentReduceBackward0Class;
  addClass<SegmentReduceBackward0>(module, SegmentReduceBackward0Class, "SegmentReduceBackward0", SegmentReduceBackward0_properties);
  static PyTypeObject PinMemoryBackward0Class;
  addClass<PinMemoryBackward0>(module, PinMemoryBackward0Class, "PinMemoryBackward0", PinMemoryBackward0_properties);
  static PyTypeObject TestAutogradMultipleDispatchBackward0Class;
  addClass<TestAutogradMultipleDispatchBackward0>(module, TestAutogradMultipleDispatchBackward0Class, "TestAutogradMultipleDispatchBackward0", TestAutogradMultipleDispatchBackward0_properties);
  static PyTypeObject TestAutogradMultipleDispatchBackwardAutogradNestedTensor0Class;
  addClass<TestAutogradMultipleDispatchBackwardAutogradNestedTensor0>(module, TestAutogradMultipleDispatchBackwardAutogradNestedTensor0Class, "TestAutogradMultipleDispatchBackwardAutogradNestedTensor0", TestAutogradMultipleDispatchBackwardAutogradNestedTensor0_properties);
  static PyTypeObject TestAutogradMultipleDispatchBackwardAutogradCUDA0Class;
  addClass<TestAutogradMultipleDispatchBackwardAutogradCUDA0>(module, TestAutogradMultipleDispatchBackwardAutogradCUDA0Class, "TestAutogradMultipleDispatchBackwardAutogradCUDA0", TestAutogradMultipleDispatchBackwardAutogradCUDA0_properties);
  static PyTypeObject TestAutogradMultipleDispatchBackwardAutogradNestedTensor1Class;
  addClass<TestAutogradMultipleDispatchBackwardAutogradNestedTensor1>(module, TestAutogradMultipleDispatchBackwardAutogradNestedTensor1Class, "TestAutogradMultipleDispatchBackwardAutogradNestedTensor1", TestAutogradMultipleDispatchBackwardAutogradNestedTensor1_properties);
  static PyTypeObject ForeachMinimumBackward0Class;
  addClass<ForeachMinimumBackward0>(module, ForeachMinimumBackward0Class, "ForeachMinimumBackward0", ForeachMinimumBackward0_properties);
  static PyTypeObject ForeachMinimumBackward1Class;
  addClass<ForeachMinimumBackward1>(module, ForeachMinimumBackward1Class, "ForeachMinimumBackward1", ForeachMinimumBackward1_properties);
  static PyTypeObject AsStridedBackward0_copyClass;
  addClass<AsStridedBackward0_copy>(module, AsStridedBackward0_copyClass, "AsStridedBackward0_copy", AsStridedBackward0_copy_properties);
  static PyTypeObject ConjBackward0_copyClass;
  addClass<ConjBackward0_copy>(module, ConjBackward0_copyClass, "ConjBackward0_copy", ConjBackward0_copy_properties);
  static PyTypeObject NegViewBackward0_copyClass;
  addClass<NegViewBackward0_copy>(module, NegViewBackward0_copyClass, "NegViewBackward0_copy", NegViewBackward0_copy_properties);
  static PyTypeObject ViewAsComplexBackward0_copyClass;
  addClass<ViewAsComplexBackward0_copy>(module, ViewAsComplexBackward0_copyClass, "ViewAsComplexBackward0_copy", ViewAsComplexBackward0_copy_properties);
  static PyTypeObject NestedViewFromBufferBackward0_copyClass;
  addClass<NestedViewFromBufferBackward0_copy>(module, NestedViewFromBufferBackward0_copyClass, "NestedViewFromBufferBackward0_copy", NestedViewFromBufferBackward0_copy_properties);
}

} // namespace torch::autograd::generated
