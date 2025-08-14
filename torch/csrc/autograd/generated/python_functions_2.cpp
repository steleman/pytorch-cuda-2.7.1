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

PyObject* THPAddBackward0_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddBackward0*>(self->cdata.get())->alpha;
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

static struct PyGetSetDef AddBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_alpha", (getter)THPAddBackward0_alpha_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef AddBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPAddcdivBackward0_tensor1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddcdivBackward0*>(self->cdata.get())->tensor1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddcdivBackward0_tensor1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddcdivBackward0*>(self->cdata.get())->tensor1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddcdivBackward0_tensor2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddcdivBackward0*>(self->cdata.get())->tensor2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddcdivBackward0_tensor2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddcdivBackward0*>(self->cdata.get())->tensor2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddcdivBackward0_value_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddcdivBackward0*>(self->cdata.get())->value;
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

static struct PyGetSetDef AddcdivBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_tensor1", (getter)THPAddcdivBackward0_tensor1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_tensor1", (getter)THPAddcdivBackward0_tensor1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_tensor2", (getter)THPAddcdivBackward0_tensor2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_tensor2", (getter)THPAddcdivBackward0_tensor2_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_value", (getter)THPAddcdivBackward0_value_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAngleBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AngleBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAngleBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AngleBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AngleBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPAngleBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPAngleBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef AcoshBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPAsinhBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AsinhBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAsinhBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AsinhBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AsinhBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPAsinhBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPAsinhBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef AsinhBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPAsStridedBackward1_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AsStridedBackward1*>(self->cdata.get())->size;
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

PyObject* THPAsStridedBackward1_storage_offset_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<AsStridedBackward1*>(self->cdata.get())->storage_offset;
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

PyObject* THPAsStridedBackward1_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AsStridedBackward1*>(self->cdata.get())->stride;
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

static struct PyGetSetDef AsStridedBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_size", (getter)THPAsStridedBackward1_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_storage_offset", (getter)THPAsStridedBackward1_storage_offset_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPAsStridedBackward1_stride_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAtan2Backward0_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<Atan2Backward0*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAtan2Backward0_other_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<Atan2Backward0*>(self->cdata.get())->other_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPAtan2Backward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<Atan2Backward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAtan2Backward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<Atan2Backward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef Atan2Backward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPAtan2Backward0_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_other", (getter)THPAtan2Backward0_other_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPAtan2Backward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPAtan2Backward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPBaddbmmBackward0_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<BaddbmmBackward0*>(self->cdata.get())->alpha;
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

PyObject* THPBaddbmmBackward0_batch1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BaddbmmBackward0*>(self->cdata.get())->batch1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBaddbmmBackward0_batch1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BaddbmmBackward0*>(self->cdata.get())->batch1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBaddbmmBackward0_batch2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BaddbmmBackward0*>(self->cdata.get())->batch2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBaddbmmBackward0_batch2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BaddbmmBackward0*>(self->cdata.get())->batch2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBaddbmmBackward0_beta_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<BaddbmmBackward0*>(self->cdata.get())->beta;
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

static struct PyGetSetDef BaddbmmBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_alpha", (getter)THPBaddbmmBackward0_alpha_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_batch1", (getter)THPBaddbmmBackward0_batch1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_batch1", (getter)THPBaddbmmBackward0_batch1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_batch2", (getter)THPBaddbmmBackward0_batch2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_batch2", (getter)THPBaddbmmBackward0_batch2_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_beta", (getter)THPBaddbmmBackward0_beta_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef BernoulliBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef ChunkBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPChunkBackwardAutogradNestedTensor0_chunks_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ChunkBackwardAutogradNestedTensor0*>(self->cdata.get())->chunks;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPChunkBackwardAutogradNestedTensor0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ChunkBackwardAutogradNestedTensor0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPChunkBackwardAutogradNestedTensor0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ChunkBackwardAutogradNestedTensor0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPChunkBackwardAutogradNestedTensor0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ChunkBackwardAutogradNestedTensor0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ChunkBackwardAutogradNestedTensor0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_chunks", (getter)THPChunkBackwardAutogradNestedTensor0_chunks_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPChunkBackwardAutogradNestedTensor0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPChunkBackwardAutogradNestedTensor0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPChunkBackwardAutogradNestedTensor0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCholeskyInverseBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CholeskyInverseBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCholeskyInverseBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CholeskyInverseBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCholeskyInverseBackward0_upper_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CholeskyInverseBackward0*>(self->cdata.get())->upper;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCholeskyInverseBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CholeskyInverseBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCholeskyInverseBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CholeskyInverseBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CholeskyInverseBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPCholeskyInverseBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPCholeskyInverseBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_upper", (getter)THPCholeskyInverseBackward0_upper_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPCholeskyInverseBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPCholeskyInverseBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPClampBackward0_max_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ClampBackward0*>(self->cdata.get())->max_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPClampBackward0_max_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ClampBackward0*>(self->cdata.get())->max_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPClampBackward0_min_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ClampBackward0*>(self->cdata.get())->min_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPClampBackward0_min_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ClampBackward0*>(self->cdata.get())->min_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPClampBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ClampBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPClampBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ClampBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ClampBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_max", (getter)THPClampBackward0_max_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_max", (getter)THPClampBackward0_max_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_min", (getter)THPClampBackward0_min_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_min", (getter)THPClampBackward0_min_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPClampBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPClampBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPClampBackward1_max_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<ClampBackward1*>(self->cdata.get())->max;
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

PyObject* THPClampBackward1_min_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<ClampBackward1*>(self->cdata.get())->min;
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

PyObject* THPClampBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ClampBackward1*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPClampBackward1_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ClampBackward1*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ClampBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_max", (getter)THPClampBackward1_max_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_min", (getter)THPClampBackward1_min_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPClampBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPClampBackward1_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPClampMaxBackward0_max_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ClampMaxBackward0*>(self->cdata.get())->max;
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

PyObject* THPClampMaxBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ClampMaxBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPClampMaxBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ClampMaxBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ClampMaxBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_max", (getter)THPClampMaxBackward0_max_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPClampMaxBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPClampMaxBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPClampMaxBackward1_max_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ClampMaxBackward1*>(self->cdata.get())->max_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPClampMaxBackward1_max_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ClampMaxBackward1*>(self->cdata.get())->max_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPClampMaxBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ClampMaxBackward1*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPClampMaxBackward1_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ClampMaxBackward1*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ClampMaxBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_max", (getter)THPClampMaxBackward1_max_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_max", (getter)THPClampMaxBackward1_max_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPClampMaxBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPClampMaxBackward1_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef LazyCloneBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPComplexBackward0_imag_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ComplexBackward0*>(self->cdata.get())->imag_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPComplexBackward0_imag_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ComplexBackward0*>(self->cdata.get())->imag_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPComplexBackward0_real_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ComplexBackward0*>(self->cdata.get())->real_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPComplexBackward0_real_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ComplexBackward0*>(self->cdata.get())->real_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ComplexBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_imag", (getter)THPComplexBackward0_imag_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_imag", (getter)THPComplexBackward0_imag_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_real", (getter)THPComplexBackward0_real_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_real", (getter)THPComplexBackward0_real_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPPolarBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PolarBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPPolarBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PolarBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef PolarBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_result", (getter)THPPolarBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPPolarBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCummaxBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CummaxBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCummaxBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CummaxBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCummaxBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CummaxBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCummaxBackward0_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CummaxBackward0*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCummaxBackward0_indices_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CummaxBackward0*>(self->cdata.get())->indices_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CummaxBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPCummaxBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPCummaxBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPCummaxBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_indices", (getter)THPCummaxBackward0_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_indices", (getter)THPCummaxBackward0_indices_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef BlockDiagBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPDiagonalBackwardBackward0_dim1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<DiagonalBackwardBackward0*>(self->cdata.get())->dim1;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPDiagonalBackwardBackward0_dim2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<DiagonalBackwardBackward0*>(self->cdata.get())->dim2;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPDiagonalBackwardBackward0_offset_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<DiagonalBackwardBackward0*>(self->cdata.get())->offset;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef DiagonalBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim1", (getter)THPDiagonalBackwardBackward0_dim1_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim2", (getter)THPDiagonalBackwardBackward0_dim2_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_offset", (getter)THPDiagonalBackwardBackward0_offset_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPDotBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<DotBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPDotBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<DotBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPDotBackward0_tensor_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<DotBackward0*>(self->cdata.get())->tensor_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPDotBackward0_tensor_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<DotBackward0*>(self->cdata.get())->tensor_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef DotBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPDotBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPDotBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_tensor", (getter)THPDotBackward0_tensor_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_tensor", (getter)THPDotBackward0_tensor_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNativeDropoutBackward0_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NativeDropoutBackward0*>(self->cdata.get())->p;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeDropoutBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeDropoutBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeDropoutBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeDropoutBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NativeDropoutBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_p", (getter)THPNativeDropoutBackward0_p_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPNativeDropoutBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPNativeDropoutBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPExpBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ExpBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPExpBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ExpBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ExpBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_result", (getter)THPExpBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPExpBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPExp2Backward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<Exp2Backward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPExp2Backward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<Exp2Backward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef Exp2Backward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_result", (getter)THPExp2Backward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPExp2Backward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPFakeQuantizePerTensorAffineCachemaskBackward0_mask_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FakeQuantizePerTensorAffineCachemaskBackward0*>(self->cdata.get())->mask_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFakeQuantizePerTensorAffineCachemaskBackward0_mask_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FakeQuantizePerTensorAffineCachemaskBackward0*>(self->cdata.get())->mask_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FakeQuantizePerTensorAffineCachemaskBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_mask", (getter)THPFakeQuantizePerTensorAffineCachemaskBackward0_mask_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_mask", (getter)THPFakeQuantizePerTensorAffineCachemaskBackward0_mask_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPFakeQuantizePerTensorAffineCachemaskTensorQparamsBackward0_mask_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FakeQuantizePerTensorAffineCachemaskTensorQparamsBackward0*>(self->cdata.get())->mask_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFakeQuantizePerTensorAffineCachemaskTensorQparamsBackward0_mask_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FakeQuantizePerTensorAffineCachemaskTensorQparamsBackward0*>(self->cdata.get())->mask_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FakeQuantizePerTensorAffineCachemaskTensorQparamsBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_mask", (getter)THPFakeQuantizePerTensorAffineCachemaskTensorQparamsBackward0_mask_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_mask", (getter)THPFakeQuantizePerTensorAffineCachemaskTensorQparamsBackward0_mask_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPFakeQuantizeLearnablePerTensorAffineBackward0_grad_factor_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FakeQuantizeLearnablePerTensorAffineBackward0*>(self->cdata.get())->grad_factor;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPFakeQuantizeLearnablePerTensorAffineBackward0_quant_max_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FakeQuantizeLearnablePerTensorAffineBackward0*>(self->cdata.get())->quant_max;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPFakeQuantizeLearnablePerTensorAffineBackward0_quant_min_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FakeQuantizeLearnablePerTensorAffineBackward0*>(self->cdata.get())->quant_min;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPFakeQuantizeLearnablePerTensorAffineBackward0_scale_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FakeQuantizeLearnablePerTensorAffineBackward0*>(self->cdata.get())->scale_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFakeQuantizeLearnablePerTensorAffineBackward0_scale_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FakeQuantizeLearnablePerTensorAffineBackward0*>(self->cdata.get())->scale_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPFakeQuantizeLearnablePerTensorAffineBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FakeQuantizeLearnablePerTensorAffineBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFakeQuantizeLearnablePerTensorAffineBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FakeQuantizeLearnablePerTensorAffineBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPFakeQuantizeLearnablePerTensorAffineBackward0_zero_point_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FakeQuantizeLearnablePerTensorAffineBackward0*>(self->cdata.get())->zero_point_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFakeQuantizeLearnablePerTensorAffineBackward0_zero_point_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FakeQuantizeLearnablePerTensorAffineBackward0*>(self->cdata.get())->zero_point_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FakeQuantizeLearnablePerTensorAffineBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_factor", (getter)THPFakeQuantizeLearnablePerTensorAffineBackward0_grad_factor_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_quant_max", (getter)THPFakeQuantizeLearnablePerTensorAffineBackward0_quant_max_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_quant_min", (getter)THPFakeQuantizeLearnablePerTensorAffineBackward0_quant_min_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale", (getter)THPFakeQuantizeLearnablePerTensorAffineBackward0_scale_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_scale", (getter)THPFakeQuantizeLearnablePerTensorAffineBackward0_scale_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPFakeQuantizeLearnablePerTensorAffineBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPFakeQuantizeLearnablePerTensorAffineBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_zero_point", (getter)THPFakeQuantizeLearnablePerTensorAffineBackward0_zero_point_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_zero_point", (getter)THPFakeQuantizeLearnablePerTensorAffineBackward0_zero_point_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef FracBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPHardswishBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HardswishBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPHardswishBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HardswishBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef HardswishBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPHardswishBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPHardswishBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPHypotBackward0_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HypotBackward0*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPHypotBackward0_other_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HypotBackward0*>(self->cdata.get())->other_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPHypotBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HypotBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPHypotBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HypotBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPHypotBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HypotBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPHypotBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HypotBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef HypotBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPHypotBackward0_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_other", (getter)THPHypotBackward0_other_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPHypotBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPHypotBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPHypotBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPHypotBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPIgammacBackward0_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IgammacBackward0*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPIgammacBackward0_other_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IgammacBackward0*>(self->cdata.get())->other_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPIgammacBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IgammacBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPIgammacBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IgammacBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef IgammacBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPIgammacBackward0_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_other", (getter)THPIgammacBackward0_other_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPIgammacBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPIgammacBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUnsafeMaskedIndexBackward0_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<UnsafeMaskedIndexBackward0*>(self->cdata.get());
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

PyObject* THPUnsafeMaskedIndexBackward0_indices_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<UnsafeMaskedIndexBackward0*>(self->cdata.get());
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

PyObject* THPUnsafeMaskedIndexBackward0_mask_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<UnsafeMaskedIndexBackward0*>(self->cdata.get())->mask_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPUnsafeMaskedIndexBackward0_mask_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<UnsafeMaskedIndexBackward0*>(self->cdata.get())->mask_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPUnsafeMaskedIndexBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnsafeMaskedIndexBackward0*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef UnsafeMaskedIndexBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_indices", (getter)THPUnsafeMaskedIndexBackward0_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_indices", (getter)THPUnsafeMaskedIndexBackward0_indices_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mask", (getter)THPUnsafeMaskedIndexBackward0_mask_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_mask", (getter)THPUnsafeMaskedIndexBackward0_mask_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPUnsafeMaskedIndexBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLogBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LogBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPLogBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPLogBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLog2Backward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<Log2Backward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLog2Backward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<Log2Backward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef Log2Backward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPLog2Backward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPLog2Backward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLogaddexpBackward0_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogaddexpBackward0*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogaddexpBackward0_other_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogaddexpBackward0*>(self->cdata.get())->other_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogaddexpBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogaddexpBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogaddexpBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogaddexpBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LogaddexpBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPLogaddexpBackward0_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_other", (getter)THPLogaddexpBackward0_other_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPLogaddexpBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPLogaddexpBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSpecialZetaBackward0_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialZetaBackward0*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialZetaBackward0_other_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialZetaBackward0*>(self->cdata.get())->other_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialZetaBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialZetaBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialZetaBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialZetaBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SpecialZetaBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPSpecialZetaBackward0_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_other", (getter)THPSpecialZetaBackward0_other_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSpecialZetaBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSpecialZetaBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSpecialZetaBackward1_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialZetaBackward1*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialZetaBackward1_other_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialZetaBackward1*>(self->cdata.get())->other_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialZetaBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SpecialZetaBackward1*>(self->cdata.get())->self;
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

static struct PyGetSetDef SpecialZetaBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPSpecialZetaBackward1_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_other", (getter)THPSpecialZetaBackward1_other_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSpecialZetaBackward1_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef SpecialZetaBackward2_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPMaskedScatterBackwardBackward0_mask_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaskedScatterBackwardBackward0*>(self->cdata.get())->mask_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaskedScatterBackwardBackward0_mask_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaskedScatterBackwardBackward0*>(self->cdata.get())->mask_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MaskedScatterBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_mask", (getter)THPMaskedScatterBackwardBackward0_mask_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_mask", (getter)THPMaskedScatterBackwardBackward0_mask_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMaskedSelectBackward0_mask_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaskedSelectBackward0*>(self->cdata.get())->mask_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaskedSelectBackward0_mask_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaskedSelectBackward0*>(self->cdata.get())->mask_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaskedSelectBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaskedSelectBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaskedSelectBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaskedSelectBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MaskedSelectBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_mask", (getter)THPMaskedSelectBackward0_mask_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_mask", (getter)THPMaskedSelectBackward0_mask_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPMaskedSelectBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPMaskedSelectBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPFmaxBackward0_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FmaxBackward0*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFmaxBackward0_other_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FmaxBackward0*>(self->cdata.get())->other_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPFmaxBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FmaxBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFmaxBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FmaxBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FmaxBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPFmaxBackward0_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_other", (getter)THPFmaxBackward0_other_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPFmaxBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPFmaxBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNativeBatchNormLegitBackward0_eps_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NativeBatchNormLegitBackward0*>(self->cdata.get())->eps;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormLegitBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormLegitBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitBackward0_running_mean_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormLegitBackward0*>(self->cdata.get())->running_mean_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitBackward0_running_mean_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormLegitBackward0*>(self->cdata.get())->running_mean_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitBackward0_running_var_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormLegitBackward0*>(self->cdata.get())->running_var_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitBackward0_running_var_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormLegitBackward0*>(self->cdata.get())->running_var_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitBackward0_training_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NativeBatchNormLegitBackward0*>(self->cdata.get())->training;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormLegitBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormLegitBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormLegitBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormLegitBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitBackward0_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormLegitBackward0*>(self->cdata.get())->result2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitBackward0_result2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormLegitBackward0*>(self->cdata.get())->result2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NativeBatchNormLegitBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_eps", (getter)THPNativeBatchNormLegitBackward0_eps_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPNativeBatchNormLegitBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPNativeBatchNormLegitBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_mean", (getter)THPNativeBatchNormLegitBackward0_running_mean_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_running_mean", (getter)THPNativeBatchNormLegitBackward0_running_mean_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_var", (getter)THPNativeBatchNormLegitBackward0_running_var_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_running_var", (getter)THPNativeBatchNormLegitBackward0_running_var_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_training", (getter)THPNativeBatchNormLegitBackward0_training_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPNativeBatchNormLegitBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPNativeBatchNormLegitBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPNativeBatchNormLegitBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPNativeBatchNormLegitBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPNativeBatchNormLegitBackward0_result2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result2", (getter)THPNativeBatchNormLegitBackward0_result2_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNativeBatchNormLegitBackward1_eps_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NativeBatchNormLegitBackward1*>(self->cdata.get())->eps;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitBackward1_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormLegitBackward1*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitBackward1_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormLegitBackward1*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitBackward1_training_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NativeBatchNormLegitBackward1*>(self->cdata.get())->training;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitBackward1_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormLegitBackward1*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitBackward1_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormLegitBackward1*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitBackward1_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormLegitBackward1*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitBackward1_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormLegitBackward1*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitBackward1_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormLegitBackward1*>(self->cdata.get())->result2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitBackward1_result2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormLegitBackward1*>(self->cdata.get())->result2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NativeBatchNormLegitBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_eps", (getter)THPNativeBatchNormLegitBackward1_eps_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPNativeBatchNormLegitBackward1_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPNativeBatchNormLegitBackward1_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_training", (getter)THPNativeBatchNormLegitBackward1_training_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPNativeBatchNormLegitBackward1_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPNativeBatchNormLegitBackward1_weight_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPNativeBatchNormLegitBackward1_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPNativeBatchNormLegitBackward1_result1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPNativeBatchNormLegitBackward1_result2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result2", (getter)THPNativeBatchNormLegitBackward1_result2_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNativeLayerNormBackwardBackward0_grad_out_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeLayerNormBackwardBackward0*>(self->cdata.get())->grad_out_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeLayerNormBackwardBackward0_grad_out_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeLayerNormBackwardBackward0*>(self->cdata.get())->grad_out_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeLayerNormBackwardBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeLayerNormBackwardBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeLayerNormBackwardBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeLayerNormBackwardBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeLayerNormBackwardBackward0_mean_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeLayerNormBackwardBackward0*>(self->cdata.get())->mean_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeLayerNormBackwardBackward0_mean_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeLayerNormBackwardBackward0*>(self->cdata.get())->mean_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeLayerNormBackwardBackward0_normalized_shape_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NativeLayerNormBackwardBackward0*>(self->cdata.get())->normalized_shape;
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

PyObject* THPNativeLayerNormBackwardBackward0_rstd_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeLayerNormBackwardBackward0*>(self->cdata.get())->rstd_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeLayerNormBackwardBackward0_rstd_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeLayerNormBackwardBackward0*>(self->cdata.get())->rstd_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeLayerNormBackwardBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeLayerNormBackwardBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeLayerNormBackwardBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeLayerNormBackwardBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NativeLayerNormBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_out", (getter)THPNativeLayerNormBackwardBackward0_grad_out_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_grad_out", (getter)THPNativeLayerNormBackwardBackward0_grad_out_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPNativeLayerNormBackwardBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPNativeLayerNormBackwardBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mean", (getter)THPNativeLayerNormBackwardBackward0_mean_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_mean", (getter)THPNativeLayerNormBackwardBackward0_mean_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_normalized_shape", (getter)THPNativeLayerNormBackwardBackward0_normalized_shape_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_rstd", (getter)THPNativeLayerNormBackwardBackward0_rstd_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_rstd", (getter)THPNativeLayerNormBackwardBackward0_rstd_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPNativeLayerNormBackwardBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPNativeLayerNormBackwardBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPBatchNormWithUpdateBackward0_eps_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<BatchNormWithUpdateBackward0*>(self->cdata.get())->eps;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormWithUpdateBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormWithUpdateBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormWithUpdateBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormWithUpdateBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormWithUpdateBackward0_running_mean_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormWithUpdateBackward0*>(self->cdata.get())->running_mean_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormWithUpdateBackward0_running_mean_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormWithUpdateBackward0*>(self->cdata.get())->running_mean_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormWithUpdateBackward0_running_var_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormWithUpdateBackward0*>(self->cdata.get())->running_var_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormWithUpdateBackward0_running_var_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormWithUpdateBackward0*>(self->cdata.get())->running_var_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormWithUpdateBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormWithUpdateBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormWithUpdateBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormWithUpdateBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormWithUpdateBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormWithUpdateBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormWithUpdateBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormWithUpdateBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormWithUpdateBackward0_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormWithUpdateBackward0*>(self->cdata.get())->result2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormWithUpdateBackward0_result2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormWithUpdateBackward0*>(self->cdata.get())->result2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormWithUpdateBackward0_result3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormWithUpdateBackward0*>(self->cdata.get())->result3_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormWithUpdateBackward0_result3_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormWithUpdateBackward0*>(self->cdata.get())->result3_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef BatchNormWithUpdateBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_eps", (getter)THPBatchNormWithUpdateBackward0_eps_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPBatchNormWithUpdateBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPBatchNormWithUpdateBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_mean", (getter)THPBatchNormWithUpdateBackward0_running_mean_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_running_mean", (getter)THPBatchNormWithUpdateBackward0_running_mean_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_var", (getter)THPBatchNormWithUpdateBackward0_running_var_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_running_var", (getter)THPBatchNormWithUpdateBackward0_running_var_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPBatchNormWithUpdateBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPBatchNormWithUpdateBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPBatchNormWithUpdateBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPBatchNormWithUpdateBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPBatchNormWithUpdateBackward0_result2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result2", (getter)THPBatchNormWithUpdateBackward0_result2_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result3", (getter)THPBatchNormWithUpdateBackward0_result3_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result3", (getter)THPBatchNormWithUpdateBackward0_result3_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNormBackward0_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NormBackward0*>(self->cdata.get())->p;
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

PyObject* THPNormBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NormBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNormBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NormBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNormBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NormBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNormBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NormBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NormBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_p", (getter)THPNormBackward0_p_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPNormBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPNormBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPNormBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPNormBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNormBackward1_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NormBackward1*>(self->cdata.get())->dim;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPNormBackward1_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NormBackward1*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNormBackward1_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<NormBackward1*>(self->cdata.get())->p;
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

PyObject* THPNormBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NormBackward1*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNormBackward1_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NormBackward1*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNormBackward1_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NormBackward1*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNormBackward1_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NormBackward1*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NormBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPNormBackward1_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPNormBackward1_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_p", (getter)THPNormBackward1_p_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPNormBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPNormBackward1_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPNormBackward1_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPNormBackward1_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNormBackward2_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<NormBackward2*>(self->cdata.get())->p;
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

PyObject* THPNormBackward2_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NormBackward2*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNormBackward2_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NormBackward2*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNormBackward2_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NormBackward2*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNormBackward2_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NormBackward2*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NormBackward2_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_p", (getter)THPNormBackward2_p_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPNormBackward2_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPNormBackward2_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPNormBackward2_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPNormBackward2_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNormBackward3_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NormBackward3*>(self->cdata.get())->dim;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPNormBackward3_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NormBackward3*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNormBackward3_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<NormBackward3*>(self->cdata.get())->p;
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

PyObject* THPNormBackward3_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NormBackward3*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNormBackward3_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NormBackward3*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNormBackward3_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NormBackward3*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNormBackward3_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NormBackward3*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NormBackward3_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPNormBackward3_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPNormBackward3_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_p", (getter)THPNormBackward3_p_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPNormBackward3_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPNormBackward3_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPNormBackward3_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPNormBackward3_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPPdistBackward0_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<PdistBackward0*>(self->cdata.get())->p;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPPdistBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PdistBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPPdistBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PdistBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPPdistBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PdistBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPPdistBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PdistBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef PdistBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_p", (getter)THPPdistBackward0_p_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPPdistBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPPdistBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPPdistBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPPdistBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPEuclideanDistBackward0_x1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EuclideanDistBackward0*>(self->cdata.get())->x1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEuclideanDistBackward0_x1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EuclideanDistBackward0*>(self->cdata.get())->x1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPEuclideanDistBackward0_x2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EuclideanDistBackward0*>(self->cdata.get())->x2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEuclideanDistBackward0_x2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EuclideanDistBackward0*>(self->cdata.get())->x2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPEuclideanDistBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EuclideanDistBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEuclideanDistBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EuclideanDistBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef EuclideanDistBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_x1", (getter)THPEuclideanDistBackward0_x1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_x1", (getter)THPEuclideanDistBackward0_x1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_x2", (getter)THPEuclideanDistBackward0_x2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_x2", (getter)THPEuclideanDistBackward0_x2_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPEuclideanDistBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPEuclideanDistBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef CdistBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPLinalgHouseholderProductBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgHouseholderProductBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgHouseholderProductBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgHouseholderProductBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgHouseholderProductBackward0_tau_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgHouseholderProductBackward0*>(self->cdata.get())->tau_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgHouseholderProductBackward0_tau_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgHouseholderProductBackward0*>(self->cdata.get())->tau_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgHouseholderProductBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgHouseholderProductBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgHouseholderProductBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgHouseholderProductBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LinalgHouseholderProductBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_input", (getter)THPLinalgHouseholderProductBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPLinalgHouseholderProductBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_tau", (getter)THPLinalgHouseholderProductBackward0_tau_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_tau", (getter)THPLinalgHouseholderProductBackward0_tau_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPLinalgHouseholderProductBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPLinalgHouseholderProductBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPPutBackward0_accumulate_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<PutBackward0*>(self->cdata.get())->accumulate;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPPutBackward0_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PutBackward0*>(self->cdata.get())->index_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPPutBackward0_index_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PutBackward0*>(self->cdata.get())->index_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPPutBackward0_source_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PutBackward0*>(self->cdata.get())->source_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPPutBackward0_source_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PutBackward0*>(self->cdata.get())->source_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef PutBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_accumulate", (getter)THPPutBackward0_accumulate_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_index", (getter)THPPutBackward0_index_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_index", (getter)THPPutBackward0_index_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_source", (getter)THPPutBackward0_source_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_source", (getter)THPPutBackward0_source_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef Rad2DegBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef RandomBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef RandomBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef RandomBackward2_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPRepeatBackward0_repeats_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RepeatBackward0*>(self->cdata.get())->repeats;
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

PyObject* THPRepeatBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RepeatBackward0*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef RepeatBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_repeats", (getter)THPRepeatBackward0_repeats_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPRepeatBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPRsqrtBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RsqrtBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPRsqrtBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RsqrtBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef RsqrtBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_result", (getter)THPRsqrtBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPRsqrtBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPScatterBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ScatterBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPScatterBackward0_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScatterBackward0*>(self->cdata.get())->index_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScatterBackward0_index_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScatterBackward0*>(self->cdata.get())->index_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ScatterBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPScatterBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_index", (getter)THPScatterBackward0_index_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_index", (getter)THPScatterBackward0_index_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPScatterBackward1_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ScatterBackward1*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPScatterBackward1_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScatterBackward1*>(self->cdata.get())->index_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScatterBackward1_index_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScatterBackward1*>(self->cdata.get())->index_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ScatterBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPScatterBackward1_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_index", (getter)THPScatterBackward1_index_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_index", (getter)THPScatterBackward1_index_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSortBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SortBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSortBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SortBackward0*>(self->cdata.get())->self_sym_sizes;
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

PyObject* THPSortBackward0_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SortBackward0*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSortBackward0_indices_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SortBackward0*>(self->cdata.get())->indices_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SortBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSortBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPSortBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_indices", (getter)THPSortBackward0_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_indices", (getter)THPSortBackward0_indices_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSortBackward1_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SortBackward1*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSortBackward1_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SortBackward1*>(self->cdata.get())->self_sym_sizes;
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

PyObject* THPSortBackward1_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SortBackward1*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSortBackward1_indices_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SortBackward1*>(self->cdata.get())->indices_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SortBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSortBackward1_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPSortBackward1_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_indices", (getter)THPSortBackward1_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_indices", (getter)THPSortBackward1_indices_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSqueezeBackward3_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SqueezeBackward3*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef SqueezeBackward3_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sym_sizes", (getter)THPSqueezeBackward3_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSqueezeBackward4_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SqueezeBackward4*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSqueezeBackward4_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SqueezeBackward4*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef SqueezeBackward4_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSqueezeBackward4_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPSqueezeBackward4_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSqueezeBackward5_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SqueezeBackward5*>(self->cdata.get())->dim;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSqueezeBackward5_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SqueezeBackward5*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef SqueezeBackward5_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSqueezeBackward5_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPSqueezeBackward5_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSumBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SumBackward0*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef SumBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sym_sizes", (getter)THPSumBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSumBackwardAutogradNestedTensor0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SumBackwardAutogradNestedTensor0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSumBackwardAutogradNestedTensor0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SumBackwardAutogradNestedTensor0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SumBackwardAutogradNestedTensor0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSumBackwardAutogradNestedTensor0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSumBackwardAutogradNestedTensor0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSumBackward1_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<SumBackward1*>(self->cdata.get())->dim;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSumBackward1_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SumBackward1*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPSumBackward1_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SumBackward1*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef SumBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSumBackward1_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPSumBackward1_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPSumBackward1_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSumBackwardAutogradNestedTensor1_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<SumBackwardAutogradNestedTensor1*>(self->cdata.get())->dim;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSumBackwardAutogradNestedTensor1_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SumBackwardAutogradNestedTensor1*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPSumBackwardAutogradNestedTensor1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SumBackwardAutogradNestedTensor1*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSumBackwardAutogradNestedTensor1_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SumBackwardAutogradNestedTensor1*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SumBackwardAutogradNestedTensor1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSumBackwardAutogradNestedTensor1_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPSumBackwardAutogradNestedTensor1_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSumBackwardAutogradNestedTensor1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSumBackwardAutogradNestedTensor1_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNansumBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<NansumBackward0*>(self->cdata.get())->dim;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPNansumBackward0_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NansumBackward0*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNansumBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NansumBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNansumBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NansumBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NansumBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPNansumBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPNansumBackward0_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPNansumBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPNansumBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLinalgEigBackward0_eigenvalues_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgEigBackward0*>(self->cdata.get())->eigenvalues_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgEigBackward0_eigenvalues_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgEigBackward0*>(self->cdata.get())->eigenvalues_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgEigBackward0_eigenvectors_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgEigBackward0*>(self->cdata.get())->eigenvectors_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgEigBackward0_eigenvectors_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgEigBackward0*>(self->cdata.get())->eigenvectors_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LinalgEigBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_eigenvalues", (getter)THPLinalgEigBackward0_eigenvalues_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_eigenvalues", (getter)THPLinalgEigBackward0_eigenvalues_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_eigenvectors", (getter)THPLinalgEigBackward0_eigenvectors_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_eigenvectors", (getter)THPLinalgEigBackward0_eigenvectors_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPRot90Backward0_dims_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Rot90Backward0*>(self->cdata.get())->dims;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPRot90Backward0_k_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Rot90Backward0*>(self->cdata.get())->k;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef Rot90Backward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dims", (getter)THPRot90Backward0_dims_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_k", (getter)THPRot90Backward0_k_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPTrilBackward0_diagonal_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TrilBackward0*>(self->cdata.get())->diagonal;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef TrilBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_diagonal", (getter)THPTrilBackward0_diagonal_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef TruncBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPToDenseBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ToDenseBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPToDenseBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ToDenseBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ToDenseBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPToDenseBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPToDenseBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPVarBackward0_correction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<VarBackward0*>(self->cdata.get())->correction;
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

PyObject* THPVarBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<VarBackward0*>(self->cdata.get())->dim;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPVarBackward0_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<VarBackward0*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPVarBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<VarBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPVarBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<VarBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef VarBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_correction", (getter)THPVarBackward0_correction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPVarBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPVarBackward0_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPVarBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPVarBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPVarMeanBackward0_correction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<VarMeanBackward0*>(self->cdata.get())->correction;
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

PyObject* THPVarMeanBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<VarMeanBackward0*>(self->cdata.get())->dim;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPVarMeanBackward0_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<VarMeanBackward0*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPVarMeanBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<VarMeanBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPVarMeanBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<VarMeanBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef VarMeanBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_correction", (getter)THPVarMeanBackward0_correction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPVarMeanBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPVarMeanBackward0_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPVarMeanBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPVarMeanBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSparseCooTensorWithDimsAndTensorsBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseCooTensorWithDimsAndTensorsBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseCooTensorWithDimsAndTensorsBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseCooTensorWithDimsAndTensorsBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SparseCooTensorWithDimsAndTensorsBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_result", (getter)THPSparseCooTensorWithDimsAndTensorsBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPSparseCooTensorWithDimsAndTensorsBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef StandardGammaGradBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPBinaryCrossEntropyWithLogitsBackward0_pos_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyWithLogitsBackward0*>(self->cdata.get())->pos_weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyWithLogitsBackward0_pos_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyWithLogitsBackward0*>(self->cdata.get())->pos_weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyWithLogitsBackward0_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<BinaryCrossEntropyWithLogitsBackward0*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyWithLogitsBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyWithLogitsBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyWithLogitsBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyWithLogitsBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyWithLogitsBackward0_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyWithLogitsBackward0*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyWithLogitsBackward0_target_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyWithLogitsBackward0*>(self->cdata.get())->target_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyWithLogitsBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyWithLogitsBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyWithLogitsBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyWithLogitsBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef BinaryCrossEntropyWithLogitsBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_pos_weight", (getter)THPBinaryCrossEntropyWithLogitsBackward0_pos_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_pos_weight", (getter)THPBinaryCrossEntropyWithLogitsBackward0_pos_weight_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPBinaryCrossEntropyWithLogitsBackward0_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPBinaryCrossEntropyWithLogitsBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPBinaryCrossEntropyWithLogitsBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPBinaryCrossEntropyWithLogitsBackward0_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_target", (getter)THPBinaryCrossEntropyWithLogitsBackward0_target_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPBinaryCrossEntropyWithLogitsBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPBinaryCrossEntropyWithLogitsBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMultiMarginLossBackward0_margin_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MultiMarginLossBackward0*>(self->cdata.get())->margin;
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

PyObject* THPMultiMarginLossBackward0_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MultiMarginLossBackward0*>(self->cdata.get())->p;
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

PyObject* THPMultiMarginLossBackward0_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MultiMarginLossBackward0*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMultiMarginLossBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MultiMarginLossBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMultiMarginLossBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MultiMarginLossBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMultiMarginLossBackward0_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MultiMarginLossBackward0*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMultiMarginLossBackward0_target_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MultiMarginLossBackward0*>(self->cdata.get())->target_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMultiMarginLossBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MultiMarginLossBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMultiMarginLossBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MultiMarginLossBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MultiMarginLossBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_margin", (getter)THPMultiMarginLossBackward0_margin_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_p", (getter)THPMultiMarginLossBackward0_p_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPMultiMarginLossBackward0_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPMultiMarginLossBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPMultiMarginLossBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPMultiMarginLossBackward0_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_target", (getter)THPMultiMarginLossBackward0_target_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPMultiMarginLossBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPMultiMarginLossBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMultilabelMarginLossBackward0_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MultilabelMarginLossBackward0*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMultilabelMarginLossBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MultilabelMarginLossBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMultilabelMarginLossBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MultilabelMarginLossBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMultilabelMarginLossBackward0_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MultilabelMarginLossBackward0*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMultilabelMarginLossBackward0_target_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MultilabelMarginLossBackward0*>(self->cdata.get())->target_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMultilabelMarginLossBackward0_is_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MultilabelMarginLossBackward0*>(self->cdata.get())->is_target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMultilabelMarginLossBackward0_is_target_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MultilabelMarginLossBackward0*>(self->cdata.get())->is_target_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MultilabelMarginLossBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_reduction", (getter)THPMultilabelMarginLossBackward0_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPMultilabelMarginLossBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPMultilabelMarginLossBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPMultilabelMarginLossBackward0_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_target", (getter)THPMultilabelMarginLossBackward0_target_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_is_target", (getter)THPMultilabelMarginLossBackward0_is_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_is_target", (getter)THPMultilabelMarginLossBackward0_is_target_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNllLossBackward0_ignore_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NllLossBackward0*>(self->cdata.get())->ignore_index;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLossBackward0_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NllLossBackward0*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLossBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLossBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLossBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLossBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLossBackward0_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLossBackward0*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLossBackward0_target_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLossBackward0*>(self->cdata.get())->target_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLossBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLossBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLossBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLossBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLossBackward0_total_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLossBackward0*>(self->cdata.get())->total_weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLossBackward0_total_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLossBackward0*>(self->cdata.get())->total_weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NllLossBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_ignore_index", (getter)THPNllLossBackward0_ignore_index_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPNllLossBackward0_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPNllLossBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPNllLossBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPNllLossBackward0_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_target", (getter)THPNllLossBackward0_target_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPNllLossBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPNllLossBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_total_weight", (getter)THPNllLossBackward0_total_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_total_weight", (getter)THPNllLossBackward0_total_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPPreluKernelBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PreluKernelBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPPreluKernelBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PreluKernelBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPPreluKernelBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PreluKernelBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPPreluKernelBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PreluKernelBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef PreluKernelBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPPreluKernelBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPPreluKernelBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPPreluKernelBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPPreluKernelBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPRreluWithNoiseBackward0_lower_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RreluWithNoiseBackward0*>(self->cdata.get())->lower;
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

PyObject* THPRreluWithNoiseBackward0_noise_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RreluWithNoiseBackward0*>(self->cdata.get())->noise_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPRreluWithNoiseBackward0_noise_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RreluWithNoiseBackward0*>(self->cdata.get())->noise_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPRreluWithNoiseBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RreluWithNoiseBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPRreluWithNoiseBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RreluWithNoiseBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPRreluWithNoiseBackward0_training_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RreluWithNoiseBackward0*>(self->cdata.get())->training;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPRreluWithNoiseBackward0_upper_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RreluWithNoiseBackward0*>(self->cdata.get())->upper;
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

static struct PyGetSetDef RreluWithNoiseBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_lower", (getter)THPRreluWithNoiseBackward0_lower_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_noise", (getter)THPRreluWithNoiseBackward0_noise_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_noise", (getter)THPRreluWithNoiseBackward0_noise_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPRreluWithNoiseBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPRreluWithNoiseBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_training", (getter)THPRreluWithNoiseBackward0_training_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_upper", (getter)THPRreluWithNoiseBackward0_upper_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSparseSoftmaxBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SparseSoftmaxBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseSoftmaxBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseSoftmaxBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseSoftmaxBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseSoftmaxBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseSoftmaxBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseSoftmaxBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseSoftmaxBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseSoftmaxBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SparseSoftmaxBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSparseSoftmaxBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSparseSoftmaxBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSparseSoftmaxBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPSparseSoftmaxBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPSparseSoftmaxBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSparseSparseMatmulBackward0_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseSparseMatmulBackward0*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseSparseMatmulBackward0_other_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseSparseMatmulBackward0*>(self->cdata.get())->other_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseSparseMatmulBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseSparseMatmulBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseSparseMatmulBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseSparseMatmulBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SparseSparseMatmulBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPSparseSparseMatmulBackward0_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_other", (getter)THPSparseSparseMatmulBackward0_other_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSparseSparseMatmulBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSparseSparseMatmulBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPThresholdBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThresholdBackward1*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThresholdBackward1_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThresholdBackward1*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPThresholdBackward1_threshold_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ThresholdBackward1*>(self->cdata.get())->threshold;
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

static struct PyGetSetDef ThresholdBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPThresholdBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPThresholdBackward1_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_threshold", (getter)THPThresholdBackward1_threshold_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleNearestExact2DBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleNearestExact2DBackward0*>(self->cdata.get())->output_size;
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

PyObject* THPUpsampleNearestExact2DBackward0_scales_h_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearestExact2DBackward0*>(self->cdata.get())->scales_h;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearestExact2DBackward0_scales_w_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearestExact2DBackward0*>(self->cdata.get())->scales_w;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearestExact2DBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleNearestExact2DBackward0*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef UpsampleNearestExact2DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_output_size", (getter)THPUpsampleNearestExact2DBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_h", (getter)THPUpsampleNearestExact2DBackward0_scales_h_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_w", (getter)THPUpsampleNearestExact2DBackward0_scales_w_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPUpsampleNearestExact2DBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleNearestExact3DBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleNearestExact3DBackward0*>(self->cdata.get())->output_size;
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

PyObject* THPUpsampleNearestExact3DBackward0_scales_d_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearestExact3DBackward0*>(self->cdata.get())->scales_d;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearestExact3DBackward0_scales_h_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearestExact3DBackward0*>(self->cdata.get())->scales_h;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearestExact3DBackward0_scales_w_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearestExact3DBackward0*>(self->cdata.get())->scales_w;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearestExact3DBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleNearestExact3DBackward0*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef UpsampleNearestExact3DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_output_size", (getter)THPUpsampleNearestExact3DBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_d", (getter)THPUpsampleNearestExact3DBackward0_scales_d_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_h", (getter)THPUpsampleNearestExact3DBackward0_scales_h_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_w", (getter)THPUpsampleNearestExact3DBackward0_scales_w_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPUpsampleNearestExact3DBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAvgPool2DBackward0_ceil_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool2DBackward0*>(self->cdata.get())->ceil_mode;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool2DBackward0_count_include_pad_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool2DBackward0*>(self->cdata.get())->count_include_pad;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool2DBackward0_divisor_override_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<AvgPool2DBackward0*>(self->cdata.get())->divisor_override;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool2DBackward0_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool2DBackward0*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool2DBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool2DBackward0*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool2DBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AvgPool2DBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool2DBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AvgPool2DBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool2DBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool2DBackward0*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AvgPool2DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_ceil_mode", (getter)THPAvgPool2DBackward0_ceil_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_count_include_pad", (getter)THPAvgPool2DBackward0_count_include_pad_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_divisor_override", (getter)THPAvgPool2DBackward0_divisor_override_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_kernel_size", (getter)THPAvgPool2DBackward0_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPAvgPool2DBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPAvgPool2DBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPAvgPool2DBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPAvgPool2DBackward0_stride_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMpsConvolutionBackwardBackward0_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MpsConvolutionBackwardBackward0*>(self->cdata.get())->dilation;
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

PyObject* THPMpsConvolutionBackwardBackward0_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MpsConvolutionBackwardBackward0*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMpsConvolutionBackwardBackward0_grad_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MpsConvolutionBackwardBackward0*>(self->cdata.get())->grad_output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMpsConvolutionBackwardBackward0_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MpsConvolutionBackwardBackward0*>(self->cdata.get())->groups;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMpsConvolutionBackwardBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MpsConvolutionBackwardBackward0*>(self->cdata.get())->padding;
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

PyObject* THPMpsConvolutionBackwardBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MpsConvolutionBackwardBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMpsConvolutionBackwardBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MpsConvolutionBackwardBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMpsConvolutionBackwardBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MpsConvolutionBackwardBackward0*>(self->cdata.get())->stride;
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

PyObject* THPMpsConvolutionBackwardBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MpsConvolutionBackwardBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMpsConvolutionBackwardBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MpsConvolutionBackwardBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MpsConvolutionBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dilation", (getter)THPMpsConvolutionBackwardBackward0_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output", (getter)THPMpsConvolutionBackwardBackward0_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_grad_output", (getter)THPMpsConvolutionBackwardBackward0_grad_output_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_groups", (getter)THPMpsConvolutionBackwardBackward0_groups_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPMpsConvolutionBackwardBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPMpsConvolutionBackwardBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPMpsConvolutionBackwardBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPMpsConvolutionBackwardBackward0_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPMpsConvolutionBackwardBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPMpsConvolutionBackwardBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSlowConvDilated3DBackward0_bias_sym_sizes_opt_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<SlowConvDilated3DBackward0*>(self->cdata.get())->bias_sym_sizes_opt;
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

PyObject* THPSlowConvDilated3DBackward0_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvDilated3DBackward0*>(self->cdata.get())->dilation;
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

PyObject* THPSlowConvDilated3DBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvDilated3DBackward0*>(self->cdata.get())->padding;
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

PyObject* THPSlowConvDilated3DBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvDilated3DBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvDilated3DBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvDilated3DBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvDilated3DBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvDilated3DBackward0*>(self->cdata.get())->stride;
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

PyObject* THPSlowConvDilated3DBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvDilated3DBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvDilated3DBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvDilated3DBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SlowConvDilated3DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_bias_sym_sizes_opt", (getter)THPSlowConvDilated3DBackward0_bias_sym_sizes_opt_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPSlowConvDilated3DBackward0_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPSlowConvDilated3DBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSlowConvDilated3DBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSlowConvDilated3DBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPSlowConvDilated3DBackward0_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPSlowConvDilated3DBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPSlowConvDilated3DBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAdaptiveAvgPool2DBackwardBackward0_grad_output_sym_argsize_minus_1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AdaptiveAvgPool2DBackwardBackward0*>(self->cdata.get())->grad_output_sym_argsize_minus_1;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAdaptiveAvgPool2DBackwardBackward0_grad_output_sym_argsize_minus_2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AdaptiveAvgPool2DBackwardBackward0*>(self->cdata.get())->grad_output_sym_argsize_minus_2;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AdaptiveAvgPool2DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_output_sym_argsize_minus_1", (getter)THPAdaptiveAvgPool2DBackwardBackward0_grad_output_sym_argsize_minus_1_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output_sym_argsize_minus_2", (getter)THPAdaptiveAvgPool2DBackwardBackward0_grad_output_sym_argsize_minus_2_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPFractionalMaxPool2DBackwardBackward0_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FractionalMaxPool2DBackwardBackward0*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFractionalMaxPool2DBackwardBackward0_indices_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FractionalMaxPool2DBackwardBackward0*>(self->cdata.get())->indices_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FractionalMaxPool2DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_indices", (getter)THPFractionalMaxPool2DBackwardBackward0_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_indices", (getter)THPFractionalMaxPool2DBackwardBackward0_indices_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMaxPool3DWithIndicesBackwardBackward0_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxPool3DWithIndicesBackwardBackward0*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool3DWithIndicesBackwardBackward0_indices_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxPool3DWithIndicesBackwardBackward0*>(self->cdata.get())->indices_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MaxPool3DWithIndicesBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_indices", (getter)THPMaxPool3DWithIndicesBackwardBackward0_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_indices", (getter)THPMaxPool3DWithIndicesBackwardBackward0_indices_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPReflectionPad1DBackwardBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ReflectionPad1DBackwardBackward0*>(self->cdata.get())->padding;
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

static struct PyGetSetDef ReflectionPad1DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_padding", (getter)THPReflectionPad1DBackwardBackward0_padding_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPReflectionPad2DBackwardBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ReflectionPad2DBackwardBackward0*>(self->cdata.get())->padding;
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

static struct PyGetSetDef ReflectionPad2DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_padding", (getter)THPReflectionPad2DBackwardBackward0_padding_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPReplicationPad1DBackwardBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ReplicationPad1DBackwardBackward0*>(self->cdata.get())->padding;
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

static struct PyGetSetDef ReplicationPad1DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_padding", (getter)THPReplicationPad1DBackwardBackward0_padding_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPReplicationPad3DBackwardBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ReplicationPad3DBackwardBackward0*>(self->cdata.get())->padding;
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

static struct PyGetSetDef ReplicationPad3DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_padding", (getter)THPReplicationPad3DBackwardBackward0_padding_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSoftMarginLossBackwardBackward0_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftMarginLossBackwardBackward0*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftMarginLossBackwardBackward0_grad_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftMarginLossBackwardBackward0*>(self->cdata.get())->grad_output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftMarginLossBackwardBackward0_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SoftMarginLossBackwardBackward0*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftMarginLossBackwardBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftMarginLossBackwardBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftMarginLossBackwardBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftMarginLossBackwardBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftMarginLossBackwardBackward0_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftMarginLossBackwardBackward0*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftMarginLossBackwardBackward0_target_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftMarginLossBackwardBackward0*>(self->cdata.get())->target_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SoftMarginLossBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_output", (getter)THPSoftMarginLossBackwardBackward0_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_grad_output", (getter)THPSoftMarginLossBackwardBackward0_grad_output_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPSoftMarginLossBackwardBackward0_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSoftMarginLossBackwardBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSoftMarginLossBackwardBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPSoftMarginLossBackwardBackward0_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_target", (getter)THPSoftMarginLossBackwardBackward0_target_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSoftshrinkBackwardBackward0_lambd_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SoftshrinkBackwardBackward0*>(self->cdata.get())->lambd;
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

PyObject* THPSoftshrinkBackwardBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftshrinkBackwardBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftshrinkBackwardBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftshrinkBackwardBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SoftshrinkBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_lambd", (getter)THPSoftshrinkBackwardBackward0_lambd_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSoftshrinkBackwardBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSoftshrinkBackwardBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleBilinear2DBackwardBackward0_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBilinear2DBackwardBackward0*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBilinear2DBackwardBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBilinear2DBackwardBackward0*>(self->cdata.get())->output_size;
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

PyObject* THPUpsampleBilinear2DBackwardBackward0_scales_h_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleBilinear2DBackwardBackward0*>(self->cdata.get())->scales_h;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBilinear2DBackwardBackward0_scales_w_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleBilinear2DBackwardBackward0*>(self->cdata.get())->scales_w;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleBilinear2DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_align_corners", (getter)THPUpsampleBilinear2DBackwardBackward0_align_corners_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPUpsampleBilinear2DBackwardBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_h", (getter)THPUpsampleBilinear2DBackwardBackward0_scales_h_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_w", (getter)THPUpsampleBilinear2DBackwardBackward0_scales_w_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleBicubic2DAaBackwardBackward0_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBicubic2DAaBackwardBackward0*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBicubic2DAaBackwardBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBicubic2DAaBackwardBackward0*>(self->cdata.get())->output_size;
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

PyObject* THPUpsampleBicubic2DAaBackwardBackward0_scales_h_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleBicubic2DAaBackwardBackward0*>(self->cdata.get())->scales_h;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBicubic2DAaBackwardBackward0_scales_w_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleBicubic2DAaBackwardBackward0*>(self->cdata.get())->scales_w;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleBicubic2DAaBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_align_corners", (getter)THPUpsampleBicubic2DAaBackwardBackward0_align_corners_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPUpsampleBicubic2DAaBackwardBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_h", (getter)THPUpsampleBicubic2DAaBackwardBackward0_scales_h_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_w", (getter)THPUpsampleBicubic2DAaBackwardBackward0_scales_w_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleNearestExact1DBackwardBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleNearestExact1DBackwardBackward0*>(self->cdata.get())->output_size;
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

PyObject* THPUpsampleNearestExact1DBackwardBackward0_scales_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearestExact1DBackwardBackward0*>(self->cdata.get())->scales;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleNearestExact1DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_output_size", (getter)THPUpsampleNearestExact1DBackwardBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales", (getter)THPUpsampleNearestExact1DBackwardBackward0_scales_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleNearestExact2DBackwardBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleNearestExact2DBackwardBackward0*>(self->cdata.get())->output_size;
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

PyObject* THPUpsampleNearestExact2DBackwardBackward0_scales_h_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearestExact2DBackwardBackward0*>(self->cdata.get())->scales_h;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearestExact2DBackwardBackward0_scales_w_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearestExact2DBackwardBackward0*>(self->cdata.get())->scales_w;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleNearestExact2DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_output_size", (getter)THPUpsampleNearestExact2DBackwardBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_h", (getter)THPUpsampleNearestExact2DBackwardBackward0_scales_h_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_w", (getter)THPUpsampleNearestExact2DBackwardBackward0_scales_w_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMiopenConvolutionBackward0_bias_sym_sizes_opt_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<MiopenConvolutionBackward0*>(self->cdata.get())->bias_sym_sizes_opt;
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

PyObject* THPMiopenConvolutionBackward0_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionBackward0*>(self->cdata.get())->dilation;
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

PyObject* THPMiopenConvolutionBackward0_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionBackward0*>(self->cdata.get())->groups;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionBackward0*>(self->cdata.get())->padding;
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

PyObject* THPMiopenConvolutionBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenConvolutionBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenConvolutionBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionBackward0*>(self->cdata.get())->stride;
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

PyObject* THPMiopenConvolutionBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenConvolutionBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenConvolutionBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MiopenConvolutionBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_bias_sym_sizes_opt", (getter)THPMiopenConvolutionBackward0_bias_sym_sizes_opt_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPMiopenConvolutionBackward0_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_groups", (getter)THPMiopenConvolutionBackward0_groups_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPMiopenConvolutionBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPMiopenConvolutionBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPMiopenConvolutionBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPMiopenConvolutionBackward0_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPMiopenConvolutionBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPMiopenConvolutionBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMiopenRnnBackward0_batch_first_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenRnnBackward0*>(self->cdata.get())->batch_first;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward0_batch_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenRnnBackward0*>(self->cdata.get())->batch_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward0_bidirectional_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenRnnBackward0*>(self->cdata.get())->bidirectional;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward0_cx_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenRnnBackward0*>(self->cdata.get())->cx_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward0_cx_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenRnnBackward0*>(self->cdata.get())->cx_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward0_dropout_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenRnnBackward0*>(self->cdata.get())->dropout;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward0_dropout_state_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenRnnBackward0*>(self->cdata.get())->dropout_state_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward0_dropout_state_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenRnnBackward0*>(self->cdata.get())->dropout_state_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward0_hidden_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenRnnBackward0*>(self->cdata.get())->hidden_size;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward0_hx_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenRnnBackward0*>(self->cdata.get())->hx_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward0_hx_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenRnnBackward0*>(self->cdata.get())->hx_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenRnnBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenRnnBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward0_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenRnnBackward0*>(self->cdata.get())->mode;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward0_num_layers_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenRnnBackward0*>(self->cdata.get())->num_layers;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward0_train_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenRnnBackward0*>(self->cdata.get())->train;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<MiopenRnnBackward0*>(self->cdata.get());
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

PyObject* THPMiopenRnnBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<MiopenRnnBackward0*>(self->cdata.get());
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

PyObject* THPMiopenRnnBackward0_weight_stride0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenRnnBackward0*>(self->cdata.get())->weight_stride0;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward0_result0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenRnnBackward0*>(self->cdata.get())->result0_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward0_result0_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenRnnBackward0*>(self->cdata.get())->result0_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward0_result3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenRnnBackward0*>(self->cdata.get())->result3_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward0_result3_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenRnnBackward0*>(self->cdata.get())->result3_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward0_result4_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenRnnBackward0*>(self->cdata.get())->result4_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward0_result4_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenRnnBackward0*>(self->cdata.get())->result4_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MiopenRnnBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_batch_first", (getter)THPMiopenRnnBackward0_batch_first_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_batch_sizes", (getter)THPMiopenRnnBackward0_batch_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_bidirectional", (getter)THPMiopenRnnBackward0_bidirectional_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_cx", (getter)THPMiopenRnnBackward0_cx_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_cx", (getter)THPMiopenRnnBackward0_cx_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dropout", (getter)THPMiopenRnnBackward0_dropout_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dropout_state", (getter)THPMiopenRnnBackward0_dropout_state_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_dropout_state", (getter)THPMiopenRnnBackward0_dropout_state_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_hidden_size", (getter)THPMiopenRnnBackward0_hidden_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_hx", (getter)THPMiopenRnnBackward0_hx_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_hx", (getter)THPMiopenRnnBackward0_hx_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPMiopenRnnBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPMiopenRnnBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mode", (getter)THPMiopenRnnBackward0_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_num_layers", (getter)THPMiopenRnnBackward0_num_layers_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_train", (getter)THPMiopenRnnBackward0_train_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPMiopenRnnBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPMiopenRnnBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight_stride0", (getter)THPMiopenRnnBackward0_weight_stride0_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result0", (getter)THPMiopenRnnBackward0_result0_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result0", (getter)THPMiopenRnnBackward0_result0_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result3", (getter)THPMiopenRnnBackward0_result3_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result3", (getter)THPMiopenRnnBackward0_result3_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result4", (getter)THPMiopenRnnBackward0_result4_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result4", (getter)THPMiopenRnnBackward0_result4_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMkldnnRnnLayerBackward0_batch_first_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnRnnLayerBackward0*>(self->cdata.get())->batch_first;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnRnnLayerBackward0_batch_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnRnnLayerBackward0*>(self->cdata.get())->batch_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnRnnLayerBackward0_bidirectional_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnRnnLayerBackward0*>(self->cdata.get())->bidirectional;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnRnnLayerBackward0_cx__getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnRnnLayerBackward0*>(self->cdata.get())->cx__;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnRnnLayerBackward0_cx__raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnRnnLayerBackward0*>(self->cdata.get())->cx__;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnRnnLayerBackward0_has_biases_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnRnnLayerBackward0*>(self->cdata.get())->has_biases;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnRnnLayerBackward0_hidden_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnRnnLayerBackward0*>(self->cdata.get())->hidden_size;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnRnnLayerBackward0_hx__getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnRnnLayerBackward0*>(self->cdata.get())->hx__;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnRnnLayerBackward0_hx__raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnRnnLayerBackward0*>(self->cdata.get())->hx__;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnRnnLayerBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnRnnLayerBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnRnnLayerBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnRnnLayerBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnRnnLayerBackward0_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnRnnLayerBackward0*>(self->cdata.get())->mode;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnRnnLayerBackward0_num_layers_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnRnnLayerBackward0*>(self->cdata.get())->num_layers;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnRnnLayerBackward0_reverse_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnRnnLayerBackward0*>(self->cdata.get())->reverse;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnRnnLayerBackward0_train_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnRnnLayerBackward0*>(self->cdata.get())->train;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnRnnLayerBackward0_weight0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnRnnLayerBackward0*>(self->cdata.get())->weight0_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnRnnLayerBackward0_weight0_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnRnnLayerBackward0*>(self->cdata.get())->weight0_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnRnnLayerBackward0_weight1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnRnnLayerBackward0*>(self->cdata.get())->weight1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnRnnLayerBackward0_weight1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnRnnLayerBackward0*>(self->cdata.get())->weight1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnRnnLayerBackward0_weight2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnRnnLayerBackward0*>(self->cdata.get())->weight2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnRnnLayerBackward0_weight2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnRnnLayerBackward0*>(self->cdata.get())->weight2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnRnnLayerBackward0_weight3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnRnnLayerBackward0*>(self->cdata.get())->weight3_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnRnnLayerBackward0_weight3_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnRnnLayerBackward0*>(self->cdata.get())->weight3_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnRnnLayerBackward0_result0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnRnnLayerBackward0*>(self->cdata.get())->result0_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnRnnLayerBackward0_result0_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnRnnLayerBackward0*>(self->cdata.get())->result0_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnRnnLayerBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnRnnLayerBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnRnnLayerBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnRnnLayerBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnRnnLayerBackward0_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnRnnLayerBackward0*>(self->cdata.get())->result2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnRnnLayerBackward0_result2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnRnnLayerBackward0*>(self->cdata.get())->result2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnRnnLayerBackward0_result3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnRnnLayerBackward0*>(self->cdata.get())->result3_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnRnnLayerBackward0_result3_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnRnnLayerBackward0*>(self->cdata.get())->result3_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MkldnnRnnLayerBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_batch_first", (getter)THPMkldnnRnnLayerBackward0_batch_first_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_batch_sizes", (getter)THPMkldnnRnnLayerBackward0_batch_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_bidirectional", (getter)THPMkldnnRnnLayerBackward0_bidirectional_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_cx_", (getter)THPMkldnnRnnLayerBackward0_cx__getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_cx_", (getter)THPMkldnnRnnLayerBackward0_cx__raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_has_biases", (getter)THPMkldnnRnnLayerBackward0_has_biases_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_hidden_size", (getter)THPMkldnnRnnLayerBackward0_hidden_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_hx_", (getter)THPMkldnnRnnLayerBackward0_hx__getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_hx_", (getter)THPMkldnnRnnLayerBackward0_hx__raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPMkldnnRnnLayerBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPMkldnnRnnLayerBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mode", (getter)THPMkldnnRnnLayerBackward0_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_num_layers", (getter)THPMkldnnRnnLayerBackward0_num_layers_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reverse", (getter)THPMkldnnRnnLayerBackward0_reverse_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_train", (getter)THPMkldnnRnnLayerBackward0_train_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight0", (getter)THPMkldnnRnnLayerBackward0_weight0_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight0", (getter)THPMkldnnRnnLayerBackward0_weight0_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight1", (getter)THPMkldnnRnnLayerBackward0_weight1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight1", (getter)THPMkldnnRnnLayerBackward0_weight1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight2", (getter)THPMkldnnRnnLayerBackward0_weight2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight2", (getter)THPMkldnnRnnLayerBackward0_weight2_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight3", (getter)THPMkldnnRnnLayerBackward0_weight3_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight3", (getter)THPMkldnnRnnLayerBackward0_weight3_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result0", (getter)THPMkldnnRnnLayerBackward0_result0_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result0", (getter)THPMkldnnRnnLayerBackward0_result0_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPMkldnnRnnLayerBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPMkldnnRnnLayerBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPMkldnnRnnLayerBackward0_result2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result2", (getter)THPMkldnnRnnLayerBackward0_result2_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result3", (getter)THPMkldnnRnnLayerBackward0_result3_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result3", (getter)THPMkldnnRnnLayerBackward0_result3_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMkldnnMaxPool3DBackward0_ceil_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnMaxPool3DBackward0*>(self->cdata.get())->ceil_mode;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnMaxPool3DBackward0_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnMaxPool3DBackward0*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnMaxPool3DBackward0_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnMaxPool3DBackward0*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnMaxPool3DBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnMaxPool3DBackward0*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnMaxPool3DBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnMaxPool3DBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnMaxPool3DBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnMaxPool3DBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnMaxPool3DBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnMaxPool3DBackward0*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnMaxPool3DBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnMaxPool3DBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnMaxPool3DBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnMaxPool3DBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MkldnnMaxPool3DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_ceil_mode", (getter)THPMkldnnMaxPool3DBackward0_ceil_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPMkldnnMaxPool3DBackward0_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_kernel_size", (getter)THPMkldnnMaxPool3DBackward0_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPMkldnnMaxPool3DBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPMkldnnMaxPool3DBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPMkldnnMaxPool3DBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPMkldnnMaxPool3DBackward0_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPMkldnnMaxPool3DBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPMkldnnMaxPool3DBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMkldnnReshapeBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnReshapeBackward0*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef MkldnnReshapeBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sym_sizes", (getter)THPMkldnnReshapeBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNestedTensorFromTensorListBackward0_list_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<NestedTensorFromTensorListBackward0*>(self->cdata.get());
  const auto& prop = node->list_;
  if (node->list_released_) {
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

PyObject* THPNestedTensorFromTensorListBackward0_list_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<NestedTensorFromTensorListBackward0*>(self->cdata.get());
  const auto& prop = node->list_;
  if (node->list_released_) {
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

static struct PyGetSetDef NestedTensorFromTensorListBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_list", (getter)THPNestedTensorFromTensorListBackward0_list_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_list", (getter)THPNestedTensorFromTensorListBackward0_list_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPScaledDotProductCudnnAttentionBackward0_attn_bias_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductCudnnAttentionBackward0*>(self->cdata.get())->attn_bias_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductCudnnAttentionBackward0_attn_bias_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductCudnnAttentionBackward0*>(self->cdata.get())->attn_bias_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductCudnnAttentionBackward0_dropout_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ScaledDotProductCudnnAttentionBackward0*>(self->cdata.get())->dropout_p;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductCudnnAttentionBackward0_is_causal_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ScaledDotProductCudnnAttentionBackward0*>(self->cdata.get())->is_causal;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductCudnnAttentionBackward0_key_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductCudnnAttentionBackward0*>(self->cdata.get())->key_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductCudnnAttentionBackward0_key_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductCudnnAttentionBackward0*>(self->cdata.get())->key_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductCudnnAttentionBackward0_query_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductCudnnAttentionBackward0*>(self->cdata.get())->query_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductCudnnAttentionBackward0_query_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductCudnnAttentionBackward0*>(self->cdata.get())->query_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductCudnnAttentionBackward0_scale_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<ScaledDotProductCudnnAttentionBackward0*>(self->cdata.get())->scale;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductCudnnAttentionBackward0_value_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductCudnnAttentionBackward0*>(self->cdata.get())->value_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductCudnnAttentionBackward0_value_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductCudnnAttentionBackward0*>(self->cdata.get())->value_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductCudnnAttentionBackward0_cum_seq_k_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductCudnnAttentionBackward0*>(self->cdata.get())->cum_seq_k_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductCudnnAttentionBackward0_cum_seq_k_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductCudnnAttentionBackward0*>(self->cdata.get())->cum_seq_k_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductCudnnAttentionBackward0_cum_seq_q_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductCudnnAttentionBackward0*>(self->cdata.get())->cum_seq_q_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductCudnnAttentionBackward0_cum_seq_q_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductCudnnAttentionBackward0*>(self->cdata.get())->cum_seq_q_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductCudnnAttentionBackward0_logsumexp_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductCudnnAttentionBackward0*>(self->cdata.get())->logsumexp_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductCudnnAttentionBackward0_logsumexp_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductCudnnAttentionBackward0*>(self->cdata.get())->logsumexp_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductCudnnAttentionBackward0_max_k_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ScaledDotProductCudnnAttentionBackward0*>(self->cdata.get())->max_k;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductCudnnAttentionBackward0_max_q_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ScaledDotProductCudnnAttentionBackward0*>(self->cdata.get())->max_q;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductCudnnAttentionBackward0_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductCudnnAttentionBackward0*>(self->cdata.get())->output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductCudnnAttentionBackward0_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductCudnnAttentionBackward0*>(self->cdata.get())->output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductCudnnAttentionBackward0_philox_offset_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductCudnnAttentionBackward0*>(self->cdata.get())->philox_offset_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductCudnnAttentionBackward0_philox_offset_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductCudnnAttentionBackward0*>(self->cdata.get())->philox_offset_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductCudnnAttentionBackward0_philox_seed_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductCudnnAttentionBackward0*>(self->cdata.get())->philox_seed_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductCudnnAttentionBackward0_philox_seed_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductCudnnAttentionBackward0*>(self->cdata.get())->philox_seed_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ScaledDotProductCudnnAttentionBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_attn_bias", (getter)THPScaledDotProductCudnnAttentionBackward0_attn_bias_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_attn_bias", (getter)THPScaledDotProductCudnnAttentionBackward0_attn_bias_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dropout_p", (getter)THPScaledDotProductCudnnAttentionBackward0_dropout_p_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_is_causal", (getter)THPScaledDotProductCudnnAttentionBackward0_is_causal_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_key", (getter)THPScaledDotProductCudnnAttentionBackward0_key_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_key", (getter)THPScaledDotProductCudnnAttentionBackward0_key_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_query", (getter)THPScaledDotProductCudnnAttentionBackward0_query_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_query", (getter)THPScaledDotProductCudnnAttentionBackward0_query_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale", (getter)THPScaledDotProductCudnnAttentionBackward0_scale_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_value", (getter)THPScaledDotProductCudnnAttentionBackward0_value_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_value", (getter)THPScaledDotProductCudnnAttentionBackward0_value_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_cum_seq_k", (getter)THPScaledDotProductCudnnAttentionBackward0_cum_seq_k_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_cum_seq_k", (getter)THPScaledDotProductCudnnAttentionBackward0_cum_seq_k_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_cum_seq_q", (getter)THPScaledDotProductCudnnAttentionBackward0_cum_seq_q_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_cum_seq_q", (getter)THPScaledDotProductCudnnAttentionBackward0_cum_seq_q_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_logsumexp", (getter)THPScaledDotProductCudnnAttentionBackward0_logsumexp_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_logsumexp", (getter)THPScaledDotProductCudnnAttentionBackward0_logsumexp_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_max_k", (getter)THPScaledDotProductCudnnAttentionBackward0_max_k_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_max_q", (getter)THPScaledDotProductCudnnAttentionBackward0_max_q_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output", (getter)THPScaledDotProductCudnnAttentionBackward0_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_output", (getter)THPScaledDotProductCudnnAttentionBackward0_output_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_philox_offset", (getter)THPScaledDotProductCudnnAttentionBackward0_philox_offset_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_philox_offset", (getter)THPScaledDotProductCudnnAttentionBackward0_philox_offset_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_philox_seed", (getter)THPScaledDotProductCudnnAttentionBackward0_philox_seed_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_philox_seed", (getter)THPScaledDotProductCudnnAttentionBackward0_philox_seed_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPScatterReduceBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ScatterReduceBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPScatterReduceBackward0_include_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ScatterReduceBackward0*>(self->cdata.get())->include_self;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPScatterReduceBackward0_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScatterReduceBackward0*>(self->cdata.get())->index_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScatterReduceBackward0_index_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScatterReduceBackward0*>(self->cdata.get())->index_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScatterReduceBackward0_reduce_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ScatterReduceBackward0*>(self->cdata.get())->reduce;
  return PyUnicode_FromStringAndSize(prop.data(), prop.size());
  END_HANDLE_TH_ERRORS
}

PyObject* THPScatterReduceBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScatterReduceBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScatterReduceBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScatterReduceBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScatterReduceBackward0_src_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScatterReduceBackward0*>(self->cdata.get())->src_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScatterReduceBackward0_src_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScatterReduceBackward0*>(self->cdata.get())->src_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScatterReduceBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScatterReduceBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScatterReduceBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScatterReduceBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ScatterReduceBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPScatterReduceBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_include_self", (getter)THPScatterReduceBackward0_include_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_index", (getter)THPScatterReduceBackward0_index_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_index", (getter)THPScatterReduceBackward0_index_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduce", (getter)THPScatterReduceBackward0_reduce_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPScatterReduceBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPScatterReduceBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_src", (getter)THPScatterReduceBackward0_src_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_src", (getter)THPScatterReduceBackward0_src_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPScatterReduceBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPScatterReduceBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPForeachPowBackward0_exponent_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<ForeachPowBackward0*>(self->cdata.get());
  const auto& prop = node->exponent_;
  if (node->exponent_released_) {
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

PyObject* THPForeachPowBackward0_exponent_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<ForeachPowBackward0*>(self->cdata.get());
  const auto& prop = node->exponent_;
  if (node->exponent_released_) {
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

PyObject* THPForeachPowBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<ForeachPowBackward0*>(self->cdata.get());
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

PyObject* THPForeachPowBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<ForeachPowBackward0*>(self->cdata.get());
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

PyObject* THPForeachPowBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<ForeachPowBackward0*>(self->cdata.get());
  const auto& prop = node->result_;
  if (node->result_released_) {
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

PyObject* THPForeachPowBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<ForeachPowBackward0*>(self->cdata.get());
  const auto& prop = node->result_;
  if (node->result_released_) {
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

static struct PyGetSetDef ForeachPowBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_exponent", (getter)THPForeachPowBackward0_exponent_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_exponent", (getter)THPForeachPowBackward0_exponent_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPForeachPowBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPForeachPowBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPForeachPowBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPForeachPowBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPForeachPowBackward1_exponent_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<ForeachPowBackward1*>(self->cdata.get());
  const auto& prop = node->exponent;
  if (node->exponent_released_) {
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
                            
PyObject* THPForeachPowBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<ForeachPowBackward1*>(self->cdata.get());
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

PyObject* THPForeachPowBackward1_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<ForeachPowBackward1*>(self->cdata.get());
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

static struct PyGetSetDef ForeachPowBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_exponent", (getter)THPForeachPowBackward1_exponent_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPForeachPowBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPForeachPowBackward1_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPForeachPowBackward2_exponent_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<ForeachPowBackward2*>(self->cdata.get());
  const auto& prop = node->exponent_;
  if (node->exponent_released_) {
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

PyObject* THPForeachPowBackward2_exponent_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<ForeachPowBackward2*>(self->cdata.get());
  const auto& prop = node->exponent_;
  if (node->exponent_released_) {
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

PyObject* THPForeachPowBackward2_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ForeachPowBackward2*>(self->cdata.get())->self;
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

PyObject* THPForeachPowBackward2_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<ForeachPowBackward2*>(self->cdata.get());
  const auto& prop = node->result_;
  if (node->result_released_) {
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

PyObject* THPForeachPowBackward2_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<ForeachPowBackward2*>(self->cdata.get());
  const auto& prop = node->result_;
  if (node->result_released_) {
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

static struct PyGetSetDef ForeachPowBackward2_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_exponent", (getter)THPForeachPowBackward2_exponent_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_exponent", (getter)THPForeachPowBackward2_exponent_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPForeachPowBackward2_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPForeachPowBackward2_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPForeachPowBackward2_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPForeachNormBackward0_ord_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ForeachNormBackward0*>(self->cdata.get())->ord;
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

PyObject* THPForeachNormBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<ForeachNormBackward0*>(self->cdata.get());
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

PyObject* THPForeachNormBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<ForeachNormBackward0*>(self->cdata.get());
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

PyObject* THPForeachNormBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<ForeachNormBackward0*>(self->cdata.get());
  const auto& prop = node->result_;
  if (node->result_released_) {
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

PyObject* THPForeachNormBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<ForeachNormBackward0*>(self->cdata.get());
  const auto& prop = node->result_;
  if (node->result_released_) {
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

static struct PyGetSetDef ForeachNormBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_ord", (getter)THPForeachNormBackward0_ord_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPForeachNormBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPForeachNormBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPForeachNormBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPForeachNormBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPDiagonalBackward0_copy_dim1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<DiagonalBackward0_copy*>(self->cdata.get())->dim1;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPDiagonalBackward0_copy_dim2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<DiagonalBackward0_copy*>(self->cdata.get())->dim2;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPDiagonalBackward0_copy_offset_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<DiagonalBackward0_copy*>(self->cdata.get())->offset;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPDiagonalBackward0_copy_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<DiagonalBackward0_copy*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef DiagonalBackward0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim1", (getter)THPDiagonalBackward0_copy_dim1_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim2", (getter)THPDiagonalBackward0_copy_dim2_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_offset", (getter)THPDiagonalBackward0_copy_offset_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPDiagonalBackward0_copy_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPExpandBackward0_copy_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ExpandBackward0_copy*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef ExpandBackward0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sym_sizes", (getter)THPExpandBackward0_copy_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPPermuteBackward0_copy_dims_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<PermuteBackward0_copy*>(self->cdata.get())->dims;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef PermuteBackward0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dims", (getter)THPPermuteBackward0_copy_dims_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPReshapeAliasBackward0_copy_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ReshapeAliasBackward0_copy*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef ReshapeAliasBackward0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sym_sizes", (getter)THPReshapeAliasBackward0_copy_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef TBackward0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPUnsqueezeBackward0_copy_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnsqueezeBackward0_copy*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UnsqueezeBackward0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPUnsqueezeBackward0_copy_dim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPViewBackward0_copy_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ViewBackward0_copy*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef ViewBackward0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sym_sizes", (getter)THPViewBackward0_copy_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPViewBackwardAutogradNestedTensor0_copy_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ViewBackwardAutogradNestedTensor0_copy*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPViewBackwardAutogradNestedTensor0_copy_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ViewBackwardAutogradNestedTensor0_copy*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ViewBackwardAutogradNestedTensor0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPViewBackwardAutogradNestedTensor0_copy_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPViewBackwardAutogradNestedTensor0_copy_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPTestAutogradMultipleDispatchViewBackward0_copy_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TestAutogradMultipleDispatchViewBackward0_copy*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPTestAutogradMultipleDispatchViewBackward0_copy_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TestAutogradMultipleDispatchViewBackward0_copy*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef TestAutogradMultipleDispatchViewBackward0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPTestAutogradMultipleDispatchViewBackward0_copy_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPTestAutogradMultipleDispatchViewBackward0_copy_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPTestAutogradMultipleDispatchViewBackwardAutogradCUDA0_copy_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TestAutogradMultipleDispatchViewBackwardAutogradCUDA0_copy*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPTestAutogradMultipleDispatchViewBackwardAutogradCUDA0_copy_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TestAutogradMultipleDispatchViewBackwardAutogradCUDA0_copy*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef TestAutogradMultipleDispatchViewBackwardAutogradCUDA0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPTestAutogradMultipleDispatchViewBackwardAutogradCUDA0_copy_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPTestAutogradMultipleDispatchViewBackwardAutogradCUDA0_copy_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

void initialize_autogenerated_functions_2(PyObject* module) {
  static PyTypeObject AddBackward0Class;
  addClass<AddBackward0>(module, AddBackward0Class, "AddBackward0", AddBackward0_properties);
  static PyTypeObject AddBackward1Class;
  addClass<AddBackward1>(module, AddBackward1Class, "AddBackward1", AddBackward1_properties);
  static PyTypeObject AddcdivBackward0Class;
  addClass<AddcdivBackward0>(module, AddcdivBackward0Class, "AddcdivBackward0", AddcdivBackward0_properties);
  static PyTypeObject AngleBackward0Class;
  addClass<AngleBackward0>(module, AngleBackward0Class, "AngleBackward0", AngleBackward0_properties);
  static PyTypeObject AcoshBackward1Class;
  addClass<AcoshBackward1>(module, AcoshBackward1Class, "AcoshBackward1", AcoshBackward1_properties);
  static PyTypeObject AsinhBackward0Class;
  addClass<AsinhBackward0>(module, AsinhBackward0Class, "AsinhBackward0", AsinhBackward0_properties);
  static PyTypeObject AsinhBackward1Class;
  addClass<AsinhBackward1>(module, AsinhBackward1Class, "AsinhBackward1", AsinhBackward1_properties);
  static PyTypeObject AsStridedBackward1Class;
  addClass<AsStridedBackward1>(module, AsStridedBackward1Class, "AsStridedBackward1", AsStridedBackward1_properties);
  static PyTypeObject Atan2Backward0Class;
  addClass<Atan2Backward0>(module, Atan2Backward0Class, "Atan2Backward0", Atan2Backward0_properties);
  static PyTypeObject BaddbmmBackward0Class;
  addClass<BaddbmmBackward0>(module, BaddbmmBackward0Class, "BaddbmmBackward0", BaddbmmBackward0_properties);
  static PyTypeObject BernoulliBackward0Class;
  addClass<BernoulliBackward0>(module, BernoulliBackward0Class, "BernoulliBackward0", BernoulliBackward0_properties);
  static PyTypeObject ChunkBackward0Class;
  addClass<ChunkBackward0>(module, ChunkBackward0Class, "ChunkBackward0", ChunkBackward0_properties);
  static PyTypeObject ChunkBackwardAutogradNestedTensor0Class;
  addClass<ChunkBackwardAutogradNestedTensor0>(module, ChunkBackwardAutogradNestedTensor0Class, "ChunkBackwardAutogradNestedTensor0", ChunkBackwardAutogradNestedTensor0_properties);
  static PyTypeObject CholeskyInverseBackward0Class;
  addClass<CholeskyInverseBackward0>(module, CholeskyInverseBackward0Class, "CholeskyInverseBackward0", CholeskyInverseBackward0_properties);
  static PyTypeObject ClampBackward0Class;
  addClass<ClampBackward0>(module, ClampBackward0Class, "ClampBackward0", ClampBackward0_properties);
  static PyTypeObject ClampBackward1Class;
  addClass<ClampBackward1>(module, ClampBackward1Class, "ClampBackward1", ClampBackward1_properties);
  static PyTypeObject ClampMaxBackward0Class;
  addClass<ClampMaxBackward0>(module, ClampMaxBackward0Class, "ClampMaxBackward0", ClampMaxBackward0_properties);
  static PyTypeObject ClampMaxBackward1Class;
  addClass<ClampMaxBackward1>(module, ClampMaxBackward1Class, "ClampMaxBackward1", ClampMaxBackward1_properties);
  static PyTypeObject LazyCloneBackward0Class;
  addClass<LazyCloneBackward0>(module, LazyCloneBackward0Class, "LazyCloneBackward0", LazyCloneBackward0_properties);
  static PyTypeObject ComplexBackward0Class;
  addClass<ComplexBackward0>(module, ComplexBackward0Class, "ComplexBackward0", ComplexBackward0_properties);
  static PyTypeObject PolarBackward0Class;
  addClass<PolarBackward0>(module, PolarBackward0Class, "PolarBackward0", PolarBackward0_properties);
  static PyTypeObject CummaxBackward0Class;
  addClass<CummaxBackward0>(module, CummaxBackward0Class, "CummaxBackward0", CummaxBackward0_properties);
  static PyTypeObject BlockDiagBackward0Class;
  addClass<BlockDiagBackward0>(module, BlockDiagBackward0Class, "BlockDiagBackward0", BlockDiagBackward0_properties);
  static PyTypeObject DiagonalBackwardBackward0Class;
  addClass<DiagonalBackwardBackward0>(module, DiagonalBackwardBackward0Class, "DiagonalBackwardBackward0", DiagonalBackwardBackward0_properties);
  static PyTypeObject DotBackward0Class;
  addClass<DotBackward0>(module, DotBackward0Class, "DotBackward0", DotBackward0_properties);
  static PyTypeObject NativeDropoutBackward0Class;
  addClass<NativeDropoutBackward0>(module, NativeDropoutBackward0Class, "NativeDropoutBackward0", NativeDropoutBackward0_properties);
  static PyTypeObject ExpBackward0Class;
  addClass<ExpBackward0>(module, ExpBackward0Class, "ExpBackward0", ExpBackward0_properties);
  static PyTypeObject Exp2Backward0Class;
  addClass<Exp2Backward0>(module, Exp2Backward0Class, "Exp2Backward0", Exp2Backward0_properties);
  static PyTypeObject FakeQuantizePerTensorAffineCachemaskBackward0Class;
  addClass<FakeQuantizePerTensorAffineCachemaskBackward0>(module, FakeQuantizePerTensorAffineCachemaskBackward0Class, "FakeQuantizePerTensorAffineCachemaskBackward0", FakeQuantizePerTensorAffineCachemaskBackward0_properties);
  static PyTypeObject FakeQuantizePerTensorAffineCachemaskTensorQparamsBackward0Class;
  addClass<FakeQuantizePerTensorAffineCachemaskTensorQparamsBackward0>(module, FakeQuantizePerTensorAffineCachemaskTensorQparamsBackward0Class, "FakeQuantizePerTensorAffineCachemaskTensorQparamsBackward0", FakeQuantizePerTensorAffineCachemaskTensorQparamsBackward0_properties);
  static PyTypeObject FakeQuantizeLearnablePerTensorAffineBackward0Class;
  addClass<FakeQuantizeLearnablePerTensorAffineBackward0>(module, FakeQuantizeLearnablePerTensorAffineBackward0Class, "FakeQuantizeLearnablePerTensorAffineBackward0", FakeQuantizeLearnablePerTensorAffineBackward0_properties);
  static PyTypeObject FracBackward0Class;
  addClass<FracBackward0>(module, FracBackward0Class, "FracBackward0", FracBackward0_properties);
  static PyTypeObject HardswishBackward0Class;
  addClass<HardswishBackward0>(module, HardswishBackward0Class, "HardswishBackward0", HardswishBackward0_properties);
  static PyTypeObject HypotBackward0Class;
  addClass<HypotBackward0>(module, HypotBackward0Class, "HypotBackward0", HypotBackward0_properties);
  static PyTypeObject IgammacBackward0Class;
  addClass<IgammacBackward0>(module, IgammacBackward0Class, "IgammacBackward0", IgammacBackward0_properties);
  static PyTypeObject UnsafeMaskedIndexBackward0Class;
  addClass<UnsafeMaskedIndexBackward0>(module, UnsafeMaskedIndexBackward0Class, "UnsafeMaskedIndexBackward0", UnsafeMaskedIndexBackward0_properties);
  static PyTypeObject LogBackward0Class;
  addClass<LogBackward0>(module, LogBackward0Class, "LogBackward0", LogBackward0_properties);
  static PyTypeObject Log2Backward0Class;
  addClass<Log2Backward0>(module, Log2Backward0Class, "Log2Backward0", Log2Backward0_properties);
  static PyTypeObject LogaddexpBackward0Class;
  addClass<LogaddexpBackward0>(module, LogaddexpBackward0Class, "LogaddexpBackward0", LogaddexpBackward0_properties);
  static PyTypeObject SpecialZetaBackward0Class;
  addClass<SpecialZetaBackward0>(module, SpecialZetaBackward0Class, "SpecialZetaBackward0", SpecialZetaBackward0_properties);
  static PyTypeObject SpecialZetaBackward1Class;
  addClass<SpecialZetaBackward1>(module, SpecialZetaBackward1Class, "SpecialZetaBackward1", SpecialZetaBackward1_properties);
  static PyTypeObject SpecialZetaBackward2Class;
  addClass<SpecialZetaBackward2>(module, SpecialZetaBackward2Class, "SpecialZetaBackward2", SpecialZetaBackward2_properties);
  static PyTypeObject MaskedScatterBackwardBackward0Class;
  addClass<MaskedScatterBackwardBackward0>(module, MaskedScatterBackwardBackward0Class, "MaskedScatterBackwardBackward0", MaskedScatterBackwardBackward0_properties);
  static PyTypeObject MaskedSelectBackward0Class;
  addClass<MaskedSelectBackward0>(module, MaskedSelectBackward0Class, "MaskedSelectBackward0", MaskedSelectBackward0_properties);
  static PyTypeObject FmaxBackward0Class;
  addClass<FmaxBackward0>(module, FmaxBackward0Class, "FmaxBackward0", FmaxBackward0_properties);
  static PyTypeObject NativeBatchNormLegitBackward0Class;
  addClass<NativeBatchNormLegitBackward0>(module, NativeBatchNormLegitBackward0Class, "NativeBatchNormLegitBackward0", NativeBatchNormLegitBackward0_properties);
  static PyTypeObject NativeBatchNormLegitBackward1Class;
  addClass<NativeBatchNormLegitBackward1>(module, NativeBatchNormLegitBackward1Class, "NativeBatchNormLegitBackward1", NativeBatchNormLegitBackward1_properties);
  static PyTypeObject NativeLayerNormBackwardBackward0Class;
  addClass<NativeLayerNormBackwardBackward0>(module, NativeLayerNormBackwardBackward0Class, "NativeLayerNormBackwardBackward0", NativeLayerNormBackwardBackward0_properties);
  static PyTypeObject BatchNormWithUpdateBackward0Class;
  addClass<BatchNormWithUpdateBackward0>(module, BatchNormWithUpdateBackward0Class, "BatchNormWithUpdateBackward0", BatchNormWithUpdateBackward0_properties);
  static PyTypeObject NormBackward0Class;
  addClass<NormBackward0>(module, NormBackward0Class, "NormBackward0", NormBackward0_properties);
  static PyTypeObject NormBackward1Class;
  addClass<NormBackward1>(module, NormBackward1Class, "NormBackward1", NormBackward1_properties);
  static PyTypeObject NormBackward2Class;
  addClass<NormBackward2>(module, NormBackward2Class, "NormBackward2", NormBackward2_properties);
  static PyTypeObject NormBackward3Class;
  addClass<NormBackward3>(module, NormBackward3Class, "NormBackward3", NormBackward3_properties);
  static PyTypeObject PdistBackward0Class;
  addClass<PdistBackward0>(module, PdistBackward0Class, "PdistBackward0", PdistBackward0_properties);
  static PyTypeObject EuclideanDistBackward0Class;
  addClass<EuclideanDistBackward0>(module, EuclideanDistBackward0Class, "EuclideanDistBackward0", EuclideanDistBackward0_properties);
  static PyTypeObject CdistBackwardBackward0Class;
  addClass<CdistBackwardBackward0>(module, CdistBackwardBackward0Class, "CdistBackwardBackward0", CdistBackwardBackward0_properties);
  static PyTypeObject LinalgHouseholderProductBackward0Class;
  addClass<LinalgHouseholderProductBackward0>(module, LinalgHouseholderProductBackward0Class, "LinalgHouseholderProductBackward0", LinalgHouseholderProductBackward0_properties);
  static PyTypeObject PutBackward0Class;
  addClass<PutBackward0>(module, PutBackward0Class, "PutBackward0", PutBackward0_properties);
  static PyTypeObject Rad2DegBackward0Class;
  addClass<Rad2DegBackward0>(module, Rad2DegBackward0Class, "Rad2DegBackward0", Rad2DegBackward0_properties);
  static PyTypeObject RandomBackward0Class;
  addClass<RandomBackward0>(module, RandomBackward0Class, "RandomBackward0", RandomBackward0_properties);
  static PyTypeObject RandomBackward1Class;
  addClass<RandomBackward1>(module, RandomBackward1Class, "RandomBackward1", RandomBackward1_properties);
  static PyTypeObject RandomBackward2Class;
  addClass<RandomBackward2>(module, RandomBackward2Class, "RandomBackward2", RandomBackward2_properties);
  static PyTypeObject RepeatBackward0Class;
  addClass<RepeatBackward0>(module, RepeatBackward0Class, "RepeatBackward0", RepeatBackward0_properties);
  static PyTypeObject RsqrtBackward0Class;
  addClass<RsqrtBackward0>(module, RsqrtBackward0Class, "RsqrtBackward0", RsqrtBackward0_properties);
  static PyTypeObject ScatterBackward0Class;
  addClass<ScatterBackward0>(module, ScatterBackward0Class, "ScatterBackward0", ScatterBackward0_properties);
  static PyTypeObject ScatterBackward1Class;
  addClass<ScatterBackward1>(module, ScatterBackward1Class, "ScatterBackward1", ScatterBackward1_properties);
  static PyTypeObject SortBackward0Class;
  addClass<SortBackward0>(module, SortBackward0Class, "SortBackward0", SortBackward0_properties);
  static PyTypeObject SortBackward1Class;
  addClass<SortBackward1>(module, SortBackward1Class, "SortBackward1", SortBackward1_properties);
  static PyTypeObject SqueezeBackward3Class;
  addClass<SqueezeBackward3>(module, SqueezeBackward3Class, "SqueezeBackward3", SqueezeBackward3_properties);
  static PyTypeObject SqueezeBackward4Class;
  addClass<SqueezeBackward4>(module, SqueezeBackward4Class, "SqueezeBackward4", SqueezeBackward4_properties);
  static PyTypeObject SqueezeBackward5Class;
  addClass<SqueezeBackward5>(module, SqueezeBackward5Class, "SqueezeBackward5", SqueezeBackward5_properties);
  static PyTypeObject SumBackward0Class;
  addClass<SumBackward0>(module, SumBackward0Class, "SumBackward0", SumBackward0_properties);
  static PyTypeObject SumBackwardAutogradNestedTensor0Class;
  addClass<SumBackwardAutogradNestedTensor0>(module, SumBackwardAutogradNestedTensor0Class, "SumBackwardAutogradNestedTensor0", SumBackwardAutogradNestedTensor0_properties);
  static PyTypeObject SumBackward1Class;
  addClass<SumBackward1>(module, SumBackward1Class, "SumBackward1", SumBackward1_properties);
  static PyTypeObject SumBackwardAutogradNestedTensor1Class;
  addClass<SumBackwardAutogradNestedTensor1>(module, SumBackwardAutogradNestedTensor1Class, "SumBackwardAutogradNestedTensor1", SumBackwardAutogradNestedTensor1_properties);
  static PyTypeObject NansumBackward0Class;
  addClass<NansumBackward0>(module, NansumBackward0Class, "NansumBackward0", NansumBackward0_properties);
  static PyTypeObject LinalgEigBackward0Class;
  addClass<LinalgEigBackward0>(module, LinalgEigBackward0Class, "LinalgEigBackward0", LinalgEigBackward0_properties);
  static PyTypeObject Rot90Backward0Class;
  addClass<Rot90Backward0>(module, Rot90Backward0Class, "Rot90Backward0", Rot90Backward0_properties);
  static PyTypeObject TrilBackward0Class;
  addClass<TrilBackward0>(module, TrilBackward0Class, "TrilBackward0", TrilBackward0_properties);
  static PyTypeObject TruncBackward0Class;
  addClass<TruncBackward0>(module, TruncBackward0Class, "TruncBackward0", TruncBackward0_properties);
  static PyTypeObject ToDenseBackward0Class;
  addClass<ToDenseBackward0>(module, ToDenseBackward0Class, "ToDenseBackward0", ToDenseBackward0_properties);
  static PyTypeObject VarBackward0Class;
  addClass<VarBackward0>(module, VarBackward0Class, "VarBackward0", VarBackward0_properties);
  static PyTypeObject VarMeanBackward0Class;
  addClass<VarMeanBackward0>(module, VarMeanBackward0Class, "VarMeanBackward0", VarMeanBackward0_properties);
  static PyTypeObject SparseCooTensorWithDimsAndTensorsBackward0Class;
  addClass<SparseCooTensorWithDimsAndTensorsBackward0>(module, SparseCooTensorWithDimsAndTensorsBackward0Class, "SparseCooTensorWithDimsAndTensorsBackward0", SparseCooTensorWithDimsAndTensorsBackward0_properties);
  static PyTypeObject StandardGammaGradBackward0Class;
  addClass<StandardGammaGradBackward0>(module, StandardGammaGradBackward0Class, "StandardGammaGradBackward0", StandardGammaGradBackward0_properties);
  static PyTypeObject BinaryCrossEntropyWithLogitsBackward0Class;
  addClass<BinaryCrossEntropyWithLogitsBackward0>(module, BinaryCrossEntropyWithLogitsBackward0Class, "BinaryCrossEntropyWithLogitsBackward0", BinaryCrossEntropyWithLogitsBackward0_properties);
  static PyTypeObject MultiMarginLossBackward0Class;
  addClass<MultiMarginLossBackward0>(module, MultiMarginLossBackward0Class, "MultiMarginLossBackward0", MultiMarginLossBackward0_properties);
  static PyTypeObject MultilabelMarginLossBackward0Class;
  addClass<MultilabelMarginLossBackward0>(module, MultilabelMarginLossBackward0Class, "MultilabelMarginLossBackward0", MultilabelMarginLossBackward0_properties);
  static PyTypeObject NllLossBackward0Class;
  addClass<NllLossBackward0>(module, NllLossBackward0Class, "NllLossBackward0", NllLossBackward0_properties);
  static PyTypeObject PreluKernelBackward0Class;
  addClass<PreluKernelBackward0>(module, PreluKernelBackward0Class, "PreluKernelBackward0", PreluKernelBackward0_properties);
  static PyTypeObject RreluWithNoiseBackward0Class;
  addClass<RreluWithNoiseBackward0>(module, RreluWithNoiseBackward0Class, "RreluWithNoiseBackward0", RreluWithNoiseBackward0_properties);
  static PyTypeObject SparseSoftmaxBackward0Class;
  addClass<SparseSoftmaxBackward0>(module, SparseSoftmaxBackward0Class, "SparseSoftmaxBackward0", SparseSoftmaxBackward0_properties);
  static PyTypeObject SparseSparseMatmulBackward0Class;
  addClass<SparseSparseMatmulBackward0>(module, SparseSparseMatmulBackward0Class, "SparseSparseMatmulBackward0", SparseSparseMatmulBackward0_properties);
  static PyTypeObject ThresholdBackward1Class;
  addClass<ThresholdBackward1>(module, ThresholdBackward1Class, "ThresholdBackward1", ThresholdBackward1_properties);
  static PyTypeObject UpsampleNearestExact2DBackward0Class;
  addClass<UpsampleNearestExact2DBackward0>(module, UpsampleNearestExact2DBackward0Class, "UpsampleNearestExact2DBackward0", UpsampleNearestExact2DBackward0_properties);
  static PyTypeObject UpsampleNearestExact3DBackward0Class;
  addClass<UpsampleNearestExact3DBackward0>(module, UpsampleNearestExact3DBackward0Class, "UpsampleNearestExact3DBackward0", UpsampleNearestExact3DBackward0_properties);
  static PyTypeObject AvgPool2DBackward0Class;
  addClass<AvgPool2DBackward0>(module, AvgPool2DBackward0Class, "AvgPool2DBackward0", AvgPool2DBackward0_properties);
  static PyTypeObject MpsConvolutionBackwardBackward0Class;
  addClass<MpsConvolutionBackwardBackward0>(module, MpsConvolutionBackwardBackward0Class, "MpsConvolutionBackwardBackward0", MpsConvolutionBackwardBackward0_properties);
  static PyTypeObject SlowConvDilated3DBackward0Class;
  addClass<SlowConvDilated3DBackward0>(module, SlowConvDilated3DBackward0Class, "SlowConvDilated3DBackward0", SlowConvDilated3DBackward0_properties);
  static PyTypeObject AdaptiveAvgPool2DBackwardBackward0Class;
  addClass<AdaptiveAvgPool2DBackwardBackward0>(module, AdaptiveAvgPool2DBackwardBackward0Class, "AdaptiveAvgPool2DBackwardBackward0", AdaptiveAvgPool2DBackwardBackward0_properties);
  static PyTypeObject FractionalMaxPool2DBackwardBackward0Class;
  addClass<FractionalMaxPool2DBackwardBackward0>(module, FractionalMaxPool2DBackwardBackward0Class, "FractionalMaxPool2DBackwardBackward0", FractionalMaxPool2DBackwardBackward0_properties);
  static PyTypeObject MaxPool3DWithIndicesBackwardBackward0Class;
  addClass<MaxPool3DWithIndicesBackwardBackward0>(module, MaxPool3DWithIndicesBackwardBackward0Class, "MaxPool3DWithIndicesBackwardBackward0", MaxPool3DWithIndicesBackwardBackward0_properties);
  static PyTypeObject ReflectionPad1DBackwardBackward0Class;
  addClass<ReflectionPad1DBackwardBackward0>(module, ReflectionPad1DBackwardBackward0Class, "ReflectionPad1DBackwardBackward0", ReflectionPad1DBackwardBackward0_properties);
  static PyTypeObject ReflectionPad2DBackwardBackward0Class;
  addClass<ReflectionPad2DBackwardBackward0>(module, ReflectionPad2DBackwardBackward0Class, "ReflectionPad2DBackwardBackward0", ReflectionPad2DBackwardBackward0_properties);
  static PyTypeObject ReplicationPad1DBackwardBackward0Class;
  addClass<ReplicationPad1DBackwardBackward0>(module, ReplicationPad1DBackwardBackward0Class, "ReplicationPad1DBackwardBackward0", ReplicationPad1DBackwardBackward0_properties);
  static PyTypeObject ReplicationPad3DBackwardBackward0Class;
  addClass<ReplicationPad3DBackwardBackward0>(module, ReplicationPad3DBackwardBackward0Class, "ReplicationPad3DBackwardBackward0", ReplicationPad3DBackwardBackward0_properties);
  static PyTypeObject SoftMarginLossBackwardBackward0Class;
  addClass<SoftMarginLossBackwardBackward0>(module, SoftMarginLossBackwardBackward0Class, "SoftMarginLossBackwardBackward0", SoftMarginLossBackwardBackward0_properties);
  static PyTypeObject SoftshrinkBackwardBackward0Class;
  addClass<SoftshrinkBackwardBackward0>(module, SoftshrinkBackwardBackward0Class, "SoftshrinkBackwardBackward0", SoftshrinkBackwardBackward0_properties);
  static PyTypeObject UpsampleBilinear2DBackwardBackward0Class;
  addClass<UpsampleBilinear2DBackwardBackward0>(module, UpsampleBilinear2DBackwardBackward0Class, "UpsampleBilinear2DBackwardBackward0", UpsampleBilinear2DBackwardBackward0_properties);
  static PyTypeObject UpsampleBicubic2DAaBackwardBackward0Class;
  addClass<UpsampleBicubic2DAaBackwardBackward0>(module, UpsampleBicubic2DAaBackwardBackward0Class, "UpsampleBicubic2DAaBackwardBackward0", UpsampleBicubic2DAaBackwardBackward0_properties);
  static PyTypeObject UpsampleNearestExact1DBackwardBackward0Class;
  addClass<UpsampleNearestExact1DBackwardBackward0>(module, UpsampleNearestExact1DBackwardBackward0Class, "UpsampleNearestExact1DBackwardBackward0", UpsampleNearestExact1DBackwardBackward0_properties);
  static PyTypeObject UpsampleNearestExact2DBackwardBackward0Class;
  addClass<UpsampleNearestExact2DBackwardBackward0>(module, UpsampleNearestExact2DBackwardBackward0Class, "UpsampleNearestExact2DBackwardBackward0", UpsampleNearestExact2DBackwardBackward0_properties);
  static PyTypeObject MiopenConvolutionBackward0Class;
  addClass<MiopenConvolutionBackward0>(module, MiopenConvolutionBackward0Class, "MiopenConvolutionBackward0", MiopenConvolutionBackward0_properties);
  static PyTypeObject MiopenRnnBackward0Class;
  addClass<MiopenRnnBackward0>(module, MiopenRnnBackward0Class, "MiopenRnnBackward0", MiopenRnnBackward0_properties);
  static PyTypeObject MkldnnRnnLayerBackward0Class;
  addClass<MkldnnRnnLayerBackward0>(module, MkldnnRnnLayerBackward0Class, "MkldnnRnnLayerBackward0", MkldnnRnnLayerBackward0_properties);
  static PyTypeObject MkldnnMaxPool3DBackward0Class;
  addClass<MkldnnMaxPool3DBackward0>(module, MkldnnMaxPool3DBackward0Class, "MkldnnMaxPool3DBackward0", MkldnnMaxPool3DBackward0_properties);
  static PyTypeObject MkldnnReshapeBackward0Class;
  addClass<MkldnnReshapeBackward0>(module, MkldnnReshapeBackward0Class, "MkldnnReshapeBackward0", MkldnnReshapeBackward0_properties);
  static PyTypeObject NestedTensorFromTensorListBackward0Class;
  addClass<NestedTensorFromTensorListBackward0>(module, NestedTensorFromTensorListBackward0Class, "NestedTensorFromTensorListBackward0", NestedTensorFromTensorListBackward0_properties);
  static PyTypeObject ScaledDotProductCudnnAttentionBackward0Class;
  addClass<ScaledDotProductCudnnAttentionBackward0>(module, ScaledDotProductCudnnAttentionBackward0Class, "ScaledDotProductCudnnAttentionBackward0", ScaledDotProductCudnnAttentionBackward0_properties);
  static PyTypeObject ScatterReduceBackward0Class;
  addClass<ScatterReduceBackward0>(module, ScatterReduceBackward0Class, "ScatterReduceBackward0", ScatterReduceBackward0_properties);
  static PyTypeObject ForeachPowBackward0Class;
  addClass<ForeachPowBackward0>(module, ForeachPowBackward0Class, "ForeachPowBackward0", ForeachPowBackward0_properties);
  static PyTypeObject ForeachPowBackward1Class;
  addClass<ForeachPowBackward1>(module, ForeachPowBackward1Class, "ForeachPowBackward1", ForeachPowBackward1_properties);
  static PyTypeObject ForeachPowBackward2Class;
  addClass<ForeachPowBackward2>(module, ForeachPowBackward2Class, "ForeachPowBackward2", ForeachPowBackward2_properties);
  static PyTypeObject ForeachNormBackward0Class;
  addClass<ForeachNormBackward0>(module, ForeachNormBackward0Class, "ForeachNormBackward0", ForeachNormBackward0_properties);
  static PyTypeObject DiagonalBackward0_copyClass;
  addClass<DiagonalBackward0_copy>(module, DiagonalBackward0_copyClass, "DiagonalBackward0_copy", DiagonalBackward0_copy_properties);
  static PyTypeObject ExpandBackward0_copyClass;
  addClass<ExpandBackward0_copy>(module, ExpandBackward0_copyClass, "ExpandBackward0_copy", ExpandBackward0_copy_properties);
  static PyTypeObject PermuteBackward0_copyClass;
  addClass<PermuteBackward0_copy>(module, PermuteBackward0_copyClass, "PermuteBackward0_copy", PermuteBackward0_copy_properties);
  static PyTypeObject ReshapeAliasBackward0_copyClass;
  addClass<ReshapeAliasBackward0_copy>(module, ReshapeAliasBackward0_copyClass, "ReshapeAliasBackward0_copy", ReshapeAliasBackward0_copy_properties);
  static PyTypeObject TBackward0_copyClass;
  addClass<TBackward0_copy>(module, TBackward0_copyClass, "TBackward0_copy", TBackward0_copy_properties);
  static PyTypeObject UnsqueezeBackward0_copyClass;
  addClass<UnsqueezeBackward0_copy>(module, UnsqueezeBackward0_copyClass, "UnsqueezeBackward0_copy", UnsqueezeBackward0_copy_properties);
  static PyTypeObject ViewBackward0_copyClass;
  addClass<ViewBackward0_copy>(module, ViewBackward0_copyClass, "ViewBackward0_copy", ViewBackward0_copy_properties);
  static PyTypeObject ViewBackwardAutogradNestedTensor0_copyClass;
  addClass<ViewBackwardAutogradNestedTensor0_copy>(module, ViewBackwardAutogradNestedTensor0_copyClass, "ViewBackwardAutogradNestedTensor0_copy", ViewBackwardAutogradNestedTensor0_copy_properties);
  static PyTypeObject TestAutogradMultipleDispatchViewBackward0_copyClass;
  addClass<TestAutogradMultipleDispatchViewBackward0_copy>(module, TestAutogradMultipleDispatchViewBackward0_copyClass, "TestAutogradMultipleDispatchViewBackward0_copy", TestAutogradMultipleDispatchViewBackward0_copy_properties);
  static PyTypeObject TestAutogradMultipleDispatchViewBackwardAutogradCUDA0_copyClass;
  addClass<TestAutogradMultipleDispatchViewBackwardAutogradCUDA0_copy>(module, TestAutogradMultipleDispatchViewBackwardAutogradCUDA0_copyClass, "TestAutogradMultipleDispatchViewBackwardAutogradCUDA0_copy", TestAutogradMultipleDispatchViewBackwardAutogradCUDA0_copy_properties);
}

} // namespace torch::autograd::generated
