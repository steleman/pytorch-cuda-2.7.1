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

PyObject* THPAbsBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AbsBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAbsBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AbsBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AbsBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPAbsBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPAbsBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAddmvBackward0_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddmvBackward0*>(self->cdata.get())->alpha;
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

PyObject* THPAddmvBackward0_beta_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddmvBackward0*>(self->cdata.get())->beta;
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

PyObject* THPAddmvBackward0_mat_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddmvBackward0*>(self->cdata.get())->mat_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddmvBackward0_mat_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddmvBackward0*>(self->cdata.get())->mat_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddmvBackward0_vec_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddmvBackward0*>(self->cdata.get())->vec_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddmvBackward0_vec_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddmvBackward0*>(self->cdata.get())->vec_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AddmvBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_alpha", (getter)THPAddmvBackward0_alpha_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_beta", (getter)THPAddmvBackward0_beta_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mat", (getter)THPAddmvBackward0_mat_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_mat", (getter)THPAddmvBackward0_mat_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_vec", (getter)THPAddmvBackward0_vec_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_vec", (getter)THPAddmvBackward0_vec_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAddrBackward0_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddrBackward0*>(self->cdata.get())->alpha;
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

PyObject* THPAddrBackward0_beta_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddrBackward0*>(self->cdata.get())->beta;
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

PyObject* THPAddrBackward0_vec1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddrBackward0*>(self->cdata.get())->vec1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddrBackward0_vec1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddrBackward0*>(self->cdata.get())->vec1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddrBackward0_vec2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddrBackward0*>(self->cdata.get())->vec2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddrBackward0_vec2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddrBackward0*>(self->cdata.get())->vec2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AddrBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_alpha", (getter)THPAddrBackward0_alpha_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_beta", (getter)THPAddrBackward0_beta_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_vec1", (getter)THPAddrBackward0_vec1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_vec1", (getter)THPAddrBackward0_vec1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_vec2", (getter)THPAddrBackward0_vec2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_vec2", (getter)THPAddrBackward0_vec2_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef AliasBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPAtanBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AtanBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAtanBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AtanBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AtanBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPAtanBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPAtanBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef CeilBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPLinalgCholeskyExBackward0_upper_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LinalgCholeskyExBackward0*>(self->cdata.get())->upper;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgCholeskyExBackward0_L_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgCholeskyExBackward0*>(self->cdata.get())->L_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgCholeskyExBackward0_L_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgCholeskyExBackward0*>(self->cdata.get())->L_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LinalgCholeskyExBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_upper", (getter)THPLinalgCholeskyExBackward0_upper_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_L", (getter)THPLinalgCholeskyExBackward0_L_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_L", (getter)THPLinalgCholeskyExBackward0_L_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCholeskySolveBackward0_input2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CholeskySolveBackward0*>(self->cdata.get())->input2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCholeskySolveBackward0_input2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CholeskySolveBackward0*>(self->cdata.get())->input2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCholeskySolveBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CholeskySolveBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCholeskySolveBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CholeskySolveBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCholeskySolveBackward0_upper_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CholeskySolveBackward0*>(self->cdata.get())->upper;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCholeskySolveBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CholeskySolveBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCholeskySolveBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CholeskySolveBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CholeskySolveBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_input2", (getter)THPCholeskySolveBackward0_input2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input2", (getter)THPCholeskySolveBackward0_input2_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPCholeskySolveBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPCholeskySolveBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_upper", (getter)THPCholeskySolveBackward0_upper_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPCholeskySolveBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPCholeskySolveBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef CloneBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPCoshBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CoshBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCoshBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CoshBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CoshBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPCoshBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPCoshBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLinalgCrossBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LinalgCrossBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgCrossBackward0_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgCrossBackward0*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgCrossBackward0_other_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgCrossBackward0*>(self->cdata.get())->other_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgCrossBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgCrossBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgCrossBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgCrossBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LinalgCrossBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPLinalgCrossBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_other", (getter)THPLinalgCrossBackward0_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_other", (getter)THPLinalgCrossBackward0_other_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPLinalgCrossBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPLinalgCrossBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLogcumsumexpBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LogcumsumexpBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogcumsumexpBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogcumsumexpBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogcumsumexpBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogcumsumexpBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogcumsumexpBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogcumsumexpBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogcumsumexpBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogcumsumexpBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LogcumsumexpBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPLogcumsumexpBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPLogcumsumexpBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPLogcumsumexpBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPLogcumsumexpBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPLogcumsumexpBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCumminBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CumminBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCumminBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CumminBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCumminBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CumminBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCumminBackward0_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CumminBackward0*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCumminBackward0_indices_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CumminBackward0*>(self->cdata.get())->indices_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CumminBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPCumminBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPCumminBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPCumminBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_indices", (getter)THPCumminBackward0_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_indices", (getter)THPCumminBackward0_indices_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPConvTbcBackward0_bias_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvTbcBackward0*>(self->cdata.get())->bias_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvTbcBackward0_bias_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvTbcBackward0*>(self->cdata.get())->bias_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvTbcBackward0_pad_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvTbcBackward0*>(self->cdata.get())->pad;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvTbcBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvTbcBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvTbcBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvTbcBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvTbcBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvTbcBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvTbcBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvTbcBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ConvTbcBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_bias", (getter)THPConvTbcBackward0_bias_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_bias", (getter)THPConvTbcBackward0_bias_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_pad", (getter)THPConvTbcBackward0_pad_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPConvTbcBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPConvTbcBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPConvTbcBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPConvTbcBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPDivBackward0_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<DivBackward0*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPDivBackward0_other_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<DivBackward0*>(self->cdata.get())->other_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPDivBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<DivBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPDivBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<DivBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef DivBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPDivBackward0_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_other", (getter)THPDivBackward0_other_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPDivBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPDivBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPDivBackward1_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<DivBackward1*>(self->cdata.get())->other;
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

static struct PyGetSetDef DivBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPDivBackward1_other_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPDivBackward2_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<DivBackward2*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPDivBackward2_other_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<DivBackward2*>(self->cdata.get())->other_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPDivBackward2_rounding_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<DivBackward2*>(self->cdata.get())->rounding_mode;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyUnicode_FromStringAndSize(prop.data(), prop.size());
  END_HANDLE_TH_ERRORS
}

PyObject* THPDivBackward2_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<DivBackward2*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPDivBackward2_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<DivBackward2*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef DivBackward2_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPDivBackward2_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_other", (getter)THPDivBackward2_other_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_rounding_mode", (getter)THPDivBackward2_rounding_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPDivBackward2_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPDivBackward2_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPDivBackward3_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<DivBackward3*>(self->cdata.get())->other;
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

PyObject* THPDivBackward3_rounding_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<DivBackward3*>(self->cdata.get())->rounding_mode;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyUnicode_FromStringAndSize(prop.data(), prop.size());
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef DivBackward3_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPDivBackward3_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_rounding_mode", (getter)THPDivBackward3_rounding_mode_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPErfBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ErfBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPErfBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ErfBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ErfBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPErfBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPErfBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPFakeQuantizeLearnablePerChannelAffineBackward0_axis_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FakeQuantizeLearnablePerChannelAffineBackward0*>(self->cdata.get())->axis;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPFakeQuantizeLearnablePerChannelAffineBackward0_grad_factor_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FakeQuantizeLearnablePerChannelAffineBackward0*>(self->cdata.get())->grad_factor;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPFakeQuantizeLearnablePerChannelAffineBackward0_quant_max_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FakeQuantizeLearnablePerChannelAffineBackward0*>(self->cdata.get())->quant_max;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPFakeQuantizeLearnablePerChannelAffineBackward0_quant_min_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FakeQuantizeLearnablePerChannelAffineBackward0*>(self->cdata.get())->quant_min;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPFakeQuantizeLearnablePerChannelAffineBackward0_scale_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FakeQuantizeLearnablePerChannelAffineBackward0*>(self->cdata.get())->scale_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFakeQuantizeLearnablePerChannelAffineBackward0_scale_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FakeQuantizeLearnablePerChannelAffineBackward0*>(self->cdata.get())->scale_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPFakeQuantizeLearnablePerChannelAffineBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FakeQuantizeLearnablePerChannelAffineBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFakeQuantizeLearnablePerChannelAffineBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FakeQuantizeLearnablePerChannelAffineBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPFakeQuantizeLearnablePerChannelAffineBackward0_zero_point_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FakeQuantizeLearnablePerChannelAffineBackward0*>(self->cdata.get())->zero_point_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFakeQuantizeLearnablePerChannelAffineBackward0_zero_point_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FakeQuantizeLearnablePerChannelAffineBackward0*>(self->cdata.get())->zero_point_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FakeQuantizeLearnablePerChannelAffineBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_axis", (getter)THPFakeQuantizeLearnablePerChannelAffineBackward0_axis_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_factor", (getter)THPFakeQuantizeLearnablePerChannelAffineBackward0_grad_factor_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_quant_max", (getter)THPFakeQuantizeLearnablePerChannelAffineBackward0_quant_max_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_quant_min", (getter)THPFakeQuantizeLearnablePerChannelAffineBackward0_quant_min_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale", (getter)THPFakeQuantizeLearnablePerChannelAffineBackward0_scale_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_scale", (getter)THPFakeQuantizeLearnablePerChannelAffineBackward0_scale_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPFakeQuantizeLearnablePerChannelAffineBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPFakeQuantizeLearnablePerChannelAffineBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_zero_point", (getter)THPFakeQuantizeLearnablePerChannelAffineBackward0_zero_point_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_zero_point", (getter)THPFakeQuantizeLearnablePerChannelAffineBackward0_zero_point_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef FillBackward2_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef FillBackward3_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPGridSampler3DBackward0_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<GridSampler3DBackward0*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPGridSampler3DBackward0_grid_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GridSampler3DBackward0*>(self->cdata.get())->grid_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPGridSampler3DBackward0_grid_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GridSampler3DBackward0*>(self->cdata.get())->grid_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPGridSampler3DBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GridSampler3DBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPGridSampler3DBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GridSampler3DBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPGridSampler3DBackward0_interpolation_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<GridSampler3DBackward0*>(self->cdata.get())->interpolation_mode;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPGridSampler3DBackward0_padding_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<GridSampler3DBackward0*>(self->cdata.get())->padding_mode;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef GridSampler3DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_align_corners", (getter)THPGridSampler3DBackward0_align_corners_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grid", (getter)THPGridSampler3DBackward0_grid_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_grid", (getter)THPGridSampler3DBackward0_grid_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPGridSampler3DBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPGridSampler3DBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_interpolation_mode", (getter)THPGridSampler3DBackward0_interpolation_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding_mode", (getter)THPGridSampler3DBackward0_padding_mode_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPGridSampler2DCpuFallbackBackward0_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<GridSampler2DCpuFallbackBackward0*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPGridSampler2DCpuFallbackBackward0_grid_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GridSampler2DCpuFallbackBackward0*>(self->cdata.get())->grid_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPGridSampler2DCpuFallbackBackward0_grid_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GridSampler2DCpuFallbackBackward0*>(self->cdata.get())->grid_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPGridSampler2DCpuFallbackBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GridSampler2DCpuFallbackBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPGridSampler2DCpuFallbackBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GridSampler2DCpuFallbackBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPGridSampler2DCpuFallbackBackward0_interpolation_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<GridSampler2DCpuFallbackBackward0*>(self->cdata.get())->interpolation_mode;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPGridSampler2DCpuFallbackBackward0_padding_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<GridSampler2DCpuFallbackBackward0*>(self->cdata.get())->padding_mode;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef GridSampler2DCpuFallbackBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_align_corners", (getter)THPGridSampler2DCpuFallbackBackward0_align_corners_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grid", (getter)THPGridSampler2DCpuFallbackBackward0_grid_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_grid", (getter)THPGridSampler2DCpuFallbackBackward0_grid_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPGridSampler2DCpuFallbackBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPGridSampler2DCpuFallbackBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_interpolation_mode", (getter)THPGridSampler2DCpuFallbackBackward0_interpolation_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding_mode", (getter)THPGridSampler2DCpuFallbackBackward0_padding_mode_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPHardswishBackwardBackward0_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HardswishBackwardBackward0*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPHardswishBackwardBackward0_grad_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HardswishBackwardBackward0*>(self->cdata.get())->grad_output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPHardswishBackwardBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HardswishBackwardBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPHardswishBackwardBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HardswishBackwardBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef HardswishBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_output", (getter)THPHardswishBackwardBackward0_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_grad_output", (getter)THPHardswishBackwardBackward0_grad_output_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPHardswishBackwardBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPHardswishBackwardBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPIndexBackward0_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<IndexBackward0*>(self->cdata.get());
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

PyObject* THPIndexBackward0_indices_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<IndexBackward0*>(self->cdata.get());
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

PyObject* THPIndexBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<IndexBackward0*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef IndexBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_indices", (getter)THPIndexBackward0_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_indices", (getter)THPIndexBackward0_indices_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPIndexBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPIndexReduceBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<IndexReduceBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexReduceBackward0_include_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<IndexReduceBackward0*>(self->cdata.get())->include_self;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexReduceBackward0_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IndexReduceBackward0*>(self->cdata.get())->index_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexReduceBackward0_index_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IndexReduceBackward0*>(self->cdata.get())->index_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexReduceBackward0_reduce_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<IndexReduceBackward0*>(self->cdata.get())->reduce;
  return PyUnicode_FromStringAndSize(prop.data(), prop.size());
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexReduceBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IndexReduceBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexReduceBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IndexReduceBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexReduceBackward0_source_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IndexReduceBackward0*>(self->cdata.get())->source_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexReduceBackward0_source_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IndexReduceBackward0*>(self->cdata.get())->source_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexReduceBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IndexReduceBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexReduceBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IndexReduceBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef IndexReduceBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPIndexReduceBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_include_self", (getter)THPIndexReduceBackward0_include_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_index", (getter)THPIndexReduceBackward0_index_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_index", (getter)THPIndexReduceBackward0_index_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduce", (getter)THPIndexReduceBackward0_reduce_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPIndexReduceBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPIndexReduceBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_source", (getter)THPIndexReduceBackward0_source_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_source", (getter)THPIndexReduceBackward0_source_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPIndexReduceBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPIndexReduceBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPIndexFillBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<IndexFillBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexFillBackward0_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IndexFillBackward0*>(self->cdata.get())->index_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexFillBackward0_index_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IndexFillBackward0*>(self->cdata.get())->index_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef IndexFillBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPIndexFillBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_index", (getter)THPIndexFillBackward0_index_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_index", (getter)THPIndexFillBackward0_index_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPIndexFillBackward1_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<IndexFillBackward1*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexFillBackward1_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IndexFillBackward1*>(self->cdata.get())->index_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexFillBackward1_index_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IndexFillBackward1*>(self->cdata.get())->index_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef IndexFillBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPIndexFillBackward1_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_index", (getter)THPIndexFillBackward1_index_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_index", (getter)THPIndexFillBackward1_index_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPIndexPutImplBackward0_accumulate_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<IndexPutImplBackward0*>(self->cdata.get())->accumulate;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexPutImplBackward0_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<IndexPutImplBackward0*>(self->cdata.get());
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

PyObject* THPIndexPutImplBackward0_indices_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<IndexPutImplBackward0*>(self->cdata.get());
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

static struct PyGetSetDef IndexPutImplBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_accumulate", (getter)THPIndexPutImplBackward0_accumulate_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_indices", (getter)THPIndexPutImplBackward0_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_indices", (getter)THPIndexPutImplBackward0_indices_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLinalgPinvBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgPinvBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgPinvBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgPinvBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgPinvBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgPinvBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgPinvBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgPinvBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LinalgPinvBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPLinalgPinvBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPLinalgPinvBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPLinalgPinvBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPLinalgPinvBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPKthvalueBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<KthvalueBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPKthvalueBackward0_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<KthvalueBackward0*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPKthvalueBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<KthvalueBackward0*>(self->cdata.get())->self_sym_sizes;
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

PyObject* THPKthvalueBackward0_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<KthvalueBackward0*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPKthvalueBackward0_indices_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<KthvalueBackward0*>(self->cdata.get())->indices_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef KthvalueBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPKthvalueBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPKthvalueBackward0_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPKthvalueBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_indices", (getter)THPKthvalueBackward0_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_indices", (getter)THPKthvalueBackward0_indices_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPDigammaBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<DigammaBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPDigammaBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<DigammaBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef DigammaBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPDigammaBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPDigammaBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPPolygammaBackward0_n_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<PolygammaBackward0*>(self->cdata.get())->n;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPPolygammaBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PolygammaBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPPolygammaBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PolygammaBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef PolygammaBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_n", (getter)THPPolygammaBackward0_n_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPPolygammaBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPPolygammaBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPPolygammaBackward1_n_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<PolygammaBackward1*>(self->cdata.get())->n;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPPolygammaBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PolygammaBackward1*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPPolygammaBackward1_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PolygammaBackward1*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef PolygammaBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_n", (getter)THPPolygammaBackward1_n_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPPolygammaBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPPolygammaBackward1_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPXlogyBackward0_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<XlogyBackward0*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPXlogyBackward0_other_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<XlogyBackward0*>(self->cdata.get())->other_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPXlogyBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<XlogyBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPXlogyBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<XlogyBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef XlogyBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPXlogyBackward0_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_other", (getter)THPXlogyBackward0_other_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPXlogyBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPXlogyBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPXlogyBackward1_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<XlogyBackward1*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPXlogyBackward1_other_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<XlogyBackward1*>(self->cdata.get())->other_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPXlogyBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<XlogyBackward1*>(self->cdata.get())->self;
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

static struct PyGetSetDef XlogyBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPXlogyBackward1_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_other", (getter)THPXlogyBackward1_other_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPXlogyBackward1_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPXlogyBackward2_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<XlogyBackward2*>(self->cdata.get())->other;
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

PyObject* THPXlogyBackward2_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<XlogyBackward2*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPXlogyBackward2_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<XlogyBackward2*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef XlogyBackward2_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPXlogyBackward2_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPXlogyBackward2_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPXlogyBackward2_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLinalgLuFactorExBackward0_pivot_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LinalgLuFactorExBackward0*>(self->cdata.get())->pivot;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgLuFactorExBackward0_LU_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgLuFactorExBackward0*>(self->cdata.get())->LU_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgLuFactorExBackward0_LU_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgLuFactorExBackward0*>(self->cdata.get())->LU_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgLuFactorExBackward0_pivots_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgLuFactorExBackward0*>(self->cdata.get())->pivots_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgLuFactorExBackward0_pivots_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgLuFactorExBackward0*>(self->cdata.get())->pivots_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LinalgLuFactorExBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_pivot", (getter)THPLinalgLuFactorExBackward0_pivot_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_LU", (getter)THPLinalgLuFactorExBackward0_LU_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_LU", (getter)THPLinalgLuFactorExBackward0_LU_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_pivots", (getter)THPLinalgLuFactorExBackward0_pivots_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_pivots", (getter)THPLinalgLuFactorExBackward0_pivots_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMaximumBackward0_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaximumBackward0*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaximumBackward0_other_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaximumBackward0*>(self->cdata.get())->other_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaximumBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaximumBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaximumBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaximumBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MaximumBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPMaximumBackward0_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_other", (getter)THPMaximumBackward0_other_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPMaximumBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPMaximumBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMinimumBackward0_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MinimumBackward0*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMinimumBackward0_other_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MinimumBackward0*>(self->cdata.get())->other_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMinimumBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MinimumBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMinimumBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MinimumBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MinimumBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPMinimumBackward0_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_other", (getter)THPMinimumBackward0_other_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPMinimumBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPMinimumBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAmaxBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AmaxBackward0*>(self->cdata.get())->dim;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAmaxBackward0_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AmaxBackward0*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAmaxBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AmaxBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAmaxBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AmaxBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPAmaxBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AmaxBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAmaxBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AmaxBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AmaxBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPAmaxBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPAmaxBackward0_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPAmaxBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPAmaxBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPAmaxBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPAmaxBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAminBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AminBackward0*>(self->cdata.get())->dim;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAminBackward0_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AminBackward0*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAminBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AminBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAminBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AminBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPAminBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AminBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAminBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AminBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AminBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPAminBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPAminBackward0_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPAminBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPAminBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPAminBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPAminBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNanToNumBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NanToNumBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNanToNumBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NanToNumBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NanToNumBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPNanToNumBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPNanToNumBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNativeBatchNormBackward0_eps_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NativeBatchNormBackward0*>(self->cdata.get())->eps;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackward0_running_mean_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackward0*>(self->cdata.get())->running_mean_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackward0_running_mean_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackward0*>(self->cdata.get())->running_mean_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackward0_running_var_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackward0*>(self->cdata.get())->running_var_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackward0_running_var_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackward0*>(self->cdata.get())->running_var_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackward0_training_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NativeBatchNormBackward0*>(self->cdata.get())->training;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackward0_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackward0*>(self->cdata.get())->result2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackward0_result2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackward0*>(self->cdata.get())->result2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NativeBatchNormBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_eps", (getter)THPNativeBatchNormBackward0_eps_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPNativeBatchNormBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPNativeBatchNormBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_mean", (getter)THPNativeBatchNormBackward0_running_mean_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_running_mean", (getter)THPNativeBatchNormBackward0_running_mean_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_var", (getter)THPNativeBatchNormBackward0_running_var_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_running_var", (getter)THPNativeBatchNormBackward0_running_var_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_training", (getter)THPNativeBatchNormBackward0_training_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPNativeBatchNormBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPNativeBatchNormBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPNativeBatchNormBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPNativeBatchNormBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPNativeBatchNormBackward0_result2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result2", (getter)THPNativeBatchNormBackward0_result2_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNativeBatchNormBackwardBackward0_eps_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NativeBatchNormBackwardBackward0*>(self->cdata.get())->eps;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackwardBackward0_grad_out_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackwardBackward0*>(self->cdata.get())->grad_out_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackwardBackward0_grad_out_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackwardBackward0*>(self->cdata.get())->grad_out_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackwardBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackwardBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackwardBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackwardBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackwardBackward0_running_mean_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackwardBackward0*>(self->cdata.get())->running_mean_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackwardBackward0_running_mean_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackwardBackward0*>(self->cdata.get())->running_mean_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackwardBackward0_running_var_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackwardBackward0*>(self->cdata.get())->running_var_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackwardBackward0_running_var_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackwardBackward0*>(self->cdata.get())->running_var_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackwardBackward0_save_invstd_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackwardBackward0*>(self->cdata.get())->save_invstd_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackwardBackward0_save_invstd_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackwardBackward0*>(self->cdata.get())->save_invstd_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackwardBackward0_save_mean_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackwardBackward0*>(self->cdata.get())->save_mean_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackwardBackward0_save_mean_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackwardBackward0*>(self->cdata.get())->save_mean_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackwardBackward0_train_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NativeBatchNormBackwardBackward0*>(self->cdata.get())->train;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackwardBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackwardBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackwardBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackwardBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NativeBatchNormBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_eps", (getter)THPNativeBatchNormBackwardBackward0_eps_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_out", (getter)THPNativeBatchNormBackwardBackward0_grad_out_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_grad_out", (getter)THPNativeBatchNormBackwardBackward0_grad_out_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPNativeBatchNormBackwardBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPNativeBatchNormBackwardBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_mean", (getter)THPNativeBatchNormBackwardBackward0_running_mean_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_running_mean", (getter)THPNativeBatchNormBackwardBackward0_running_mean_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_var", (getter)THPNativeBatchNormBackwardBackward0_running_var_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_running_var", (getter)THPNativeBatchNormBackwardBackward0_running_var_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_save_invstd", (getter)THPNativeBatchNormBackwardBackward0_save_invstd_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_save_invstd", (getter)THPNativeBatchNormBackwardBackward0_save_invstd_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_save_mean", (getter)THPNativeBatchNormBackwardBackward0_save_mean_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_save_mean", (getter)THPNativeBatchNormBackwardBackward0_save_mean_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_train", (getter)THPNativeBatchNormBackwardBackward0_train_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPNativeBatchNormBackwardBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPNativeBatchNormBackwardBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNativeLayerNormBackward0_bias_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeLayerNormBackward0*>(self->cdata.get())->bias_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeLayerNormBackward0_bias_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeLayerNormBackward0*>(self->cdata.get())->bias_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeLayerNormBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeLayerNormBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeLayerNormBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeLayerNormBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeLayerNormBackward0_normalized_shape_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NativeLayerNormBackward0*>(self->cdata.get())->normalized_shape;
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

PyObject* THPNativeLayerNormBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeLayerNormBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeLayerNormBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeLayerNormBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeLayerNormBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeLayerNormBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeLayerNormBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeLayerNormBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeLayerNormBackward0_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeLayerNormBackward0*>(self->cdata.get())->result2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeLayerNormBackward0_result2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeLayerNormBackward0*>(self->cdata.get())->result2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NativeLayerNormBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_bias", (getter)THPNativeLayerNormBackward0_bias_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_bias", (getter)THPNativeLayerNormBackward0_bias_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPNativeLayerNormBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPNativeLayerNormBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_normalized_shape", (getter)THPNativeLayerNormBackward0_normalized_shape_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPNativeLayerNormBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPNativeLayerNormBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPNativeLayerNormBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPNativeLayerNormBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPNativeLayerNormBackward0_result2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result2", (getter)THPNativeLayerNormBackward0_result2_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef NeBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef NeBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPBatchNormBackwardBackward0_eps_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<BatchNormBackwardBackward0*>(self->cdata.get())->eps;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormBackwardBackward0_grad_out_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormBackwardBackward0*>(self->cdata.get())->grad_out_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormBackwardBackward0_grad_out_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormBackwardBackward0*>(self->cdata.get())->grad_out_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormBackwardBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormBackwardBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormBackwardBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormBackwardBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormBackwardBackward0_reserve_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormBackwardBackward0*>(self->cdata.get())->reserve_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormBackwardBackward0_reserve_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormBackwardBackward0*>(self->cdata.get())->reserve_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormBackwardBackward0_running_mean_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormBackwardBackward0*>(self->cdata.get())->running_mean_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormBackwardBackward0_running_mean_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormBackwardBackward0*>(self->cdata.get())->running_mean_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormBackwardBackward0_running_var_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormBackwardBackward0*>(self->cdata.get())->running_var_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormBackwardBackward0_running_var_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormBackwardBackward0*>(self->cdata.get())->running_var_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormBackwardBackward0_save_mean_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormBackwardBackward0*>(self->cdata.get())->save_mean_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormBackwardBackward0_save_mean_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormBackwardBackward0*>(self->cdata.get())->save_mean_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormBackwardBackward0_save_var_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormBackwardBackward0*>(self->cdata.get())->save_var_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormBackwardBackward0_save_var_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormBackwardBackward0*>(self->cdata.get())->save_var_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormBackwardBackward0_update_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<BatchNormBackwardBackward0*>(self->cdata.get())->update;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormBackwardBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormBackwardBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormBackwardBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormBackwardBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef BatchNormBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_eps", (getter)THPBatchNormBackwardBackward0_eps_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_out", (getter)THPBatchNormBackwardBackward0_grad_out_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_grad_out", (getter)THPBatchNormBackwardBackward0_grad_out_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPBatchNormBackwardBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPBatchNormBackwardBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reserve", (getter)THPBatchNormBackwardBackward0_reserve_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_reserve", (getter)THPBatchNormBackwardBackward0_reserve_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_mean", (getter)THPBatchNormBackwardBackward0_running_mean_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_running_mean", (getter)THPBatchNormBackwardBackward0_running_mean_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_var", (getter)THPBatchNormBackwardBackward0_running_var_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_running_var", (getter)THPBatchNormBackwardBackward0_running_var_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_save_mean", (getter)THPBatchNormBackwardBackward0_save_mean_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_save_mean", (getter)THPBatchNormBackwardBackward0_save_mean_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_save_var", (getter)THPBatchNormBackwardBackward0_save_var_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_save_var", (getter)THPBatchNormBackwardBackward0_save_var_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_update", (getter)THPBatchNormBackwardBackward0_update_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPBatchNormBackwardBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPBatchNormBackwardBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef NextafterBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPCdistBackward0_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CdistBackward0*>(self->cdata.get())->p;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCdistBackward0_x1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CdistBackward0*>(self->cdata.get())->x1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCdistBackward0_x1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CdistBackward0*>(self->cdata.get())->x1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCdistBackward0_x2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CdistBackward0*>(self->cdata.get())->x2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCdistBackward0_x2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CdistBackward0*>(self->cdata.get())->x2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCdistBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CdistBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCdistBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CdistBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CdistBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_p", (getter)THPCdistBackward0_p_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_x1", (getter)THPCdistBackward0_x1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_x1", (getter)THPCdistBackward0_x1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_x2", (getter)THPCdistBackward0_x2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_x2", (getter)THPCdistBackward0_x2_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPCdistBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPCdistBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNormalBackward1_mean_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NormalBackward1*>(self->cdata.get())->mean_sym_sizes;
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

static struct PyGetSetDef NormalBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_mean_sym_sizes", (getter)THPNormalBackward1_mean_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNormalBackward2_std_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NormalBackward2*>(self->cdata.get())->std_sym_sizes;
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

static struct PyGetSetDef NormalBackward2_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_std_sym_sizes", (getter)THPNormalBackward2_std_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNormalBackward3_mean_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NormalBackward3*>(self->cdata.get())->mean_sym_sizes;
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

PyObject* THPNormalBackward3_std_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NormalBackward3*>(self->cdata.get())->std_sym_sizes;
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

static struct PyGetSetDef NormalBackward3_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_mean_sym_sizes", (getter)THPNormalBackward3_mean_sym_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_std_sym_sizes", (getter)THPNormalBackward3_std_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSpecialLogNdtrBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialLogNdtrBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialLogNdtrBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialLogNdtrBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialLogNdtrBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialLogNdtrBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialLogNdtrBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialLogNdtrBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SpecialLogNdtrBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSpecialLogNdtrBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSpecialLogNdtrBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPSpecialLogNdtrBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPSpecialLogNdtrBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef RoundBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef RoundBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPScatterAddBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ScatterAddBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPScatterAddBackward0_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScatterAddBackward0*>(self->cdata.get())->index_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScatterAddBackward0_index_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScatterAddBackward0*>(self->cdata.get())->index_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ScatterAddBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPScatterAddBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_index", (getter)THPScatterAddBackward0_index_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_index", (getter)THPScatterAddBackward0_index_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSelectBackwardBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SelectBackwardBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSelectBackwardBackward0_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SelectBackwardBackward0*>(self->cdata.get())->index;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SelectBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSelectBackwardBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_index", (getter)THPSelectBackwardBackward0_index_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSigmoidBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SigmoidBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSigmoidBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SigmoidBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SigmoidBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_result", (getter)THPSigmoidBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPSigmoidBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSliceScatterBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SliceScatterBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSliceScatterBackward0_end_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<SliceScatterBackward0*>(self->cdata.get())->end;
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

PyObject* THPSliceScatterBackward0_start_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<SliceScatterBackward0*>(self->cdata.get())->start;
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

PyObject* THPSliceScatterBackward0_step_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SliceScatterBackward0*>(self->cdata.get())->step;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SliceScatterBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSliceScatterBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_end", (getter)THPSliceScatterBackward0_end_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_start", (getter)THPSliceScatterBackward0_start_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_step", (getter)THPSliceScatterBackward0_step_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSplitWithSizesBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SplitWithSizesBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSplitWithSizesBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SplitWithSizesBackward0*>(self->cdata.get())->self_sym_sizes;
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

PyObject* THPSplitWithSizesBackward0_split_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SplitWithSizesBackward0*>(self->cdata.get())->split_sizes;
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

static struct PyGetSetDef SplitWithSizesBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSplitWithSizesBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPSplitWithSizesBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_split_sizes", (getter)THPSplitWithSizesBackward0_split_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSplitWithSizesBackwardAutogradNestedTensor0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SplitWithSizesBackwardAutogradNestedTensor0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSplitWithSizesBackwardAutogradNestedTensor0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SplitWithSizesBackwardAutogradNestedTensor0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSplitWithSizesBackwardAutogradNestedTensor0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SplitWithSizesBackwardAutogradNestedTensor0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSplitWithSizesBackwardAutogradNestedTensor0_split_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SplitWithSizesBackwardAutogradNestedTensor0*>(self->cdata.get())->split_sizes;
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

static struct PyGetSetDef SplitWithSizesBackwardAutogradNestedTensor0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSplitWithSizesBackwardAutogradNestedTensor0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSplitWithSizesBackwardAutogradNestedTensor0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSplitWithSizesBackwardAutogradNestedTensor0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_split_sizes", (getter)THPSplitWithSizesBackwardAutogradNestedTensor0_split_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef TBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPTanhBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TanhBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPTanhBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TanhBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef TanhBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_result", (getter)THPTanhBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPTanhBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPTopkBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TopkBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPTopkBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TopkBackward0*>(self->cdata.get())->self_sym_sizes;
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

PyObject* THPTopkBackward0_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TopkBackward0*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPTopkBackward0_indices_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TopkBackward0*>(self->cdata.get())->indices_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef TopkBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPTopkBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPTopkBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_indices", (getter)THPTopkBackward0_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_indices", (getter)THPTopkBackward0_indices_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPToSparseCscBackward0_self_self_sym_blocksize_opt_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<ToSparseCscBackward0*>(self->cdata.get())->self_self_sym_blocksize_opt;
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

static struct PyGetSetDef ToSparseCscBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_self_sym_blocksize_opt", (getter)THPToSparseCscBackward0_self_self_sym_blocksize_opt_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPToSparseBscBackward0_self_self_sym_blocksize_opt_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<ToSparseBscBackward0*>(self->cdata.get())->self_self_sym_blocksize_opt;
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

static struct PyGetSetDef ToSparseBscBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_self_sym_blocksize_opt", (getter)THPToSparseBscBackward0_self_self_sym_blocksize_opt_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUnfoldBackwardBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnfoldBackwardBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUnfoldBackwardBackward0_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnfoldBackwardBackward0*>(self->cdata.get())->size;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUnfoldBackwardBackward0_step_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnfoldBackwardBackward0*>(self->cdata.get())->step;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UnfoldBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPUnfoldBackwardBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_size", (getter)THPUnfoldBackwardBackward0_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_step", (getter)THPUnfoldBackwardBackward0_step_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef UniformBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef UniqueDimBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef UniqueConsecutiveBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPBinaryCrossEntropyBackwardBackward0_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyBackwardBackward0*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyBackwardBackward0_grad_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyBackwardBackward0*>(self->cdata.get())->grad_output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyBackwardBackward0_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<BinaryCrossEntropyBackwardBackward0*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyBackwardBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyBackwardBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyBackwardBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyBackwardBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyBackwardBackward0_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyBackwardBackward0*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyBackwardBackward0_target_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyBackwardBackward0*>(self->cdata.get())->target_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyBackwardBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyBackwardBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyBackwardBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyBackwardBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef BinaryCrossEntropyBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_output", (getter)THPBinaryCrossEntropyBackwardBackward0_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_grad_output", (getter)THPBinaryCrossEntropyBackwardBackward0_grad_output_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPBinaryCrossEntropyBackwardBackward0_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPBinaryCrossEntropyBackwardBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPBinaryCrossEntropyBackwardBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPBinaryCrossEntropyBackwardBackward0_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_target", (getter)THPBinaryCrossEntropyBackwardBackward0_target_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPBinaryCrossEntropyBackwardBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPBinaryCrossEntropyBackwardBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMseLossBackward0_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MseLossBackward0*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMseLossBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MseLossBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMseLossBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MseLossBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMseLossBackward0_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MseLossBackward0*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMseLossBackward0_target_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MseLossBackward0*>(self->cdata.get())->target_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MseLossBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_reduction", (getter)THPMseLossBackward0_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPMseLossBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPMseLossBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPMseLossBackward0_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_target", (getter)THPMseLossBackward0_target_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNllLoss2DBackward0_ignore_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NllLoss2DBackward0*>(self->cdata.get())->ignore_index;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLoss2DBackward0_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NllLoss2DBackward0*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLoss2DBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLoss2DBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLoss2DBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLoss2DBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLoss2DBackward0_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLoss2DBackward0*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLoss2DBackward0_target_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLoss2DBackward0*>(self->cdata.get())->target_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLoss2DBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLoss2DBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLoss2DBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLoss2DBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLoss2DBackward0_total_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLoss2DBackward0*>(self->cdata.get())->total_weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLoss2DBackward0_total_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLoss2DBackward0*>(self->cdata.get())->total_weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NllLoss2DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_ignore_index", (getter)THPNllLoss2DBackward0_ignore_index_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPNllLoss2DBackward0_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPNllLoss2DBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPNllLoss2DBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPNllLoss2DBackward0_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_target", (getter)THPNllLoss2DBackward0_target_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPNllLoss2DBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPNllLoss2DBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_total_weight", (getter)THPNllLoss2DBackward0_total_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_total_weight", (getter)THPNllLoss2DBackward0_total_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSoftMarginLossBackward0_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SoftMarginLossBackward0*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftMarginLossBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftMarginLossBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftMarginLossBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftMarginLossBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftMarginLossBackward0_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftMarginLossBackward0*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftMarginLossBackward0_target_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftMarginLossBackward0*>(self->cdata.get())->target_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SoftMarginLossBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_reduction", (getter)THPSoftMarginLossBackward0_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSoftMarginLossBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSoftMarginLossBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPSoftMarginLossBackward0_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_target", (getter)THPSoftMarginLossBackward0_target_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMishBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MishBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMishBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MishBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MishBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMishBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPMishBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPGeluBackward0_approximate_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<GeluBackward0*>(self->cdata.get())->approximate;
  return PyUnicode_FromStringAndSize(prop.data(), prop.size());
  END_HANDLE_TH_ERRORS
}

PyObject* THPGeluBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GeluBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPGeluBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GeluBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef GeluBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_approximate", (getter)THPGeluBackward0_approximate_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPGeluBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPGeluBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPGluBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<GluBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPGluBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GluBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPGluBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GluBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef GluBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPGluBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPGluBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPGluBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPHardshrinkBackward0_lambd_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<HardshrinkBackward0*>(self->cdata.get())->lambd;
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

PyObject* THPHardshrinkBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HardshrinkBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPHardshrinkBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HardshrinkBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef HardshrinkBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_lambd", (getter)THPHardshrinkBackward0_lambd_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPHardshrinkBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPHardshrinkBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPHardtanhBackward0_max_val_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<HardtanhBackward0*>(self->cdata.get())->max_val;
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

PyObject* THPHardtanhBackward0_min_val_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<HardtanhBackward0*>(self->cdata.get())->min_val;
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

PyObject* THPHardtanhBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HardtanhBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPHardtanhBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HardtanhBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef HardtanhBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_max_val", (getter)THPHardtanhBackward0_max_val_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_min_val", (getter)THPHardtanhBackward0_min_val_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPHardtanhBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPHardtanhBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLeakyReluBackward0_negative_slope_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LeakyReluBackward0*>(self->cdata.get())->negative_slope;
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

PyObject* THPLeakyReluBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LeakyReluBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLeakyReluBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LeakyReluBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LeakyReluBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_negative_slope", (getter)THPLeakyReluBackward0_negative_slope_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPLeakyReluBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPLeakyReluBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLogSigmoidBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogSigmoidBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogSigmoidBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogSigmoidBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogSigmoidBackward0_buffer_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogSigmoidBackward0*>(self->cdata.get())->buffer_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogSigmoidBackward0_buffer_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogSigmoidBackward0*>(self->cdata.get())->buffer_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LogSigmoidBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPLogSigmoidBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPLogSigmoidBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_buffer", (getter)THPLogSigmoidBackward0_buffer_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_buffer", (getter)THPLogSigmoidBackward0_buffer_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPRreluWithNoiseBackward1_lower_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RreluWithNoiseBackward1*>(self->cdata.get())->lower;
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

PyObject* THPRreluWithNoiseBackward1_noise_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RreluWithNoiseBackward1*>(self->cdata.get())->noise_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPRreluWithNoiseBackward1_noise_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RreluWithNoiseBackward1*>(self->cdata.get())->noise_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPRreluWithNoiseBackward1_training_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RreluWithNoiseBackward1*>(self->cdata.get())->training;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPRreluWithNoiseBackward1_upper_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RreluWithNoiseBackward1*>(self->cdata.get())->upper;
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

PyObject* THPRreluWithNoiseBackward1_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RreluWithNoiseBackward1*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPRreluWithNoiseBackward1_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RreluWithNoiseBackward1*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef RreluWithNoiseBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_lower", (getter)THPRreluWithNoiseBackward1_lower_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_noise", (getter)THPRreluWithNoiseBackward1_noise_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_noise", (getter)THPRreluWithNoiseBackward1_noise_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_training", (getter)THPRreluWithNoiseBackward1_training_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_upper", (getter)THPRreluWithNoiseBackward1_upper_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPRreluWithNoiseBackward1_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPRreluWithNoiseBackward1_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSoftshrinkBackward0_lambd_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SoftshrinkBackward0*>(self->cdata.get())->lambd;
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

PyObject* THPSoftshrinkBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftshrinkBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftshrinkBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftshrinkBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SoftshrinkBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_lambd", (getter)THPSoftshrinkBackward0_lambd_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSoftshrinkBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSoftshrinkBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPReflectionPad3DBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ReflectionPad3DBackward0*>(self->cdata.get())->padding;
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

PyObject* THPReflectionPad3DBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ReflectionPad3DBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPReflectionPad3DBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ReflectionPad3DBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ReflectionPad3DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_padding", (getter)THPReflectionPad3DBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPReflectionPad3DBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPReflectionPad3DBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPReplicationPad1DBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ReplicationPad1DBackward0*>(self->cdata.get())->padding;
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

PyObject* THPReplicationPad1DBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ReplicationPad1DBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPReplicationPad1DBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ReplicationPad1DBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ReplicationPad1DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_padding", (getter)THPReplicationPad1DBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPReplicationPad1DBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPReplicationPad1DBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPReplicationPad2DBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ReplicationPad2DBackward0*>(self->cdata.get())->padding;
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

PyObject* THPReplicationPad2DBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ReplicationPad2DBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPReplicationPad2DBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ReplicationPad2DBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ReplicationPad2DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_padding", (getter)THPReplicationPad2DBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPReplicationPad2DBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPReplicationPad2DBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleNearest1DBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleNearest1DBackward0*>(self->cdata.get())->output_size;
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

PyObject* THPUpsampleNearest1DBackward0_scales_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest1DBackward0*>(self->cdata.get())->scales;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest1DBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleNearest1DBackward0*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef UpsampleNearest1DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_output_size", (getter)THPUpsampleNearest1DBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales", (getter)THPUpsampleNearest1DBackward0_scales_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPUpsampleNearest1DBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleNearestExact1DBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleNearestExact1DBackward0*>(self->cdata.get())->output_size;
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

PyObject* THPUpsampleNearestExact1DBackward0_scales_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearestExact1DBackward0*>(self->cdata.get())->scales;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearestExact1DBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleNearestExact1DBackward0*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef UpsampleNearestExact1DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_output_size", (getter)THPUpsampleNearestExact1DBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales", (getter)THPUpsampleNearestExact1DBackward0_scales_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPUpsampleNearestExact1DBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPFractionalMaxPool3DBackward0_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FractionalMaxPool3DBackward0*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPFractionalMaxPool3DBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FractionalMaxPool3DBackward0*>(self->cdata.get())->output_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPFractionalMaxPool3DBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FractionalMaxPool3DBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFractionalMaxPool3DBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FractionalMaxPool3DBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPFractionalMaxPool3DBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FractionalMaxPool3DBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFractionalMaxPool3DBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FractionalMaxPool3DBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FractionalMaxPool3DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_kernel_size", (getter)THPFractionalMaxPool3DBackward0_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPFractionalMaxPool3DBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPFractionalMaxPool3DBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPFractionalMaxPool3DBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPFractionalMaxPool3DBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPFractionalMaxPool3DBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMpsConvolutionBackward0_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MpsConvolutionBackward0*>(self->cdata.get())->dilation;
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

PyObject* THPMpsConvolutionBackward0_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MpsConvolutionBackward0*>(self->cdata.get())->groups;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMpsConvolutionBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MpsConvolutionBackward0*>(self->cdata.get())->padding;
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

PyObject* THPMpsConvolutionBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MpsConvolutionBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMpsConvolutionBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MpsConvolutionBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMpsConvolutionBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MpsConvolutionBackward0*>(self->cdata.get())->stride;
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

PyObject* THPMpsConvolutionBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MpsConvolutionBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMpsConvolutionBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MpsConvolutionBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MpsConvolutionBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dilation", (getter)THPMpsConvolutionBackward0_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_groups", (getter)THPMpsConvolutionBackward0_groups_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPMpsConvolutionBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPMpsConvolutionBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPMpsConvolutionBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPMpsConvolutionBackward0_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPMpsConvolutionBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPMpsConvolutionBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPConvDepthwise2DBackward0_bias_sym_sizes_opt_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<ConvDepthwise2DBackward0*>(self->cdata.get())->bias_sym_sizes_opt;
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

PyObject* THPConvDepthwise2DBackward0_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvDepthwise2DBackward0*>(self->cdata.get())->dilation;
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

PyObject* THPConvDepthwise2DBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvDepthwise2DBackward0*>(self->cdata.get())->padding;
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

PyObject* THPConvDepthwise2DBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvDepthwise2DBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvDepthwise2DBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvDepthwise2DBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvDepthwise2DBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvDepthwise2DBackward0*>(self->cdata.get())->stride;
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

PyObject* THPConvDepthwise2DBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvDepthwise2DBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvDepthwise2DBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvDepthwise2DBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ConvDepthwise2DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_bias_sym_sizes_opt", (getter)THPConvDepthwise2DBackward0_bias_sym_sizes_opt_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPConvDepthwise2DBackward0_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPConvDepthwise2DBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPConvDepthwise2DBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPConvDepthwise2DBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPConvDepthwise2DBackward0_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPConvDepthwise2DBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPConvDepthwise2DBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAdaptiveAvgPool3DBackwardBackward0_grad_output_sym_argsize_minus_1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AdaptiveAvgPool3DBackwardBackward0*>(self->cdata.get())->grad_output_sym_argsize_minus_1;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAdaptiveAvgPool3DBackwardBackward0_grad_output_sym_argsize_minus_2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AdaptiveAvgPool3DBackwardBackward0*>(self->cdata.get())->grad_output_sym_argsize_minus_2;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAdaptiveAvgPool3DBackwardBackward0_grad_output_sym_argsize_minus_3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AdaptiveAvgPool3DBackwardBackward0*>(self->cdata.get())->grad_output_sym_argsize_minus_3;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AdaptiveAvgPool3DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_output_sym_argsize_minus_1", (getter)THPAdaptiveAvgPool3DBackwardBackward0_grad_output_sym_argsize_minus_1_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output_sym_argsize_minus_2", (getter)THPAdaptiveAvgPool3DBackwardBackward0_grad_output_sym_argsize_minus_2_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output_sym_argsize_minus_3", (getter)THPAdaptiveAvgPool3DBackwardBackward0_grad_output_sym_argsize_minus_3_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAdaptiveMaxPool2DBackwardBackward0_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AdaptiveMaxPool2DBackwardBackward0*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAdaptiveMaxPool2DBackwardBackward0_indices_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AdaptiveMaxPool2DBackwardBackward0*>(self->cdata.get())->indices_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AdaptiveMaxPool2DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_indices", (getter)THPAdaptiveMaxPool2DBackwardBackward0_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_indices", (getter)THPAdaptiveMaxPool2DBackwardBackward0_indices_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAdaptiveMaxPool3DBackwardBackward0_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AdaptiveMaxPool3DBackwardBackward0*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAdaptiveMaxPool3DBackwardBackward0_indices_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AdaptiveMaxPool3DBackwardBackward0*>(self->cdata.get())->indices_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AdaptiveMaxPool3DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_indices", (getter)THPAdaptiveMaxPool3DBackwardBackward0_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_indices", (getter)THPAdaptiveMaxPool3DBackwardBackward0_indices_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLogSigmoidBackwardBackward0_buffer_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogSigmoidBackwardBackward0*>(self->cdata.get())->buffer_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogSigmoidBackwardBackward0_buffer_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogSigmoidBackwardBackward0*>(self->cdata.get())->buffer_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogSigmoidBackwardBackward0_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogSigmoidBackwardBackward0*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogSigmoidBackwardBackward0_grad_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogSigmoidBackwardBackward0*>(self->cdata.get())->grad_output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogSigmoidBackwardBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogSigmoidBackwardBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogSigmoidBackwardBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogSigmoidBackwardBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LogSigmoidBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_buffer", (getter)THPLogSigmoidBackwardBackward0_buffer_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_buffer", (getter)THPLogSigmoidBackwardBackward0_buffer_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output", (getter)THPLogSigmoidBackwardBackward0_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_grad_output", (getter)THPLogSigmoidBackwardBackward0_grad_output_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPLogSigmoidBackwardBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPLogSigmoidBackwardBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLogSoftmaxBackwardDataBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LogSoftmaxBackwardDataBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogSoftmaxBackwardDataBackward0_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogSoftmaxBackwardDataBackward0*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogSoftmaxBackwardDataBackward0_grad_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogSoftmaxBackwardDataBackward0*>(self->cdata.get())->grad_output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogSoftmaxBackwardDataBackward0_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogSoftmaxBackwardDataBackward0*>(self->cdata.get())->output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogSoftmaxBackwardDataBackward0_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogSoftmaxBackwardDataBackward0*>(self->cdata.get())->output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LogSoftmaxBackwardDataBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPLogSoftmaxBackwardDataBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output", (getter)THPLogSoftmaxBackwardDataBackward0_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_grad_output", (getter)THPLogSoftmaxBackwardDataBackward0_grad_output_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output", (getter)THPLogSoftmaxBackwardDataBackward0_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_output", (getter)THPLogSoftmaxBackwardDataBackward0_output_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNllLoss2DBackwardBackward0_ignore_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NllLoss2DBackwardBackward0*>(self->cdata.get())->ignore_index;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLoss2DBackwardBackward0_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NllLoss2DBackwardBackward0*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLoss2DBackwardBackward0_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLoss2DBackwardBackward0*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLoss2DBackwardBackward0_target_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLoss2DBackwardBackward0*>(self->cdata.get())->target_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLoss2DBackwardBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLoss2DBackwardBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLoss2DBackwardBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLoss2DBackwardBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NllLoss2DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_ignore_index", (getter)THPNllLoss2DBackwardBackward0_ignore_index_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPNllLoss2DBackwardBackward0_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPNllLoss2DBackwardBackward0_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_target", (getter)THPNllLoss2DBackwardBackward0_target_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPNllLoss2DBackwardBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPNllLoss2DBackwardBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSparseSampledAddmmBackward0_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SparseSampledAddmmBackward0*>(self->cdata.get())->alpha;
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

PyObject* THPSparseSampledAddmmBackward0_beta_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SparseSampledAddmmBackward0*>(self->cdata.get())->beta;
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

PyObject* THPSparseSampledAddmmBackward0_mat1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseSampledAddmmBackward0*>(self->cdata.get())->mat1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseSampledAddmmBackward0_mat1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseSampledAddmmBackward0*>(self->cdata.get())->mat1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseSampledAddmmBackward0_mat2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseSampledAddmmBackward0*>(self->cdata.get())->mat2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseSampledAddmmBackward0_mat2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseSampledAddmmBackward0*>(self->cdata.get())->mat2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseSampledAddmmBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseSampledAddmmBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseSampledAddmmBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseSampledAddmmBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SparseSampledAddmmBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_alpha", (getter)THPSparseSampledAddmmBackward0_alpha_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_beta", (getter)THPSparseSampledAddmmBackward0_beta_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mat1", (getter)THPSparseSampledAddmmBackward0_mat1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_mat1", (getter)THPSparseSampledAddmmBackward0_mat1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mat2", (getter)THPSparseSampledAddmmBackward0_mat2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_mat2", (getter)THPSparseSampledAddmmBackward0_mat2_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSparseSampledAddmmBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSparseSampledAddmmBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSoftmaxBackwardDataBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SoftmaxBackwardDataBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftmaxBackwardDataBackward0_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftmaxBackwardDataBackward0*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftmaxBackwardDataBackward0_grad_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftmaxBackwardDataBackward0*>(self->cdata.get())->grad_output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftmaxBackwardDataBackward0_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftmaxBackwardDataBackward0*>(self->cdata.get())->output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftmaxBackwardDataBackward0_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftmaxBackwardDataBackward0*>(self->cdata.get())->output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SoftmaxBackwardDataBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSoftmaxBackwardDataBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output", (getter)THPSoftmaxBackwardDataBackward0_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_grad_output", (getter)THPSoftmaxBackwardDataBackward0_grad_output_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output", (getter)THPSoftmaxBackwardDataBackward0_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_output", (getter)THPSoftmaxBackwardDataBackward0_output_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCudnnGridSamplerBackward0_grid_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnGridSamplerBackward0*>(self->cdata.get())->grid_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnGridSamplerBackward0_grid_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnGridSamplerBackward0*>(self->cdata.get())->grid_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnGridSamplerBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnGridSamplerBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnGridSamplerBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnGridSamplerBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CudnnGridSamplerBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grid", (getter)THPCudnnGridSamplerBackward0_grid_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_grid", (getter)THPCudnnGridSamplerBackward0_grid_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPCudnnGridSamplerBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPCudnnGridSamplerBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNnpackSpatialConvolutionBackward0_bias_sym_sizes_opt_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<NnpackSpatialConvolutionBackward0*>(self->cdata.get())->bias_sym_sizes_opt;
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

PyObject* THPNnpackSpatialConvolutionBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NnpackSpatialConvolutionBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNnpackSpatialConvolutionBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NnpackSpatialConvolutionBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNnpackSpatialConvolutionBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NnpackSpatialConvolutionBackward0*>(self->cdata.get())->padding;
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

PyObject* THPNnpackSpatialConvolutionBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NnpackSpatialConvolutionBackward0*>(self->cdata.get())->stride;
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

PyObject* THPNnpackSpatialConvolutionBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NnpackSpatialConvolutionBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNnpackSpatialConvolutionBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NnpackSpatialConvolutionBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NnpackSpatialConvolutionBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_bias_sym_sizes_opt", (getter)THPNnpackSpatialConvolutionBackward0_bias_sym_sizes_opt_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPNnpackSpatialConvolutionBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPNnpackSpatialConvolutionBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPNnpackSpatialConvolutionBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPNnpackSpatialConvolutionBackward0_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPNnpackSpatialConvolutionBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPNnpackSpatialConvolutionBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLstmMpsBackward0_batch_first_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LstmMpsBackward0*>(self->cdata.get())->batch_first;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPLstmMpsBackward0_bidirectional_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LstmMpsBackward0*>(self->cdata.get())->bidirectional;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPLstmMpsBackward0_dropout_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LstmMpsBackward0*>(self->cdata.get())->dropout;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPLstmMpsBackward0_has_biases_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LstmMpsBackward0*>(self->cdata.get())->has_biases;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPLstmMpsBackward0_hx_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<LstmMpsBackward0*>(self->cdata.get());
  const auto& prop = node->hx_;
  if (node->hx_released_) {
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

PyObject* THPLstmMpsBackward0_hx_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<LstmMpsBackward0*>(self->cdata.get());
  const auto& prop = node->hx_;
  if (node->hx_released_) {
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

PyObject* THPLstmMpsBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LstmMpsBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLstmMpsBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LstmMpsBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLstmMpsBackward0_num_layers_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LstmMpsBackward0*>(self->cdata.get())->num_layers;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPLstmMpsBackward0_params_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<LstmMpsBackward0*>(self->cdata.get());
  const auto& prop = node->params_;
  if (node->params_released_) {
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

PyObject* THPLstmMpsBackward0_params_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<LstmMpsBackward0*>(self->cdata.get());
  const auto& prop = node->params_;
  if (node->params_released_) {
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

PyObject* THPLstmMpsBackward0_train_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LstmMpsBackward0*>(self->cdata.get())->train;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPLstmMpsBackward0_result3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LstmMpsBackward0*>(self->cdata.get())->result3_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLstmMpsBackward0_result3_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LstmMpsBackward0*>(self->cdata.get())->result3_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLstmMpsBackward0_result4_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LstmMpsBackward0*>(self->cdata.get())->result4_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLstmMpsBackward0_result4_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LstmMpsBackward0*>(self->cdata.get())->result4_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLstmMpsBackward0_result5_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LstmMpsBackward0*>(self->cdata.get())->result5_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLstmMpsBackward0_result5_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LstmMpsBackward0*>(self->cdata.get())->result5_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LstmMpsBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_batch_first", (getter)THPLstmMpsBackward0_batch_first_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_bidirectional", (getter)THPLstmMpsBackward0_bidirectional_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dropout", (getter)THPLstmMpsBackward0_dropout_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_has_biases", (getter)THPLstmMpsBackward0_has_biases_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_hx", (getter)THPLstmMpsBackward0_hx_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_hx", (getter)THPLstmMpsBackward0_hx_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPLstmMpsBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPLstmMpsBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_num_layers", (getter)THPLstmMpsBackward0_num_layers_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_params", (getter)THPLstmMpsBackward0_params_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_params", (getter)THPLstmMpsBackward0_params_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_train", (getter)THPLstmMpsBackward0_train_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result3", (getter)THPLstmMpsBackward0_result3_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result3", (getter)THPLstmMpsBackward0_result3_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result4", (getter)THPLstmMpsBackward0_result4_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result4", (getter)THPLstmMpsBackward0_result4_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result5", (getter)THPLstmMpsBackward0_result5_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result5", (getter)THPLstmMpsBackward0_result5_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMiopenDepthwiseConvolutionBackward0_bias_sym_sizes_opt_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<MiopenDepthwiseConvolutionBackward0*>(self->cdata.get())->bias_sym_sizes_opt;
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

PyObject* THPMiopenDepthwiseConvolutionBackward0_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenDepthwiseConvolutionBackward0*>(self->cdata.get())->dilation;
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

PyObject* THPMiopenDepthwiseConvolutionBackward0_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenDepthwiseConvolutionBackward0*>(self->cdata.get())->groups;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenDepthwiseConvolutionBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenDepthwiseConvolutionBackward0*>(self->cdata.get())->padding;
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

PyObject* THPMiopenDepthwiseConvolutionBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenDepthwiseConvolutionBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenDepthwiseConvolutionBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenDepthwiseConvolutionBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenDepthwiseConvolutionBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenDepthwiseConvolutionBackward0*>(self->cdata.get())->stride;
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

PyObject* THPMiopenDepthwiseConvolutionBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenDepthwiseConvolutionBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenDepthwiseConvolutionBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenDepthwiseConvolutionBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MiopenDepthwiseConvolutionBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_bias_sym_sizes_opt", (getter)THPMiopenDepthwiseConvolutionBackward0_bias_sym_sizes_opt_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPMiopenDepthwiseConvolutionBackward0_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_groups", (getter)THPMiopenDepthwiseConvolutionBackward0_groups_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPMiopenDepthwiseConvolutionBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPMiopenDepthwiseConvolutionBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPMiopenDepthwiseConvolutionBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPMiopenDepthwiseConvolutionBackward0_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPMiopenDepthwiseConvolutionBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPMiopenDepthwiseConvolutionBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMkldnnMaxPool2DBackward0_ceil_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnMaxPool2DBackward0*>(self->cdata.get())->ceil_mode;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnMaxPool2DBackward0_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnMaxPool2DBackward0*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnMaxPool2DBackward0_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnMaxPool2DBackward0*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnMaxPool2DBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnMaxPool2DBackward0*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnMaxPool2DBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnMaxPool2DBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnMaxPool2DBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnMaxPool2DBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnMaxPool2DBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnMaxPool2DBackward0*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnMaxPool2DBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnMaxPool2DBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnMaxPool2DBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnMaxPool2DBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MkldnnMaxPool2DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_ceil_mode", (getter)THPMkldnnMaxPool2DBackward0_ceil_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPMkldnnMaxPool2DBackward0_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_kernel_size", (getter)THPMkldnnMaxPool2DBackward0_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPMkldnnMaxPool2DBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPMkldnnMaxPool2DBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPMkldnnMaxPool2DBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPMkldnnMaxPool2DBackward0_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPMkldnnMaxPool2DBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPMkldnnMaxPool2DBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPScaledDotProductFlashAttentionBackward0_dropout_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ScaledDotProductFlashAttentionBackward0*>(self->cdata.get())->dropout_p;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionBackward0_is_causal_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ScaledDotProductFlashAttentionBackward0*>(self->cdata.get())->is_causal;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionBackward0_key_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFlashAttentionBackward0*>(self->cdata.get())->key_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionBackward0_key_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFlashAttentionBackward0*>(self->cdata.get())->key_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionBackward0_query_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFlashAttentionBackward0*>(self->cdata.get())->query_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionBackward0_query_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFlashAttentionBackward0*>(self->cdata.get())->query_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionBackward0_scale_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<ScaledDotProductFlashAttentionBackward0*>(self->cdata.get())->scale;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionBackward0_value_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFlashAttentionBackward0*>(self->cdata.get())->value_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionBackward0_value_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFlashAttentionBackward0*>(self->cdata.get())->value_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionBackward0_cum_seq_k_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFlashAttentionBackward0*>(self->cdata.get())->cum_seq_k_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionBackward0_cum_seq_k_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFlashAttentionBackward0*>(self->cdata.get())->cum_seq_k_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionBackward0_cum_seq_q_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFlashAttentionBackward0*>(self->cdata.get())->cum_seq_q_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionBackward0_cum_seq_q_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFlashAttentionBackward0*>(self->cdata.get())->cum_seq_q_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionBackward0_logsumexp_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFlashAttentionBackward0*>(self->cdata.get())->logsumexp_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionBackward0_logsumexp_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFlashAttentionBackward0*>(self->cdata.get())->logsumexp_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionBackward0_max_k_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ScaledDotProductFlashAttentionBackward0*>(self->cdata.get())->max_k;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionBackward0_max_q_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ScaledDotProductFlashAttentionBackward0*>(self->cdata.get())->max_q;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionBackward0_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFlashAttentionBackward0*>(self->cdata.get())->output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionBackward0_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFlashAttentionBackward0*>(self->cdata.get())->output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionBackward0_rng_state_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFlashAttentionBackward0*>(self->cdata.get())->rng_state_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionBackward0_rng_state_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFlashAttentionBackward0*>(self->cdata.get())->rng_state_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionBackward0_unused_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFlashAttentionBackward0*>(self->cdata.get())->unused_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionBackward0_unused_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFlashAttentionBackward0*>(self->cdata.get())->unused_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ScaledDotProductFlashAttentionBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dropout_p", (getter)THPScaledDotProductFlashAttentionBackward0_dropout_p_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_is_causal", (getter)THPScaledDotProductFlashAttentionBackward0_is_causal_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_key", (getter)THPScaledDotProductFlashAttentionBackward0_key_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_key", (getter)THPScaledDotProductFlashAttentionBackward0_key_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_query", (getter)THPScaledDotProductFlashAttentionBackward0_query_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_query", (getter)THPScaledDotProductFlashAttentionBackward0_query_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale", (getter)THPScaledDotProductFlashAttentionBackward0_scale_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_value", (getter)THPScaledDotProductFlashAttentionBackward0_value_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_value", (getter)THPScaledDotProductFlashAttentionBackward0_value_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_cum_seq_k", (getter)THPScaledDotProductFlashAttentionBackward0_cum_seq_k_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_cum_seq_k", (getter)THPScaledDotProductFlashAttentionBackward0_cum_seq_k_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_cum_seq_q", (getter)THPScaledDotProductFlashAttentionBackward0_cum_seq_q_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_cum_seq_q", (getter)THPScaledDotProductFlashAttentionBackward0_cum_seq_q_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_logsumexp", (getter)THPScaledDotProductFlashAttentionBackward0_logsumexp_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_logsumexp", (getter)THPScaledDotProductFlashAttentionBackward0_logsumexp_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_max_k", (getter)THPScaledDotProductFlashAttentionBackward0_max_k_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_max_q", (getter)THPScaledDotProductFlashAttentionBackward0_max_q_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output", (getter)THPScaledDotProductFlashAttentionBackward0_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_output", (getter)THPScaledDotProductFlashAttentionBackward0_output_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_rng_state", (getter)THPScaledDotProductFlashAttentionBackward0_rng_state_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_rng_state", (getter)THPScaledDotProductFlashAttentionBackward0_rng_state_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_unused", (getter)THPScaledDotProductFlashAttentionBackward0_unused_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_unused", (getter)THPScaledDotProductFlashAttentionBackward0_unused_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPScaledDotProductFlashAttentionForCpuBackward0_attn_mask_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFlashAttentionForCpuBackward0*>(self->cdata.get())->attn_mask_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionForCpuBackward0_attn_mask_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFlashAttentionForCpuBackward0*>(self->cdata.get())->attn_mask_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionForCpuBackward0_dropout_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ScaledDotProductFlashAttentionForCpuBackward0*>(self->cdata.get())->dropout_p;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionForCpuBackward0_is_causal_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ScaledDotProductFlashAttentionForCpuBackward0*>(self->cdata.get())->is_causal;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionForCpuBackward0_key_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFlashAttentionForCpuBackward0*>(self->cdata.get())->key_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionForCpuBackward0_key_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFlashAttentionForCpuBackward0*>(self->cdata.get())->key_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionForCpuBackward0_query_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFlashAttentionForCpuBackward0*>(self->cdata.get())->query_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionForCpuBackward0_query_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFlashAttentionForCpuBackward0*>(self->cdata.get())->query_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionForCpuBackward0_scale_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<ScaledDotProductFlashAttentionForCpuBackward0*>(self->cdata.get())->scale;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionForCpuBackward0_value_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFlashAttentionForCpuBackward0*>(self->cdata.get())->value_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionForCpuBackward0_value_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFlashAttentionForCpuBackward0*>(self->cdata.get())->value_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionForCpuBackward0_logsumexp_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFlashAttentionForCpuBackward0*>(self->cdata.get())->logsumexp_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionForCpuBackward0_logsumexp_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFlashAttentionForCpuBackward0*>(self->cdata.get())->logsumexp_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionForCpuBackward0_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFlashAttentionForCpuBackward0*>(self->cdata.get())->output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFlashAttentionForCpuBackward0_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFlashAttentionForCpuBackward0*>(self->cdata.get())->output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ScaledDotProductFlashAttentionForCpuBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_attn_mask", (getter)THPScaledDotProductFlashAttentionForCpuBackward0_attn_mask_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_attn_mask", (getter)THPScaledDotProductFlashAttentionForCpuBackward0_attn_mask_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dropout_p", (getter)THPScaledDotProductFlashAttentionForCpuBackward0_dropout_p_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_is_causal", (getter)THPScaledDotProductFlashAttentionForCpuBackward0_is_causal_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_key", (getter)THPScaledDotProductFlashAttentionForCpuBackward0_key_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_key", (getter)THPScaledDotProductFlashAttentionForCpuBackward0_key_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_query", (getter)THPScaledDotProductFlashAttentionForCpuBackward0_query_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_query", (getter)THPScaledDotProductFlashAttentionForCpuBackward0_query_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale", (getter)THPScaledDotProductFlashAttentionForCpuBackward0_scale_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_value", (getter)THPScaledDotProductFlashAttentionForCpuBackward0_value_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_value", (getter)THPScaledDotProductFlashAttentionForCpuBackward0_value_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_logsumexp", (getter)THPScaledDotProductFlashAttentionForCpuBackward0_logsumexp_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_logsumexp", (getter)THPScaledDotProductFlashAttentionForCpuBackward0_logsumexp_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output", (getter)THPScaledDotProductFlashAttentionForCpuBackward0_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_output", (getter)THPScaledDotProductFlashAttentionForCpuBackward0_output_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPFftC2CBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FftC2CBackward0*>(self->cdata.get())->dim;
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

PyObject* THPFftC2CBackward0_forward_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FftC2CBackward0*>(self->cdata.get())->forward;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPFftC2CBackward0_normalization_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FftC2CBackward0*>(self->cdata.get())->normalization;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FftC2CBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPFftC2CBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_forward", (getter)THPFftC2CBackward0_forward_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_normalization", (getter)THPFftC2CBackward0_normalization_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPStackBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<StackBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef StackBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPStackBackward0_dim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPThnnFusedLstmCellBackward0_cx_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedLstmCellBackward0*>(self->cdata.get())->cx_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedLstmCellBackward0_cx_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedLstmCellBackward0*>(self->cdata.get())->cx_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedLstmCellBackward0_hidden_bias_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedLstmCellBackward0*>(self->cdata.get())->hidden_bias_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedLstmCellBackward0_hidden_bias_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedLstmCellBackward0*>(self->cdata.get())->hidden_bias_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedLstmCellBackward0_hidden_gates_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedLstmCellBackward0*>(self->cdata.get())->hidden_gates_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedLstmCellBackward0_hidden_gates_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedLstmCellBackward0*>(self->cdata.get())->hidden_gates_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedLstmCellBackward0_input_bias_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedLstmCellBackward0*>(self->cdata.get())->input_bias_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedLstmCellBackward0_input_bias_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedLstmCellBackward0*>(self->cdata.get())->input_bias_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedLstmCellBackward0_input_gates_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedLstmCellBackward0*>(self->cdata.get())->input_gates_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedLstmCellBackward0_input_gates_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedLstmCellBackward0*>(self->cdata.get())->input_gates_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedLstmCellBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedLstmCellBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedLstmCellBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedLstmCellBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedLstmCellBackward0_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedLstmCellBackward0*>(self->cdata.get())->result2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedLstmCellBackward0_result2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedLstmCellBackward0*>(self->cdata.get())->result2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ThnnFusedLstmCellBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_cx", (getter)THPThnnFusedLstmCellBackward0_cx_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_cx", (getter)THPThnnFusedLstmCellBackward0_cx_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_hidden_bias", (getter)THPThnnFusedLstmCellBackward0_hidden_bias_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_hidden_bias", (getter)THPThnnFusedLstmCellBackward0_hidden_bias_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_hidden_gates", (getter)THPThnnFusedLstmCellBackward0_hidden_gates_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_hidden_gates", (getter)THPThnnFusedLstmCellBackward0_hidden_gates_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input_bias", (getter)THPThnnFusedLstmCellBackward0_input_bias_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input_bias", (getter)THPThnnFusedLstmCellBackward0_input_bias_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input_gates", (getter)THPThnnFusedLstmCellBackward0_input_gates_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input_gates", (getter)THPThnnFusedLstmCellBackward0_input_gates_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPThnnFusedLstmCellBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPThnnFusedLstmCellBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPThnnFusedLstmCellBackward0_result2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result2", (getter)THPThnnFusedLstmCellBackward0_result2_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPForeachMaximumBackward0_scalar_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ForeachMaximumBackward0*>(self->cdata.get())->scalar;
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

PyObject* THPForeachMaximumBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<ForeachMaximumBackward0*>(self->cdata.get());
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

PyObject* THPForeachMaximumBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<ForeachMaximumBackward0*>(self->cdata.get());
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

static struct PyGetSetDef ForeachMaximumBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_scalar", (getter)THPForeachMaximumBackward0_scalar_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPForeachMaximumBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPForeachMaximumBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPForeachMaximumBackward1_scalars_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<ForeachMaximumBackward1*>(self->cdata.get());
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
                            
PyObject* THPForeachMaximumBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<ForeachMaximumBackward1*>(self->cdata.get());
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

PyObject* THPForeachMaximumBackward1_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<ForeachMaximumBackward1*>(self->cdata.get());
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

static struct PyGetSetDef ForeachMaximumBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_scalars", (getter)THPForeachMaximumBackward1_scalars_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPForeachMaximumBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPForeachMaximumBackward1_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef AliasBackward0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPSplitBackward0_copy_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SplitBackward0_copy*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSplitBackward0_copy_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SplitBackward0_copy*>(self->cdata.get())->self_sym_sizes;
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

PyObject* THPSplitBackward0_copy_split_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SplitBackward0_copy*>(self->cdata.get())->split_size;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SplitBackward0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSplitBackward0_copy_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPSplitBackward0_copy_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_split_size", (getter)THPSplitBackward0_copy_split_size_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPTransposeBackward0_copy_dim0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TransposeBackward0_copy*>(self->cdata.get())->dim0;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPTransposeBackward0_copy_dim1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TransposeBackward0_copy*>(self->cdata.get())->dim1;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef TransposeBackward0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim0", (getter)THPTransposeBackward0_copy_dim0_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim1", (getter)THPTransposeBackward0_copy_dim1_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef LiftFreshBackward0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef ViewAsRealBackward0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPNestedGetValuesBackward0_copy_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NestedGetValuesBackward0_copy*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNestedGetValuesBackward0_copy_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NestedGetValuesBackward0_copy*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NestedGetValuesBackward0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPNestedGetValuesBackward0_copy_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPNestedGetValuesBackward0_copy_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUnbindBackward0_copy_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnbindBackward0_copy*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UnbindBackward0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPUnbindBackward0_copy_dim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUnbindBackwardAutogradNestedTensor0_copy_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnbindBackwardAutogradNestedTensor0_copy*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUnbindBackwardAutogradNestedTensor0_copy_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<UnbindBackwardAutogradNestedTensor0_copy*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPUnbindBackwardAutogradNestedTensor0_copy_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<UnbindBackwardAutogradNestedTensor0_copy*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UnbindBackwardAutogradNestedTensor0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPUnbindBackwardAutogradNestedTensor0_copy_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPUnbindBackwardAutogradNestedTensor0_copy_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPUnbindBackwardAutogradNestedTensor0_copy_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

void initialize_autogenerated_functions_1(PyObject* module) {
  static PyTypeObject AbsBackward0Class;
  addClass<AbsBackward0>(module, AbsBackward0Class, "AbsBackward0", AbsBackward0_properties);
  static PyTypeObject AddmvBackward0Class;
  addClass<AddmvBackward0>(module, AddmvBackward0Class, "AddmvBackward0", AddmvBackward0_properties);
  static PyTypeObject AddrBackward0Class;
  addClass<AddrBackward0>(module, AddrBackward0Class, "AddrBackward0", AddrBackward0_properties);
  static PyTypeObject AliasBackward0Class;
  addClass<AliasBackward0>(module, AliasBackward0Class, "AliasBackward0", AliasBackward0_properties);
  static PyTypeObject AtanBackward0Class;
  addClass<AtanBackward0>(module, AtanBackward0Class, "AtanBackward0", AtanBackward0_properties);
  static PyTypeObject CeilBackward0Class;
  addClass<CeilBackward0>(module, CeilBackward0Class, "CeilBackward0", CeilBackward0_properties);
  static PyTypeObject LinalgCholeskyExBackward0Class;
  addClass<LinalgCholeskyExBackward0>(module, LinalgCholeskyExBackward0Class, "LinalgCholeskyExBackward0", LinalgCholeskyExBackward0_properties);
  static PyTypeObject CholeskySolveBackward0Class;
  addClass<CholeskySolveBackward0>(module, CholeskySolveBackward0Class, "CholeskySolveBackward0", CholeskySolveBackward0_properties);
  static PyTypeObject CloneBackward0Class;
  addClass<CloneBackward0>(module, CloneBackward0Class, "CloneBackward0", CloneBackward0_properties);
  static PyTypeObject CoshBackward0Class;
  addClass<CoshBackward0>(module, CoshBackward0Class, "CoshBackward0", CoshBackward0_properties);
  static PyTypeObject LinalgCrossBackward0Class;
  addClass<LinalgCrossBackward0>(module, LinalgCrossBackward0Class, "LinalgCrossBackward0", LinalgCrossBackward0_properties);
  static PyTypeObject LogcumsumexpBackward0Class;
  addClass<LogcumsumexpBackward0>(module, LogcumsumexpBackward0Class, "LogcumsumexpBackward0", LogcumsumexpBackward0_properties);
  static PyTypeObject CumminBackward0Class;
  addClass<CumminBackward0>(module, CumminBackward0Class, "CumminBackward0", CumminBackward0_properties);
  static PyTypeObject ConvTbcBackward0Class;
  addClass<ConvTbcBackward0>(module, ConvTbcBackward0Class, "ConvTbcBackward0", ConvTbcBackward0_properties);
  static PyTypeObject DivBackward0Class;
  addClass<DivBackward0>(module, DivBackward0Class, "DivBackward0", DivBackward0_properties);
  static PyTypeObject DivBackward1Class;
  addClass<DivBackward1>(module, DivBackward1Class, "DivBackward1", DivBackward1_properties);
  static PyTypeObject DivBackward2Class;
  addClass<DivBackward2>(module, DivBackward2Class, "DivBackward2", DivBackward2_properties);
  static PyTypeObject DivBackward3Class;
  addClass<DivBackward3>(module, DivBackward3Class, "DivBackward3", DivBackward3_properties);
  static PyTypeObject ErfBackward0Class;
  addClass<ErfBackward0>(module, ErfBackward0Class, "ErfBackward0", ErfBackward0_properties);
  static PyTypeObject FakeQuantizeLearnablePerChannelAffineBackward0Class;
  addClass<FakeQuantizeLearnablePerChannelAffineBackward0>(module, FakeQuantizeLearnablePerChannelAffineBackward0Class, "FakeQuantizeLearnablePerChannelAffineBackward0", FakeQuantizeLearnablePerChannelAffineBackward0_properties);
  static PyTypeObject FillBackward2Class;
  addClass<FillBackward2>(module, FillBackward2Class, "FillBackward2", FillBackward2_properties);
  static PyTypeObject FillBackward3Class;
  addClass<FillBackward3>(module, FillBackward3Class, "FillBackward3", FillBackward3_properties);
  static PyTypeObject GridSampler3DBackward0Class;
  addClass<GridSampler3DBackward0>(module, GridSampler3DBackward0Class, "GridSampler3DBackward0", GridSampler3DBackward0_properties);
  static PyTypeObject GridSampler2DCpuFallbackBackward0Class;
  addClass<GridSampler2DCpuFallbackBackward0>(module, GridSampler2DCpuFallbackBackward0Class, "GridSampler2DCpuFallbackBackward0", GridSampler2DCpuFallbackBackward0_properties);
  static PyTypeObject HardswishBackwardBackward0Class;
  addClass<HardswishBackwardBackward0>(module, HardswishBackwardBackward0Class, "HardswishBackwardBackward0", HardswishBackwardBackward0_properties);
  static PyTypeObject IndexBackward0Class;
  addClass<IndexBackward0>(module, IndexBackward0Class, "IndexBackward0", IndexBackward0_properties);
  static PyTypeObject IndexReduceBackward0Class;
  addClass<IndexReduceBackward0>(module, IndexReduceBackward0Class, "IndexReduceBackward0", IndexReduceBackward0_properties);
  static PyTypeObject IndexFillBackward0Class;
  addClass<IndexFillBackward0>(module, IndexFillBackward0Class, "IndexFillBackward0", IndexFillBackward0_properties);
  static PyTypeObject IndexFillBackward1Class;
  addClass<IndexFillBackward1>(module, IndexFillBackward1Class, "IndexFillBackward1", IndexFillBackward1_properties);
  static PyTypeObject IndexPutImplBackward0Class;
  addClass<IndexPutImplBackward0>(module, IndexPutImplBackward0Class, "IndexPutImplBackward0", IndexPutImplBackward0_properties);
  static PyTypeObject LinalgPinvBackward0Class;
  addClass<LinalgPinvBackward0>(module, LinalgPinvBackward0Class, "LinalgPinvBackward0", LinalgPinvBackward0_properties);
  static PyTypeObject KthvalueBackward0Class;
  addClass<KthvalueBackward0>(module, KthvalueBackward0Class, "KthvalueBackward0", KthvalueBackward0_properties);
  static PyTypeObject DigammaBackward0Class;
  addClass<DigammaBackward0>(module, DigammaBackward0Class, "DigammaBackward0", DigammaBackward0_properties);
  static PyTypeObject PolygammaBackward0Class;
  addClass<PolygammaBackward0>(module, PolygammaBackward0Class, "PolygammaBackward0", PolygammaBackward0_properties);
  static PyTypeObject PolygammaBackward1Class;
  addClass<PolygammaBackward1>(module, PolygammaBackward1Class, "PolygammaBackward1", PolygammaBackward1_properties);
  static PyTypeObject XlogyBackward0Class;
  addClass<XlogyBackward0>(module, XlogyBackward0Class, "XlogyBackward0", XlogyBackward0_properties);
  static PyTypeObject XlogyBackward1Class;
  addClass<XlogyBackward1>(module, XlogyBackward1Class, "XlogyBackward1", XlogyBackward1_properties);
  static PyTypeObject XlogyBackward2Class;
  addClass<XlogyBackward2>(module, XlogyBackward2Class, "XlogyBackward2", XlogyBackward2_properties);
  static PyTypeObject LinalgLuFactorExBackward0Class;
  addClass<LinalgLuFactorExBackward0>(module, LinalgLuFactorExBackward0Class, "LinalgLuFactorExBackward0", LinalgLuFactorExBackward0_properties);
  static PyTypeObject MaximumBackward0Class;
  addClass<MaximumBackward0>(module, MaximumBackward0Class, "MaximumBackward0", MaximumBackward0_properties);
  static PyTypeObject MinimumBackward0Class;
  addClass<MinimumBackward0>(module, MinimumBackward0Class, "MinimumBackward0", MinimumBackward0_properties);
  static PyTypeObject AmaxBackward0Class;
  addClass<AmaxBackward0>(module, AmaxBackward0Class, "AmaxBackward0", AmaxBackward0_properties);
  static PyTypeObject AminBackward0Class;
  addClass<AminBackward0>(module, AminBackward0Class, "AminBackward0", AminBackward0_properties);
  static PyTypeObject NanToNumBackward0Class;
  addClass<NanToNumBackward0>(module, NanToNumBackward0Class, "NanToNumBackward0", NanToNumBackward0_properties);
  static PyTypeObject NativeBatchNormBackward0Class;
  addClass<NativeBatchNormBackward0>(module, NativeBatchNormBackward0Class, "NativeBatchNormBackward0", NativeBatchNormBackward0_properties);
  static PyTypeObject NativeBatchNormBackwardBackward0Class;
  addClass<NativeBatchNormBackwardBackward0>(module, NativeBatchNormBackwardBackward0Class, "NativeBatchNormBackwardBackward0", NativeBatchNormBackwardBackward0_properties);
  static PyTypeObject NativeLayerNormBackward0Class;
  addClass<NativeLayerNormBackward0>(module, NativeLayerNormBackward0Class, "NativeLayerNormBackward0", NativeLayerNormBackward0_properties);
  static PyTypeObject NeBackward0Class;
  addClass<NeBackward0>(module, NeBackward0Class, "NeBackward0", NeBackward0_properties);
  static PyTypeObject NeBackward1Class;
  addClass<NeBackward1>(module, NeBackward1Class, "NeBackward1", NeBackward1_properties);
  static PyTypeObject BatchNormBackwardBackward0Class;
  addClass<BatchNormBackwardBackward0>(module, BatchNormBackwardBackward0Class, "BatchNormBackwardBackward0", BatchNormBackwardBackward0_properties);
  static PyTypeObject NextafterBackward0Class;
  addClass<NextafterBackward0>(module, NextafterBackward0Class, "NextafterBackward0", NextafterBackward0_properties);
  static PyTypeObject CdistBackward0Class;
  addClass<CdistBackward0>(module, CdistBackward0Class, "CdistBackward0", CdistBackward0_properties);
  static PyTypeObject NormalBackward1Class;
  addClass<NormalBackward1>(module, NormalBackward1Class, "NormalBackward1", NormalBackward1_properties);
  static PyTypeObject NormalBackward2Class;
  addClass<NormalBackward2>(module, NormalBackward2Class, "NormalBackward2", NormalBackward2_properties);
  static PyTypeObject NormalBackward3Class;
  addClass<NormalBackward3>(module, NormalBackward3Class, "NormalBackward3", NormalBackward3_properties);
  static PyTypeObject SpecialLogNdtrBackward0Class;
  addClass<SpecialLogNdtrBackward0>(module, SpecialLogNdtrBackward0Class, "SpecialLogNdtrBackward0", SpecialLogNdtrBackward0_properties);
  static PyTypeObject RoundBackward0Class;
  addClass<RoundBackward0>(module, RoundBackward0Class, "RoundBackward0", RoundBackward0_properties);
  static PyTypeObject RoundBackward1Class;
  addClass<RoundBackward1>(module, RoundBackward1Class, "RoundBackward1", RoundBackward1_properties);
  static PyTypeObject ScatterAddBackward0Class;
  addClass<ScatterAddBackward0>(module, ScatterAddBackward0Class, "ScatterAddBackward0", ScatterAddBackward0_properties);
  static PyTypeObject SelectBackwardBackward0Class;
  addClass<SelectBackwardBackward0>(module, SelectBackwardBackward0Class, "SelectBackwardBackward0", SelectBackwardBackward0_properties);
  static PyTypeObject SigmoidBackward0Class;
  addClass<SigmoidBackward0>(module, SigmoidBackward0Class, "SigmoidBackward0", SigmoidBackward0_properties);
  static PyTypeObject SliceScatterBackward0Class;
  addClass<SliceScatterBackward0>(module, SliceScatterBackward0Class, "SliceScatterBackward0", SliceScatterBackward0_properties);
  static PyTypeObject SplitWithSizesBackward0Class;
  addClass<SplitWithSizesBackward0>(module, SplitWithSizesBackward0Class, "SplitWithSizesBackward0", SplitWithSizesBackward0_properties);
  static PyTypeObject SplitWithSizesBackwardAutogradNestedTensor0Class;
  addClass<SplitWithSizesBackwardAutogradNestedTensor0>(module, SplitWithSizesBackwardAutogradNestedTensor0Class, "SplitWithSizesBackwardAutogradNestedTensor0", SplitWithSizesBackwardAutogradNestedTensor0_properties);
  static PyTypeObject TBackward1Class;
  addClass<TBackward1>(module, TBackward1Class, "TBackward1", TBackward1_properties);
  static PyTypeObject TanhBackward0Class;
  addClass<TanhBackward0>(module, TanhBackward0Class, "TanhBackward0", TanhBackward0_properties);
  static PyTypeObject TopkBackward0Class;
  addClass<TopkBackward0>(module, TopkBackward0Class, "TopkBackward0", TopkBackward0_properties);
  static PyTypeObject ToSparseCscBackward0Class;
  addClass<ToSparseCscBackward0>(module, ToSparseCscBackward0Class, "ToSparseCscBackward0", ToSparseCscBackward0_properties);
  static PyTypeObject ToSparseBscBackward0Class;
  addClass<ToSparseBscBackward0>(module, ToSparseBscBackward0Class, "ToSparseBscBackward0", ToSparseBscBackward0_properties);
  static PyTypeObject UnfoldBackwardBackward0Class;
  addClass<UnfoldBackwardBackward0>(module, UnfoldBackwardBackward0Class, "UnfoldBackwardBackward0", UnfoldBackwardBackward0_properties);
  static PyTypeObject UniformBackward0Class;
  addClass<UniformBackward0>(module, UniformBackward0Class, "UniformBackward0", UniformBackward0_properties);
  static PyTypeObject UniqueDimBackward0Class;
  addClass<UniqueDimBackward0>(module, UniqueDimBackward0Class, "UniqueDimBackward0", UniqueDimBackward0_properties);
  static PyTypeObject UniqueConsecutiveBackward0Class;
  addClass<UniqueConsecutiveBackward0>(module, UniqueConsecutiveBackward0Class, "UniqueConsecutiveBackward0", UniqueConsecutiveBackward0_properties);
  static PyTypeObject BinaryCrossEntropyBackwardBackward0Class;
  addClass<BinaryCrossEntropyBackwardBackward0>(module, BinaryCrossEntropyBackwardBackward0Class, "BinaryCrossEntropyBackwardBackward0", BinaryCrossEntropyBackwardBackward0_properties);
  static PyTypeObject MseLossBackward0Class;
  addClass<MseLossBackward0>(module, MseLossBackward0Class, "MseLossBackward0", MseLossBackward0_properties);
  static PyTypeObject NllLoss2DBackward0Class;
  addClass<NllLoss2DBackward0>(module, NllLoss2DBackward0Class, "NllLoss2DBackward0", NllLoss2DBackward0_properties);
  static PyTypeObject SoftMarginLossBackward0Class;
  addClass<SoftMarginLossBackward0>(module, SoftMarginLossBackward0Class, "SoftMarginLossBackward0", SoftMarginLossBackward0_properties);
  static PyTypeObject MishBackward0Class;
  addClass<MishBackward0>(module, MishBackward0Class, "MishBackward0", MishBackward0_properties);
  static PyTypeObject GeluBackward0Class;
  addClass<GeluBackward0>(module, GeluBackward0Class, "GeluBackward0", GeluBackward0_properties);
  static PyTypeObject GluBackward0Class;
  addClass<GluBackward0>(module, GluBackward0Class, "GluBackward0", GluBackward0_properties);
  static PyTypeObject HardshrinkBackward0Class;
  addClass<HardshrinkBackward0>(module, HardshrinkBackward0Class, "HardshrinkBackward0", HardshrinkBackward0_properties);
  static PyTypeObject HardtanhBackward0Class;
  addClass<HardtanhBackward0>(module, HardtanhBackward0Class, "HardtanhBackward0", HardtanhBackward0_properties);
  static PyTypeObject LeakyReluBackward0Class;
  addClass<LeakyReluBackward0>(module, LeakyReluBackward0Class, "LeakyReluBackward0", LeakyReluBackward0_properties);
  static PyTypeObject LogSigmoidBackward0Class;
  addClass<LogSigmoidBackward0>(module, LogSigmoidBackward0Class, "LogSigmoidBackward0", LogSigmoidBackward0_properties);
  static PyTypeObject RreluWithNoiseBackward1Class;
  addClass<RreluWithNoiseBackward1>(module, RreluWithNoiseBackward1Class, "RreluWithNoiseBackward1", RreluWithNoiseBackward1_properties);
  static PyTypeObject SoftshrinkBackward0Class;
  addClass<SoftshrinkBackward0>(module, SoftshrinkBackward0Class, "SoftshrinkBackward0", SoftshrinkBackward0_properties);
  static PyTypeObject ReflectionPad3DBackward0Class;
  addClass<ReflectionPad3DBackward0>(module, ReflectionPad3DBackward0Class, "ReflectionPad3DBackward0", ReflectionPad3DBackward0_properties);
  static PyTypeObject ReplicationPad1DBackward0Class;
  addClass<ReplicationPad1DBackward0>(module, ReplicationPad1DBackward0Class, "ReplicationPad1DBackward0", ReplicationPad1DBackward0_properties);
  static PyTypeObject ReplicationPad2DBackward0Class;
  addClass<ReplicationPad2DBackward0>(module, ReplicationPad2DBackward0Class, "ReplicationPad2DBackward0", ReplicationPad2DBackward0_properties);
  static PyTypeObject UpsampleNearest1DBackward0Class;
  addClass<UpsampleNearest1DBackward0>(module, UpsampleNearest1DBackward0Class, "UpsampleNearest1DBackward0", UpsampleNearest1DBackward0_properties);
  static PyTypeObject UpsampleNearestExact1DBackward0Class;
  addClass<UpsampleNearestExact1DBackward0>(module, UpsampleNearestExact1DBackward0Class, "UpsampleNearestExact1DBackward0", UpsampleNearestExact1DBackward0_properties);
  static PyTypeObject FractionalMaxPool3DBackward0Class;
  addClass<FractionalMaxPool3DBackward0>(module, FractionalMaxPool3DBackward0Class, "FractionalMaxPool3DBackward0", FractionalMaxPool3DBackward0_properties);
  static PyTypeObject MpsConvolutionBackward0Class;
  addClass<MpsConvolutionBackward0>(module, MpsConvolutionBackward0Class, "MpsConvolutionBackward0", MpsConvolutionBackward0_properties);
  static PyTypeObject ConvDepthwise2DBackward0Class;
  addClass<ConvDepthwise2DBackward0>(module, ConvDepthwise2DBackward0Class, "ConvDepthwise2DBackward0", ConvDepthwise2DBackward0_properties);
  static PyTypeObject AdaptiveAvgPool3DBackwardBackward0Class;
  addClass<AdaptiveAvgPool3DBackwardBackward0>(module, AdaptiveAvgPool3DBackwardBackward0Class, "AdaptiveAvgPool3DBackwardBackward0", AdaptiveAvgPool3DBackwardBackward0_properties);
  static PyTypeObject AdaptiveMaxPool2DBackwardBackward0Class;
  addClass<AdaptiveMaxPool2DBackwardBackward0>(module, AdaptiveMaxPool2DBackwardBackward0Class, "AdaptiveMaxPool2DBackwardBackward0", AdaptiveMaxPool2DBackwardBackward0_properties);
  static PyTypeObject AdaptiveMaxPool3DBackwardBackward0Class;
  addClass<AdaptiveMaxPool3DBackwardBackward0>(module, AdaptiveMaxPool3DBackwardBackward0Class, "AdaptiveMaxPool3DBackwardBackward0", AdaptiveMaxPool3DBackwardBackward0_properties);
  static PyTypeObject LogSigmoidBackwardBackward0Class;
  addClass<LogSigmoidBackwardBackward0>(module, LogSigmoidBackwardBackward0Class, "LogSigmoidBackwardBackward0", LogSigmoidBackwardBackward0_properties);
  static PyTypeObject LogSoftmaxBackwardDataBackward0Class;
  addClass<LogSoftmaxBackwardDataBackward0>(module, LogSoftmaxBackwardDataBackward0Class, "LogSoftmaxBackwardDataBackward0", LogSoftmaxBackwardDataBackward0_properties);
  static PyTypeObject NllLoss2DBackwardBackward0Class;
  addClass<NllLoss2DBackwardBackward0>(module, NllLoss2DBackwardBackward0Class, "NllLoss2DBackwardBackward0", NllLoss2DBackwardBackward0_properties);
  static PyTypeObject SparseSampledAddmmBackward0Class;
  addClass<SparseSampledAddmmBackward0>(module, SparseSampledAddmmBackward0Class, "SparseSampledAddmmBackward0", SparseSampledAddmmBackward0_properties);
  static PyTypeObject SoftmaxBackwardDataBackward0Class;
  addClass<SoftmaxBackwardDataBackward0>(module, SoftmaxBackwardDataBackward0Class, "SoftmaxBackwardDataBackward0", SoftmaxBackwardDataBackward0_properties);
  static PyTypeObject CudnnGridSamplerBackward0Class;
  addClass<CudnnGridSamplerBackward0>(module, CudnnGridSamplerBackward0Class, "CudnnGridSamplerBackward0", CudnnGridSamplerBackward0_properties);
  static PyTypeObject NnpackSpatialConvolutionBackward0Class;
  addClass<NnpackSpatialConvolutionBackward0>(module, NnpackSpatialConvolutionBackward0Class, "NnpackSpatialConvolutionBackward0", NnpackSpatialConvolutionBackward0_properties);
  static PyTypeObject LstmMpsBackward0Class;
  addClass<LstmMpsBackward0>(module, LstmMpsBackward0Class, "LstmMpsBackward0", LstmMpsBackward0_properties);
  static PyTypeObject MiopenDepthwiseConvolutionBackward0Class;
  addClass<MiopenDepthwiseConvolutionBackward0>(module, MiopenDepthwiseConvolutionBackward0Class, "MiopenDepthwiseConvolutionBackward0", MiopenDepthwiseConvolutionBackward0_properties);
  static PyTypeObject MkldnnMaxPool2DBackward0Class;
  addClass<MkldnnMaxPool2DBackward0>(module, MkldnnMaxPool2DBackward0Class, "MkldnnMaxPool2DBackward0", MkldnnMaxPool2DBackward0_properties);
  static PyTypeObject ScaledDotProductFlashAttentionBackward0Class;
  addClass<ScaledDotProductFlashAttentionBackward0>(module, ScaledDotProductFlashAttentionBackward0Class, "ScaledDotProductFlashAttentionBackward0", ScaledDotProductFlashAttentionBackward0_properties);
  static PyTypeObject ScaledDotProductFlashAttentionForCpuBackward0Class;
  addClass<ScaledDotProductFlashAttentionForCpuBackward0>(module, ScaledDotProductFlashAttentionForCpuBackward0Class, "ScaledDotProductFlashAttentionForCpuBackward0", ScaledDotProductFlashAttentionForCpuBackward0_properties);
  static PyTypeObject FftC2CBackward0Class;
  addClass<FftC2CBackward0>(module, FftC2CBackward0Class, "FftC2CBackward0", FftC2CBackward0_properties);
  static PyTypeObject StackBackward0Class;
  addClass<StackBackward0>(module, StackBackward0Class, "StackBackward0", StackBackward0_properties);
  static PyTypeObject ThnnFusedLstmCellBackward0Class;
  addClass<ThnnFusedLstmCellBackward0>(module, ThnnFusedLstmCellBackward0Class, "ThnnFusedLstmCellBackward0", ThnnFusedLstmCellBackward0_properties);
  static PyTypeObject ForeachMaximumBackward0Class;
  addClass<ForeachMaximumBackward0>(module, ForeachMaximumBackward0Class, "ForeachMaximumBackward0", ForeachMaximumBackward0_properties);
  static PyTypeObject ForeachMaximumBackward1Class;
  addClass<ForeachMaximumBackward1>(module, ForeachMaximumBackward1Class, "ForeachMaximumBackward1", ForeachMaximumBackward1_properties);
  static PyTypeObject AliasBackward0_copyClass;
  addClass<AliasBackward0_copy>(module, AliasBackward0_copyClass, "AliasBackward0_copy", AliasBackward0_copy_properties);
  static PyTypeObject SplitBackward0_copyClass;
  addClass<SplitBackward0_copy>(module, SplitBackward0_copyClass, "SplitBackward0_copy", SplitBackward0_copy_properties);
  static PyTypeObject TransposeBackward0_copyClass;
  addClass<TransposeBackward0_copy>(module, TransposeBackward0_copyClass, "TransposeBackward0_copy", TransposeBackward0_copy_properties);
  static PyTypeObject LiftFreshBackward0_copyClass;
  addClass<LiftFreshBackward0_copy>(module, LiftFreshBackward0_copyClass, "LiftFreshBackward0_copy", LiftFreshBackward0_copy_properties);
  static PyTypeObject ViewAsRealBackward0_copyClass;
  addClass<ViewAsRealBackward0_copy>(module, ViewAsRealBackward0_copyClass, "ViewAsRealBackward0_copy", ViewAsRealBackward0_copy_properties);
  static PyTypeObject NestedGetValuesBackward0_copyClass;
  addClass<NestedGetValuesBackward0_copy>(module, NestedGetValuesBackward0_copyClass, "NestedGetValuesBackward0_copy", NestedGetValuesBackward0_copy_properties);
  static PyTypeObject UnbindBackward0_copyClass;
  addClass<UnbindBackward0_copy>(module, UnbindBackward0_copyClass, "UnbindBackward0_copy", UnbindBackward0_copy_properties);
  static PyTypeObject UnbindBackwardAutogradNestedTensor0_copyClass;
  addClass<UnbindBackwardAutogradNestedTensor0_copy>(module, UnbindBackwardAutogradNestedTensor0_copyClass, "UnbindBackwardAutogradNestedTensor0_copy", UnbindBackwardAutogradNestedTensor0_copy_properties);
}

} // namespace torch::autograd::generated
