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

PyObject* THPAddbmmBackward0_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddbmmBackward0*>(self->cdata.get())->alpha;
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

PyObject* THPAddbmmBackward0_batch1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddbmmBackward0*>(self->cdata.get())->batch1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddbmmBackward0_batch1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddbmmBackward0*>(self->cdata.get())->batch1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddbmmBackward0_batch1_sym_argsize_0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddbmmBackward0*>(self->cdata.get())->batch1_sym_argsize_0;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddbmmBackward0_batch1_sym_argsize_1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddbmmBackward0*>(self->cdata.get())->batch1_sym_argsize_1;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddbmmBackward0_batch2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddbmmBackward0*>(self->cdata.get())->batch2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddbmmBackward0_batch2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddbmmBackward0*>(self->cdata.get())->batch2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddbmmBackward0_batch2_sym_argsize_2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddbmmBackward0*>(self->cdata.get())->batch2_sym_argsize_2;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddbmmBackward0_beta_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddbmmBackward0*>(self->cdata.get())->beta;
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

static struct PyGetSetDef AddbmmBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_alpha", (getter)THPAddbmmBackward0_alpha_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_batch1", (getter)THPAddbmmBackward0_batch1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_batch1", (getter)THPAddbmmBackward0_batch1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_batch1_sym_argsize_0", (getter)THPAddbmmBackward0_batch1_sym_argsize_0_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_batch1_sym_argsize_1", (getter)THPAddbmmBackward0_batch1_sym_argsize_1_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_batch2", (getter)THPAddbmmBackward0_batch2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_batch2", (getter)THPAddbmmBackward0_batch2_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_batch2_sym_argsize_2", (getter)THPAddbmmBackward0_batch2_sym_argsize_2_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_beta", (getter)THPAddbmmBackward0_beta_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAffineGridGeneratorBackward0_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AffineGridGeneratorBackward0*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAffineGridGeneratorBackward0_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AffineGridGeneratorBackward0*>(self->cdata.get())->size;
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

static struct PyGetSetDef AffineGridGeneratorBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_align_corners", (getter)THPAffineGridGeneratorBackward0_align_corners_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_size", (getter)THPAffineGridGeneratorBackward0_size_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAtanhBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AtanhBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAtanhBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AtanhBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AtanhBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPAtanhBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPAtanhBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef BernoulliBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef BernoulliBackward2_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPBmmBackward0_mat2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BmmBackward0*>(self->cdata.get())->mat2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBmmBackward0_mat2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BmmBackward0*>(self->cdata.get())->mat2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBmmBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BmmBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBmmBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BmmBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef BmmBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_mat2", (getter)THPBmmBackward0_mat2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_mat2", (getter)THPBmmBackward0_mat2_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPBmmBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPBmmBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef CauchyBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPClampMinBackward0_min_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ClampMinBackward0*>(self->cdata.get())->min;
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

PyObject* THPClampMinBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ClampMinBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPClampMinBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ClampMinBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ClampMinBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_min", (getter)THPClampMinBackward0_min_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPClampMinBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPClampMinBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPClampMinBackward1_min_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ClampMinBackward1*>(self->cdata.get())->min_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPClampMinBackward1_min_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ClampMinBackward1*>(self->cdata.get())->min_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPClampMinBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ClampMinBackward1*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPClampMinBackward1_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ClampMinBackward1*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ClampMinBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_min", (getter)THPClampMinBackward1_min_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_min", (getter)THPClampMinBackward1_min_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPClampMinBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPClampMinBackward1_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef CoalesceBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef ConjPhysicalBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPCumprodBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CumprodBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCumprodBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CumprodBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCumprodBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CumprodBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCumprodBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CumprodBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCumprodBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CumprodBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CumprodBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPCumprodBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPCumprodBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPCumprodBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPCumprodBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPCumprodBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLinalgDetBackward0_A_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgDetBackward0*>(self->cdata.get())->A_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgDetBackward0_A_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgDetBackward0*>(self->cdata.get())->A_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgDetBackward0_LU_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgDetBackward0*>(self->cdata.get())->LU_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgDetBackward0_LU_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgDetBackward0*>(self->cdata.get())->LU_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgDetBackward0_pivots_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgDetBackward0*>(self->cdata.get())->pivots_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgDetBackward0_pivots_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgDetBackward0*>(self->cdata.get())->pivots_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgDetBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgDetBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgDetBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgDetBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LinalgDetBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_A", (getter)THPLinalgDetBackward0_A_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_A", (getter)THPLinalgDetBackward0_A_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_LU", (getter)THPLinalgDetBackward0_LU_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_LU", (getter)THPLinalgDetBackward0_LU_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_pivots", (getter)THPLinalgDetBackward0_pivots_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_pivots", (getter)THPLinalgDetBackward0_pivots_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPLinalgDetBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPLinalgDetBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPDiagEmbedBackward0_dim1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<DiagEmbedBackward0*>(self->cdata.get())->dim1;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPDiagEmbedBackward0_dim2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<DiagEmbedBackward0*>(self->cdata.get())->dim2;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPDiagEmbedBackward0_offset_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<DiagEmbedBackward0*>(self->cdata.get())->offset;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef DiagEmbedBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim1", (getter)THPDiagEmbedBackward0_dim1_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim2", (getter)THPDiagEmbedBackward0_dim2_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_offset", (getter)THPDiagEmbedBackward0_offset_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPDiagonalBackward0_dim1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<DiagonalBackward0*>(self->cdata.get())->dim1;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPDiagonalBackward0_dim2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<DiagonalBackward0*>(self->cdata.get())->dim2;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPDiagonalBackward0_offset_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<DiagonalBackward0*>(self->cdata.get())->offset;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPDiagonalBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<DiagonalBackward0*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef DiagonalBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim1", (getter)THPDiagonalBackward0_dim1_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim2", (getter)THPDiagonalBackward0_dim2_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_offset", (getter)THPDiagonalBackward0_offset_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPDiagonalBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNativeDropoutBackwardBackward0_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeDropoutBackwardBackward0*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeDropoutBackwardBackward0_grad_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeDropoutBackwardBackward0*>(self->cdata.get())->grad_output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeDropoutBackwardBackward0_mask_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeDropoutBackwardBackward0*>(self->cdata.get())->mask_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeDropoutBackwardBackward0_mask_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeDropoutBackwardBackward0*>(self->cdata.get())->mask_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeDropoutBackwardBackward0_scale_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NativeDropoutBackwardBackward0*>(self->cdata.get())->scale;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NativeDropoutBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_output", (getter)THPNativeDropoutBackwardBackward0_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_grad_output", (getter)THPNativeDropoutBackwardBackward0_grad_output_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mask", (getter)THPNativeDropoutBackwardBackward0_mask_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_mask", (getter)THPNativeDropoutBackwardBackward0_mask_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale", (getter)THPNativeDropoutBackwardBackward0_scale_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef EqBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef EqBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPExpandBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ExpandBackward0*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef ExpandBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sym_sizes", (getter)THPExpandBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef FloorBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPFrexpBackward0_exponent_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FrexpBackward0*>(self->cdata.get())->exponent_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFrexpBackward0_exponent_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FrexpBackward0*>(self->cdata.get())->exponent_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FrexpBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_exponent", (getter)THPFrexpBackward0_exponent_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_exponent", (getter)THPFrexpBackward0_exponent_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPGatherBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<GatherBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPGatherBackward0_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GatherBackward0*>(self->cdata.get())->index_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPGatherBackward0_index_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GatherBackward0*>(self->cdata.get())->index_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPGatherBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GatherBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPGatherBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GatherBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPGatherBackward0_sparse_grad_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<GatherBackward0*>(self->cdata.get())->sparse_grad;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef GatherBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPGatherBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_index", (getter)THPGatherBackward0_index_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_index", (getter)THPGatherBackward0_index_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPGatherBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPGatherBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_sparse_grad", (getter)THPGatherBackward0_sparse_grad_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef GeBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef GeBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPHardsigmoidBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HardsigmoidBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPHardsigmoidBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HardsigmoidBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef HardsigmoidBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPHardsigmoidBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPHardsigmoidBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPI0Backward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<I0Backward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPI0Backward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<I0Backward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef I0Backward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPI0Backward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPI0Backward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUnsafeIndexBackward0_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<UnsafeIndexBackward0*>(self->cdata.get());
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

PyObject* THPUnsafeIndexBackward0_indices_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<UnsafeIndexBackward0*>(self->cdata.get());
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

PyObject* THPUnsafeIndexBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnsafeIndexBackward0*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef UnsafeIndexBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_indices", (getter)THPUnsafeIndexBackward0_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_indices", (getter)THPUnsafeIndexBackward0_indices_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPUnsafeIndexBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLinalgInvExBackward0_inverse_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgInvExBackward0*>(self->cdata.get())->inverse_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgInvExBackward0_inverse_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgInvExBackward0*>(self->cdata.get())->inverse_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LinalgInvExBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_inverse", (getter)THPLinalgInvExBackward0_inverse_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_inverse", (getter)THPLinalgInvExBackward0_inverse_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLerpBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LerpBackward0*>(self->cdata.get())->weight;
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

static struct PyGetSetDef LerpBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_weight", (getter)THPLerpBackward0_weight_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLerpBackward1_end_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LerpBackward1*>(self->cdata.get())->end_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLerpBackward1_end_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LerpBackward1*>(self->cdata.get())->end_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLerpBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LerpBackward1*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLerpBackward1_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LerpBackward1*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLerpBackward1_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LerpBackward1*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLerpBackward1_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LerpBackward1*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LerpBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_end", (getter)THPLerpBackward1_end_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_end", (getter)THPLerpBackward1_end_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPLerpBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPLerpBackward1_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPLerpBackward1_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPLerpBackward1_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSpecialXlog1PyBackward0_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialXlog1PyBackward0*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialXlog1PyBackward0_other_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialXlog1PyBackward0*>(self->cdata.get())->other_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialXlog1PyBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialXlog1PyBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialXlog1PyBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialXlog1PyBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SpecialXlog1PyBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPSpecialXlog1PyBackward0_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_other", (getter)THPSpecialXlog1PyBackward0_other_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSpecialXlog1PyBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSpecialXlog1PyBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSpecialXlog1PyBackward1_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialXlog1PyBackward1*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialXlog1PyBackward1_other_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialXlog1PyBackward1*>(self->cdata.get())->other_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialXlog1PyBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SpecialXlog1PyBackward1*>(self->cdata.get())->self;
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

static struct PyGetSetDef SpecialXlog1PyBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPSpecialXlog1PyBackward1_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_other", (getter)THPSpecialXlog1PyBackward1_other_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSpecialXlog1PyBackward1_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSpecialXlog1PyBackward2_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SpecialXlog1PyBackward2*>(self->cdata.get())->other;
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

PyObject* THPSpecialXlog1PyBackward2_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialXlog1PyBackward2*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialXlog1PyBackward2_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialXlog1PyBackward2*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SpecialXlog1PyBackward2_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPSpecialXlog1PyBackward2_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSpecialXlog1PyBackward2_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSpecialXlog1PyBackward2_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef LtBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef LtBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPMeanBackward0_self_sym_numel_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MeanBackward0*>(self->cdata.get())->self_sym_numel;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMeanBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MeanBackward0*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef MeanBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sym_numel", (getter)THPMeanBackward0_self_sym_numel_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPMeanBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMeanBackwardAutogradNestedTensor0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MeanBackwardAutogradNestedTensor0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMeanBackwardAutogradNestedTensor0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MeanBackwardAutogradNestedTensor0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMeanBackwardAutogradNestedTensor0_self_sym_numel_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MeanBackwardAutogradNestedTensor0*>(self->cdata.get())->self_sym_numel;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MeanBackwardAutogradNestedTensor0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMeanBackwardAutogradNestedTensor0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPMeanBackwardAutogradNestedTensor0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_numel", (getter)THPMeanBackwardAutogradNestedTensor0_self_sym_numel_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMeanBackward1_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<MeanBackward1*>(self->cdata.get())->dim;
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

PyObject* THPMeanBackward1_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MeanBackward1*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMeanBackward1_self_sym_numel_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MeanBackward1*>(self->cdata.get())->self_sym_numel;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMeanBackward1_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MeanBackward1*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef MeanBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPMeanBackward1_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPMeanBackward1_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_numel", (getter)THPMeanBackward1_self_sym_numel_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPMeanBackward1_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMinBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MinBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMinBackward0_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MinBackward0*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMinBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MinBackward0*>(self->cdata.get())->self_sym_sizes;
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

PyObject* THPMinBackward0_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MinBackward0*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMinBackward0_indices_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MinBackward0*>(self->cdata.get())->indices_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MinBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPMinBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPMinBackward0_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPMinBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_indices", (getter)THPMinBackward0_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_indices", (getter)THPMinBackward0_indices_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMinBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MinBackward1*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMinBackward1_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MinBackward1*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMinBackward1_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MinBackward1*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMinBackward1_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MinBackward1*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MinBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMinBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPMinBackward1_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPMinBackward1_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPMinBackward1_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPFminBackward0_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FminBackward0*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFminBackward0_other_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FminBackward0*>(self->cdata.get())->other_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPFminBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FminBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFminBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FminBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FminBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPFminBackward0_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_other", (getter)THPFminBackward0_other_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPFminBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPFminBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMmBackward0_mat2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MmBackward0*>(self->cdata.get())->mat2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMmBackward0_mat2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MmBackward0*>(self->cdata.get())->mat2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMmBackward0_mat2_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MmBackward0*>(self->cdata.get())->mat2_sym_sizes;
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

PyObject* THPMmBackward0_mat2_sym_strides_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MmBackward0*>(self->cdata.get())->mat2_sym_strides;
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

PyObject* THPMmBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MmBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMmBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MmBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMmBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MmBackward0*>(self->cdata.get())->self_sym_sizes;
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

PyObject* THPMmBackward0_self_sym_strides_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MmBackward0*>(self->cdata.get())->self_sym_strides;
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

static struct PyGetSetDef MmBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_mat2", (getter)THPMmBackward0_mat2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_mat2", (getter)THPMmBackward0_mat2_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mat2_sym_sizes", (getter)THPMmBackward0_mat2_sym_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mat2_sym_strides", (getter)THPMmBackward0_mat2_sym_strides_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPMmBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPMmBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPMmBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_strides", (getter)THPMmBackward0_self_sym_strides_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMvBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MvBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMvBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MvBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMvBackward0_vec_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MvBackward0*>(self->cdata.get())->vec_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMvBackward0_vec_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MvBackward0*>(self->cdata.get())->vec_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MvBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMvBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPMvBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_vec", (getter)THPMvBackward0_vec_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_vec", (getter)THPMvBackward0_vec_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNativeBatchNormLegitNoTrainingBackward0_eps_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NativeBatchNormLegitNoTrainingBackward0*>(self->cdata.get())->eps;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitNoTrainingBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormLegitNoTrainingBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitNoTrainingBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormLegitNoTrainingBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitNoTrainingBackward0_running_mean_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormLegitNoTrainingBackward0*>(self->cdata.get())->running_mean_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitNoTrainingBackward0_running_mean_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormLegitNoTrainingBackward0*>(self->cdata.get())->running_mean_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitNoTrainingBackward0_running_var_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormLegitNoTrainingBackward0*>(self->cdata.get())->running_var_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitNoTrainingBackward0_running_var_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormLegitNoTrainingBackward0*>(self->cdata.get())->running_var_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitNoTrainingBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormLegitNoTrainingBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitNoTrainingBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormLegitNoTrainingBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitNoTrainingBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormLegitNoTrainingBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitNoTrainingBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormLegitNoTrainingBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitNoTrainingBackward0_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormLegitNoTrainingBackward0*>(self->cdata.get())->result2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormLegitNoTrainingBackward0_result2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormLegitNoTrainingBackward0*>(self->cdata.get())->result2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NativeBatchNormLegitNoTrainingBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_eps", (getter)THPNativeBatchNormLegitNoTrainingBackward0_eps_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPNativeBatchNormLegitNoTrainingBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPNativeBatchNormLegitNoTrainingBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_mean", (getter)THPNativeBatchNormLegitNoTrainingBackward0_running_mean_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_running_mean", (getter)THPNativeBatchNormLegitNoTrainingBackward0_running_mean_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_var", (getter)THPNativeBatchNormLegitNoTrainingBackward0_running_var_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_running_var", (getter)THPNativeBatchNormLegitNoTrainingBackward0_running_var_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPNativeBatchNormLegitNoTrainingBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPNativeBatchNormLegitNoTrainingBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPNativeBatchNormLegitNoTrainingBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPNativeBatchNormLegitNoTrainingBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPNativeBatchNormLegitNoTrainingBackward0_result2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result2", (getter)THPNativeBatchNormLegitNoTrainingBackward0_result2_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNativeGroupNormBackward0_C_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NativeGroupNormBackward0*>(self->cdata.get())->C;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeGroupNormBackward0_HxW_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NativeGroupNormBackward0*>(self->cdata.get())->HxW;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeGroupNormBackward0_N_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NativeGroupNormBackward0*>(self->cdata.get())->N;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeGroupNormBackward0_eps_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NativeGroupNormBackward0*>(self->cdata.get())->eps;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeGroupNormBackward0_group_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NativeGroupNormBackward0*>(self->cdata.get())->group;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeGroupNormBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeGroupNormBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeGroupNormBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeGroupNormBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeGroupNormBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeGroupNormBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeGroupNormBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeGroupNormBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeGroupNormBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeGroupNormBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeGroupNormBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeGroupNormBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeGroupNormBackward0_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeGroupNormBackward0*>(self->cdata.get())->result2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeGroupNormBackward0_result2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeGroupNormBackward0*>(self->cdata.get())->result2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NativeGroupNormBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_C", (getter)THPNativeGroupNormBackward0_C_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_HxW", (getter)THPNativeGroupNormBackward0_HxW_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_N", (getter)THPNativeGroupNormBackward0_N_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_eps", (getter)THPNativeGroupNormBackward0_eps_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_group", (getter)THPNativeGroupNormBackward0_group_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPNativeGroupNormBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPNativeGroupNormBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPNativeGroupNormBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPNativeGroupNormBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPNativeGroupNormBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPNativeGroupNormBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPNativeGroupNormBackward0_result2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result2", (getter)THPNativeGroupNormBackward0_result2_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLinalgVectorNormBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<LinalgVectorNormBackward0*>(self->cdata.get())->dim;
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

PyObject* THPLinalgVectorNormBackward0_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LinalgVectorNormBackward0*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgVectorNormBackward0_ord_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LinalgVectorNormBackward0*>(self->cdata.get())->ord;
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

PyObject* THPLinalgVectorNormBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgVectorNormBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgVectorNormBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgVectorNormBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgVectorNormBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgVectorNormBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgVectorNormBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgVectorNormBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LinalgVectorNormBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPLinalgVectorNormBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPLinalgVectorNormBackward0_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_ord", (getter)THPLinalgVectorNormBackward0_ord_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPLinalgVectorNormBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPLinalgVectorNormBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPLinalgVectorNormBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPLinalgVectorNormBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef PdistBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPOrmqrBackward0_input2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<OrmqrBackward0*>(self->cdata.get())->input2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPOrmqrBackward0_input2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<OrmqrBackward0*>(self->cdata.get())->input2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPOrmqrBackward0_input3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<OrmqrBackward0*>(self->cdata.get())->input3_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPOrmqrBackward0_input3_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<OrmqrBackward0*>(self->cdata.get())->input3_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPOrmqrBackward0_left_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<OrmqrBackward0*>(self->cdata.get())->left;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPOrmqrBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<OrmqrBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPOrmqrBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<OrmqrBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPOrmqrBackward0_transpose_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<OrmqrBackward0*>(self->cdata.get())->transpose;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPOrmqrBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<OrmqrBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPOrmqrBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<OrmqrBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef OrmqrBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_input2", (getter)THPOrmqrBackward0_input2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input2", (getter)THPOrmqrBackward0_input2_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input3", (getter)THPOrmqrBackward0_input3_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input3", (getter)THPOrmqrBackward0_input3_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_left", (getter)THPOrmqrBackward0_left_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPOrmqrBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPOrmqrBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_transpose", (getter)THPOrmqrBackward0_transpose_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPOrmqrBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPOrmqrBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPPermuteBackward0_dims_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<PermuteBackward0*>(self->cdata.get())->dims;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef PermuteBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dims", (getter)THPPermuteBackward0_dims_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPReshapeAliasBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ReshapeAliasBackward0*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef ReshapeAliasBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sym_sizes", (getter)THPReshapeAliasBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSelectBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SelectBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSelectBackward0_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SelectBackward0*>(self->cdata.get())->index;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPSelectBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SelectBackward0*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef SelectBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSelectBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_index", (getter)THPSelectBackward0_index_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPSelectBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSelectBackwardAutogradNestedTensor0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SelectBackwardAutogradNestedTensor0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSelectBackwardAutogradNestedTensor0_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SelectBackwardAutogradNestedTensor0*>(self->cdata.get())->index;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPSelectBackwardAutogradNestedTensor0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SelectBackwardAutogradNestedTensor0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSelectBackwardAutogradNestedTensor0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SelectBackwardAutogradNestedTensor0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SelectBackwardAutogradNestedTensor0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSelectBackwardAutogradNestedTensor0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_index", (getter)THPSelectBackwardAutogradNestedTensor0_index_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSelectBackwardAutogradNestedTensor0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSelectBackwardAutogradNestedTensor0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLogitBackward0_eps_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<LogitBackward0*>(self->cdata.get())->eps;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogitBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogitBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogitBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogitBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LogitBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_eps", (getter)THPLogitBackward0_eps_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPLogitBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPLogitBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef SignBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPSinhBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SinhBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSinhBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SinhBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SinhBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSinhBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSinhBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSliceBackwardBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SliceBackwardBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSliceBackwardBackward0_end_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SliceBackwardBackward0*>(self->cdata.get())->end;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPSliceBackwardBackward0_start_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SliceBackwardBackward0*>(self->cdata.get())->start;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPSliceBackwardBackward0_step_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SliceBackwardBackward0*>(self->cdata.get())->step;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SliceBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSliceBackwardBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_end", (getter)THPSliceBackwardBackward0_end_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_start", (getter)THPSliceBackwardBackward0_start_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_step", (getter)THPSliceBackwardBackward0_step_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUnsafeSplitBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnsafeSplitBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUnsafeSplitBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnsafeSplitBackward0*>(self->cdata.get())->self_sym_sizes;
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

PyObject* THPUnsafeSplitBackward0_split_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnsafeSplitBackward0*>(self->cdata.get())->split_size;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UnsafeSplitBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPUnsafeSplitBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPUnsafeSplitBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_split_size", (getter)THPUnsafeSplitBackward0_split_size_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSqrtBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SqrtBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSqrtBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SqrtBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SqrtBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_result", (getter)THPSqrtBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPSqrtBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPStdBackward0_correction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<StdBackward0*>(self->cdata.get())->correction;
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

PyObject* THPStdBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<StdBackward0*>(self->cdata.get())->dim;
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

PyObject* THPStdBackward0_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<StdBackward0*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPStdBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<StdBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPStdBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<StdBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPStdBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<StdBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPStdBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<StdBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef StdBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_correction", (getter)THPStdBackward0_correction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPStdBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPStdBackward0_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPStdBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPStdBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPStdBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPStdBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPStdMeanBackward0_correction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<StdMeanBackward0*>(self->cdata.get())->correction;
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

PyObject* THPStdMeanBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<StdMeanBackward0*>(self->cdata.get())->dim;
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

PyObject* THPStdMeanBackward0_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<StdMeanBackward0*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPStdMeanBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<StdMeanBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPStdMeanBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<StdMeanBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPStdMeanBackward0_result0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<StdMeanBackward0*>(self->cdata.get())->result0_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPStdMeanBackward0_result0_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<StdMeanBackward0*>(self->cdata.get())->result0_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef StdMeanBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_correction", (getter)THPStdMeanBackward0_correction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPStdMeanBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPStdMeanBackward0_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPStdMeanBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPStdMeanBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result0", (getter)THPStdMeanBackward0_result0_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result0", (getter)THPStdMeanBackward0_result0_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPRsubBackward0_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RsubBackward0*>(self->cdata.get())->alpha;
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

static struct PyGetSetDef RsubBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_alpha", (getter)THPRsubBackward0_alpha_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPRsubBackward1_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RsubBackward1*>(self->cdata.get())->alpha;
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

static struct PyGetSetDef RsubBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_alpha", (getter)THPRsubBackward1_alpha_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef TBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPFlipBackward0_dims_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FlipBackward0*>(self->cdata.get())->dims;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FlipBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dims", (getter)THPFlipBackward0_dims_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPRollBackward0_dims_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RollBackward0*>(self->cdata.get())->dims;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPRollBackward0_shifts_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RollBackward0*>(self->cdata.get())->shifts;
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

static struct PyGetSetDef RollBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dims", (getter)THPRollBackward0_dims_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_shifts", (getter)THPRollBackward0_shifts_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPTransposeBackward0_dim0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TransposeBackward0*>(self->cdata.get())->dim0;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPTransposeBackward0_dim1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TransposeBackward0*>(self->cdata.get())->dim1;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef TransposeBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim0", (getter)THPTransposeBackward0_dim0_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim1", (getter)THPTransposeBackward0_dim1_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPTransposeBackward1_dim0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TransposeBackward1*>(self->cdata.get())->dim0;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPTransposeBackward1_dim1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TransposeBackward1*>(self->cdata.get())->dim1;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef TransposeBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim0", (getter)THPTransposeBackward1_dim0_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim1", (getter)THPTransposeBackward1_dim1_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPTriangularSolveBackward0_A_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TriangularSolveBackward0*>(self->cdata.get())->A_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPTriangularSolveBackward0_A_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TriangularSolveBackward0*>(self->cdata.get())->A_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPTriangularSolveBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TriangularSolveBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPTriangularSolveBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TriangularSolveBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPTriangularSolveBackward0_transpose_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TriangularSolveBackward0*>(self->cdata.get())->transpose;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPTriangularSolveBackward0_unitriangular_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TriangularSolveBackward0*>(self->cdata.get())->unitriangular;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPTriangularSolveBackward0_upper_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TriangularSolveBackward0*>(self->cdata.get())->upper;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPTriangularSolveBackward0_solution_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TriangularSolveBackward0*>(self->cdata.get())->solution_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPTriangularSolveBackward0_solution_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TriangularSolveBackward0*>(self->cdata.get())->solution_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef TriangularSolveBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_A", (getter)THPTriangularSolveBackward0_A_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_A", (getter)THPTriangularSolveBackward0_A_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPTriangularSolveBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPTriangularSolveBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_transpose", (getter)THPTriangularSolveBackward0_transpose_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_unitriangular", (getter)THPTriangularSolveBackward0_unitriangular_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_upper", (getter)THPTriangularSolveBackward0_upper_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_solution", (getter)THPTriangularSolveBackward0_solution_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_solution", (getter)THPTriangularSolveBackward0_solution_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLinalgSolveTriangularBackward0_left_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LinalgSolveTriangularBackward0*>(self->cdata.get())->left;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgSolveTriangularBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgSolveTriangularBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgSolveTriangularBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgSolveTriangularBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgSolveTriangularBackward0_unitriangular_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LinalgSolveTriangularBackward0*>(self->cdata.get())->unitriangular;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgSolveTriangularBackward0_upper_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LinalgSolveTriangularBackward0*>(self->cdata.get())->upper;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgSolveTriangularBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgSolveTriangularBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgSolveTriangularBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgSolveTriangularBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LinalgSolveTriangularBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_left", (getter)THPLinalgSolveTriangularBackward0_left_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPLinalgSolveTriangularBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPLinalgSolveTriangularBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_unitriangular", (getter)THPLinalgSolveTriangularBackward0_unitriangular_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_upper", (getter)THPLinalgSolveTriangularBackward0_upper_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPLinalgSolveTriangularBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPLinalgSolveTriangularBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPTriuBackward0_diagonal_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TriuBackward0*>(self->cdata.get())->diagonal;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef TriuBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_diagonal", (getter)THPTriuBackward0_diagonal_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPToSparseBackward0_self_self_sym_blocksize_opt_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<ToSparseBackward0*>(self->cdata.get())->self_self_sym_blocksize_opt;
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

static struct PyGetSetDef ToSparseBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_self_sym_blocksize_opt", (getter)THPToSparseBackward0_self_self_sym_blocksize_opt_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPToSparseBackward1_self_self_sym_blocksize_opt_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<ToSparseBackward1*>(self->cdata.get())->self_self_sym_blocksize_opt;
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

static struct PyGetSetDef ToSparseBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_self_sym_blocksize_opt", (getter)THPToSparseBackward1_self_self_sym_blocksize_opt_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef Unique2Backward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPViewBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ViewBackward0*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef ViewBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sym_sizes", (getter)THPViewBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPViewBackwardAutogradNestedTensor0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ViewBackwardAutogradNestedTensor0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPViewBackwardAutogradNestedTensor0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ViewBackwardAutogradNestedTensor0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ViewBackwardAutogradNestedTensor0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPViewBackwardAutogradNestedTensor0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPViewBackwardAutogradNestedTensor0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPTrilinearBackward0_expand1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TrilinearBackward0*>(self->cdata.get())->expand1;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPTrilinearBackward0_expand2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TrilinearBackward0*>(self->cdata.get())->expand2;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPTrilinearBackward0_expand3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TrilinearBackward0*>(self->cdata.get())->expand3;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPTrilinearBackward0_i1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TrilinearBackward0*>(self->cdata.get())->i1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPTrilinearBackward0_i1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TrilinearBackward0*>(self->cdata.get())->i1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPTrilinearBackward0_i2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TrilinearBackward0*>(self->cdata.get())->i2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPTrilinearBackward0_i2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TrilinearBackward0*>(self->cdata.get())->i2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPTrilinearBackward0_i3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TrilinearBackward0*>(self->cdata.get())->i3_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPTrilinearBackward0_i3_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TrilinearBackward0*>(self->cdata.get())->i3_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPTrilinearBackward0_sumdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TrilinearBackward0*>(self->cdata.get())->sumdim;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef TrilinearBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_expand1", (getter)THPTrilinearBackward0_expand1_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_expand2", (getter)THPTrilinearBackward0_expand2_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_expand3", (getter)THPTrilinearBackward0_expand3_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_i1", (getter)THPTrilinearBackward0_i1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_i1", (getter)THPTrilinearBackward0_i1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_i2", (getter)THPTrilinearBackward0_i2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_i2", (getter)THPTrilinearBackward0_i2_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_i3", (getter)THPTrilinearBackward0_i3_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_i3", (getter)THPTrilinearBackward0_i3_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_sumdim", (getter)THPTrilinearBackward0_sumdim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPBinaryCrossEntropyBackward0_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<BinaryCrossEntropyBackward0*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyBackward0_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyBackward0*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyBackward0_target_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyBackward0*>(self->cdata.get())->target_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef BinaryCrossEntropyBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_reduction", (getter)THPBinaryCrossEntropyBackward0_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPBinaryCrossEntropyBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPBinaryCrossEntropyBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPBinaryCrossEntropyBackward0_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_target", (getter)THPBinaryCrossEntropyBackward0_target_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPBinaryCrossEntropyBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPBinaryCrossEntropyBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef EmbeddingRenormBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPHuberLossBackward0_delta_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<HuberLossBackward0*>(self->cdata.get())->delta;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPHuberLossBackward0_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<HuberLossBackward0*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPHuberLossBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HuberLossBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPHuberLossBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HuberLossBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPHuberLossBackward0_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HuberLossBackward0*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPHuberLossBackward0_target_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HuberLossBackward0*>(self->cdata.get())->target_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef HuberLossBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_delta", (getter)THPHuberLossBackward0_delta_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPHuberLossBackward0_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPHuberLossBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPHuberLossBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPHuberLossBackward0_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_target", (getter)THPHuberLossBackward0_target_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSiluBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SiluBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSiluBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SiluBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SiluBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSiluBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSiluBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCeluBackward0_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CeluBackward0*>(self->cdata.get())->alpha;
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

PyObject* THPCeluBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CeluBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCeluBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CeluBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CeluBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_alpha", (getter)THPCeluBackward0_alpha_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPCeluBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPCeluBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCeluBackward1_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CeluBackward1*>(self->cdata.get())->alpha;
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

PyObject* THPCeluBackward1_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CeluBackward1*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCeluBackward1_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CeluBackward1*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CeluBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_alpha", (getter)THPCeluBackward1_alpha_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPCeluBackward1_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPCeluBackward1_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSparseLogSoftmaxBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SparseLogSoftmaxBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseLogSoftmaxBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseLogSoftmaxBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseLogSoftmaxBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseLogSoftmaxBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseLogSoftmaxBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseLogSoftmaxBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseLogSoftmaxBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseLogSoftmaxBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SparseLogSoftmaxBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSparseLogSoftmaxBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSparseLogSoftmaxBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSparseLogSoftmaxBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPSparseLogSoftmaxBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPSparseLogSoftmaxBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMaskedSoftmaxBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<MaskedSoftmaxBackward0*>(self->cdata.get())->dim;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaskedSoftmaxBackward0_mask_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaskedSoftmaxBackward0*>(self->cdata.get())->mask_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaskedSoftmaxBackward0_mask_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaskedSoftmaxBackward0*>(self->cdata.get())->mask_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaskedSoftmaxBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaskedSoftmaxBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaskedSoftmaxBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaskedSoftmaxBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MaskedSoftmaxBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPMaskedSoftmaxBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mask", (getter)THPMaskedSoftmaxBackward0_mask_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_mask", (getter)THPMaskedSoftmaxBackward0_mask_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPMaskedSoftmaxBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPMaskedSoftmaxBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSoftmaxBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SoftmaxBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftmaxBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftmaxBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftmaxBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftmaxBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SoftmaxBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSoftmaxBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPSoftmaxBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPSoftmaxBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPThresholdBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThresholdBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThresholdBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThresholdBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPThresholdBackward0_threshold_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ThresholdBackward0*>(self->cdata.get())->threshold;
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

static struct PyGetSetDef ThresholdBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPThresholdBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPThresholdBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_threshold", (getter)THPThresholdBackward0_threshold_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPReplicationPad3DBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ReplicationPad3DBackward0*>(self->cdata.get())->padding;
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

PyObject* THPReplicationPad3DBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ReplicationPad3DBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPReplicationPad3DBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ReplicationPad3DBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ReplicationPad3DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_padding", (getter)THPReplicationPad3DBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPReplicationPad3DBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPReplicationPad3DBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleLinear1DBackward0_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleLinear1DBackward0*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleLinear1DBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleLinear1DBackward0*>(self->cdata.get())->output_size;
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

PyObject* THPUpsampleLinear1DBackward0_scales_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleLinear1DBackward0*>(self->cdata.get())->scales;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleLinear1DBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleLinear1DBackward0*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef UpsampleLinear1DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_align_corners", (getter)THPUpsampleLinear1DBackward0_align_corners_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPUpsampleLinear1DBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales", (getter)THPUpsampleLinear1DBackward0_scales_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPUpsampleLinear1DBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleBilinear2DBackward0_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBilinear2DBackward0*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBilinear2DBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBilinear2DBackward0*>(self->cdata.get())->output_size;
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

PyObject* THPUpsampleBilinear2DBackward0_scales_h_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleBilinear2DBackward0*>(self->cdata.get())->scales_h;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBilinear2DBackward0_scales_w_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleBilinear2DBackward0*>(self->cdata.get())->scales_w;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBilinear2DBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBilinear2DBackward0*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef UpsampleBilinear2DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_align_corners", (getter)THPUpsampleBilinear2DBackward0_align_corners_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPUpsampleBilinear2DBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_h", (getter)THPUpsampleBilinear2DBackward0_scales_h_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_w", (getter)THPUpsampleBilinear2DBackward0_scales_w_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPUpsampleBilinear2DBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleBicubic2DBackward0_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBicubic2DBackward0*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBicubic2DBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBicubic2DBackward0*>(self->cdata.get())->output_size;
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

PyObject* THPUpsampleBicubic2DBackward0_scales_h_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleBicubic2DBackward0*>(self->cdata.get())->scales_h;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBicubic2DBackward0_scales_w_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleBicubic2DBackward0*>(self->cdata.get())->scales_w;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBicubic2DBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBicubic2DBackward0*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef UpsampleBicubic2DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_align_corners", (getter)THPUpsampleBicubic2DBackward0_align_corners_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPUpsampleBicubic2DBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_h", (getter)THPUpsampleBicubic2DBackward0_scales_h_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_w", (getter)THPUpsampleBicubic2DBackward0_scales_w_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPUpsampleBicubic2DBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleNearest2DBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleNearest2DBackward0*>(self->cdata.get())->output_size;
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

PyObject* THPUpsampleNearest2DBackward0_scales_h_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest2DBackward0*>(self->cdata.get())->scales_h;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest2DBackward0_scales_w_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest2DBackward0*>(self->cdata.get())->scales_w;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest2DBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleNearest2DBackward0*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef UpsampleNearest2DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_output_size", (getter)THPUpsampleNearest2DBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_h", (getter)THPUpsampleNearest2DBackward0_scales_h_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_w", (getter)THPUpsampleNearest2DBackward0_scales_w_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPUpsampleNearest2DBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPPixelShuffleBackward0_upscale_factor_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<PixelShuffleBackward0*>(self->cdata.get())->upscale_factor;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef PixelShuffleBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_upscale_factor", (getter)THPPixelShuffleBackward0_upscale_factor_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAdaptiveAvgPool3DBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AdaptiveAvgPool3DBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAdaptiveAvgPool3DBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AdaptiveAvgPool3DBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AdaptiveAvgPool3DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPAdaptiveAvgPool3DBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPAdaptiveAvgPool3DBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAdaptiveMaxPool2DBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AdaptiveMaxPool2DBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAdaptiveMaxPool2DBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AdaptiveMaxPool2DBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPAdaptiveMaxPool2DBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AdaptiveMaxPool2DBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAdaptiveMaxPool2DBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AdaptiveMaxPool2DBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AdaptiveMaxPool2DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPAdaptiveMaxPool2DBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPAdaptiveMaxPool2DBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPAdaptiveMaxPool2DBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPAdaptiveMaxPool2DBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAdaptiveMaxPool3DBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AdaptiveMaxPool3DBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAdaptiveMaxPool3DBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AdaptiveMaxPool3DBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPAdaptiveMaxPool3DBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AdaptiveMaxPool3DBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAdaptiveMaxPool3DBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AdaptiveMaxPool3DBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AdaptiveMaxPool3DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPAdaptiveMaxPool3DBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPAdaptiveMaxPool3DBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPAdaptiveMaxPool3DBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPAdaptiveMaxPool3DBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPFractionalMaxPool2DBackward0_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FractionalMaxPool2DBackward0*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPFractionalMaxPool2DBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FractionalMaxPool2DBackward0*>(self->cdata.get())->output_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPFractionalMaxPool2DBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FractionalMaxPool2DBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFractionalMaxPool2DBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FractionalMaxPool2DBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPFractionalMaxPool2DBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FractionalMaxPool2DBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFractionalMaxPool2DBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FractionalMaxPool2DBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FractionalMaxPool2DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_kernel_size", (getter)THPFractionalMaxPool2DBackward0_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPFractionalMaxPool2DBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPFractionalMaxPool2DBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPFractionalMaxPool2DBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPFractionalMaxPool2DBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPFractionalMaxPool2DBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLinearBackwardBackward0_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinearBackwardBackward0*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinearBackwardBackward0_grad_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinearBackwardBackward0*>(self->cdata.get())->grad_output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinearBackwardBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinearBackwardBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinearBackwardBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinearBackwardBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinearBackwardBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinearBackwardBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinearBackwardBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinearBackwardBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LinearBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_output", (getter)THPLinearBackwardBackward0_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_grad_output", (getter)THPLinearBackwardBackward0_grad_output_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPLinearBackwardBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPLinearBackwardBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPLinearBackwardBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPLinearBackwardBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMaxPool2DBackward0_ceil_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxPool2DBackward0*>(self->cdata.get())->ceil_mode;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool2DBackward0_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxPool2DBackward0*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool2DBackward0_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxPool2DBackward0*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool2DBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxPool2DBackward0*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool2DBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxPool2DBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool2DBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxPool2DBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool2DBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxPool2DBackward0*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MaxPool2DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_ceil_mode", (getter)THPMaxPool2DBackward0_ceil_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPMaxPool2DBackward0_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_kernel_size", (getter)THPMaxPool2DBackward0_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPMaxPool2DBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPMaxPool2DBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPMaxPool2DBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPMaxPool2DBackward0_stride_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMaxUnpool2DBackward0_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxUnpool2DBackward0*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxUnpool2DBackward0_indices_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxUnpool2DBackward0*>(self->cdata.get())->indices_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MaxUnpool2DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_indices", (getter)THPMaxUnpool2DBackward0_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_indices", (getter)THPMaxUnpool2DBackward0_indices_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMaxUnpool3DBackward0_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxUnpool3DBackward0*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxUnpool3DBackward0_indices_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxUnpool3DBackward0*>(self->cdata.get())->indices_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MaxUnpool3DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_indices", (getter)THPMaxUnpool3DBackward0_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_indices", (getter)THPMaxUnpool3DBackward0_indices_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSlowConvTranspose2DBackward0_bias_sym_sizes_opt_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<SlowConvTranspose2DBackward0*>(self->cdata.get())->bias_sym_sizes_opt;
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

PyObject* THPSlowConvTranspose2DBackward0_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvTranspose2DBackward0*>(self->cdata.get())->dilation;
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

PyObject* THPSlowConvTranspose2DBackward0_output_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvTranspose2DBackward0*>(self->cdata.get())->output_padding;
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

PyObject* THPSlowConvTranspose2DBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvTranspose2DBackward0*>(self->cdata.get())->padding;
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

PyObject* THPSlowConvTranspose2DBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvTranspose2DBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvTranspose2DBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvTranspose2DBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvTranspose2DBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvTranspose2DBackward0*>(self->cdata.get())->stride;
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

PyObject* THPSlowConvTranspose2DBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvTranspose2DBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvTranspose2DBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvTranspose2DBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SlowConvTranspose2DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_bias_sym_sizes_opt", (getter)THPSlowConvTranspose2DBackward0_bias_sym_sizes_opt_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPSlowConvTranspose2DBackward0_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_padding", (getter)THPSlowConvTranspose2DBackward0_output_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPSlowConvTranspose2DBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSlowConvTranspose2DBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSlowConvTranspose2DBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPSlowConvTranspose2DBackward0_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPSlowConvTranspose2DBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPSlowConvTranspose2DBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSlowConv2DBackwardBackward0_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConv2DBackwardBackward0*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConv2DBackwardBackward0_grad_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConv2DBackwardBackward0*>(self->cdata.get())->grad_output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConv2DBackwardBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConv2DBackwardBackward0*>(self->cdata.get())->padding;
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

PyObject* THPSlowConv2DBackwardBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConv2DBackwardBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConv2DBackwardBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConv2DBackwardBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConv2DBackwardBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConv2DBackwardBackward0*>(self->cdata.get())->stride;
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

PyObject* THPSlowConv2DBackwardBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConv2DBackwardBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConv2DBackwardBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConv2DBackwardBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SlowConv2DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_output", (getter)THPSlowConv2DBackwardBackward0_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_grad_output", (getter)THPSlowConv2DBackwardBackward0_grad_output_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPSlowConv2DBackwardBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSlowConv2DBackwardBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSlowConv2DBackwardBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPSlowConv2DBackwardBackward0_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPSlowConv2DBackwardBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPSlowConv2DBackwardBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPConvDepthwise3DBackward0_bias_sym_sizes_opt_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<ConvDepthwise3DBackward0*>(self->cdata.get())->bias_sym_sizes_opt;
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

PyObject* THPConvDepthwise3DBackward0_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvDepthwise3DBackward0*>(self->cdata.get())->dilation;
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

PyObject* THPConvDepthwise3DBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvDepthwise3DBackward0*>(self->cdata.get())->padding;
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

PyObject* THPConvDepthwise3DBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvDepthwise3DBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvDepthwise3DBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvDepthwise3DBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvDepthwise3DBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvDepthwise3DBackward0*>(self->cdata.get())->stride;
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

PyObject* THPConvDepthwise3DBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvDepthwise3DBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvDepthwise3DBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvDepthwise3DBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ConvDepthwise3DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_bias_sym_sizes_opt", (getter)THPConvDepthwise3DBackward0_bias_sym_sizes_opt_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPConvDepthwise3DBackward0_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPConvDepthwise3DBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPConvDepthwise3DBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPConvDepthwise3DBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPConvDepthwise3DBackward0_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPConvDepthwise3DBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPConvDepthwise3DBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSlowConvDilated2DBackward0_bias_sym_sizes_opt_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<SlowConvDilated2DBackward0*>(self->cdata.get())->bias_sym_sizes_opt;
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

PyObject* THPSlowConvDilated2DBackward0_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvDilated2DBackward0*>(self->cdata.get())->dilation;
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

PyObject* THPSlowConvDilated2DBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvDilated2DBackward0*>(self->cdata.get())->padding;
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

PyObject* THPSlowConvDilated2DBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvDilated2DBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvDilated2DBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvDilated2DBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvDilated2DBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvDilated2DBackward0*>(self->cdata.get())->stride;
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

PyObject* THPSlowConvDilated2DBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvDilated2DBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvDilated2DBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvDilated2DBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SlowConvDilated2DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_bias_sym_sizes_opt", (getter)THPSlowConvDilated2DBackward0_bias_sym_sizes_opt_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPSlowConvDilated2DBackward0_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPSlowConvDilated2DBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSlowConvDilated2DBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSlowConvDilated2DBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPSlowConvDilated2DBackward0_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPSlowConvDilated2DBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPSlowConvDilated2DBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCol2ImBackward0_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Col2ImBackward0*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCol2ImBackward0_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Col2ImBackward0*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCol2ImBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Col2ImBackward0*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCol2ImBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Col2ImBackward0*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef Col2ImBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dilation", (getter)THPCol2ImBackward0_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_kernel_size", (getter)THPCol2ImBackward0_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPCol2ImBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPCol2ImBackward0_stride_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAvgPool2DBackwardBackward0_ceil_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool2DBackwardBackward0*>(self->cdata.get())->ceil_mode;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool2DBackwardBackward0_count_include_pad_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool2DBackwardBackward0*>(self->cdata.get())->count_include_pad;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool2DBackwardBackward0_divisor_override_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<AvgPool2DBackwardBackward0*>(self->cdata.get())->divisor_override;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool2DBackwardBackward0_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool2DBackwardBackward0*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool2DBackwardBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool2DBackwardBackward0*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool2DBackwardBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool2DBackwardBackward0*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AvgPool2DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_ceil_mode", (getter)THPAvgPool2DBackwardBackward0_ceil_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_count_include_pad", (getter)THPAvgPool2DBackwardBackward0_count_include_pad_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_divisor_override", (getter)THPAvgPool2DBackwardBackward0_divisor_override_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_kernel_size", (getter)THPAvgPool2DBackwardBackward0_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPAvgPool2DBackwardBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPAvgPool2DBackwardBackward0_stride_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNllLossBackwardBackward0_ignore_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NllLossBackwardBackward0*>(self->cdata.get())->ignore_index;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLossBackwardBackward0_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NllLossBackwardBackward0*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLossBackwardBackward0_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLossBackwardBackward0*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLossBackwardBackward0_target_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLossBackwardBackward0*>(self->cdata.get())->target_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLossBackwardBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLossBackwardBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLossBackwardBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLossBackwardBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NllLossBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_ignore_index", (getter)THPNllLossBackwardBackward0_ignore_index_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPNllLossBackwardBackward0_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPNllLossBackwardBackward0_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_target", (getter)THPNllLossBackwardBackward0_target_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPNllLossBackwardBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPNllLossBackwardBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPReflectionPad3DBackwardBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ReflectionPad3DBackwardBackward0*>(self->cdata.get())->padding;
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

static struct PyGetSetDef ReflectionPad3DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_padding", (getter)THPReflectionPad3DBackwardBackward0_padding_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPReplicationPad2DBackwardBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ReplicationPad2DBackwardBackward0*>(self->cdata.get())->padding;
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

static struct PyGetSetDef ReplicationPad2DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_padding", (getter)THPReplicationPad2DBackwardBackward0_padding_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSparseMmReduceImplBackward0_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseMmReduceImplBackward0*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseMmReduceImplBackward0_other_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseMmReduceImplBackward0*>(self->cdata.get())->other_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseMmReduceImplBackward0_reduce_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SparseMmReduceImplBackward0*>(self->cdata.get())->reduce;
  return PyUnicode_FromStringAndSize(prop.data(), prop.size());
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseMmReduceImplBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseMmReduceImplBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseMmReduceImplBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseMmReduceImplBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseMmReduceImplBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseMmReduceImplBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseMmReduceImplBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseMmReduceImplBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SparseMmReduceImplBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPSparseMmReduceImplBackward0_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_other", (getter)THPSparseMmReduceImplBackward0_other_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduce", (getter)THPSparseMmReduceImplBackward0_reduce_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSparseMmReduceImplBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSparseMmReduceImplBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPSparseMmReduceImplBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPSparseMmReduceImplBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSmoothL1LossBackwardBackward0_beta_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SmoothL1LossBackwardBackward0*>(self->cdata.get())->beta;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSmoothL1LossBackwardBackward0_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SmoothL1LossBackwardBackward0*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSmoothL1LossBackwardBackward0_grad_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SmoothL1LossBackwardBackward0*>(self->cdata.get())->grad_output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSmoothL1LossBackwardBackward0_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SmoothL1LossBackwardBackward0*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSmoothL1LossBackwardBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SmoothL1LossBackwardBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSmoothL1LossBackwardBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SmoothL1LossBackwardBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSmoothL1LossBackwardBackward0_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SmoothL1LossBackwardBackward0*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSmoothL1LossBackwardBackward0_target_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SmoothL1LossBackwardBackward0*>(self->cdata.get())->target_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SmoothL1LossBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_beta", (getter)THPSmoothL1LossBackwardBackward0_beta_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output", (getter)THPSmoothL1LossBackwardBackward0_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_grad_output", (getter)THPSmoothL1LossBackwardBackward0_grad_output_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPSmoothL1LossBackwardBackward0_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSmoothL1LossBackwardBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSmoothL1LossBackwardBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPSmoothL1LossBackwardBackward0_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_target", (getter)THPSmoothL1LossBackwardBackward0_target_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPHuberLossBackwardBackward0_delta_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<HuberLossBackwardBackward0*>(self->cdata.get())->delta;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPHuberLossBackwardBackward0_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HuberLossBackwardBackward0*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPHuberLossBackwardBackward0_grad_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HuberLossBackwardBackward0*>(self->cdata.get())->grad_output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPHuberLossBackwardBackward0_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<HuberLossBackwardBackward0*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPHuberLossBackwardBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HuberLossBackwardBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPHuberLossBackwardBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HuberLossBackwardBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPHuberLossBackwardBackward0_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HuberLossBackwardBackward0*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPHuberLossBackwardBackward0_target_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HuberLossBackwardBackward0*>(self->cdata.get())->target_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef HuberLossBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_delta", (getter)THPHuberLossBackwardBackward0_delta_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output", (getter)THPHuberLossBackwardBackward0_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_grad_output", (getter)THPHuberLossBackwardBackward0_grad_output_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPHuberLossBackwardBackward0_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPHuberLossBackwardBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPHuberLossBackwardBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPHuberLossBackwardBackward0_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_target", (getter)THPHuberLossBackwardBackward0_target_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleBicubic2DBackwardBackward0_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBicubic2DBackwardBackward0*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBicubic2DBackwardBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBicubic2DBackwardBackward0*>(self->cdata.get())->output_size;
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

PyObject* THPUpsampleBicubic2DBackwardBackward0_scales_h_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleBicubic2DBackwardBackward0*>(self->cdata.get())->scales_h;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBicubic2DBackwardBackward0_scales_w_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleBicubic2DBackwardBackward0*>(self->cdata.get())->scales_w;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleBicubic2DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_align_corners", (getter)THPUpsampleBicubic2DBackwardBackward0_align_corners_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPUpsampleBicubic2DBackwardBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_h", (getter)THPUpsampleBicubic2DBackwardBackward0_scales_h_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_w", (getter)THPUpsampleBicubic2DBackwardBackward0_scales_w_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleTrilinear3DBackwardBackward0_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleTrilinear3DBackwardBackward0*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleTrilinear3DBackwardBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleTrilinear3DBackwardBackward0*>(self->cdata.get())->output_size;
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

PyObject* THPUpsampleTrilinear3DBackwardBackward0_scales_d_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleTrilinear3DBackwardBackward0*>(self->cdata.get())->scales_d;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleTrilinear3DBackwardBackward0_scales_h_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleTrilinear3DBackwardBackward0*>(self->cdata.get())->scales_h;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleTrilinear3DBackwardBackward0_scales_w_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleTrilinear3DBackwardBackward0*>(self->cdata.get())->scales_w;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleTrilinear3DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_align_corners", (getter)THPUpsampleTrilinear3DBackwardBackward0_align_corners_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPUpsampleTrilinear3DBackwardBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_d", (getter)THPUpsampleTrilinear3DBackwardBackward0_scales_d_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_h", (getter)THPUpsampleTrilinear3DBackwardBackward0_scales_h_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_w", (getter)THPUpsampleTrilinear3DBackwardBackward0_scales_w_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleNearest3DBackwardBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleNearest3DBackwardBackward0*>(self->cdata.get())->output_size;
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

PyObject* THPUpsampleNearest3DBackwardBackward0_scales_d_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest3DBackwardBackward0*>(self->cdata.get())->scales_d;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest3DBackwardBackward0_scales_h_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest3DBackwardBackward0*>(self->cdata.get())->scales_h;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest3DBackwardBackward0_scales_w_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest3DBackwardBackward0*>(self->cdata.get())->scales_w;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleNearest3DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_output_size", (getter)THPUpsampleNearest3DBackwardBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_d", (getter)THPUpsampleNearest3DBackwardBackward0_scales_d_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_h", (getter)THPUpsampleNearest3DBackwardBackward0_scales_h_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_w", (getter)THPUpsampleNearest3DBackwardBackward0_scales_w_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleNearestExact3DBackwardBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleNearestExact3DBackwardBackward0*>(self->cdata.get())->output_size;
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

PyObject* THPUpsampleNearestExact3DBackwardBackward0_scales_d_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearestExact3DBackwardBackward0*>(self->cdata.get())->scales_d;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearestExact3DBackwardBackward0_scales_h_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearestExact3DBackwardBackward0*>(self->cdata.get())->scales_h;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearestExact3DBackwardBackward0_scales_w_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearestExact3DBackwardBackward0*>(self->cdata.get())->scales_w;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleNearestExact3DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_output_size", (getter)THPUpsampleNearestExact3DBackwardBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_d", (getter)THPUpsampleNearestExact3DBackwardBackward0_scales_d_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_h", (getter)THPUpsampleNearestExact3DBackwardBackward0_scales_h_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_w", (getter)THPUpsampleNearestExact3DBackwardBackward0_scales_w_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCudnnConvolutionBackward0_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionBackward0*>(self->cdata.get())->dilation;
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

PyObject* THPCudnnConvolutionBackward0_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionBackward0*>(self->cdata.get())->groups;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionBackward0*>(self->cdata.get())->padding;
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

PyObject* THPCudnnConvolutionBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnConvolutionBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnConvolutionBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionBackward0*>(self->cdata.get())->stride;
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

PyObject* THPCudnnConvolutionBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnConvolutionBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnConvolutionBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CudnnConvolutionBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dilation", (getter)THPCudnnConvolutionBackward0_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_groups", (getter)THPCudnnConvolutionBackward0_groups_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPCudnnConvolutionBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPCudnnConvolutionBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPCudnnConvolutionBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPCudnnConvolutionBackward0_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPCudnnConvolutionBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPCudnnConvolutionBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef CudnnRnnBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPNestedFromPaddedBackward0_fuse_transform_0213_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NestedFromPaddedBackward0*>(self->cdata.get())->fuse_transform_0213;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNestedFromPaddedBackward0_padded_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NestedFromPaddedBackward0*>(self->cdata.get())->padded_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNestedFromPaddedBackward0_padded_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NestedFromPaddedBackward0*>(self->cdata.get())->padded_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NestedFromPaddedBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_fuse_transform_0213", (getter)THPNestedFromPaddedBackward0_fuse_transform_0213_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padded", (getter)THPNestedFromPaddedBackward0_padded_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_padded", (getter)THPNestedFromPaddedBackward0_padded_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef NestedViewFromBufferBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef NestedViewFromJaggedBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPSafeSoftmaxBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SafeSoftmaxBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSafeSoftmaxBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SafeSoftmaxBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSafeSoftmaxBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SafeSoftmaxBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SafeSoftmaxBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSafeSoftmaxBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPSafeSoftmaxBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPSafeSoftmaxBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPScaledDotProductEfficientAttentionBackward0_attn_bias_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductEfficientAttentionBackward0*>(self->cdata.get())->attn_bias_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductEfficientAttentionBackward0_attn_bias_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductEfficientAttentionBackward0*>(self->cdata.get())->attn_bias_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductEfficientAttentionBackward0_dropout_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ScaledDotProductEfficientAttentionBackward0*>(self->cdata.get())->dropout_p;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductEfficientAttentionBackward0_is_causal_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ScaledDotProductEfficientAttentionBackward0*>(self->cdata.get())->is_causal;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductEfficientAttentionBackward0_key_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductEfficientAttentionBackward0*>(self->cdata.get())->key_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductEfficientAttentionBackward0_key_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductEfficientAttentionBackward0*>(self->cdata.get())->key_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductEfficientAttentionBackward0_query_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductEfficientAttentionBackward0*>(self->cdata.get())->query_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductEfficientAttentionBackward0_query_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductEfficientAttentionBackward0*>(self->cdata.get())->query_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductEfficientAttentionBackward0_scale_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<ScaledDotProductEfficientAttentionBackward0*>(self->cdata.get())->scale;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductEfficientAttentionBackward0_value_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductEfficientAttentionBackward0*>(self->cdata.get())->value_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductEfficientAttentionBackward0_value_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductEfficientAttentionBackward0*>(self->cdata.get())->value_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductEfficientAttentionBackward0_log_sumexp_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductEfficientAttentionBackward0*>(self->cdata.get())->log_sumexp_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductEfficientAttentionBackward0_log_sumexp_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductEfficientAttentionBackward0*>(self->cdata.get())->log_sumexp_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductEfficientAttentionBackward0_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductEfficientAttentionBackward0*>(self->cdata.get())->output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductEfficientAttentionBackward0_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductEfficientAttentionBackward0*>(self->cdata.get())->output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductEfficientAttentionBackward0_philox_offset_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductEfficientAttentionBackward0*>(self->cdata.get())->philox_offset_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductEfficientAttentionBackward0_philox_offset_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductEfficientAttentionBackward0*>(self->cdata.get())->philox_offset_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductEfficientAttentionBackward0_philox_seed_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductEfficientAttentionBackward0*>(self->cdata.get())->philox_seed_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductEfficientAttentionBackward0_philox_seed_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductEfficientAttentionBackward0*>(self->cdata.get())->philox_seed_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ScaledDotProductEfficientAttentionBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_attn_bias", (getter)THPScaledDotProductEfficientAttentionBackward0_attn_bias_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_attn_bias", (getter)THPScaledDotProductEfficientAttentionBackward0_attn_bias_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dropout_p", (getter)THPScaledDotProductEfficientAttentionBackward0_dropout_p_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_is_causal", (getter)THPScaledDotProductEfficientAttentionBackward0_is_causal_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_key", (getter)THPScaledDotProductEfficientAttentionBackward0_key_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_key", (getter)THPScaledDotProductEfficientAttentionBackward0_key_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_query", (getter)THPScaledDotProductEfficientAttentionBackward0_query_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_query", (getter)THPScaledDotProductEfficientAttentionBackward0_query_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale", (getter)THPScaledDotProductEfficientAttentionBackward0_scale_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_value", (getter)THPScaledDotProductEfficientAttentionBackward0_value_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_value", (getter)THPScaledDotProductEfficientAttentionBackward0_value_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_log_sumexp", (getter)THPScaledDotProductEfficientAttentionBackward0_log_sumexp_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_log_sumexp", (getter)THPScaledDotProductEfficientAttentionBackward0_log_sumexp_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output", (getter)THPScaledDotProductEfficientAttentionBackward0_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_output", (getter)THPScaledDotProductEfficientAttentionBackward0_output_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_philox_offset", (getter)THPScaledDotProductEfficientAttentionBackward0_philox_offset_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_philox_offset", (getter)THPScaledDotProductEfficientAttentionBackward0_philox_offset_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_philox_seed", (getter)THPScaledDotProductEfficientAttentionBackward0_philox_seed_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_philox_seed", (getter)THPScaledDotProductEfficientAttentionBackward0_philox_seed_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPEfficientAttentionBackward0_bias_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EfficientAttentionBackward0*>(self->cdata.get())->bias_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEfficientAttentionBackward0_bias_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EfficientAttentionBackward0*>(self->cdata.get())->bias_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPEfficientAttentionBackward0_cu_seqlens_k_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EfficientAttentionBackward0*>(self->cdata.get())->cu_seqlens_k_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEfficientAttentionBackward0_cu_seqlens_k_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EfficientAttentionBackward0*>(self->cdata.get())->cu_seqlens_k_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPEfficientAttentionBackward0_cu_seqlens_q_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EfficientAttentionBackward0*>(self->cdata.get())->cu_seqlens_q_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEfficientAttentionBackward0_cu_seqlens_q_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EfficientAttentionBackward0*>(self->cdata.get())->cu_seqlens_q_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPEfficientAttentionBackward0_custom_mask_type_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EfficientAttentionBackward0*>(self->cdata.get())->custom_mask_type;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPEfficientAttentionBackward0_dropout_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EfficientAttentionBackward0*>(self->cdata.get())->dropout_p;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPEfficientAttentionBackward0_key_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EfficientAttentionBackward0*>(self->cdata.get())->key_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEfficientAttentionBackward0_key_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EfficientAttentionBackward0*>(self->cdata.get())->key_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPEfficientAttentionBackward0_query_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EfficientAttentionBackward0*>(self->cdata.get())->query_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEfficientAttentionBackward0_query_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EfficientAttentionBackward0*>(self->cdata.get())->query_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPEfficientAttentionBackward0_scale_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<EfficientAttentionBackward0*>(self->cdata.get())->scale;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPEfficientAttentionBackward0_value_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EfficientAttentionBackward0*>(self->cdata.get())->value_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEfficientAttentionBackward0_value_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EfficientAttentionBackward0*>(self->cdata.get())->value_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPEfficientAttentionBackward0_logsumexp_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EfficientAttentionBackward0*>(self->cdata.get())->logsumexp_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEfficientAttentionBackward0_logsumexp_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EfficientAttentionBackward0*>(self->cdata.get())->logsumexp_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPEfficientAttentionBackward0_max_seqlen_batch_k_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EfficientAttentionBackward0*>(self->cdata.get())->max_seqlen_batch_k;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPEfficientAttentionBackward0_max_seqlen_batch_q_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EfficientAttentionBackward0*>(self->cdata.get())->max_seqlen_batch_q;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPEfficientAttentionBackward0_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EfficientAttentionBackward0*>(self->cdata.get())->output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEfficientAttentionBackward0_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EfficientAttentionBackward0*>(self->cdata.get())->output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPEfficientAttentionBackward0_philox_offset_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EfficientAttentionBackward0*>(self->cdata.get())->philox_offset_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEfficientAttentionBackward0_philox_offset_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EfficientAttentionBackward0*>(self->cdata.get())->philox_offset_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPEfficientAttentionBackward0_philox_seed_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EfficientAttentionBackward0*>(self->cdata.get())->philox_seed_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEfficientAttentionBackward0_philox_seed_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EfficientAttentionBackward0*>(self->cdata.get())->philox_seed_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef EfficientAttentionBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_bias", (getter)THPEfficientAttentionBackward0_bias_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_bias", (getter)THPEfficientAttentionBackward0_bias_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_cu_seqlens_k", (getter)THPEfficientAttentionBackward0_cu_seqlens_k_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_cu_seqlens_k", (getter)THPEfficientAttentionBackward0_cu_seqlens_k_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_cu_seqlens_q", (getter)THPEfficientAttentionBackward0_cu_seqlens_q_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_cu_seqlens_q", (getter)THPEfficientAttentionBackward0_cu_seqlens_q_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_custom_mask_type", (getter)THPEfficientAttentionBackward0_custom_mask_type_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dropout_p", (getter)THPEfficientAttentionBackward0_dropout_p_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_key", (getter)THPEfficientAttentionBackward0_key_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_key", (getter)THPEfficientAttentionBackward0_key_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_query", (getter)THPEfficientAttentionBackward0_query_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_query", (getter)THPEfficientAttentionBackward0_query_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale", (getter)THPEfficientAttentionBackward0_scale_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_value", (getter)THPEfficientAttentionBackward0_value_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_value", (getter)THPEfficientAttentionBackward0_value_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_logsumexp", (getter)THPEfficientAttentionBackward0_logsumexp_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_logsumexp", (getter)THPEfficientAttentionBackward0_logsumexp_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_max_seqlen_batch_k", (getter)THPEfficientAttentionBackward0_max_seqlen_batch_k_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_max_seqlen_batch_q", (getter)THPEfficientAttentionBackward0_max_seqlen_batch_q_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output", (getter)THPEfficientAttentionBackward0_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_output", (getter)THPEfficientAttentionBackward0_output_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_philox_offset", (getter)THPEfficientAttentionBackward0_philox_offset_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_philox_offset", (getter)THPEfficientAttentionBackward0_philox_offset_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_philox_seed", (getter)THPEfficientAttentionBackward0_philox_seed_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_philox_seed", (getter)THPEfficientAttentionBackward0_philox_seed_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPFftR2CBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FftR2CBackward0*>(self->cdata.get())->dim;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPFftR2CBackward0_normalization_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FftR2CBackward0*>(self->cdata.get())->normalization;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPFftR2CBackward0_onesided_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FftR2CBackward0*>(self->cdata.get())->onesided;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPFftR2CBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FftR2CBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFftR2CBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FftR2CBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FftR2CBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPFftR2CBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_normalization", (getter)THPFftR2CBackward0_normalization_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_onesided", (getter)THPFftR2CBackward0_onesided_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPFftR2CBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPFftR2CBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPThnnFusedGruCellBackward0_hidden_bias_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedGruCellBackward0*>(self->cdata.get())->hidden_bias_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedGruCellBackward0_hidden_bias_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedGruCellBackward0*>(self->cdata.get())->hidden_bias_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedGruCellBackward0_hidden_gates_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedGruCellBackward0*>(self->cdata.get())->hidden_gates_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedGruCellBackward0_hidden_gates_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedGruCellBackward0*>(self->cdata.get())->hidden_gates_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedGruCellBackward0_hx_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedGruCellBackward0*>(self->cdata.get())->hx_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedGruCellBackward0_hx_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedGruCellBackward0*>(self->cdata.get())->hx_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedGruCellBackward0_input_bias_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedGruCellBackward0*>(self->cdata.get())->input_bias_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedGruCellBackward0_input_bias_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedGruCellBackward0*>(self->cdata.get())->input_bias_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedGruCellBackward0_input_gates_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedGruCellBackward0*>(self->cdata.get())->input_gates_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedGruCellBackward0_input_gates_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedGruCellBackward0*>(self->cdata.get())->input_gates_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedGruCellBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedGruCellBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedGruCellBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedGruCellBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ThnnFusedGruCellBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_hidden_bias", (getter)THPThnnFusedGruCellBackward0_hidden_bias_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_hidden_bias", (getter)THPThnnFusedGruCellBackward0_hidden_bias_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_hidden_gates", (getter)THPThnnFusedGruCellBackward0_hidden_gates_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_hidden_gates", (getter)THPThnnFusedGruCellBackward0_hidden_gates_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_hx", (getter)THPThnnFusedGruCellBackward0_hx_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_hx", (getter)THPThnnFusedGruCellBackward0_hx_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input_bias", (getter)THPThnnFusedGruCellBackward0_input_bias_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input_bias", (getter)THPThnnFusedGruCellBackward0_input_bias_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input_gates", (getter)THPThnnFusedGruCellBackward0_input_gates_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input_gates", (getter)THPThnnFusedGruCellBackward0_input_gates_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPThnnFusedGruCellBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPThnnFusedGruCellBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPPackPaddedSequenceBackward0_batch_first_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<PackPaddedSequenceBackward0*>(self->cdata.get())->batch_first;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPPackPaddedSequenceBackward0_input_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<PackPaddedSequenceBackward0*>(self->cdata.get())->input_sym_sizes;
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

PyObject* THPPackPaddedSequenceBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PackPaddedSequenceBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPPackPaddedSequenceBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PackPaddedSequenceBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef PackPaddedSequenceBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_batch_first", (getter)THPPackPaddedSequenceBackward0_batch_first_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input_sym_sizes", (getter)THPPackPaddedSequenceBackward0_input_sym_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPPackPaddedSequenceBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPPackPaddedSequenceBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPForeachDivBackward0_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<ForeachDivBackward0*>(self->cdata.get());
  const auto& prop = node->other_;
  if (node->other_released_) {
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

PyObject* THPForeachDivBackward0_other_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<ForeachDivBackward0*>(self->cdata.get());
  const auto& prop = node->other_;
  if (node->other_released_) {
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

PyObject* THPForeachDivBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<ForeachDivBackward0*>(self->cdata.get());
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

PyObject* THPForeachDivBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<ForeachDivBackward0*>(self->cdata.get());
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

static struct PyGetSetDef ForeachDivBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPForeachDivBackward0_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_other", (getter)THPForeachDivBackward0_other_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPForeachDivBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPForeachDivBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSelectBackward0_copy_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SelectBackward0_copy*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSelectBackward0_copy_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SelectBackward0_copy*>(self->cdata.get())->index;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPSelectBackward0_copy_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SelectBackward0_copy*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef SelectBackward0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSelectBackward0_copy_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_index", (getter)THPSelectBackward0_copy_index_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPSelectBackward0_copy_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSelectBackwardAutogradNestedTensor0_copy_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SelectBackwardAutogradNestedTensor0_copy*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSelectBackwardAutogradNestedTensor0_copy_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SelectBackwardAutogradNestedTensor0_copy*>(self->cdata.get())->index;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPSelectBackwardAutogradNestedTensor0_copy_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SelectBackwardAutogradNestedTensor0_copy*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSelectBackwardAutogradNestedTensor0_copy_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SelectBackwardAutogradNestedTensor0_copy*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SelectBackwardAutogradNestedTensor0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSelectBackwardAutogradNestedTensor0_copy_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_index", (getter)THPSelectBackwardAutogradNestedTensor0_copy_index_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSelectBackwardAutogradNestedTensor0_copy_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSelectBackwardAutogradNestedTensor0_copy_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUnfoldBackward0_copy_dimension_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnfoldBackward0_copy*>(self->cdata.get())->dimension;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUnfoldBackward0_copy_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnfoldBackward0_copy*>(self->cdata.get())->self_sym_sizes;
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

PyObject* THPUnfoldBackward0_copy_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnfoldBackward0_copy*>(self->cdata.get())->size;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUnfoldBackward0_copy_step_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnfoldBackward0_copy*>(self->cdata.get())->step;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UnfoldBackward0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dimension", (getter)THPUnfoldBackward0_copy_dimension_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPUnfoldBackward0_copy_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_size", (getter)THPUnfoldBackward0_copy_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_step", (getter)THPUnfoldBackward0_copy_step_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef NestedViewFromJaggedBackward0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

void initialize_autogenerated_functions_3(PyObject* module) {
  static PyTypeObject AddbmmBackward0Class;
  addClass<AddbmmBackward0>(module, AddbmmBackward0Class, "AddbmmBackward0", AddbmmBackward0_properties);
  static PyTypeObject AffineGridGeneratorBackward0Class;
  addClass<AffineGridGeneratorBackward0>(module, AffineGridGeneratorBackward0Class, "AffineGridGeneratorBackward0", AffineGridGeneratorBackward0_properties);
  static PyTypeObject AtanhBackward0Class;
  addClass<AtanhBackward0>(module, AtanhBackward0Class, "AtanhBackward0", AtanhBackward0_properties);
  static PyTypeObject BernoulliBackward1Class;
  addClass<BernoulliBackward1>(module, BernoulliBackward1Class, "BernoulliBackward1", BernoulliBackward1_properties);
  static PyTypeObject BernoulliBackward2Class;
  addClass<BernoulliBackward2>(module, BernoulliBackward2Class, "BernoulliBackward2", BernoulliBackward2_properties);
  static PyTypeObject BmmBackward0Class;
  addClass<BmmBackward0>(module, BmmBackward0Class, "BmmBackward0", BmmBackward0_properties);
  static PyTypeObject CauchyBackward0Class;
  addClass<CauchyBackward0>(module, CauchyBackward0Class, "CauchyBackward0", CauchyBackward0_properties);
  static PyTypeObject ClampMinBackward0Class;
  addClass<ClampMinBackward0>(module, ClampMinBackward0Class, "ClampMinBackward0", ClampMinBackward0_properties);
  static PyTypeObject ClampMinBackward1Class;
  addClass<ClampMinBackward1>(module, ClampMinBackward1Class, "ClampMinBackward1", ClampMinBackward1_properties);
  static PyTypeObject CoalesceBackward0Class;
  addClass<CoalesceBackward0>(module, CoalesceBackward0Class, "CoalesceBackward0", CoalesceBackward0_properties);
  static PyTypeObject ConjPhysicalBackward1Class;
  addClass<ConjPhysicalBackward1>(module, ConjPhysicalBackward1Class, "ConjPhysicalBackward1", ConjPhysicalBackward1_properties);
  static PyTypeObject CumprodBackward0Class;
  addClass<CumprodBackward0>(module, CumprodBackward0Class, "CumprodBackward0", CumprodBackward0_properties);
  static PyTypeObject LinalgDetBackward0Class;
  addClass<LinalgDetBackward0>(module, LinalgDetBackward0Class, "LinalgDetBackward0", LinalgDetBackward0_properties);
  static PyTypeObject DiagEmbedBackward0Class;
  addClass<DiagEmbedBackward0>(module, DiagEmbedBackward0Class, "DiagEmbedBackward0", DiagEmbedBackward0_properties);
  static PyTypeObject DiagonalBackward0Class;
  addClass<DiagonalBackward0>(module, DiagonalBackward0Class, "DiagonalBackward0", DiagonalBackward0_properties);
  static PyTypeObject NativeDropoutBackwardBackward0Class;
  addClass<NativeDropoutBackwardBackward0>(module, NativeDropoutBackwardBackward0Class, "NativeDropoutBackwardBackward0", NativeDropoutBackwardBackward0_properties);
  static PyTypeObject EqBackward0Class;
  addClass<EqBackward0>(module, EqBackward0Class, "EqBackward0", EqBackward0_properties);
  static PyTypeObject EqBackward1Class;
  addClass<EqBackward1>(module, EqBackward1Class, "EqBackward1", EqBackward1_properties);
  static PyTypeObject ExpandBackward0Class;
  addClass<ExpandBackward0>(module, ExpandBackward0Class, "ExpandBackward0", ExpandBackward0_properties);
  static PyTypeObject FloorBackward0Class;
  addClass<FloorBackward0>(module, FloorBackward0Class, "FloorBackward0", FloorBackward0_properties);
  static PyTypeObject FrexpBackward0Class;
  addClass<FrexpBackward0>(module, FrexpBackward0Class, "FrexpBackward0", FrexpBackward0_properties);
  static PyTypeObject GatherBackward0Class;
  addClass<GatherBackward0>(module, GatherBackward0Class, "GatherBackward0", GatherBackward0_properties);
  static PyTypeObject GeBackward0Class;
  addClass<GeBackward0>(module, GeBackward0Class, "GeBackward0", GeBackward0_properties);
  static PyTypeObject GeBackward1Class;
  addClass<GeBackward1>(module, GeBackward1Class, "GeBackward1", GeBackward1_properties);
  static PyTypeObject HardsigmoidBackward0Class;
  addClass<HardsigmoidBackward0>(module, HardsigmoidBackward0Class, "HardsigmoidBackward0", HardsigmoidBackward0_properties);
  static PyTypeObject I0Backward0Class;
  addClass<I0Backward0>(module, I0Backward0Class, "I0Backward0", I0Backward0_properties);
  static PyTypeObject UnsafeIndexBackward0Class;
  addClass<UnsafeIndexBackward0>(module, UnsafeIndexBackward0Class, "UnsafeIndexBackward0", UnsafeIndexBackward0_properties);
  static PyTypeObject LinalgInvExBackward0Class;
  addClass<LinalgInvExBackward0>(module, LinalgInvExBackward0Class, "LinalgInvExBackward0", LinalgInvExBackward0_properties);
  static PyTypeObject LerpBackward0Class;
  addClass<LerpBackward0>(module, LerpBackward0Class, "LerpBackward0", LerpBackward0_properties);
  static PyTypeObject LerpBackward1Class;
  addClass<LerpBackward1>(module, LerpBackward1Class, "LerpBackward1", LerpBackward1_properties);
  static PyTypeObject SpecialXlog1PyBackward0Class;
  addClass<SpecialXlog1PyBackward0>(module, SpecialXlog1PyBackward0Class, "SpecialXlog1PyBackward0", SpecialXlog1PyBackward0_properties);
  static PyTypeObject SpecialXlog1PyBackward1Class;
  addClass<SpecialXlog1PyBackward1>(module, SpecialXlog1PyBackward1Class, "SpecialXlog1PyBackward1", SpecialXlog1PyBackward1_properties);
  static PyTypeObject SpecialXlog1PyBackward2Class;
  addClass<SpecialXlog1PyBackward2>(module, SpecialXlog1PyBackward2Class, "SpecialXlog1PyBackward2", SpecialXlog1PyBackward2_properties);
  static PyTypeObject LtBackward0Class;
  addClass<LtBackward0>(module, LtBackward0Class, "LtBackward0", LtBackward0_properties);
  static PyTypeObject LtBackward1Class;
  addClass<LtBackward1>(module, LtBackward1Class, "LtBackward1", LtBackward1_properties);
  static PyTypeObject MeanBackward0Class;
  addClass<MeanBackward0>(module, MeanBackward0Class, "MeanBackward0", MeanBackward0_properties);
  static PyTypeObject MeanBackwardAutogradNestedTensor0Class;
  addClass<MeanBackwardAutogradNestedTensor0>(module, MeanBackwardAutogradNestedTensor0Class, "MeanBackwardAutogradNestedTensor0", MeanBackwardAutogradNestedTensor0_properties);
  static PyTypeObject MeanBackward1Class;
  addClass<MeanBackward1>(module, MeanBackward1Class, "MeanBackward1", MeanBackward1_properties);
  static PyTypeObject MinBackward0Class;
  addClass<MinBackward0>(module, MinBackward0Class, "MinBackward0", MinBackward0_properties);
  static PyTypeObject MinBackward1Class;
  addClass<MinBackward1>(module, MinBackward1Class, "MinBackward1", MinBackward1_properties);
  static PyTypeObject FminBackward0Class;
  addClass<FminBackward0>(module, FminBackward0Class, "FminBackward0", FminBackward0_properties);
  static PyTypeObject MmBackward0Class;
  addClass<MmBackward0>(module, MmBackward0Class, "MmBackward0", MmBackward0_properties);
  static PyTypeObject MvBackward0Class;
  addClass<MvBackward0>(module, MvBackward0Class, "MvBackward0", MvBackward0_properties);
  static PyTypeObject NativeBatchNormLegitNoTrainingBackward0Class;
  addClass<NativeBatchNormLegitNoTrainingBackward0>(module, NativeBatchNormLegitNoTrainingBackward0Class, "NativeBatchNormLegitNoTrainingBackward0", NativeBatchNormLegitNoTrainingBackward0_properties);
  static PyTypeObject NativeGroupNormBackward0Class;
  addClass<NativeGroupNormBackward0>(module, NativeGroupNormBackward0Class, "NativeGroupNormBackward0", NativeGroupNormBackward0_properties);
  static PyTypeObject LinalgVectorNormBackward0Class;
  addClass<LinalgVectorNormBackward0>(module, LinalgVectorNormBackward0Class, "LinalgVectorNormBackward0", LinalgVectorNormBackward0_properties);
  static PyTypeObject PdistBackwardBackward0Class;
  addClass<PdistBackwardBackward0>(module, PdistBackwardBackward0Class, "PdistBackwardBackward0", PdistBackwardBackward0_properties);
  static PyTypeObject OrmqrBackward0Class;
  addClass<OrmqrBackward0>(module, OrmqrBackward0Class, "OrmqrBackward0", OrmqrBackward0_properties);
  static PyTypeObject PermuteBackward0Class;
  addClass<PermuteBackward0>(module, PermuteBackward0Class, "PermuteBackward0", PermuteBackward0_properties);
  static PyTypeObject ReshapeAliasBackward0Class;
  addClass<ReshapeAliasBackward0>(module, ReshapeAliasBackward0Class, "ReshapeAliasBackward0", ReshapeAliasBackward0_properties);
  static PyTypeObject SelectBackward0Class;
  addClass<SelectBackward0>(module, SelectBackward0Class, "SelectBackward0", SelectBackward0_properties);
  static PyTypeObject SelectBackwardAutogradNestedTensor0Class;
  addClass<SelectBackwardAutogradNestedTensor0>(module, SelectBackwardAutogradNestedTensor0Class, "SelectBackwardAutogradNestedTensor0", SelectBackwardAutogradNestedTensor0_properties);
  static PyTypeObject LogitBackward0Class;
  addClass<LogitBackward0>(module, LogitBackward0Class, "LogitBackward0", LogitBackward0_properties);
  static PyTypeObject SignBackward0Class;
  addClass<SignBackward0>(module, SignBackward0Class, "SignBackward0", SignBackward0_properties);
  static PyTypeObject SinhBackward0Class;
  addClass<SinhBackward0>(module, SinhBackward0Class, "SinhBackward0", SinhBackward0_properties);
  static PyTypeObject SliceBackwardBackward0Class;
  addClass<SliceBackwardBackward0>(module, SliceBackwardBackward0Class, "SliceBackwardBackward0", SliceBackwardBackward0_properties);
  static PyTypeObject UnsafeSplitBackward0Class;
  addClass<UnsafeSplitBackward0>(module, UnsafeSplitBackward0Class, "UnsafeSplitBackward0", UnsafeSplitBackward0_properties);
  static PyTypeObject SqrtBackward0Class;
  addClass<SqrtBackward0>(module, SqrtBackward0Class, "SqrtBackward0", SqrtBackward0_properties);
  static PyTypeObject StdBackward0Class;
  addClass<StdBackward0>(module, StdBackward0Class, "StdBackward0", StdBackward0_properties);
  static PyTypeObject StdMeanBackward0Class;
  addClass<StdMeanBackward0>(module, StdMeanBackward0Class, "StdMeanBackward0", StdMeanBackward0_properties);
  static PyTypeObject RsubBackward0Class;
  addClass<RsubBackward0>(module, RsubBackward0Class, "RsubBackward0", RsubBackward0_properties);
  static PyTypeObject RsubBackward1Class;
  addClass<RsubBackward1>(module, RsubBackward1Class, "RsubBackward1", RsubBackward1_properties);
  static PyTypeObject TBackward0Class;
  addClass<TBackward0>(module, TBackward0Class, "TBackward0", TBackward0_properties);
  static PyTypeObject FlipBackward0Class;
  addClass<FlipBackward0>(module, FlipBackward0Class, "FlipBackward0", FlipBackward0_properties);
  static PyTypeObject RollBackward0Class;
  addClass<RollBackward0>(module, RollBackward0Class, "RollBackward0", RollBackward0_properties);
  static PyTypeObject TransposeBackward0Class;
  addClass<TransposeBackward0>(module, TransposeBackward0Class, "TransposeBackward0", TransposeBackward0_properties);
  static PyTypeObject TransposeBackward1Class;
  addClass<TransposeBackward1>(module, TransposeBackward1Class, "TransposeBackward1", TransposeBackward1_properties);
  static PyTypeObject TriangularSolveBackward0Class;
  addClass<TriangularSolveBackward0>(module, TriangularSolveBackward0Class, "TriangularSolveBackward0", TriangularSolveBackward0_properties);
  static PyTypeObject LinalgSolveTriangularBackward0Class;
  addClass<LinalgSolveTriangularBackward0>(module, LinalgSolveTriangularBackward0Class, "LinalgSolveTriangularBackward0", LinalgSolveTriangularBackward0_properties);
  static PyTypeObject TriuBackward0Class;
  addClass<TriuBackward0>(module, TriuBackward0Class, "TriuBackward0", TriuBackward0_properties);
  static PyTypeObject ToSparseBackward0Class;
  addClass<ToSparseBackward0>(module, ToSparseBackward0Class, "ToSparseBackward0", ToSparseBackward0_properties);
  static PyTypeObject ToSparseBackward1Class;
  addClass<ToSparseBackward1>(module, ToSparseBackward1Class, "ToSparseBackward1", ToSparseBackward1_properties);
  static PyTypeObject Unique2Backward0Class;
  addClass<Unique2Backward0>(module, Unique2Backward0Class, "Unique2Backward0", Unique2Backward0_properties);
  static PyTypeObject ViewBackward0Class;
  addClass<ViewBackward0>(module, ViewBackward0Class, "ViewBackward0", ViewBackward0_properties);
  static PyTypeObject ViewBackwardAutogradNestedTensor0Class;
  addClass<ViewBackwardAutogradNestedTensor0>(module, ViewBackwardAutogradNestedTensor0Class, "ViewBackwardAutogradNestedTensor0", ViewBackwardAutogradNestedTensor0_properties);
  static PyTypeObject TrilinearBackward0Class;
  addClass<TrilinearBackward0>(module, TrilinearBackward0Class, "TrilinearBackward0", TrilinearBackward0_properties);
  static PyTypeObject BinaryCrossEntropyBackward0Class;
  addClass<BinaryCrossEntropyBackward0>(module, BinaryCrossEntropyBackward0Class, "BinaryCrossEntropyBackward0", BinaryCrossEntropyBackward0_properties);
  static PyTypeObject EmbeddingRenormBackward0Class;
  addClass<EmbeddingRenormBackward0>(module, EmbeddingRenormBackward0Class, "EmbeddingRenormBackward0", EmbeddingRenormBackward0_properties);
  static PyTypeObject HuberLossBackward0Class;
  addClass<HuberLossBackward0>(module, HuberLossBackward0Class, "HuberLossBackward0", HuberLossBackward0_properties);
  static PyTypeObject SiluBackward0Class;
  addClass<SiluBackward0>(module, SiluBackward0Class, "SiluBackward0", SiluBackward0_properties);
  static PyTypeObject CeluBackward0Class;
  addClass<CeluBackward0>(module, CeluBackward0Class, "CeluBackward0", CeluBackward0_properties);
  static PyTypeObject CeluBackward1Class;
  addClass<CeluBackward1>(module, CeluBackward1Class, "CeluBackward1", CeluBackward1_properties);
  static PyTypeObject SparseLogSoftmaxBackward0Class;
  addClass<SparseLogSoftmaxBackward0>(module, SparseLogSoftmaxBackward0Class, "SparseLogSoftmaxBackward0", SparseLogSoftmaxBackward0_properties);
  static PyTypeObject MaskedSoftmaxBackward0Class;
  addClass<MaskedSoftmaxBackward0>(module, MaskedSoftmaxBackward0Class, "MaskedSoftmaxBackward0", MaskedSoftmaxBackward0_properties);
  static PyTypeObject SoftmaxBackward0Class;
  addClass<SoftmaxBackward0>(module, SoftmaxBackward0Class, "SoftmaxBackward0", SoftmaxBackward0_properties);
  static PyTypeObject ThresholdBackward0Class;
  addClass<ThresholdBackward0>(module, ThresholdBackward0Class, "ThresholdBackward0", ThresholdBackward0_properties);
  static PyTypeObject ReplicationPad3DBackward0Class;
  addClass<ReplicationPad3DBackward0>(module, ReplicationPad3DBackward0Class, "ReplicationPad3DBackward0", ReplicationPad3DBackward0_properties);
  static PyTypeObject UpsampleLinear1DBackward0Class;
  addClass<UpsampleLinear1DBackward0>(module, UpsampleLinear1DBackward0Class, "UpsampleLinear1DBackward0", UpsampleLinear1DBackward0_properties);
  static PyTypeObject UpsampleBilinear2DBackward0Class;
  addClass<UpsampleBilinear2DBackward0>(module, UpsampleBilinear2DBackward0Class, "UpsampleBilinear2DBackward0", UpsampleBilinear2DBackward0_properties);
  static PyTypeObject UpsampleBicubic2DBackward0Class;
  addClass<UpsampleBicubic2DBackward0>(module, UpsampleBicubic2DBackward0Class, "UpsampleBicubic2DBackward0", UpsampleBicubic2DBackward0_properties);
  static PyTypeObject UpsampleNearest2DBackward0Class;
  addClass<UpsampleNearest2DBackward0>(module, UpsampleNearest2DBackward0Class, "UpsampleNearest2DBackward0", UpsampleNearest2DBackward0_properties);
  static PyTypeObject PixelShuffleBackward0Class;
  addClass<PixelShuffleBackward0>(module, PixelShuffleBackward0Class, "PixelShuffleBackward0", PixelShuffleBackward0_properties);
  static PyTypeObject AdaptiveAvgPool3DBackward0Class;
  addClass<AdaptiveAvgPool3DBackward0>(module, AdaptiveAvgPool3DBackward0Class, "AdaptiveAvgPool3DBackward0", AdaptiveAvgPool3DBackward0_properties);
  static PyTypeObject AdaptiveMaxPool2DBackward0Class;
  addClass<AdaptiveMaxPool2DBackward0>(module, AdaptiveMaxPool2DBackward0Class, "AdaptiveMaxPool2DBackward0", AdaptiveMaxPool2DBackward0_properties);
  static PyTypeObject AdaptiveMaxPool3DBackward0Class;
  addClass<AdaptiveMaxPool3DBackward0>(module, AdaptiveMaxPool3DBackward0Class, "AdaptiveMaxPool3DBackward0", AdaptiveMaxPool3DBackward0_properties);
  static PyTypeObject FractionalMaxPool2DBackward0Class;
  addClass<FractionalMaxPool2DBackward0>(module, FractionalMaxPool2DBackward0Class, "FractionalMaxPool2DBackward0", FractionalMaxPool2DBackward0_properties);
  static PyTypeObject LinearBackwardBackward0Class;
  addClass<LinearBackwardBackward0>(module, LinearBackwardBackward0Class, "LinearBackwardBackward0", LinearBackwardBackward0_properties);
  static PyTypeObject MaxPool2DBackward0Class;
  addClass<MaxPool2DBackward0>(module, MaxPool2DBackward0Class, "MaxPool2DBackward0", MaxPool2DBackward0_properties);
  static PyTypeObject MaxUnpool2DBackward0Class;
  addClass<MaxUnpool2DBackward0>(module, MaxUnpool2DBackward0Class, "MaxUnpool2DBackward0", MaxUnpool2DBackward0_properties);
  static PyTypeObject MaxUnpool3DBackward0Class;
  addClass<MaxUnpool3DBackward0>(module, MaxUnpool3DBackward0Class, "MaxUnpool3DBackward0", MaxUnpool3DBackward0_properties);
  static PyTypeObject SlowConvTranspose2DBackward0Class;
  addClass<SlowConvTranspose2DBackward0>(module, SlowConvTranspose2DBackward0Class, "SlowConvTranspose2DBackward0", SlowConvTranspose2DBackward0_properties);
  static PyTypeObject SlowConv2DBackwardBackward0Class;
  addClass<SlowConv2DBackwardBackward0>(module, SlowConv2DBackwardBackward0Class, "SlowConv2DBackwardBackward0", SlowConv2DBackwardBackward0_properties);
  static PyTypeObject ConvDepthwise3DBackward0Class;
  addClass<ConvDepthwise3DBackward0>(module, ConvDepthwise3DBackward0Class, "ConvDepthwise3DBackward0", ConvDepthwise3DBackward0_properties);
  static PyTypeObject SlowConvDilated2DBackward0Class;
  addClass<SlowConvDilated2DBackward0>(module, SlowConvDilated2DBackward0Class, "SlowConvDilated2DBackward0", SlowConvDilated2DBackward0_properties);
  static PyTypeObject Col2ImBackward0Class;
  addClass<Col2ImBackward0>(module, Col2ImBackward0Class, "Col2ImBackward0", Col2ImBackward0_properties);
  static PyTypeObject AvgPool2DBackwardBackward0Class;
  addClass<AvgPool2DBackwardBackward0>(module, AvgPool2DBackwardBackward0Class, "AvgPool2DBackwardBackward0", AvgPool2DBackwardBackward0_properties);
  static PyTypeObject NllLossBackwardBackward0Class;
  addClass<NllLossBackwardBackward0>(module, NllLossBackwardBackward0Class, "NllLossBackwardBackward0", NllLossBackwardBackward0_properties);
  static PyTypeObject ReflectionPad3DBackwardBackward0Class;
  addClass<ReflectionPad3DBackwardBackward0>(module, ReflectionPad3DBackwardBackward0Class, "ReflectionPad3DBackwardBackward0", ReflectionPad3DBackwardBackward0_properties);
  static PyTypeObject ReplicationPad2DBackwardBackward0Class;
  addClass<ReplicationPad2DBackwardBackward0>(module, ReplicationPad2DBackwardBackward0Class, "ReplicationPad2DBackwardBackward0", ReplicationPad2DBackwardBackward0_properties);
  static PyTypeObject SparseMmReduceImplBackward0Class;
  addClass<SparseMmReduceImplBackward0>(module, SparseMmReduceImplBackward0Class, "SparseMmReduceImplBackward0", SparseMmReduceImplBackward0_properties);
  static PyTypeObject SmoothL1LossBackwardBackward0Class;
  addClass<SmoothL1LossBackwardBackward0>(module, SmoothL1LossBackwardBackward0Class, "SmoothL1LossBackwardBackward0", SmoothL1LossBackwardBackward0_properties);
  static PyTypeObject HuberLossBackwardBackward0Class;
  addClass<HuberLossBackwardBackward0>(module, HuberLossBackwardBackward0Class, "HuberLossBackwardBackward0", HuberLossBackwardBackward0_properties);
  static PyTypeObject UpsampleBicubic2DBackwardBackward0Class;
  addClass<UpsampleBicubic2DBackwardBackward0>(module, UpsampleBicubic2DBackwardBackward0Class, "UpsampleBicubic2DBackwardBackward0", UpsampleBicubic2DBackwardBackward0_properties);
  static PyTypeObject UpsampleTrilinear3DBackwardBackward0Class;
  addClass<UpsampleTrilinear3DBackwardBackward0>(module, UpsampleTrilinear3DBackwardBackward0Class, "UpsampleTrilinear3DBackwardBackward0", UpsampleTrilinear3DBackwardBackward0_properties);
  static PyTypeObject UpsampleNearest3DBackwardBackward0Class;
  addClass<UpsampleNearest3DBackwardBackward0>(module, UpsampleNearest3DBackwardBackward0Class, "UpsampleNearest3DBackwardBackward0", UpsampleNearest3DBackwardBackward0_properties);
  static PyTypeObject UpsampleNearestExact3DBackwardBackward0Class;
  addClass<UpsampleNearestExact3DBackwardBackward0>(module, UpsampleNearestExact3DBackwardBackward0Class, "UpsampleNearestExact3DBackwardBackward0", UpsampleNearestExact3DBackwardBackward0_properties);
  static PyTypeObject CudnnConvolutionBackward0Class;
  addClass<CudnnConvolutionBackward0>(module, CudnnConvolutionBackward0Class, "CudnnConvolutionBackward0", CudnnConvolutionBackward0_properties);
  static PyTypeObject CudnnRnnBackwardBackward0Class;
  addClass<CudnnRnnBackwardBackward0>(module, CudnnRnnBackwardBackward0Class, "CudnnRnnBackwardBackward0", CudnnRnnBackwardBackward0_properties);
  static PyTypeObject NestedFromPaddedBackward0Class;
  addClass<NestedFromPaddedBackward0>(module, NestedFromPaddedBackward0Class, "NestedFromPaddedBackward0", NestedFromPaddedBackward0_properties);
  static PyTypeObject NestedViewFromBufferBackward0Class;
  addClass<NestedViewFromBufferBackward0>(module, NestedViewFromBufferBackward0Class, "NestedViewFromBufferBackward0", NestedViewFromBufferBackward0_properties);
  static PyTypeObject NestedViewFromJaggedBackward0Class;
  addClass<NestedViewFromJaggedBackward0>(module, NestedViewFromJaggedBackward0Class, "NestedViewFromJaggedBackward0", NestedViewFromJaggedBackward0_properties);
  static PyTypeObject SafeSoftmaxBackward0Class;
  addClass<SafeSoftmaxBackward0>(module, SafeSoftmaxBackward0Class, "SafeSoftmaxBackward0", SafeSoftmaxBackward0_properties);
  static PyTypeObject ScaledDotProductEfficientAttentionBackward0Class;
  addClass<ScaledDotProductEfficientAttentionBackward0>(module, ScaledDotProductEfficientAttentionBackward0Class, "ScaledDotProductEfficientAttentionBackward0", ScaledDotProductEfficientAttentionBackward0_properties);
  static PyTypeObject EfficientAttentionBackward0Class;
  addClass<EfficientAttentionBackward0>(module, EfficientAttentionBackward0Class, "EfficientAttentionBackward0", EfficientAttentionBackward0_properties);
  static PyTypeObject FftR2CBackward0Class;
  addClass<FftR2CBackward0>(module, FftR2CBackward0Class, "FftR2CBackward0", FftR2CBackward0_properties);
  static PyTypeObject ThnnFusedGruCellBackward0Class;
  addClass<ThnnFusedGruCellBackward0>(module, ThnnFusedGruCellBackward0Class, "ThnnFusedGruCellBackward0", ThnnFusedGruCellBackward0_properties);
  static PyTypeObject PackPaddedSequenceBackward0Class;
  addClass<PackPaddedSequenceBackward0>(module, PackPaddedSequenceBackward0Class, "PackPaddedSequenceBackward0", PackPaddedSequenceBackward0_properties);
  static PyTypeObject ForeachDivBackward0Class;
  addClass<ForeachDivBackward0>(module, ForeachDivBackward0Class, "ForeachDivBackward0", ForeachDivBackward0_properties);
  static PyTypeObject SelectBackward0_copyClass;
  addClass<SelectBackward0_copy>(module, SelectBackward0_copyClass, "SelectBackward0_copy", SelectBackward0_copy_properties);
  static PyTypeObject SelectBackwardAutogradNestedTensor0_copyClass;
  addClass<SelectBackwardAutogradNestedTensor0_copy>(module, SelectBackwardAutogradNestedTensor0_copyClass, "SelectBackwardAutogradNestedTensor0_copy", SelectBackwardAutogradNestedTensor0_copy_properties);
  static PyTypeObject UnfoldBackward0_copyClass;
  addClass<UnfoldBackward0_copy>(module, UnfoldBackward0_copyClass, "UnfoldBackward0_copy", UnfoldBackward0_copy_properties);
  static PyTypeObject NestedViewFromJaggedBackward0_copyClass;
  addClass<NestedViewFromJaggedBackward0_copy>(module, NestedViewFromJaggedBackward0_copyClass, "NestedViewFromJaggedBackward0_copy", NestedViewFromJaggedBackward0_copy_properties);
}

} // namespace torch::autograd::generated
