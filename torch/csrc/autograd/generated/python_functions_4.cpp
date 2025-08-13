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

PyObject* THPAcosBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AcosBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAcosBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AcosBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AcosBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPAcosBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPAcosBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAsinBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AsinBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAsinBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AsinBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AsinBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPAsinBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPAsinBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMatmulBackward0_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MatmulBackward0*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMatmulBackward0_other_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MatmulBackward0*>(self->cdata.get())->other_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMatmulBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MatmulBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMatmulBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MatmulBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MatmulBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPMatmulBackward0_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_other", (getter)THPMatmulBackward0_other_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPMatmulBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPMatmulBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCopysignBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CopysignBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCopysignBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CopysignBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCopysignBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CopysignBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCopysignBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CopysignBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CopysignBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPCopysignBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPCopysignBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPCopysignBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPCopysignBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCopysignBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CopysignBackward1*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCopysignBackward1_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CopysignBackward1*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCopysignBackward1_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CopysignBackward1*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCopysignBackward1_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CopysignBackward1*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CopysignBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPCopysignBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPCopysignBackward1_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPCopysignBackward1_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPCopysignBackward1_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef Deg2RadBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPLinalgSlogdetBackward0_A_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgSlogdetBackward0*>(self->cdata.get())->A_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgSlogdetBackward0_A_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgSlogdetBackward0*>(self->cdata.get())->A_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgSlogdetBackward0_LU_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgSlogdetBackward0*>(self->cdata.get())->LU_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgSlogdetBackward0_LU_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgSlogdetBackward0*>(self->cdata.get())->LU_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgSlogdetBackward0_pivots_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgSlogdetBackward0*>(self->cdata.get())->pivots_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgSlogdetBackward0_pivots_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgSlogdetBackward0*>(self->cdata.get())->pivots_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgSlogdetBackward0_sign_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgSlogdetBackward0*>(self->cdata.get())->sign_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgSlogdetBackward0_sign_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgSlogdetBackward0*>(self->cdata.get())->sign_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LinalgSlogdetBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_A", (getter)THPLinalgSlogdetBackward0_A_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_A", (getter)THPLinalgSlogdetBackward0_A_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_LU", (getter)THPLinalgSlogdetBackward0_LU_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_LU", (getter)THPLinalgSlogdetBackward0_LU_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_pivots", (getter)THPLinalgSlogdetBackward0_pivots_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_pivots", (getter)THPLinalgSlogdetBackward0_pivots_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_sign", (getter)THPLinalgSlogdetBackward0_sign_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_sign", (getter)THPLinalgSlogdetBackward0_sign_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPDistBackward0_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<DistBackward0*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPDistBackward0_other_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<DistBackward0*>(self->cdata.get())->other_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPDistBackward0_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<DistBackward0*>(self->cdata.get())->p;
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

PyObject* THPDistBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<DistBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPDistBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<DistBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPDistBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<DistBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPDistBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<DistBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef DistBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPDistBackward0_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_other", (getter)THPDistBackward0_other_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_p", (getter)THPDistBackward0_p_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPDistBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPDistBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPDistBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPDistBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPErfcBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ErfcBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPErfcBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ErfcBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ErfcBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPErfcBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPErfcBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSpecialErfcxBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialErfcxBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialErfcxBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialErfcxBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialErfcxBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialErfcxBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialErfcxBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialErfcxBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SpecialErfcxBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSpecialErfcxBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSpecialErfcxBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPSpecialErfcxBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPSpecialErfcxBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPErfinvBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ErfinvBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPErfinvBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ErfinvBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ErfinvBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPErfinvBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPErfinvBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef FmodBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPFmodBackward1_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FmodBackward1*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFmodBackward1_other_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FmodBackward1*>(self->cdata.get())->other_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPFmodBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FmodBackward1*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFmodBackward1_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FmodBackward1*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FmodBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPFmodBackward1_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_other", (getter)THPFmodBackward1_other_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPFmodBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPFmodBackward1_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef GeqrfBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPGridSampler2DBackward0_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<GridSampler2DBackward0*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPGridSampler2DBackward0_grid_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GridSampler2DBackward0*>(self->cdata.get())->grid_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPGridSampler2DBackward0_grid_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GridSampler2DBackward0*>(self->cdata.get())->grid_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPGridSampler2DBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GridSampler2DBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPGridSampler2DBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GridSampler2DBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPGridSampler2DBackward0_interpolation_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<GridSampler2DBackward0*>(self->cdata.get())->interpolation_mode;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPGridSampler2DBackward0_padding_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<GridSampler2DBackward0*>(self->cdata.get())->padding_mode;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef GridSampler2DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_align_corners", (getter)THPGridSampler2DBackward0_align_corners_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grid", (getter)THPGridSampler2DBackward0_grid_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_grid", (getter)THPGridSampler2DBackward0_grid_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPGridSampler2DBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPGridSampler2DBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_interpolation_mode", (getter)THPGridSampler2DBackward0_interpolation_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding_mode", (getter)THPGridSampler2DBackward0_padding_mode_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef GtBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef GtBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPSpecialI0EBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialI0EBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialI0EBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialI0EBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialI0EBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialI0EBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialI0EBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialI0EBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SpecialI0EBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSpecialI0EBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSpecialI0EBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPSpecialI0EBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPSpecialI0EBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSpecialI1Backward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialI1Backward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialI1Backward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialI1Backward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialI1Backward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialI1Backward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialI1Backward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialI1Backward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SpecialI1Backward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSpecialI1Backward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSpecialI1Backward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPSpecialI1Backward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPSpecialI1Backward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUnsafeMaskedIndexPutAccumulateBackward0_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<UnsafeMaskedIndexPutAccumulateBackward0*>(self->cdata.get());
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

PyObject* THPUnsafeMaskedIndexPutAccumulateBackward0_indices_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<UnsafeMaskedIndexPutAccumulateBackward0*>(self->cdata.get());
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

PyObject* THPUnsafeMaskedIndexPutAccumulateBackward0_mask_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<UnsafeMaskedIndexPutAccumulateBackward0*>(self->cdata.get())->mask_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPUnsafeMaskedIndexPutAccumulateBackward0_mask_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<UnsafeMaskedIndexPutAccumulateBackward0*>(self->cdata.get())->mask_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UnsafeMaskedIndexPutAccumulateBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_indices", (getter)THPUnsafeMaskedIndexPutAccumulateBackward0_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_indices", (getter)THPUnsafeMaskedIndexPutAccumulateBackward0_indices_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mask", (getter)THPUnsafeMaskedIndexPutAccumulateBackward0_mask_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_mask", (getter)THPUnsafeMaskedIndexPutAccumulateBackward0_mask_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPIndexAddBackward0_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<IndexAddBackward0*>(self->cdata.get())->alpha;
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

PyObject* THPIndexAddBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<IndexAddBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexAddBackward0_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IndexAddBackward0*>(self->cdata.get())->index_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexAddBackward0_index_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IndexAddBackward0*>(self->cdata.get())->index_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexAddBackward0_source_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IndexAddBackward0*>(self->cdata.get())->source_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexAddBackward0_source_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IndexAddBackward0*>(self->cdata.get())->source_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexAddBackward0_source_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<IndexAddBackward0*>(self->cdata.get())->source_dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef IndexAddBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_alpha", (getter)THPIndexAddBackward0_alpha_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPIndexAddBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_index", (getter)THPIndexAddBackward0_index_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_index", (getter)THPIndexAddBackward0_index_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_source", (getter)THPIndexAddBackward0_source_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_source", (getter)THPIndexAddBackward0_source_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_source_dim", (getter)THPIndexAddBackward0_source_dim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPIndexPutBackward0_accumulate_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<IndexPutBackward0*>(self->cdata.get())->accumulate;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexPutBackward0_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<IndexPutBackward0*>(self->cdata.get());
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

PyObject* THPIndexPutBackward0_indices_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<IndexPutBackward0*>(self->cdata.get());
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

static struct PyGetSetDef IndexPutBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_accumulate", (getter)THPIndexPutBackward0_accumulate_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_indices", (getter)THPIndexPutBackward0_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_indices", (getter)THPIndexPutBackward0_indices_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef LogNormalBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPLinalgLuFactorBackward0_pivot_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LinalgLuFactorBackward0*>(self->cdata.get())->pivot;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgLuFactorBackward0_LU_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgLuFactorBackward0*>(self->cdata.get())->LU_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgLuFactorBackward0_LU_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgLuFactorBackward0*>(self->cdata.get())->LU_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgLuFactorBackward0_pivots_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgLuFactorBackward0*>(self->cdata.get())->pivots_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgLuFactorBackward0_pivots_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgLuFactorBackward0*>(self->cdata.get())->pivots_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LinalgLuFactorBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_pivot", (getter)THPLinalgLuFactorBackward0_pivot_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_LU", (getter)THPLinalgLuFactorBackward0_LU_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_LU", (getter)THPLinalgLuFactorBackward0_LU_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_pivots", (getter)THPLinalgLuFactorBackward0_pivots_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_pivots", (getter)THPLinalgLuFactorBackward0_pivots_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLinalgLuSolveBackward0_LU_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgLuSolveBackward0*>(self->cdata.get())->LU_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgLuSolveBackward0_LU_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgLuSolveBackward0*>(self->cdata.get())->LU_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgLuSolveBackward0_adjoint_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LinalgLuSolveBackward0*>(self->cdata.get())->adjoint;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgLuSolveBackward0_left_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LinalgLuSolveBackward0*>(self->cdata.get())->left;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgLuSolveBackward0_pivots_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgLuSolveBackward0*>(self->cdata.get())->pivots_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgLuSolveBackward0_pivots_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgLuSolveBackward0*>(self->cdata.get())->pivots_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgLuSolveBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgLuSolveBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgLuSolveBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgLuSolveBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LinalgLuSolveBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_LU", (getter)THPLinalgLuSolveBackward0_LU_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_LU", (getter)THPLinalgLuSolveBackward0_LU_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_adjoint", (getter)THPLinalgLuSolveBackward0_adjoint_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_left", (getter)THPLinalgLuSolveBackward0_left_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_pivots", (getter)THPLinalgLuSolveBackward0_pivots_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_pivots", (getter)THPLinalgLuSolveBackward0_pivots_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPLinalgLuSolveBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPLinalgLuSolveBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMaskedFillBackward0_mask_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaskedFillBackward0*>(self->cdata.get())->mask_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaskedFillBackward0_mask_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaskedFillBackward0*>(self->cdata.get())->mask_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MaskedFillBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_mask", (getter)THPMaskedFillBackward0_mask_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_mask", (getter)THPMaskedFillBackward0_mask_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMaskedFillBackward1_mask_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaskedFillBackward1*>(self->cdata.get())->mask_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaskedFillBackward1_mask_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaskedFillBackward1*>(self->cdata.get())->mask_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MaskedFillBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_mask", (getter)THPMaskedFillBackward1_mask_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_mask", (getter)THPMaskedFillBackward1_mask_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMaskedScatterBackward0_mask_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaskedScatterBackward0*>(self->cdata.get())->mask_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaskedScatterBackward0_mask_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaskedScatterBackward0*>(self->cdata.get())->mask_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaskedScatterBackward0_source_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaskedScatterBackward0*>(self->cdata.get())->source_sym_sizes;
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

static struct PyGetSetDef MaskedScatterBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_mask", (getter)THPMaskedScatterBackward0_mask_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_mask", (getter)THPMaskedScatterBackward0_mask_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_source_sym_sizes", (getter)THPMaskedScatterBackward0_source_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMedianBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MedianBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMedianBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MedianBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMedianBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MedianBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMedianBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MedianBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MedianBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMedianBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPMedianBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPMedianBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPMedianBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNanmedianBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NanmedianBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNanmedianBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NanmedianBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPNanmedianBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NanmedianBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNanmedianBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NanmedianBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NanmedianBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPNanmedianBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPNanmedianBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPNanmedianBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPNanmedianBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMedianBackward1_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MedianBackward1*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMedianBackward1_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MedianBackward1*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMedianBackward1_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MedianBackward1*>(self->cdata.get())->self_sym_sizes;
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

PyObject* THPMedianBackward1_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MedianBackward1*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMedianBackward1_indices_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MedianBackward1*>(self->cdata.get())->indices_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MedianBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPMedianBackward1_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPMedianBackward1_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPMedianBackward1_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_indices", (getter)THPMedianBackward1_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_indices", (getter)THPMedianBackward1_indices_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNanmedianBackward1_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NanmedianBackward1*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNanmedianBackward1_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NanmedianBackward1*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNanmedianBackward1_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NanmedianBackward1*>(self->cdata.get())->self_sym_sizes;
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

PyObject* THPNanmedianBackward1_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NanmedianBackward1*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNanmedianBackward1_indices_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NanmedianBackward1*>(self->cdata.get())->indices_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NanmedianBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPNanmedianBackward1_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPNanmedianBackward1_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPNanmedianBackward1_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_indices", (getter)THPNanmedianBackward1_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_indices", (getter)THPNanmedianBackward1_indices_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPBatchNormNoUpdateBackward0_eps_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<BatchNormNoUpdateBackward0*>(self->cdata.get())->eps;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormNoUpdateBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormNoUpdateBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormNoUpdateBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormNoUpdateBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormNoUpdateBackward0_running_mean_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormNoUpdateBackward0*>(self->cdata.get())->running_mean_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormNoUpdateBackward0_running_mean_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormNoUpdateBackward0*>(self->cdata.get())->running_mean_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormNoUpdateBackward0_running_var_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormNoUpdateBackward0*>(self->cdata.get())->running_var_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormNoUpdateBackward0_running_var_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormNoUpdateBackward0*>(self->cdata.get())->running_var_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormNoUpdateBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormNoUpdateBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormNoUpdateBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormNoUpdateBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormNoUpdateBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormNoUpdateBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormNoUpdateBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormNoUpdateBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormNoUpdateBackward0_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormNoUpdateBackward0*>(self->cdata.get())->result2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormNoUpdateBackward0_result2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormNoUpdateBackward0*>(self->cdata.get())->result2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormNoUpdateBackward0_result3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormNoUpdateBackward0*>(self->cdata.get())->result3_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBatchNormNoUpdateBackward0_result3_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BatchNormNoUpdateBackward0*>(self->cdata.get())->result3_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef BatchNormNoUpdateBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_eps", (getter)THPBatchNormNoUpdateBackward0_eps_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPBatchNormNoUpdateBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPBatchNormNoUpdateBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_mean", (getter)THPBatchNormNoUpdateBackward0_running_mean_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_running_mean", (getter)THPBatchNormNoUpdateBackward0_running_mean_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_var", (getter)THPBatchNormNoUpdateBackward0_running_var_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_running_var", (getter)THPBatchNormNoUpdateBackward0_running_var_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPBatchNormNoUpdateBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPBatchNormNoUpdateBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPBatchNormNoUpdateBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPBatchNormNoUpdateBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPBatchNormNoUpdateBackward0_result2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result2", (getter)THPBatchNormNoUpdateBackward0_result2_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result3", (getter)THPBatchNormNoUpdateBackward0_result3_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result3", (getter)THPBatchNormNoUpdateBackward0_result3_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef PoissonBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPProdBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ProdBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPProdBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ProdBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPProdBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ProdBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPProdBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ProdBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ProdBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPProdBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPProdBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPProdBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPProdBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPProdBackward1_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ProdBackward1*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPProdBackward1_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ProdBackward1*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPProdBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ProdBackward1*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPProdBackward1_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ProdBackward1*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPProdBackward1_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ProdBackward1*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPProdBackward1_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ProdBackward1*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ProdBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPProdBackward1_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPProdBackward1_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPProdBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPProdBackward1_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPProdBackward1_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPProdBackward1_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLinalgQrBackward0_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LinalgQrBackward0*>(self->cdata.get())->mode;
  return PyUnicode_FromStringAndSize(prop.data(), prop.size());
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgQrBackward0_Q_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgQrBackward0*>(self->cdata.get())->Q_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgQrBackward0_Q_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgQrBackward0*>(self->cdata.get())->Q_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgQrBackward0_R_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgQrBackward0*>(self->cdata.get())->R_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgQrBackward0_R_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgQrBackward0*>(self->cdata.get())->R_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LinalgQrBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_mode", (getter)THPLinalgQrBackward0_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_Q", (getter)THPLinalgQrBackward0_Q_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_Q", (getter)THPLinalgQrBackward0_Q_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_R", (getter)THPLinalgQrBackward0_R_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_R", (getter)THPLinalgQrBackward0_R_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef RemainderBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPRemainderBackward1_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RemainderBackward1*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPRemainderBackward1_other_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RemainderBackward1*>(self->cdata.get())->other_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPRemainderBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RemainderBackward1*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPRemainderBackward1_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RemainderBackward1*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef RemainderBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPRemainderBackward1_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_other", (getter)THPRemainderBackward1_other_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPRemainderBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPRemainderBackward1_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPRenormBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RenormBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPRenormBackward0_maxnorm_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RenormBackward0*>(self->cdata.get())->maxnorm;
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

PyObject* THPRenormBackward0_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RenormBackward0*>(self->cdata.get())->p;
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

PyObject* THPRenormBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RenormBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPRenormBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RenormBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef RenormBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPRenormBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_maxnorm", (getter)THPRenormBackward0_maxnorm_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_p", (getter)THPRenormBackward0_p_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPRenormBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPRenormBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSincBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SincBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSincBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SincBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SincBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSincBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSincBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSliceBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SliceBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSliceBackward0_end_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<SliceBackward0*>(self->cdata.get())->end;
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

PyObject* THPSliceBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SliceBackward0*>(self->cdata.get())->self_sym_sizes;
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

PyObject* THPSliceBackward0_start_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<SliceBackward0*>(self->cdata.get())->start;
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

PyObject* THPSliceBackward0_step_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SliceBackward0*>(self->cdata.get())->step;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SliceBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSliceBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_end", (getter)THPSliceBackward0_end_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPSliceBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_start", (getter)THPSliceBackward0_start_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_step", (getter)THPSliceBackward0_step_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSliceInverseBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SliceInverseBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSliceInverseBackward0_end_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<SliceInverseBackward0*>(self->cdata.get())->end;
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

PyObject* THPSliceInverseBackward0_start_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<SliceInverseBackward0*>(self->cdata.get())->start;
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

PyObject* THPSliceInverseBackward0_step_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SliceInverseBackward0*>(self->cdata.get())->step;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SliceInverseBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSliceInverseBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_end", (getter)THPSliceInverseBackward0_end_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_start", (getter)THPSliceInverseBackward0_start_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_step", (getter)THPSliceInverseBackward0_step_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSelectScatterBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SelectScatterBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSelectScatterBackward0_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SelectScatterBackward0*>(self->cdata.get())->index;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SelectScatterBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSelectScatterBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_index", (getter)THPSelectScatterBackward0_index_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLinalgSolveExBackward0_A_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgSolveExBackward0*>(self->cdata.get())->A_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgSolveExBackward0_A_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgSolveExBackward0*>(self->cdata.get())->A_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgSolveExBackward0_left_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LinalgSolveExBackward0*>(self->cdata.get())->left;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgSolveExBackward0_LU_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgSolveExBackward0*>(self->cdata.get())->LU_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgSolveExBackward0_LU_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgSolveExBackward0*>(self->cdata.get())->LU_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgSolveExBackward0_pivots_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgSolveExBackward0*>(self->cdata.get())->pivots_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgSolveExBackward0_pivots_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgSolveExBackward0*>(self->cdata.get())->pivots_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgSolveExBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgSolveExBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgSolveExBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgSolveExBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LinalgSolveExBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_A", (getter)THPLinalgSolveExBackward0_A_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_A", (getter)THPLinalgSolveExBackward0_A_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_left", (getter)THPLinalgSolveExBackward0_left_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_LU", (getter)THPLinalgSolveExBackward0_LU_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_LU", (getter)THPLinalgSolveExBackward0_LU_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_pivots", (getter)THPLinalgSolveExBackward0_pivots_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_pivots", (getter)THPLinalgSolveExBackward0_pivots_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPLinalgSolveExBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPLinalgSolveExBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUnsafeSplitWithSizesBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnsafeSplitWithSizesBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUnsafeSplitWithSizesBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnsafeSplitWithSizesBackward0*>(self->cdata.get())->self_sym_sizes;
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

PyObject* THPUnsafeSplitWithSizesBackward0_split_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnsafeSplitWithSizesBackward0*>(self->cdata.get())->split_sizes;
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

static struct PyGetSetDef UnsafeSplitWithSizesBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPUnsafeSplitWithSizesBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPUnsafeSplitWithSizesBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_split_sizes", (getter)THPUnsafeSplitWithSizesBackward0_split_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSubBackward0_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SubBackward0*>(self->cdata.get())->alpha;
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

static struct PyGetSetDef SubBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_alpha", (getter)THPSubBackward0_alpha_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef SubBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPLinalgSvdBackward0_full_matrices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LinalgSvdBackward0*>(self->cdata.get())->full_matrices;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgSvdBackward0_S_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgSvdBackward0*>(self->cdata.get())->S_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgSvdBackward0_S_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgSvdBackward0*>(self->cdata.get())->S_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgSvdBackward0_S_sym_argsize_minus_1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LinalgSvdBackward0*>(self->cdata.get())->S_sym_argsize_minus_1;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgSvdBackward0_U_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgSvdBackward0*>(self->cdata.get())->U_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgSvdBackward0_U_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgSvdBackward0*>(self->cdata.get())->U_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgSvdBackward0_Vh_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgSvdBackward0*>(self->cdata.get())->Vh_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgSvdBackward0_Vh_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgSvdBackward0*>(self->cdata.get())->Vh_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LinalgSvdBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_full_matrices", (getter)THPLinalgSvdBackward0_full_matrices_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_S", (getter)THPLinalgSvdBackward0_S_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_S", (getter)THPLinalgSvdBackward0_S_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_S_sym_argsize_minus_1", (getter)THPLinalgSvdBackward0_S_sym_argsize_minus_1_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_U", (getter)THPLinalgSvdBackward0_U_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_U", (getter)THPLinalgSvdBackward0_U_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_Vh", (getter)THPLinalgSvdBackward0_Vh_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_Vh", (getter)THPLinalgSvdBackward0_Vh_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPTakeBackward0_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TakeBackward0*>(self->cdata.get())->index_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPTakeBackward0_index_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TakeBackward0*>(self->cdata.get())->index_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPTakeBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TakeBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPTakeBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TakeBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef TakeBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_index", (getter)THPTakeBackward0_index_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_index", (getter)THPTakeBackward0_index_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPTakeBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPTakeBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPTanBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TanBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPTanBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TanBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef TanBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_result", (getter)THPTanBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPTanBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPTraceBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TraceBackward0*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef TraceBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sym_sizes", (getter)THPTraceBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPToSparseCsrBackward0_self_self_sym_blocksize_opt_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<ToSparseCsrBackward0*>(self->cdata.get())->self_self_sym_blocksize_opt;
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

static struct PyGetSetDef ToSparseCsrBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_self_sym_blocksize_opt", (getter)THPToSparseCsrBackward0_self_self_sym_blocksize_opt_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPToSparseBsrBackward0_self_self_sym_blocksize_opt_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<ToSparseBsrBackward0*>(self->cdata.get())->self_self_sym_blocksize_opt;
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

static struct PyGetSetDef ToSparseBsrBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_self_sym_blocksize_opt", (getter)THPToSparseBsrBackward0_self_self_sym_blocksize_opt_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUnfoldBackward0_dimension_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnfoldBackward0*>(self->cdata.get())->dimension;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUnfoldBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnfoldBackward0*>(self->cdata.get())->self_sym_sizes;
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

PyObject* THPUnfoldBackward0_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnfoldBackward0*>(self->cdata.get())->size;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUnfoldBackward0_step_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnfoldBackward0*>(self->cdata.get())->step;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UnfoldBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dimension", (getter)THPUnfoldBackward0_dimension_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPUnfoldBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_size", (getter)THPUnfoldBackward0_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_step", (getter)THPUnfoldBackward0_step_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef UniqueDimConsecutiveBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPUnsafeViewBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnsafeViewBackward0*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef UnsafeViewBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sym_sizes", (getter)THPUnsafeViewBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUnsqueezeBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnsqueezeBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UnsqueezeBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPUnsqueezeBackward0_dim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUnsqueezeBackward1_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnsqueezeBackward1*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UnsqueezeBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPUnsqueezeBackward1_dim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef ViewAsRealBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef ViewAsComplexBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPSparseCompressedTensorBackward0_values_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseCompressedTensorBackward0*>(self->cdata.get())->values_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseCompressedTensorBackward0_values_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseCompressedTensorBackward0*>(self->cdata.get())->values_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseCompressedTensorBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseCompressedTensorBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseCompressedTensorBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseCompressedTensorBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SparseCompressedTensorBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_values", (getter)THPSparseCompressedTensorBackward0_values_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_values", (getter)THPSparseCompressedTensorBackward0_values_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPSparseCompressedTensorBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPSparseCompressedTensorBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPConstantPadNdBackward0_pad_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConstantPadNdBackward0*>(self->cdata.get())->pad;
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

static struct PyGetSetDef ConstantPadNdBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_pad", (getter)THPConstantPadNdBackward0_pad_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPEmbeddingBackward0_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EmbeddingBackward0*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBackward0_indices_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EmbeddingBackward0*>(self->cdata.get())->indices_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBackward0_padding_idx_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EmbeddingBackward0*>(self->cdata.get())->padding_idx;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBackward0_scale_grad_by_freq_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EmbeddingBackward0*>(self->cdata.get())->scale_grad_by_freq;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBackward0_sparse_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EmbeddingBackward0*>(self->cdata.get())->sparse;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBackward0_weight_sym_argsize_0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EmbeddingBackward0*>(self->cdata.get())->weight_sym_argsize_0;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef EmbeddingBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_indices", (getter)THPEmbeddingBackward0_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_indices", (getter)THPEmbeddingBackward0_indices_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding_idx", (getter)THPEmbeddingBackward0_padding_idx_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale_grad_by_freq", (getter)THPEmbeddingBackward0_scale_grad_by_freq_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_sparse", (getter)THPEmbeddingBackward0_sparse_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight_sym_argsize_0", (getter)THPEmbeddingBackward0_weight_sym_argsize_0_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef EmbeddingBagBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef EmbeddingBagDenseBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPSmoothL1LossBackward0_beta_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SmoothL1LossBackward0*>(self->cdata.get())->beta;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSmoothL1LossBackward0_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SmoothL1LossBackward0*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSmoothL1LossBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SmoothL1LossBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSmoothL1LossBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SmoothL1LossBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSmoothL1LossBackward0_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SmoothL1LossBackward0*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSmoothL1LossBackward0_target_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SmoothL1LossBackward0*>(self->cdata.get())->target_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SmoothL1LossBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_beta", (getter)THPSmoothL1LossBackward0_beta_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPSmoothL1LossBackward0_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSmoothL1LossBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSmoothL1LossBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPSmoothL1LossBackward0_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_target", (getter)THPSmoothL1LossBackward0_target_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPReluBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ReluBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPReluBackward0_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ReluBackward0*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ReluBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_result", (getter)THPReluBackward0_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPReluBackward0_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPEluBackward0_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EluBackward0*>(self->cdata.get())->alpha;
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

PyObject* THPEluBackward0_input_scale_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EluBackward0*>(self->cdata.get())->input_scale;
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

PyObject* THPEluBackward0_scale_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EluBackward0*>(self->cdata.get())->scale;
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

PyObject* THPEluBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EluBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEluBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EluBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef EluBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_alpha", (getter)THPEluBackward0_alpha_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input_scale", (getter)THPEluBackward0_input_scale_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale", (getter)THPEluBackward0_scale_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPEluBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPEluBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPEluBackward1_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EluBackward1*>(self->cdata.get())->alpha;
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

PyObject* THPEluBackward1_input_scale_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EluBackward1*>(self->cdata.get())->input_scale;
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

PyObject* THPEluBackward1_scale_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EluBackward1*>(self->cdata.get())->scale;
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

PyObject* THPEluBackward1_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EluBackward1*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEluBackward1_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EluBackward1*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef EluBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_alpha", (getter)THPEluBackward1_alpha_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input_scale", (getter)THPEluBackward1_input_scale_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale", (getter)THPEluBackward1_scale_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPEluBackward1_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPEluBackward1_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPHardshrinkBackwardBackward0_lambd_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<HardshrinkBackwardBackward0*>(self->cdata.get())->lambd;
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

PyObject* THPHardshrinkBackwardBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HardshrinkBackwardBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPHardshrinkBackwardBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HardshrinkBackwardBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef HardshrinkBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_lambd", (getter)THPHardshrinkBackwardBackward0_lambd_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPHardshrinkBackwardBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPHardshrinkBackwardBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLeakyReluBackward1_negative_slope_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LeakyReluBackward1*>(self->cdata.get())->negative_slope;
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

PyObject* THPLeakyReluBackward1_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LeakyReluBackward1*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLeakyReluBackward1_result_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LeakyReluBackward1*>(self->cdata.get())->result_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LeakyReluBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_negative_slope", (getter)THPLeakyReluBackward1_negative_slope_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPLeakyReluBackward1_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result", (getter)THPLeakyReluBackward1_result_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPRreluWithNoiseFunctionalBackward0_lower_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RreluWithNoiseFunctionalBackward0*>(self->cdata.get())->lower;
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

PyObject* THPRreluWithNoiseFunctionalBackward0_noise_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RreluWithNoiseFunctionalBackward0*>(self->cdata.get())->noise_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPRreluWithNoiseFunctionalBackward0_noise_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RreluWithNoiseFunctionalBackward0*>(self->cdata.get())->noise_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPRreluWithNoiseFunctionalBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RreluWithNoiseFunctionalBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPRreluWithNoiseFunctionalBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RreluWithNoiseFunctionalBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPRreluWithNoiseFunctionalBackward0_training_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RreluWithNoiseFunctionalBackward0*>(self->cdata.get())->training;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPRreluWithNoiseFunctionalBackward0_upper_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RreluWithNoiseFunctionalBackward0*>(self->cdata.get())->upper;
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

static struct PyGetSetDef RreluWithNoiseFunctionalBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_lower", (getter)THPRreluWithNoiseFunctionalBackward0_lower_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_noise", (getter)THPRreluWithNoiseFunctionalBackward0_noise_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_noise", (getter)THPRreluWithNoiseFunctionalBackward0_noise_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPRreluWithNoiseFunctionalBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPRreluWithNoiseFunctionalBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_training", (getter)THPRreluWithNoiseFunctionalBackward0_training_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_upper", (getter)THPRreluWithNoiseFunctionalBackward0_upper_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSoftplusBackward0_beta_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SoftplusBackward0*>(self->cdata.get())->beta;
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

PyObject* THPSoftplusBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftplusBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftplusBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftplusBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftplusBackward0_threshold_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SoftplusBackward0*>(self->cdata.get())->threshold;
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

static struct PyGetSetDef SoftplusBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_beta", (getter)THPSoftplusBackward0_beta_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSoftplusBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSoftplusBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_threshold", (getter)THPSoftplusBackward0_threshold_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPReflectionPad2DBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ReflectionPad2DBackward0*>(self->cdata.get())->padding;
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

PyObject* THPReflectionPad2DBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ReflectionPad2DBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPReflectionPad2DBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ReflectionPad2DBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ReflectionPad2DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_padding", (getter)THPReflectionPad2DBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPReflectionPad2DBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPReflectionPad2DBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleBilinear2DAaBackward0_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBilinear2DAaBackward0*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBilinear2DAaBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBilinear2DAaBackward0*>(self->cdata.get())->output_size;
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

PyObject* THPUpsampleBilinear2DAaBackward0_scales_h_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleBilinear2DAaBackward0*>(self->cdata.get())->scales_h;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBilinear2DAaBackward0_scales_w_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleBilinear2DAaBackward0*>(self->cdata.get())->scales_w;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBilinear2DAaBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBilinear2DAaBackward0*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef UpsampleBilinear2DAaBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_align_corners", (getter)THPUpsampleBilinear2DAaBackward0_align_corners_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPUpsampleBilinear2DAaBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_h", (getter)THPUpsampleBilinear2DAaBackward0_scales_h_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_w", (getter)THPUpsampleBilinear2DAaBackward0_scales_w_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPUpsampleBilinear2DAaBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPChannelShuffleBackward0_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ChannelShuffleBackward0*>(self->cdata.get())->groups;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ChannelShuffleBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_groups", (getter)THPChannelShuffleBackward0_groups_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAdaptiveAvgPool2DBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AdaptiveAvgPool2DBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAdaptiveAvgPool2DBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AdaptiveAvgPool2DBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AdaptiveAvgPool2DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPAdaptiveAvgPool2DBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPAdaptiveAvgPool2DBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAvgPool3DBackward0_ceil_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool3DBackward0*>(self->cdata.get())->ceil_mode;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool3DBackward0_count_include_pad_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool3DBackward0*>(self->cdata.get())->count_include_pad;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool3DBackward0_divisor_override_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<AvgPool3DBackward0*>(self->cdata.get())->divisor_override;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool3DBackward0_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool3DBackward0*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool3DBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool3DBackward0*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool3DBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AvgPool3DBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool3DBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AvgPool3DBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool3DBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool3DBackward0*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AvgPool3DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_ceil_mode", (getter)THPAvgPool3DBackward0_ceil_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_count_include_pad", (getter)THPAvgPool3DBackward0_count_include_pad_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_divisor_override", (getter)THPAvgPool3DBackward0_divisor_override_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_kernel_size", (getter)THPAvgPool3DBackward0_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPAvgPool3DBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPAvgPool3DBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPAvgPool3DBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPAvgPool3DBackward0_stride_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMaxPool3DWithIndicesBackward0_ceil_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxPool3DWithIndicesBackward0*>(self->cdata.get())->ceil_mode;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool3DWithIndicesBackward0_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxPool3DWithIndicesBackward0*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool3DWithIndicesBackward0_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxPool3DWithIndicesBackward0*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool3DWithIndicesBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxPool3DWithIndicesBackward0*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool3DWithIndicesBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxPool3DWithIndicesBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool3DWithIndicesBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxPool3DWithIndicesBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool3DWithIndicesBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxPool3DWithIndicesBackward0*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool3DWithIndicesBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxPool3DWithIndicesBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool3DWithIndicesBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxPool3DWithIndicesBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MaxPool3DWithIndicesBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_ceil_mode", (getter)THPMaxPool3DWithIndicesBackward0_ceil_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPMaxPool3DWithIndicesBackward0_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_kernel_size", (getter)THPMaxPool3DWithIndicesBackward0_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPMaxPool3DWithIndicesBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPMaxPool3DWithIndicesBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPMaxPool3DWithIndicesBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPMaxPool3DWithIndicesBackward0_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPMaxPool3DWithIndicesBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPMaxPool3DWithIndicesBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPConvolutionBackwardBackward0_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionBackwardBackward0*>(self->cdata.get())->dilation;
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

PyObject* THPConvolutionBackwardBackward0_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvolutionBackwardBackward0*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackwardBackward0_grad_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvolutionBackwardBackward0*>(self->cdata.get())->grad_output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackwardBackward0_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionBackwardBackward0*>(self->cdata.get())->groups;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackwardBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvolutionBackwardBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackwardBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvolutionBackwardBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackwardBackward0_output_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionBackwardBackward0*>(self->cdata.get())->output_padding;
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

PyObject* THPConvolutionBackwardBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionBackwardBackward0*>(self->cdata.get())->padding;
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

PyObject* THPConvolutionBackwardBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionBackwardBackward0*>(self->cdata.get())->stride;
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

PyObject* THPConvolutionBackwardBackward0_transposed_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionBackwardBackward0*>(self->cdata.get())->transposed;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackwardBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvolutionBackwardBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackwardBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvolutionBackwardBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ConvolutionBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dilation", (getter)THPConvolutionBackwardBackward0_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output", (getter)THPConvolutionBackwardBackward0_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_grad_output", (getter)THPConvolutionBackwardBackward0_grad_output_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_groups", (getter)THPConvolutionBackwardBackward0_groups_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPConvolutionBackwardBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPConvolutionBackwardBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_padding", (getter)THPConvolutionBackwardBackward0_output_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPConvolutionBackwardBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPConvolutionBackwardBackward0_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_transposed", (getter)THPConvolutionBackwardBackward0_transposed_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPConvolutionBackwardBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPConvolutionBackwardBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPConvolutionOverrideableBackward0_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionOverrideableBackward0*>(self->cdata.get())->dilation;
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

PyObject* THPConvolutionOverrideableBackward0_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionOverrideableBackward0*>(self->cdata.get())->groups;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionOverrideableBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvolutionOverrideableBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionOverrideableBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvolutionOverrideableBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionOverrideableBackward0_output_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionOverrideableBackward0*>(self->cdata.get())->output_padding;
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

PyObject* THPConvolutionOverrideableBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionOverrideableBackward0*>(self->cdata.get())->padding;
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

PyObject* THPConvolutionOverrideableBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionOverrideableBackward0*>(self->cdata.get())->stride;
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

PyObject* THPConvolutionOverrideableBackward0_transposed_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionOverrideableBackward0*>(self->cdata.get())->transposed;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionOverrideableBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvolutionOverrideableBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionOverrideableBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvolutionOverrideableBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ConvolutionOverrideableBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dilation", (getter)THPConvolutionOverrideableBackward0_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_groups", (getter)THPConvolutionOverrideableBackward0_groups_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPConvolutionOverrideableBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPConvolutionOverrideableBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_padding", (getter)THPConvolutionOverrideableBackward0_output_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPConvolutionOverrideableBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPConvolutionOverrideableBackward0_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_transposed", (getter)THPConvolutionOverrideableBackward0_transposed_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPConvolutionOverrideableBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPConvolutionOverrideableBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSlowConvTranspose3DBackward0_bias_sym_sizes_opt_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<SlowConvTranspose3DBackward0*>(self->cdata.get())->bias_sym_sizes_opt;
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

PyObject* THPSlowConvTranspose3DBackward0_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvTranspose3DBackward0*>(self->cdata.get())->dilation;
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

PyObject* THPSlowConvTranspose3DBackward0_output_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvTranspose3DBackward0*>(self->cdata.get())->output_padding;
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

PyObject* THPSlowConvTranspose3DBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvTranspose3DBackward0*>(self->cdata.get())->padding;
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

PyObject* THPSlowConvTranspose3DBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvTranspose3DBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvTranspose3DBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvTranspose3DBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvTranspose3DBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvTranspose3DBackward0*>(self->cdata.get())->stride;
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

PyObject* THPSlowConvTranspose3DBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvTranspose3DBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvTranspose3DBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvTranspose3DBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SlowConvTranspose3DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_bias_sym_sizes_opt", (getter)THPSlowConvTranspose3DBackward0_bias_sym_sizes_opt_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPSlowConvTranspose3DBackward0_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_padding", (getter)THPSlowConvTranspose3DBackward0_output_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPSlowConvTranspose3DBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSlowConvTranspose3DBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSlowConvTranspose3DBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPSlowConvTranspose3DBackward0_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPSlowConvTranspose3DBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPSlowConvTranspose3DBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSlowConv3DBackward0_bias_sym_sizes_opt_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<SlowConv3DBackward0*>(self->cdata.get())->bias_sym_sizes_opt;
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

PyObject* THPSlowConv3DBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConv3DBackward0*>(self->cdata.get())->padding;
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

PyObject* THPSlowConv3DBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConv3DBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConv3DBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConv3DBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConv3DBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConv3DBackward0*>(self->cdata.get())->stride;
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

PyObject* THPSlowConv3DBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConv3DBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConv3DBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConv3DBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SlowConv3DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_bias_sym_sizes_opt", (getter)THPSlowConv3DBackward0_bias_sym_sizes_opt_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPSlowConv3DBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSlowConv3DBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSlowConv3DBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPSlowConv3DBackward0_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPSlowConv3DBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPSlowConv3DBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPIm2ColBackward0_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Im2ColBackward0*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPIm2ColBackward0_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Im2ColBackward0*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPIm2ColBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Im2ColBackward0*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPIm2ColBackward0_self_sym_argsize_minus_1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Im2ColBackward0*>(self->cdata.get())->self_sym_argsize_minus_1;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPIm2ColBackward0_self_sym_argsize_minus_2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Im2ColBackward0*>(self->cdata.get())->self_sym_argsize_minus_2;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPIm2ColBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Im2ColBackward0*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef Im2ColBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dilation", (getter)THPIm2ColBackward0_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_kernel_size", (getter)THPIm2ColBackward0_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPIm2ColBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_argsize_minus_1", (getter)THPIm2ColBackward0_self_sym_argsize_minus_1_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_argsize_minus_2", (getter)THPIm2ColBackward0_self_sym_argsize_minus_2_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPIm2ColBackward0_stride_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAvgPool3DBackwardBackward0_ceil_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool3DBackwardBackward0*>(self->cdata.get())->ceil_mode;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool3DBackwardBackward0_count_include_pad_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool3DBackwardBackward0*>(self->cdata.get())->count_include_pad;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool3DBackwardBackward0_divisor_override_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<AvgPool3DBackwardBackward0*>(self->cdata.get())->divisor_override;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool3DBackwardBackward0_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool3DBackwardBackward0*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool3DBackwardBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool3DBackwardBackward0*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool3DBackwardBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool3DBackwardBackward0*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AvgPool3DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_ceil_mode", (getter)THPAvgPool3DBackwardBackward0_ceil_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_count_include_pad", (getter)THPAvgPool3DBackwardBackward0_count_include_pad_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_divisor_override", (getter)THPAvgPool3DBackwardBackward0_divisor_override_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_kernel_size", (getter)THPAvgPool3DBackwardBackward0_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPAvgPool3DBackwardBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPAvgPool3DBackwardBackward0_stride_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPGluBackwardBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<GluBackwardBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPGluBackwardBackward0_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GluBackwardBackward0*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPGluBackwardBackward0_grad_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GluBackwardBackward0*>(self->cdata.get())->grad_output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPGluBackwardBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GluBackwardBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPGluBackwardBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GluBackwardBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef GluBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPGluBackwardBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output", (getter)THPGluBackwardBackward0_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_grad_output", (getter)THPGluBackwardBackward0_grad_output_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPGluBackwardBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPGluBackwardBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPHardtanhBackwardBackward0_max_val_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<HardtanhBackwardBackward0*>(self->cdata.get())->max_val;
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

PyObject* THPHardtanhBackwardBackward0_min_val_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<HardtanhBackwardBackward0*>(self->cdata.get())->min_val;
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

PyObject* THPHardtanhBackwardBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HardtanhBackwardBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPHardtanhBackwardBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HardtanhBackwardBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef HardtanhBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_max_val", (getter)THPHardtanhBackwardBackward0_max_val_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_min_val", (getter)THPHardtanhBackwardBackward0_min_val_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPHardtanhBackwardBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPHardtanhBackwardBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLeakyReluBackwardBackward0_negative_slope_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LeakyReluBackwardBackward0*>(self->cdata.get())->negative_slope;
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

PyObject* THPLeakyReluBackwardBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LeakyReluBackwardBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLeakyReluBackwardBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LeakyReluBackwardBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LeakyReluBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_negative_slope", (getter)THPLeakyReluBackwardBackward0_negative_slope_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPLeakyReluBackwardBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPLeakyReluBackwardBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef MaxPool2DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPMaxPool2DWithIndicesBackwardBackward0_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxPool2DWithIndicesBackwardBackward0*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool2DWithIndicesBackwardBackward0_indices_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxPool2DWithIndicesBackwardBackward0*>(self->cdata.get())->indices_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MaxPool2DWithIndicesBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_indices", (getter)THPMaxPool2DWithIndicesBackwardBackward0_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_indices", (getter)THPMaxPool2DWithIndicesBackwardBackward0_indices_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleLinear1DBackwardBackward0_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleLinear1DBackwardBackward0*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleLinear1DBackwardBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleLinear1DBackwardBackward0*>(self->cdata.get())->output_size;
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

PyObject* THPUpsampleLinear1DBackwardBackward0_scales_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleLinear1DBackwardBackward0*>(self->cdata.get())->scales;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleLinear1DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_align_corners", (getter)THPUpsampleLinear1DBackwardBackward0_align_corners_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPUpsampleLinear1DBackwardBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales", (getter)THPUpsampleLinear1DBackwardBackward0_scales_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleNearest1DBackwardBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleNearest1DBackwardBackward0*>(self->cdata.get())->output_size;
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

PyObject* THPUpsampleNearest1DBackwardBackward0_scales_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest1DBackwardBackward0*>(self->cdata.get())->scales;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleNearest1DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_output_size", (getter)THPUpsampleNearest1DBackwardBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales", (getter)THPUpsampleNearest1DBackwardBackward0_scales_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleNearest2DBackwardBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleNearest2DBackwardBackward0*>(self->cdata.get())->output_size;
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

PyObject* THPUpsampleNearest2DBackwardBackward0_scales_h_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest2DBackwardBackward0*>(self->cdata.get())->scales_h;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest2DBackwardBackward0_scales_w_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest2DBackwardBackward0*>(self->cdata.get())->scales_w;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleNearest2DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_output_size", (getter)THPUpsampleNearest2DBackwardBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_h", (getter)THPUpsampleNearest2DBackwardBackward0_scales_h_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_w", (getter)THPUpsampleNearest2DBackwardBackward0_scales_w_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMpsConvolutionTransposeBackward0_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MpsConvolutionTransposeBackward0*>(self->cdata.get())->dilation;
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

PyObject* THPMpsConvolutionTransposeBackward0_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MpsConvolutionTransposeBackward0*>(self->cdata.get())->groups;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMpsConvolutionTransposeBackward0_output_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MpsConvolutionTransposeBackward0*>(self->cdata.get())->output_padding;
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

PyObject* THPMpsConvolutionTransposeBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MpsConvolutionTransposeBackward0*>(self->cdata.get())->padding;
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

PyObject* THPMpsConvolutionTransposeBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MpsConvolutionTransposeBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMpsConvolutionTransposeBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MpsConvolutionTransposeBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMpsConvolutionTransposeBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MpsConvolutionTransposeBackward0*>(self->cdata.get())->stride;
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

PyObject* THPMpsConvolutionTransposeBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MpsConvolutionTransposeBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMpsConvolutionTransposeBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MpsConvolutionTransposeBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MpsConvolutionTransposeBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dilation", (getter)THPMpsConvolutionTransposeBackward0_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_groups", (getter)THPMpsConvolutionTransposeBackward0_groups_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_padding", (getter)THPMpsConvolutionTransposeBackward0_output_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPMpsConvolutionTransposeBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPMpsConvolutionTransposeBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPMpsConvolutionTransposeBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPMpsConvolutionTransposeBackward0_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPMpsConvolutionTransposeBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPMpsConvolutionTransposeBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCudnnBatchNormBackward0_epsilon_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnBatchNormBackward0*>(self->cdata.get())->epsilon;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackward0_running_mean_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackward0*>(self->cdata.get())->running_mean_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackward0_running_mean_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackward0*>(self->cdata.get())->running_mean_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackward0_running_var_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackward0*>(self->cdata.get())->running_var_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackward0_running_var_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackward0*>(self->cdata.get())->running_var_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackward0_training_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnBatchNormBackward0*>(self->cdata.get())->training;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackward0_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackward0*>(self->cdata.get())->result2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackward0_result2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackward0*>(self->cdata.get())->result2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackward0_result3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackward0*>(self->cdata.get())->result3_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackward0_result3_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackward0*>(self->cdata.get())->result3_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CudnnBatchNormBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_epsilon", (getter)THPCudnnBatchNormBackward0_epsilon_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPCudnnBatchNormBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPCudnnBatchNormBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_mean", (getter)THPCudnnBatchNormBackward0_running_mean_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_running_mean", (getter)THPCudnnBatchNormBackward0_running_mean_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_var", (getter)THPCudnnBatchNormBackward0_running_var_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_running_var", (getter)THPCudnnBatchNormBackward0_running_var_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_training", (getter)THPCudnnBatchNormBackward0_training_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPCudnnBatchNormBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPCudnnBatchNormBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPCudnnBatchNormBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPCudnnBatchNormBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPCudnnBatchNormBackward0_result2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result2", (getter)THPCudnnBatchNormBackward0_result2_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result3", (getter)THPCudnnBatchNormBackward0_result3_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result3", (getter)THPCudnnBatchNormBackward0_result3_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMiopenConvolutionTransposeBackward0_bias_sym_sizes_opt_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<MiopenConvolutionTransposeBackward0*>(self->cdata.get())->bias_sym_sizes_opt;
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

PyObject* THPMiopenConvolutionTransposeBackward0_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionTransposeBackward0*>(self->cdata.get())->dilation;
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

PyObject* THPMiopenConvolutionTransposeBackward0_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionTransposeBackward0*>(self->cdata.get())->groups;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionTransposeBackward0_output_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionTransposeBackward0*>(self->cdata.get())->output_padding;
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

PyObject* THPMiopenConvolutionTransposeBackward0_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionTransposeBackward0*>(self->cdata.get())->padding;
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

PyObject* THPMiopenConvolutionTransposeBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenConvolutionTransposeBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionTransposeBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenConvolutionTransposeBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionTransposeBackward0_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionTransposeBackward0*>(self->cdata.get())->stride;
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

PyObject* THPMiopenConvolutionTransposeBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenConvolutionTransposeBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionTransposeBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenConvolutionTransposeBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MiopenConvolutionTransposeBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_bias_sym_sizes_opt", (getter)THPMiopenConvolutionTransposeBackward0_bias_sym_sizes_opt_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPMiopenConvolutionTransposeBackward0_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_groups", (getter)THPMiopenConvolutionTransposeBackward0_groups_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_padding", (getter)THPMiopenConvolutionTransposeBackward0_output_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPMiopenConvolutionTransposeBackward0_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPMiopenConvolutionTransposeBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPMiopenConvolutionTransposeBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPMiopenConvolutionTransposeBackward0_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPMiopenConvolutionTransposeBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPMiopenConvolutionTransposeBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMiopenBatchNormBackward0_epsilon_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenBatchNormBackward0*>(self->cdata.get())->epsilon;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackward0_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackward0*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackward0_input_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackward0*>(self->cdata.get())->input_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackward0_running_mean_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackward0*>(self->cdata.get())->running_mean_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackward0_running_mean_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackward0*>(self->cdata.get())->running_mean_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackward0_running_var_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackward0*>(self->cdata.get())->running_var_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackward0_running_var_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackward0*>(self->cdata.get())->running_var_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackward0_training_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenBatchNormBackward0*>(self->cdata.get())->training;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackward0*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackward0_weight_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackward0*>(self->cdata.get())->weight_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackward0_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackward0*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackward0_result1_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackward0*>(self->cdata.get())->result1_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackward0_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackward0*>(self->cdata.get())->result2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackward0_result2_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackward0*>(self->cdata.get())->result2_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MiopenBatchNormBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_epsilon", (getter)THPMiopenBatchNormBackward0_epsilon_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPMiopenBatchNormBackward0_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_input", (getter)THPMiopenBatchNormBackward0_input_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_mean", (getter)THPMiopenBatchNormBackward0_running_mean_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_running_mean", (getter)THPMiopenBatchNormBackward0_running_mean_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_var", (getter)THPMiopenBatchNormBackward0_running_var_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_running_var", (getter)THPMiopenBatchNormBackward0_running_var_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_training", (getter)THPMiopenBatchNormBackward0_training_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPMiopenBatchNormBackward0_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_weight", (getter)THPMiopenBatchNormBackward0_weight_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPMiopenBatchNormBackward0_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result1", (getter)THPMiopenBatchNormBackward0_result1_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPMiopenBatchNormBackward0_result2_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_result2", (getter)THPMiopenBatchNormBackward0_result2_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMkldnnAdaptiveAvgPool2DBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnAdaptiveAvgPool2DBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnAdaptiveAvgPool2DBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnAdaptiveAvgPool2DBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MkldnnAdaptiveAvgPool2DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMkldnnAdaptiveAvgPool2DBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPMkldnnAdaptiveAvgPool2DBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNestedTensorFromMaskBackward0_t_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NestedTensorFromMaskBackward0*>(self->cdata.get())->t_sym_sizes;
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

static struct PyGetSetDef NestedTensorFromMaskBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_t_sym_sizes", (getter)THPNestedTensorFromMaskBackward0_t_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNestedFromPaddedTensorBackward0_padded_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NestedFromPaddedTensorBackward0*>(self->cdata.get())->padded_sym_sizes;
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

static struct PyGetSetDef NestedFromPaddedTensorBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_padded_sym_sizes", (getter)THPNestedFromPaddedTensorBackward0_padded_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNestedGetValuesBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NestedGetValuesBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNestedGetValuesBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NestedGetValuesBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NestedGetValuesBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPNestedGetValuesBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPNestedGetValuesBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPScaledDotProductFusedAttentionOverrideableBackward0_attn_bias_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFusedAttentionOverrideableBackward0*>(self->cdata.get())->attn_bias_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFusedAttentionOverrideableBackward0_attn_bias_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFusedAttentionOverrideableBackward0*>(self->cdata.get())->attn_bias_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFusedAttentionOverrideableBackward0_dropout_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ScaledDotProductFusedAttentionOverrideableBackward0*>(self->cdata.get())->dropout_p;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFusedAttentionOverrideableBackward0_is_causal_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ScaledDotProductFusedAttentionOverrideableBackward0*>(self->cdata.get())->is_causal;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFusedAttentionOverrideableBackward0_key_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFusedAttentionOverrideableBackward0*>(self->cdata.get())->key_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFusedAttentionOverrideableBackward0_key_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFusedAttentionOverrideableBackward0*>(self->cdata.get())->key_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFusedAttentionOverrideableBackward0_query_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFusedAttentionOverrideableBackward0*>(self->cdata.get())->query_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFusedAttentionOverrideableBackward0_query_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFusedAttentionOverrideableBackward0*>(self->cdata.get())->query_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFusedAttentionOverrideableBackward0_scale_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<ScaledDotProductFusedAttentionOverrideableBackward0*>(self->cdata.get())->scale;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFusedAttentionOverrideableBackward0_value_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFusedAttentionOverrideableBackward0*>(self->cdata.get())->value_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFusedAttentionOverrideableBackward0_value_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFusedAttentionOverrideableBackward0*>(self->cdata.get())->value_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFusedAttentionOverrideableBackward0_cum_seq_k_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFusedAttentionOverrideableBackward0*>(self->cdata.get())->cum_seq_k_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFusedAttentionOverrideableBackward0_cum_seq_k_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFusedAttentionOverrideableBackward0*>(self->cdata.get())->cum_seq_k_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFusedAttentionOverrideableBackward0_cum_seq_q_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFusedAttentionOverrideableBackward0*>(self->cdata.get())->cum_seq_q_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFusedAttentionOverrideableBackward0_cum_seq_q_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFusedAttentionOverrideableBackward0*>(self->cdata.get())->cum_seq_q_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFusedAttentionOverrideableBackward0_logsumexp_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFusedAttentionOverrideableBackward0*>(self->cdata.get())->logsumexp_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFusedAttentionOverrideableBackward0_logsumexp_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFusedAttentionOverrideableBackward0*>(self->cdata.get())->logsumexp_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFusedAttentionOverrideableBackward0_max_k_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ScaledDotProductFusedAttentionOverrideableBackward0*>(self->cdata.get())->max_k;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFusedAttentionOverrideableBackward0_max_q_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ScaledDotProductFusedAttentionOverrideableBackward0*>(self->cdata.get())->max_q;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFusedAttentionOverrideableBackward0_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFusedAttentionOverrideableBackward0*>(self->cdata.get())->output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFusedAttentionOverrideableBackward0_output_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFusedAttentionOverrideableBackward0*>(self->cdata.get())->output_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFusedAttentionOverrideableBackward0_philox_offset_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFusedAttentionOverrideableBackward0*>(self->cdata.get())->philox_offset_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFusedAttentionOverrideableBackward0_philox_offset_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFusedAttentionOverrideableBackward0*>(self->cdata.get())->philox_offset_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFusedAttentionOverrideableBackward0_philox_seed_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFusedAttentionOverrideableBackward0*>(self->cdata.get())->philox_seed_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPScaledDotProductFusedAttentionOverrideableBackward0_philox_seed_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScaledDotProductFusedAttentionOverrideableBackward0*>(self->cdata.get())->philox_seed_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ScaledDotProductFusedAttentionOverrideableBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_attn_bias", (getter)THPScaledDotProductFusedAttentionOverrideableBackward0_attn_bias_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_attn_bias", (getter)THPScaledDotProductFusedAttentionOverrideableBackward0_attn_bias_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dropout_p", (getter)THPScaledDotProductFusedAttentionOverrideableBackward0_dropout_p_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_is_causal", (getter)THPScaledDotProductFusedAttentionOverrideableBackward0_is_causal_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_key", (getter)THPScaledDotProductFusedAttentionOverrideableBackward0_key_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_key", (getter)THPScaledDotProductFusedAttentionOverrideableBackward0_key_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_query", (getter)THPScaledDotProductFusedAttentionOverrideableBackward0_query_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_query", (getter)THPScaledDotProductFusedAttentionOverrideableBackward0_query_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale", (getter)THPScaledDotProductFusedAttentionOverrideableBackward0_scale_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_value", (getter)THPScaledDotProductFusedAttentionOverrideableBackward0_value_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_value", (getter)THPScaledDotProductFusedAttentionOverrideableBackward0_value_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_cum_seq_k", (getter)THPScaledDotProductFusedAttentionOverrideableBackward0_cum_seq_k_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_cum_seq_k", (getter)THPScaledDotProductFusedAttentionOverrideableBackward0_cum_seq_k_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_cum_seq_q", (getter)THPScaledDotProductFusedAttentionOverrideableBackward0_cum_seq_q_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_cum_seq_q", (getter)THPScaledDotProductFusedAttentionOverrideableBackward0_cum_seq_q_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_logsumexp", (getter)THPScaledDotProductFusedAttentionOverrideableBackward0_logsumexp_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_logsumexp", (getter)THPScaledDotProductFusedAttentionOverrideableBackward0_logsumexp_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_max_k", (getter)THPScaledDotProductFusedAttentionOverrideableBackward0_max_k_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_max_q", (getter)THPScaledDotProductFusedAttentionOverrideableBackward0_max_q_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output", (getter)THPScaledDotProductFusedAttentionOverrideableBackward0_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_output", (getter)THPScaledDotProductFusedAttentionOverrideableBackward0_output_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_philox_offset", (getter)THPScaledDotProductFusedAttentionOverrideableBackward0_philox_offset_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_philox_offset", (getter)THPScaledDotProductFusedAttentionOverrideableBackward0_philox_offset_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_philox_seed", (getter)THPScaledDotProductFusedAttentionOverrideableBackward0_philox_seed_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_philox_seed", (getter)THPScaledDotProductFusedAttentionOverrideableBackward0_philox_seed_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPFftC2RBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FftC2RBackward0*>(self->cdata.get())->dim;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPFftC2RBackward0_normalization_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FftC2RBackward0*>(self->cdata.get())->normalization;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FftC2RBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPFftC2RBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_normalization", (getter)THPFftC2RBackward0_normalization_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef TestWarnInAutogradBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPTestAutogradMultipleDispatchViewBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TestAutogradMultipleDispatchViewBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPTestAutogradMultipleDispatchViewBackward0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TestAutogradMultipleDispatchViewBackward0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef TestAutogradMultipleDispatchViewBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPTestAutogradMultipleDispatchViewBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPTestAutogradMultipleDispatchViewBackward0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPTestAutogradMultipleDispatchViewBackwardAutogradCUDA0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TestAutogradMultipleDispatchViewBackwardAutogradCUDA0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPTestAutogradMultipleDispatchViewBackwardAutogradCUDA0_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TestAutogradMultipleDispatchViewBackwardAutogradCUDA0*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef TestAutogradMultipleDispatchViewBackwardAutogradCUDA0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPTestAutogradMultipleDispatchViewBackwardAutogradCUDA0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPTestAutogradMultipleDispatchViewBackwardAutogradCUDA0_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPReshapeCopyBackward0_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ReshapeCopyBackward0*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef ReshapeCopyBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sym_sizes", (getter)THPReshapeCopyBackward0_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSliceBackward0_copy_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SliceBackward0_copy*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSliceBackward0_copy_end_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<SliceBackward0_copy*>(self->cdata.get())->end;
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

PyObject* THPSliceBackward0_copy_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SliceBackward0_copy*>(self->cdata.get())->self_sym_sizes;
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

PyObject* THPSliceBackward0_copy_start_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<SliceBackward0_copy*>(self->cdata.get())->start;
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

PyObject* THPSliceBackward0_copy_step_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SliceBackward0_copy*>(self->cdata.get())->step;
  if (auto m = prop.maybe_as_int()) {
    return PyLong_FromUnsignedLong(*m);
  } else {
    return py::cast(prop).release().ptr();
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SliceBackward0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSliceBackward0_copy_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_end", (getter)THPSliceBackward0_copy_end_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPSliceBackward0_copy_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_start", (getter)THPSliceBackward0_copy_start_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_step", (getter)THPSliceBackward0_copy_step_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSplitWithSizesBackward0_copy_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SplitWithSizesBackward0_copy*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSplitWithSizesBackward0_copy_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SplitWithSizesBackward0_copy*>(self->cdata.get())->self_sym_sizes;
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

PyObject* THPSplitWithSizesBackward0_copy_split_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SplitWithSizesBackward0_copy*>(self->cdata.get())->split_sizes;
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

static struct PyGetSetDef SplitWithSizesBackward0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSplitWithSizesBackward0_copy_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPSplitWithSizesBackward0_copy_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_split_sizes", (getter)THPSplitWithSizesBackward0_copy_split_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSplitWithSizesBackwardAutogradNestedTensor0_copy_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SplitWithSizesBackwardAutogradNestedTensor0_copy*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSplitWithSizesBackwardAutogradNestedTensor0_copy_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SplitWithSizesBackwardAutogradNestedTensor0_copy*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSplitWithSizesBackwardAutogradNestedTensor0_copy_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SplitWithSizesBackwardAutogradNestedTensor0_copy*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THPSplitWithSizesBackwardAutogradNestedTensor0_copy_split_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SplitWithSizesBackwardAutogradNestedTensor0_copy*>(self->cdata.get())->split_sizes;
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

static struct PyGetSetDef SplitWithSizesBackwardAutogradNestedTensor0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSplitWithSizesBackwardAutogradNestedTensor0_copy_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSplitWithSizesBackwardAutogradNestedTensor0_copy_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPSplitWithSizesBackwardAutogradNestedTensor0_copy_self_raw_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_split_sizes", (getter)THPSplitWithSizesBackwardAutogradNestedTensor0_copy_split_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSqueezeBackward0_copy_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SqueezeBackward0_copy*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef SqueezeBackward0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sym_sizes", (getter)THPSqueezeBackward0_copy_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSqueezeBackward1_copy_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SqueezeBackward1_copy*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSqueezeBackward1_copy_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SqueezeBackward1_copy*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef SqueezeBackward1_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSqueezeBackward1_copy_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPSqueezeBackward1_copy_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSqueezeBackwardAutogradNestedTensor0_copy_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SqueezeBackwardAutogradNestedTensor0_copy*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SqueezeBackwardAutogradNestedTensor0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSqueezeBackwardAutogradNestedTensor0_copy_dim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSqueezeBackward2_copy_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SqueezeBackward2_copy*>(self->cdata.get())->dim;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSqueezeBackward2_copy_self_sym_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SqueezeBackward2_copy*>(self->cdata.get())->self_sym_sizes;
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

static struct PyGetSetDef SqueezeBackward2_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSqueezeBackward2_copy_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sym_sizes", (getter)THPSqueezeBackward2_copy_self_sym_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSqueezeBackwardAutogradNestedTensor1_copy_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SqueezeBackwardAutogradNestedTensor1_copy*>(self->cdata.get())->dim;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (auto i : c10::irange(prop.size())) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSqueezeBackwardAutogradNestedTensor1_copy_self_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SqueezeBackwardAutogradNestedTensor1_copy*>(self->cdata.get())->self_dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SqueezeBackwardAutogradNestedTensor1_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPSqueezeBackwardAutogradNestedTensor1_copy_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_dim", (getter)THPSqueezeBackwardAutogradNestedTensor1_copy_self_dim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPValuesBackward0_copy_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ValuesBackward0_copy*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPValuesBackward0_copy_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ValuesBackward0_copy*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ValuesBackward0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPValuesBackward0_copy_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPValuesBackward0_copy_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPValuesBackwardAutogradNestedTensor0_copy_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ValuesBackwardAutogradNestedTensor0_copy*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPValuesBackwardAutogradNestedTensor0_copy_self_raw_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ValuesBackwardAutogradNestedTensor0_copy*>(self->cdata.get())->self_;
  pybind11::object obj = pybind11::cast(prop, pybind11::return_value_policy::reference);
  return obj.release().ptr();
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ValuesBackwardAutogradNestedTensor0_copy_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPValuesBackwardAutogradNestedTensor0_copy_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_raw_saved_self", (getter)THPValuesBackwardAutogradNestedTensor0_copy_self_raw_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

void initialize_autogenerated_functions_4(PyObject* module) {
  static PyTypeObject AcosBackward0Class;
  addClass<AcosBackward0>(module, AcosBackward0Class, "AcosBackward0", AcosBackward0_properties);
  static PyTypeObject AsinBackward0Class;
  addClass<AsinBackward0>(module, AsinBackward0Class, "AsinBackward0", AsinBackward0_properties);
  static PyTypeObject MatmulBackward0Class;
  addClass<MatmulBackward0>(module, MatmulBackward0Class, "MatmulBackward0", MatmulBackward0_properties);
  static PyTypeObject CopysignBackward0Class;
  addClass<CopysignBackward0>(module, CopysignBackward0Class, "CopysignBackward0", CopysignBackward0_properties);
  static PyTypeObject CopysignBackward1Class;
  addClass<CopysignBackward1>(module, CopysignBackward1Class, "CopysignBackward1", CopysignBackward1_properties);
  static PyTypeObject Deg2RadBackward0Class;
  addClass<Deg2RadBackward0>(module, Deg2RadBackward0Class, "Deg2RadBackward0", Deg2RadBackward0_properties);
  static PyTypeObject LinalgSlogdetBackward0Class;
  addClass<LinalgSlogdetBackward0>(module, LinalgSlogdetBackward0Class, "LinalgSlogdetBackward0", LinalgSlogdetBackward0_properties);
  static PyTypeObject DistBackward0Class;
  addClass<DistBackward0>(module, DistBackward0Class, "DistBackward0", DistBackward0_properties);
  static PyTypeObject ErfcBackward0Class;
  addClass<ErfcBackward0>(module, ErfcBackward0Class, "ErfcBackward0", ErfcBackward0_properties);
  static PyTypeObject SpecialErfcxBackward0Class;
  addClass<SpecialErfcxBackward0>(module, SpecialErfcxBackward0Class, "SpecialErfcxBackward0", SpecialErfcxBackward0_properties);
  static PyTypeObject ErfinvBackward0Class;
  addClass<ErfinvBackward0>(module, ErfinvBackward0Class, "ErfinvBackward0", ErfinvBackward0_properties);
  static PyTypeObject FmodBackward0Class;
  addClass<FmodBackward0>(module, FmodBackward0Class, "FmodBackward0", FmodBackward0_properties);
  static PyTypeObject FmodBackward1Class;
  addClass<FmodBackward1>(module, FmodBackward1Class, "FmodBackward1", FmodBackward1_properties);
  static PyTypeObject GeqrfBackward0Class;
  addClass<GeqrfBackward0>(module, GeqrfBackward0Class, "GeqrfBackward0", GeqrfBackward0_properties);
  static PyTypeObject GridSampler2DBackward0Class;
  addClass<GridSampler2DBackward0>(module, GridSampler2DBackward0Class, "GridSampler2DBackward0", GridSampler2DBackward0_properties);
  static PyTypeObject GtBackward0Class;
  addClass<GtBackward0>(module, GtBackward0Class, "GtBackward0", GtBackward0_properties);
  static PyTypeObject GtBackward1Class;
  addClass<GtBackward1>(module, GtBackward1Class, "GtBackward1", GtBackward1_properties);
  static PyTypeObject SpecialI0EBackward0Class;
  addClass<SpecialI0EBackward0>(module, SpecialI0EBackward0Class, "SpecialI0EBackward0", SpecialI0EBackward0_properties);
  static PyTypeObject SpecialI1Backward0Class;
  addClass<SpecialI1Backward0>(module, SpecialI1Backward0Class, "SpecialI1Backward0", SpecialI1Backward0_properties);
  static PyTypeObject UnsafeMaskedIndexPutAccumulateBackward0Class;
  addClass<UnsafeMaskedIndexPutAccumulateBackward0>(module, UnsafeMaskedIndexPutAccumulateBackward0Class, "UnsafeMaskedIndexPutAccumulateBackward0", UnsafeMaskedIndexPutAccumulateBackward0_properties);
  static PyTypeObject IndexAddBackward0Class;
  addClass<IndexAddBackward0>(module, IndexAddBackward0Class, "IndexAddBackward0", IndexAddBackward0_properties);
  static PyTypeObject IndexPutBackward0Class;
  addClass<IndexPutBackward0>(module, IndexPutBackward0Class, "IndexPutBackward0", IndexPutBackward0_properties);
  static PyTypeObject LogNormalBackward0Class;
  addClass<LogNormalBackward0>(module, LogNormalBackward0Class, "LogNormalBackward0", LogNormalBackward0_properties);
  static PyTypeObject LinalgLuFactorBackward0Class;
  addClass<LinalgLuFactorBackward0>(module, LinalgLuFactorBackward0Class, "LinalgLuFactorBackward0", LinalgLuFactorBackward0_properties);
  static PyTypeObject LinalgLuSolveBackward0Class;
  addClass<LinalgLuSolveBackward0>(module, LinalgLuSolveBackward0Class, "LinalgLuSolveBackward0", LinalgLuSolveBackward0_properties);
  static PyTypeObject MaskedFillBackward0Class;
  addClass<MaskedFillBackward0>(module, MaskedFillBackward0Class, "MaskedFillBackward0", MaskedFillBackward0_properties);
  static PyTypeObject MaskedFillBackward1Class;
  addClass<MaskedFillBackward1>(module, MaskedFillBackward1Class, "MaskedFillBackward1", MaskedFillBackward1_properties);
  static PyTypeObject MaskedScatterBackward0Class;
  addClass<MaskedScatterBackward0>(module, MaskedScatterBackward0Class, "MaskedScatterBackward0", MaskedScatterBackward0_properties);
  static PyTypeObject MedianBackward0Class;
  addClass<MedianBackward0>(module, MedianBackward0Class, "MedianBackward0", MedianBackward0_properties);
  static PyTypeObject NanmedianBackward0Class;
  addClass<NanmedianBackward0>(module, NanmedianBackward0Class, "NanmedianBackward0", NanmedianBackward0_properties);
  static PyTypeObject MedianBackward1Class;
  addClass<MedianBackward1>(module, MedianBackward1Class, "MedianBackward1", MedianBackward1_properties);
  static PyTypeObject NanmedianBackward1Class;
  addClass<NanmedianBackward1>(module, NanmedianBackward1Class, "NanmedianBackward1", NanmedianBackward1_properties);
  static PyTypeObject BatchNormNoUpdateBackward0Class;
  addClass<BatchNormNoUpdateBackward0>(module, BatchNormNoUpdateBackward0Class, "BatchNormNoUpdateBackward0", BatchNormNoUpdateBackward0_properties);
  static PyTypeObject PoissonBackward0Class;
  addClass<PoissonBackward0>(module, PoissonBackward0Class, "PoissonBackward0", PoissonBackward0_properties);
  static PyTypeObject ProdBackward0Class;
  addClass<ProdBackward0>(module, ProdBackward0Class, "ProdBackward0", ProdBackward0_properties);
  static PyTypeObject ProdBackward1Class;
  addClass<ProdBackward1>(module, ProdBackward1Class, "ProdBackward1", ProdBackward1_properties);
  static PyTypeObject LinalgQrBackward0Class;
  addClass<LinalgQrBackward0>(module, LinalgQrBackward0Class, "LinalgQrBackward0", LinalgQrBackward0_properties);
  static PyTypeObject RemainderBackward0Class;
  addClass<RemainderBackward0>(module, RemainderBackward0Class, "RemainderBackward0", RemainderBackward0_properties);
  static PyTypeObject RemainderBackward1Class;
  addClass<RemainderBackward1>(module, RemainderBackward1Class, "RemainderBackward1", RemainderBackward1_properties);
  static PyTypeObject RenormBackward0Class;
  addClass<RenormBackward0>(module, RenormBackward0Class, "RenormBackward0", RenormBackward0_properties);
  static PyTypeObject SincBackward0Class;
  addClass<SincBackward0>(module, SincBackward0Class, "SincBackward0", SincBackward0_properties);
  static PyTypeObject SliceBackward0Class;
  addClass<SliceBackward0>(module, SliceBackward0Class, "SliceBackward0", SliceBackward0_properties);
  static PyTypeObject SliceInverseBackward0Class;
  addClass<SliceInverseBackward0>(module, SliceInverseBackward0Class, "SliceInverseBackward0", SliceInverseBackward0_properties);
  static PyTypeObject SelectScatterBackward0Class;
  addClass<SelectScatterBackward0>(module, SelectScatterBackward0Class, "SelectScatterBackward0", SelectScatterBackward0_properties);
  static PyTypeObject LinalgSolveExBackward0Class;
  addClass<LinalgSolveExBackward0>(module, LinalgSolveExBackward0Class, "LinalgSolveExBackward0", LinalgSolveExBackward0_properties);
  static PyTypeObject UnsafeSplitWithSizesBackward0Class;
  addClass<UnsafeSplitWithSizesBackward0>(module, UnsafeSplitWithSizesBackward0Class, "UnsafeSplitWithSizesBackward0", UnsafeSplitWithSizesBackward0_properties);
  static PyTypeObject SubBackward0Class;
  addClass<SubBackward0>(module, SubBackward0Class, "SubBackward0", SubBackward0_properties);
  static PyTypeObject SubBackward1Class;
  addClass<SubBackward1>(module, SubBackward1Class, "SubBackward1", SubBackward1_properties);
  static PyTypeObject LinalgSvdBackward0Class;
  addClass<LinalgSvdBackward0>(module, LinalgSvdBackward0Class, "LinalgSvdBackward0", LinalgSvdBackward0_properties);
  static PyTypeObject TakeBackward0Class;
  addClass<TakeBackward0>(module, TakeBackward0Class, "TakeBackward0", TakeBackward0_properties);
  static PyTypeObject TanBackward0Class;
  addClass<TanBackward0>(module, TanBackward0Class, "TanBackward0", TanBackward0_properties);
  static PyTypeObject TraceBackward0Class;
  addClass<TraceBackward0>(module, TraceBackward0Class, "TraceBackward0", TraceBackward0_properties);
  static PyTypeObject ToSparseCsrBackward0Class;
  addClass<ToSparseCsrBackward0>(module, ToSparseCsrBackward0Class, "ToSparseCsrBackward0", ToSparseCsrBackward0_properties);
  static PyTypeObject ToSparseBsrBackward0Class;
  addClass<ToSparseBsrBackward0>(module, ToSparseBsrBackward0Class, "ToSparseBsrBackward0", ToSparseBsrBackward0_properties);
  static PyTypeObject UnfoldBackward0Class;
  addClass<UnfoldBackward0>(module, UnfoldBackward0Class, "UnfoldBackward0", UnfoldBackward0_properties);
  static PyTypeObject UniqueDimConsecutiveBackward0Class;
  addClass<UniqueDimConsecutiveBackward0>(module, UniqueDimConsecutiveBackward0Class, "UniqueDimConsecutiveBackward0", UniqueDimConsecutiveBackward0_properties);
  static PyTypeObject UnsafeViewBackward0Class;
  addClass<UnsafeViewBackward0>(module, UnsafeViewBackward0Class, "UnsafeViewBackward0", UnsafeViewBackward0_properties);
  static PyTypeObject UnsqueezeBackward0Class;
  addClass<UnsqueezeBackward0>(module, UnsqueezeBackward0Class, "UnsqueezeBackward0", UnsqueezeBackward0_properties);
  static PyTypeObject UnsqueezeBackward1Class;
  addClass<UnsqueezeBackward1>(module, UnsqueezeBackward1Class, "UnsqueezeBackward1", UnsqueezeBackward1_properties);
  static PyTypeObject ViewAsRealBackward0Class;
  addClass<ViewAsRealBackward0>(module, ViewAsRealBackward0Class, "ViewAsRealBackward0", ViewAsRealBackward0_properties);
  static PyTypeObject ViewAsComplexBackward0Class;
  addClass<ViewAsComplexBackward0>(module, ViewAsComplexBackward0Class, "ViewAsComplexBackward0", ViewAsComplexBackward0_properties);
  static PyTypeObject SparseCompressedTensorBackward0Class;
  addClass<SparseCompressedTensorBackward0>(module, SparseCompressedTensorBackward0Class, "SparseCompressedTensorBackward0", SparseCompressedTensorBackward0_properties);
  static PyTypeObject ConstantPadNdBackward0Class;
  addClass<ConstantPadNdBackward0>(module, ConstantPadNdBackward0Class, "ConstantPadNdBackward0", ConstantPadNdBackward0_properties);
  static PyTypeObject EmbeddingBackward0Class;
  addClass<EmbeddingBackward0>(module, EmbeddingBackward0Class, "EmbeddingBackward0", EmbeddingBackward0_properties);
  static PyTypeObject EmbeddingBagBackwardBackward0Class;
  addClass<EmbeddingBagBackwardBackward0>(module, EmbeddingBagBackwardBackward0Class, "EmbeddingBagBackwardBackward0", EmbeddingBagBackwardBackward0_properties);
  static PyTypeObject EmbeddingBagDenseBackwardBackward0Class;
  addClass<EmbeddingBagDenseBackwardBackward0>(module, EmbeddingBagDenseBackwardBackward0Class, "EmbeddingBagDenseBackwardBackward0", EmbeddingBagDenseBackwardBackward0_properties);
  static PyTypeObject SmoothL1LossBackward0Class;
  addClass<SmoothL1LossBackward0>(module, SmoothL1LossBackward0Class, "SmoothL1LossBackward0", SmoothL1LossBackward0_properties);
  static PyTypeObject ReluBackward0Class;
  addClass<ReluBackward0>(module, ReluBackward0Class, "ReluBackward0", ReluBackward0_properties);
  static PyTypeObject EluBackward0Class;
  addClass<EluBackward0>(module, EluBackward0Class, "EluBackward0", EluBackward0_properties);
  static PyTypeObject EluBackward1Class;
  addClass<EluBackward1>(module, EluBackward1Class, "EluBackward1", EluBackward1_properties);
  static PyTypeObject HardshrinkBackwardBackward0Class;
  addClass<HardshrinkBackwardBackward0>(module, HardshrinkBackwardBackward0Class, "HardshrinkBackwardBackward0", HardshrinkBackwardBackward0_properties);
  static PyTypeObject LeakyReluBackward1Class;
  addClass<LeakyReluBackward1>(module, LeakyReluBackward1Class, "LeakyReluBackward1", LeakyReluBackward1_properties);
  static PyTypeObject RreluWithNoiseFunctionalBackward0Class;
  addClass<RreluWithNoiseFunctionalBackward0>(module, RreluWithNoiseFunctionalBackward0Class, "RreluWithNoiseFunctionalBackward0", RreluWithNoiseFunctionalBackward0_properties);
  static PyTypeObject SoftplusBackward0Class;
  addClass<SoftplusBackward0>(module, SoftplusBackward0Class, "SoftplusBackward0", SoftplusBackward0_properties);
  static PyTypeObject ReflectionPad2DBackward0Class;
  addClass<ReflectionPad2DBackward0>(module, ReflectionPad2DBackward0Class, "ReflectionPad2DBackward0", ReflectionPad2DBackward0_properties);
  static PyTypeObject UpsampleBilinear2DAaBackward0Class;
  addClass<UpsampleBilinear2DAaBackward0>(module, UpsampleBilinear2DAaBackward0Class, "UpsampleBilinear2DAaBackward0", UpsampleBilinear2DAaBackward0_properties);
  static PyTypeObject ChannelShuffleBackward0Class;
  addClass<ChannelShuffleBackward0>(module, ChannelShuffleBackward0Class, "ChannelShuffleBackward0", ChannelShuffleBackward0_properties);
  static PyTypeObject AdaptiveAvgPool2DBackward0Class;
  addClass<AdaptiveAvgPool2DBackward0>(module, AdaptiveAvgPool2DBackward0Class, "AdaptiveAvgPool2DBackward0", AdaptiveAvgPool2DBackward0_properties);
  static PyTypeObject AvgPool3DBackward0Class;
  addClass<AvgPool3DBackward0>(module, AvgPool3DBackward0Class, "AvgPool3DBackward0", AvgPool3DBackward0_properties);
  static PyTypeObject MaxPool3DWithIndicesBackward0Class;
  addClass<MaxPool3DWithIndicesBackward0>(module, MaxPool3DWithIndicesBackward0Class, "MaxPool3DWithIndicesBackward0", MaxPool3DWithIndicesBackward0_properties);
  static PyTypeObject ConvolutionBackwardBackward0Class;
  addClass<ConvolutionBackwardBackward0>(module, ConvolutionBackwardBackward0Class, "ConvolutionBackwardBackward0", ConvolutionBackwardBackward0_properties);
  static PyTypeObject ConvolutionOverrideableBackward0Class;
  addClass<ConvolutionOverrideableBackward0>(module, ConvolutionOverrideableBackward0Class, "ConvolutionOverrideableBackward0", ConvolutionOverrideableBackward0_properties);
  static PyTypeObject SlowConvTranspose3DBackward0Class;
  addClass<SlowConvTranspose3DBackward0>(module, SlowConvTranspose3DBackward0Class, "SlowConvTranspose3DBackward0", SlowConvTranspose3DBackward0_properties);
  static PyTypeObject SlowConv3DBackward0Class;
  addClass<SlowConv3DBackward0>(module, SlowConv3DBackward0Class, "SlowConv3DBackward0", SlowConv3DBackward0_properties);
  static PyTypeObject Im2ColBackward0Class;
  addClass<Im2ColBackward0>(module, Im2ColBackward0Class, "Im2ColBackward0", Im2ColBackward0_properties);
  static PyTypeObject AvgPool3DBackwardBackward0Class;
  addClass<AvgPool3DBackwardBackward0>(module, AvgPool3DBackwardBackward0Class, "AvgPool3DBackwardBackward0", AvgPool3DBackwardBackward0_properties);
  static PyTypeObject GluBackwardBackward0Class;
  addClass<GluBackwardBackward0>(module, GluBackwardBackward0Class, "GluBackwardBackward0", GluBackwardBackward0_properties);
  static PyTypeObject HardtanhBackwardBackward0Class;
  addClass<HardtanhBackwardBackward0>(module, HardtanhBackwardBackward0Class, "HardtanhBackwardBackward0", HardtanhBackwardBackward0_properties);
  static PyTypeObject LeakyReluBackwardBackward0Class;
  addClass<LeakyReluBackwardBackward0>(module, LeakyReluBackwardBackward0Class, "LeakyReluBackwardBackward0", LeakyReluBackwardBackward0_properties);
  static PyTypeObject MaxPool2DBackwardBackward0Class;
  addClass<MaxPool2DBackwardBackward0>(module, MaxPool2DBackwardBackward0Class, "MaxPool2DBackwardBackward0", MaxPool2DBackwardBackward0_properties);
  static PyTypeObject MaxPool2DWithIndicesBackwardBackward0Class;
  addClass<MaxPool2DWithIndicesBackwardBackward0>(module, MaxPool2DWithIndicesBackwardBackward0Class, "MaxPool2DWithIndicesBackwardBackward0", MaxPool2DWithIndicesBackwardBackward0_properties);
  static PyTypeObject UpsampleLinear1DBackwardBackward0Class;
  addClass<UpsampleLinear1DBackwardBackward0>(module, UpsampleLinear1DBackwardBackward0Class, "UpsampleLinear1DBackwardBackward0", UpsampleLinear1DBackwardBackward0_properties);
  static PyTypeObject UpsampleNearest1DBackwardBackward0Class;
  addClass<UpsampleNearest1DBackwardBackward0>(module, UpsampleNearest1DBackwardBackward0Class, "UpsampleNearest1DBackwardBackward0", UpsampleNearest1DBackwardBackward0_properties);
  static PyTypeObject UpsampleNearest2DBackwardBackward0Class;
  addClass<UpsampleNearest2DBackwardBackward0>(module, UpsampleNearest2DBackwardBackward0Class, "UpsampleNearest2DBackwardBackward0", UpsampleNearest2DBackwardBackward0_properties);
  static PyTypeObject MpsConvolutionTransposeBackward0Class;
  addClass<MpsConvolutionTransposeBackward0>(module, MpsConvolutionTransposeBackward0Class, "MpsConvolutionTransposeBackward0", MpsConvolutionTransposeBackward0_properties);
  static PyTypeObject CudnnBatchNormBackward0Class;
  addClass<CudnnBatchNormBackward0>(module, CudnnBatchNormBackward0Class, "CudnnBatchNormBackward0", CudnnBatchNormBackward0_properties);
  static PyTypeObject MiopenConvolutionTransposeBackward0Class;
  addClass<MiopenConvolutionTransposeBackward0>(module, MiopenConvolutionTransposeBackward0Class, "MiopenConvolutionTransposeBackward0", MiopenConvolutionTransposeBackward0_properties);
  static PyTypeObject MiopenBatchNormBackward0Class;
  addClass<MiopenBatchNormBackward0>(module, MiopenBatchNormBackward0Class, "MiopenBatchNormBackward0", MiopenBatchNormBackward0_properties);
  static PyTypeObject MkldnnAdaptiveAvgPool2DBackward0Class;
  addClass<MkldnnAdaptiveAvgPool2DBackward0>(module, MkldnnAdaptiveAvgPool2DBackward0Class, "MkldnnAdaptiveAvgPool2DBackward0", MkldnnAdaptiveAvgPool2DBackward0_properties);
  static PyTypeObject NestedTensorFromMaskBackward0Class;
  addClass<NestedTensorFromMaskBackward0>(module, NestedTensorFromMaskBackward0Class, "NestedTensorFromMaskBackward0", NestedTensorFromMaskBackward0_properties);
  static PyTypeObject NestedFromPaddedTensorBackward0Class;
  addClass<NestedFromPaddedTensorBackward0>(module, NestedFromPaddedTensorBackward0Class, "NestedFromPaddedTensorBackward0", NestedFromPaddedTensorBackward0_properties);
  static PyTypeObject NestedGetValuesBackward0Class;
  addClass<NestedGetValuesBackward0>(module, NestedGetValuesBackward0Class, "NestedGetValuesBackward0", NestedGetValuesBackward0_properties);
  static PyTypeObject ScaledDotProductFusedAttentionOverrideableBackward0Class;
  addClass<ScaledDotProductFusedAttentionOverrideableBackward0>(module, ScaledDotProductFusedAttentionOverrideableBackward0Class, "ScaledDotProductFusedAttentionOverrideableBackward0", ScaledDotProductFusedAttentionOverrideableBackward0_properties);
  static PyTypeObject FftC2RBackward0Class;
  addClass<FftC2RBackward0>(module, FftC2RBackward0Class, "FftC2RBackward0", FftC2RBackward0_properties);
  static PyTypeObject TestWarnInAutogradBackward0Class;
  addClass<TestWarnInAutogradBackward0>(module, TestWarnInAutogradBackward0Class, "TestWarnInAutogradBackward0", TestWarnInAutogradBackward0_properties);
  static PyTypeObject TestAutogradMultipleDispatchViewBackward0Class;
  addClass<TestAutogradMultipleDispatchViewBackward0>(module, TestAutogradMultipleDispatchViewBackward0Class, "TestAutogradMultipleDispatchViewBackward0", TestAutogradMultipleDispatchViewBackward0_properties);
  static PyTypeObject TestAutogradMultipleDispatchViewBackwardAutogradCUDA0Class;
  addClass<TestAutogradMultipleDispatchViewBackwardAutogradCUDA0>(module, TestAutogradMultipleDispatchViewBackwardAutogradCUDA0Class, "TestAutogradMultipleDispatchViewBackwardAutogradCUDA0", TestAutogradMultipleDispatchViewBackwardAutogradCUDA0_properties);
  static PyTypeObject ReshapeCopyBackward0Class;
  addClass<ReshapeCopyBackward0>(module, ReshapeCopyBackward0Class, "ReshapeCopyBackward0", ReshapeCopyBackward0_properties);
  static PyTypeObject SliceBackward0_copyClass;
  addClass<SliceBackward0_copy>(module, SliceBackward0_copyClass, "SliceBackward0_copy", SliceBackward0_copy_properties);
  static PyTypeObject SplitWithSizesBackward0_copyClass;
  addClass<SplitWithSizesBackward0_copy>(module, SplitWithSizesBackward0_copyClass, "SplitWithSizesBackward0_copy", SplitWithSizesBackward0_copy_properties);
  static PyTypeObject SplitWithSizesBackwardAutogradNestedTensor0_copyClass;
  addClass<SplitWithSizesBackwardAutogradNestedTensor0_copy>(module, SplitWithSizesBackwardAutogradNestedTensor0_copyClass, "SplitWithSizesBackwardAutogradNestedTensor0_copy", SplitWithSizesBackwardAutogradNestedTensor0_copy_properties);
  static PyTypeObject SqueezeBackward0_copyClass;
  addClass<SqueezeBackward0_copy>(module, SqueezeBackward0_copyClass, "SqueezeBackward0_copy", SqueezeBackward0_copy_properties);
  static PyTypeObject SqueezeBackward1_copyClass;
  addClass<SqueezeBackward1_copy>(module, SqueezeBackward1_copyClass, "SqueezeBackward1_copy", SqueezeBackward1_copy_properties);
  static PyTypeObject SqueezeBackwardAutogradNestedTensor0_copyClass;
  addClass<SqueezeBackwardAutogradNestedTensor0_copy>(module, SqueezeBackwardAutogradNestedTensor0_copyClass, "SqueezeBackwardAutogradNestedTensor0_copy", SqueezeBackwardAutogradNestedTensor0_copy_properties);
  static PyTypeObject SqueezeBackward2_copyClass;
  addClass<SqueezeBackward2_copy>(module, SqueezeBackward2_copyClass, "SqueezeBackward2_copy", SqueezeBackward2_copy_properties);
  static PyTypeObject SqueezeBackwardAutogradNestedTensor1_copyClass;
  addClass<SqueezeBackwardAutogradNestedTensor1_copy>(module, SqueezeBackwardAutogradNestedTensor1_copyClass, "SqueezeBackwardAutogradNestedTensor1_copy", SqueezeBackwardAutogradNestedTensor1_copy_properties);
  static PyTypeObject ValuesBackward0_copyClass;
  addClass<ValuesBackward0_copy>(module, ValuesBackward0_copyClass, "ValuesBackward0_copy", ValuesBackward0_copy_properties);
  static PyTypeObject ValuesBackwardAutogradNestedTensor0_copyClass;
  addClass<ValuesBackwardAutogradNestedTensor0_copy>(module, ValuesBackwardAutogradNestedTensor0_copyClass, "ValuesBackwardAutogradNestedTensor0_copy", ValuesBackwardAutogradNestedTensor0_copy_properties);
}

} // namespace torch::autograd::generated
