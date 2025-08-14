#include <Python.h>

#include <vector>
#include <map>
#include <string>

#include "torch/csrc/autograd/generated/python_return_types.h"
#include "torch/csrc/utils/structseq.h"
#include "torch/csrc/Exceptions.h"

namespace torch { namespace autograd { namespace generated {

PyTypeObject* get__fake_quantize_per_tensor_affine_cachemask_tensor_qparams_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"output", ""}, {"mask", ""},  {nullptr} };
    static PyTypeObject _fake_quantize_per_tensor_affine_cachemask_tensor_qparamsNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types._fake_quantize_per_tensor_affine_cachemask_tensor_qparams", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&_fake_quantize_per_tensor_affine_cachemask_tensor_qparamsNamedTuple, &desc);
        _fake_quantize_per_tensor_affine_cachemask_tensor_qparamsNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &_fake_quantize_per_tensor_affine_cachemask_tensor_qparamsNamedTuple;
}
PyTypeObject* get__fused_moving_avg_obs_fq_helper_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"output", ""}, {"mask", ""},  {nullptr} };
    static PyTypeObject _fused_moving_avg_obs_fq_helperNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types._fused_moving_avg_obs_fq_helper", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&_fused_moving_avg_obs_fq_helperNamedTuple, &desc);
        _fused_moving_avg_obs_fq_helperNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &_fused_moving_avg_obs_fq_helperNamedTuple;
}
PyTypeObject* get__linalg_det_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"result", ""}, {"LU", ""}, {"pivots", ""},  {nullptr} };
    static PyTypeObject _linalg_detNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types._linalg_det", nullptr, NamedTuple_fields, 3 };
    if (!is_initialized) {
        PyStructSequence_InitType(&_linalg_detNamedTuple, &desc);
        _linalg_detNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &_linalg_detNamedTuple;
}

PyTypeObject* get__linalg_det_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"result", ""}, {"LU", ""}, {"pivots", ""},  {nullptr} };
    static PyTypeObject _linalg_det_outNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types._linalg_det_out", nullptr, NamedTuple_fields, 3 };
    if (!is_initialized) {
        PyStructSequence_InitType(&_linalg_det_outNamedTuple1, &desc);
        _linalg_det_outNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &_linalg_det_outNamedTuple1;
}
PyTypeObject* get__linalg_eigh_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"eigenvalues", ""}, {"eigenvectors", ""},  {nullptr} };
    static PyTypeObject _linalg_eighNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types._linalg_eigh", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&_linalg_eighNamedTuple, &desc);
        _linalg_eighNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &_linalg_eighNamedTuple;
}

PyTypeObject* get__linalg_eigh_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"eigenvalues", ""}, {"eigenvectors", ""},  {nullptr} };
    static PyTypeObject _linalg_eigh_outNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types._linalg_eigh_out", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&_linalg_eigh_outNamedTuple1, &desc);
        _linalg_eigh_outNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &_linalg_eigh_outNamedTuple1;
}
PyTypeObject* get__linalg_slogdet_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"sign", ""}, {"logabsdet", ""}, {"LU", ""}, {"pivots", ""},  {nullptr} };
    static PyTypeObject _linalg_slogdetNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types._linalg_slogdet", nullptr, NamedTuple_fields, 4 };
    if (!is_initialized) {
        PyStructSequence_InitType(&_linalg_slogdetNamedTuple, &desc);
        _linalg_slogdetNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &_linalg_slogdetNamedTuple;
}

PyTypeObject* get__linalg_slogdet_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"sign", ""}, {"logabsdet", ""}, {"LU", ""}, {"pivots", ""},  {nullptr} };
    static PyTypeObject _linalg_slogdet_outNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types._linalg_slogdet_out", nullptr, NamedTuple_fields, 4 };
    if (!is_initialized) {
        PyStructSequence_InitType(&_linalg_slogdet_outNamedTuple1, &desc);
        _linalg_slogdet_outNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &_linalg_slogdet_outNamedTuple1;
}
PyTypeObject* get__linalg_solve_ex_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"result", ""}, {"LU", ""}, {"pivots", ""}, {"info", ""},  {nullptr} };
    static PyTypeObject _linalg_solve_exNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types._linalg_solve_ex", nullptr, NamedTuple_fields, 4 };
    if (!is_initialized) {
        PyStructSequence_InitType(&_linalg_solve_exNamedTuple, &desc);
        _linalg_solve_exNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &_linalg_solve_exNamedTuple;
}

PyTypeObject* get__linalg_solve_ex_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"result", ""}, {"LU", ""}, {"pivots", ""}, {"info", ""},  {nullptr} };
    static PyTypeObject _linalg_solve_ex_outNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types._linalg_solve_ex_out", nullptr, NamedTuple_fields, 4 };
    if (!is_initialized) {
        PyStructSequence_InitType(&_linalg_solve_ex_outNamedTuple1, &desc);
        _linalg_solve_ex_outNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &_linalg_solve_ex_outNamedTuple1;
}
PyTypeObject* get__linalg_svd_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"U", ""}, {"S", ""}, {"Vh", ""},  {nullptr} };
    static PyTypeObject _linalg_svdNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types._linalg_svd", nullptr, NamedTuple_fields, 3 };
    if (!is_initialized) {
        PyStructSequence_InitType(&_linalg_svdNamedTuple, &desc);
        _linalg_svdNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &_linalg_svdNamedTuple;
}

PyTypeObject* get__linalg_svd_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"U", ""}, {"S", ""}, {"Vh", ""},  {nullptr} };
    static PyTypeObject _linalg_svd_outNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types._linalg_svd_out", nullptr, NamedTuple_fields, 3 };
    if (!is_initialized) {
        PyStructSequence_InitType(&_linalg_svd_outNamedTuple1, &desc);
        _linalg_svd_outNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &_linalg_svd_outNamedTuple1;
}
PyTypeObject* get__lu_with_info_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"LU", ""}, {"pivots", ""}, {"info", ""},  {nullptr} };
    static PyTypeObject _lu_with_infoNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types._lu_with_info", nullptr, NamedTuple_fields, 3 };
    if (!is_initialized) {
        PyStructSequence_InitType(&_lu_with_infoNamedTuple, &desc);
        _lu_with_infoNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &_lu_with_infoNamedTuple;
}
PyTypeObject* get__scaled_dot_product_cudnn_attention_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"output", ""}, {"logsumexp", ""}, {"cum_seq_q", ""}, {"cum_seq_k", ""}, {"max_q", ""}, {"max_k", ""}, {"philox_seed", ""}, {"philox_offset", ""}, {"debug_attn_mask", ""},  {nullptr} };
    static PyTypeObject _scaled_dot_product_cudnn_attentionNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types._scaled_dot_product_cudnn_attention", nullptr, NamedTuple_fields, 9 };
    if (!is_initialized) {
        PyStructSequence_InitType(&_scaled_dot_product_cudnn_attentionNamedTuple, &desc);
        _scaled_dot_product_cudnn_attentionNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &_scaled_dot_product_cudnn_attentionNamedTuple;
}
PyTypeObject* get__scaled_dot_product_efficient_attention_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"output", ""}, {"log_sumexp", ""}, {"philox_seed", ""}, {"philox_offset", ""},  {nullptr} };
    static PyTypeObject _scaled_dot_product_efficient_attentionNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types._scaled_dot_product_efficient_attention", nullptr, NamedTuple_fields, 4 };
    if (!is_initialized) {
        PyStructSequence_InitType(&_scaled_dot_product_efficient_attentionNamedTuple, &desc);
        _scaled_dot_product_efficient_attentionNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &_scaled_dot_product_efficient_attentionNamedTuple;
}
PyTypeObject* get__scaled_dot_product_flash_attention_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"output", ""}, {"logsumexp", ""}, {"cum_seq_q", ""}, {"cum_seq_k", ""}, {"max_q", ""}, {"max_k", ""}, {"rng_state", ""}, {"unused", ""}, {"debug_attn_mask", ""},  {nullptr} };
    static PyTypeObject _scaled_dot_product_flash_attentionNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types._scaled_dot_product_flash_attention", nullptr, NamedTuple_fields, 9 };
    if (!is_initialized) {
        PyStructSequence_InitType(&_scaled_dot_product_flash_attentionNamedTuple, &desc);
        _scaled_dot_product_flash_attentionNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &_scaled_dot_product_flash_attentionNamedTuple;
}
PyTypeObject* get__scaled_dot_product_flash_attention_for_cpu_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"output", ""}, {"logsumexp", ""},  {nullptr} };
    static PyTypeObject _scaled_dot_product_flash_attention_for_cpuNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types._scaled_dot_product_flash_attention_for_cpu", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&_scaled_dot_product_flash_attention_for_cpuNamedTuple, &desc);
        _scaled_dot_product_flash_attention_for_cpuNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &_scaled_dot_product_flash_attention_for_cpuNamedTuple;
}
PyTypeObject* get__unpack_dual_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"primal", ""}, {"tangent", ""},  {nullptr} };
    static PyTypeObject _unpack_dualNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types._unpack_dual", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&_unpack_dualNamedTuple, &desc);
        _unpack_dualNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &_unpack_dualNamedTuple;
}
PyTypeObject* get_aminmax_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"min", ""}, {"max", ""},  {nullptr} };
    static PyTypeObject aminmaxNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.aminmax", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&aminmaxNamedTuple, &desc);
        aminmaxNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &aminmaxNamedTuple;
}

PyTypeObject* get_aminmax_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"min", ""}, {"max", ""},  {nullptr} };
    static PyTypeObject aminmax_outNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.aminmax_out", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&aminmax_outNamedTuple1, &desc);
        aminmax_outNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &aminmax_outNamedTuple1;
}
PyTypeObject* get_cummax_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"values", ""}, {"indices", ""},  {nullptr} };
    static PyTypeObject cummaxNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.cummax", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&cummaxNamedTuple, &desc);
        cummaxNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &cummaxNamedTuple;
}

PyTypeObject* get_cummax_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"values", ""}, {"indices", ""},  {nullptr} };
    static PyTypeObject cummax_outNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.cummax_out", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&cummax_outNamedTuple1, &desc);
        cummax_outNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &cummax_outNamedTuple1;
}
PyTypeObject* get_cummin_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"values", ""}, {"indices", ""},  {nullptr} };
    static PyTypeObject cumminNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.cummin", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&cumminNamedTuple, &desc);
        cumminNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &cumminNamedTuple;
}

PyTypeObject* get_cummin_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"values", ""}, {"indices", ""},  {nullptr} };
    static PyTypeObject cummin_outNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.cummin_out", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&cummin_outNamedTuple1, &desc);
        cummin_outNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &cummin_outNamedTuple1;
}
PyTypeObject* get_frexp_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"mantissa", ""}, {"exponent", ""},  {nullptr} };
    static PyTypeObject frexpNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.frexp", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&frexpNamedTuple, &desc);
        frexpNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &frexpNamedTuple;
}

PyTypeObject* get_frexp_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"mantissa", ""}, {"exponent", ""},  {nullptr} };
    static PyTypeObject frexp_outNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.frexp_out", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&frexp_outNamedTuple1, &desc);
        frexp_outNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &frexp_outNamedTuple1;
}
PyTypeObject* get_geqrf_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"a", ""}, {"tau", ""},  {nullptr} };
    static PyTypeObject geqrf_outNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.geqrf_out", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&geqrf_outNamedTuple, &desc);
        geqrf_outNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &geqrf_outNamedTuple;
}

PyTypeObject* get_geqrf_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"a", ""}, {"tau", ""},  {nullptr} };
    static PyTypeObject geqrfNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.geqrf", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&geqrfNamedTuple1, &desc);
        geqrfNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &geqrfNamedTuple1;
}
PyTypeObject* get_histogram_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"hist", ""}, {"bin_edges", ""},  {nullptr} };
    static PyTypeObject histogram_outNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.histogram_out", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&histogram_outNamedTuple, &desc);
        histogram_outNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &histogram_outNamedTuple;
}

PyTypeObject* get_histogram_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"hist", ""}, {"bin_edges", ""},  {nullptr} };
    static PyTypeObject histogramNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.histogram", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&histogramNamedTuple1, &desc);
        histogramNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &histogramNamedTuple1;
}
PyTypeObject* get_histogramdd_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"hist", ""}, {"bin_edges", ""},  {nullptr} };
    static PyTypeObject histogramddNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.histogramdd", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&histogramddNamedTuple, &desc);
        histogramddNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &histogramddNamedTuple;
}
PyTypeObject* get_kthvalue_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"values", ""}, {"indices", ""},  {nullptr} };
    static PyTypeObject kthvalueNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.kthvalue", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&kthvalueNamedTuple, &desc);
        kthvalueNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &kthvalueNamedTuple;
}

PyTypeObject* get_kthvalue_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"values", ""}, {"indices", ""},  {nullptr} };
    static PyTypeObject kthvalue_outNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.kthvalue_out", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&kthvalue_outNamedTuple1, &desc);
        kthvalue_outNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &kthvalue_outNamedTuple1;
}
PyTypeObject* get_linalg_cholesky_ex_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"L", ""}, {"info", ""},  {nullptr} };
    static PyTypeObject linalg_cholesky_exNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_cholesky_ex", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&linalg_cholesky_exNamedTuple, &desc);
        linalg_cholesky_exNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &linalg_cholesky_exNamedTuple;
}

PyTypeObject* get_linalg_cholesky_ex_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"L", ""}, {"info", ""},  {nullptr} };
    static PyTypeObject linalg_cholesky_ex_outNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_cholesky_ex_out", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&linalg_cholesky_ex_outNamedTuple1, &desc);
        linalg_cholesky_ex_outNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &linalg_cholesky_ex_outNamedTuple1;
}
PyTypeObject* get_linalg_eig_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"eigenvalues", ""}, {"eigenvectors", ""},  {nullptr} };
    static PyTypeObject linalg_eigNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_eig", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&linalg_eigNamedTuple, &desc);
        linalg_eigNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &linalg_eigNamedTuple;
}

PyTypeObject* get_linalg_eig_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"eigenvalues", ""}, {"eigenvectors", ""},  {nullptr} };
    static PyTypeObject linalg_eig_outNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_eig_out", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&linalg_eig_outNamedTuple1, &desc);
        linalg_eig_outNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &linalg_eig_outNamedTuple1;
}
PyTypeObject* get_linalg_eigh_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"eigenvalues", ""}, {"eigenvectors", ""},  {nullptr} };
    static PyTypeObject linalg_eighNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_eigh", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&linalg_eighNamedTuple, &desc);
        linalg_eighNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &linalg_eighNamedTuple;
}

PyTypeObject* get_linalg_eigh_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"eigenvalues", ""}, {"eigenvectors", ""},  {nullptr} };
    static PyTypeObject linalg_eigh_outNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_eigh_out", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&linalg_eigh_outNamedTuple1, &desc);
        linalg_eigh_outNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &linalg_eigh_outNamedTuple1;
}
PyTypeObject* get_linalg_inv_ex_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"inverse", ""}, {"info", ""},  {nullptr} };
    static PyTypeObject linalg_inv_exNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_inv_ex", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&linalg_inv_exNamedTuple, &desc);
        linalg_inv_exNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &linalg_inv_exNamedTuple;
}

PyTypeObject* get_linalg_inv_ex_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"inverse", ""}, {"info", ""},  {nullptr} };
    static PyTypeObject linalg_inv_ex_outNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_inv_ex_out", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&linalg_inv_ex_outNamedTuple1, &desc);
        linalg_inv_ex_outNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &linalg_inv_ex_outNamedTuple1;
}
PyTypeObject* get_linalg_ldl_factor_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"LD", ""}, {"pivots", ""},  {nullptr} };
    static PyTypeObject linalg_ldl_factorNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_ldl_factor", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&linalg_ldl_factorNamedTuple, &desc);
        linalg_ldl_factorNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &linalg_ldl_factorNamedTuple;
}

PyTypeObject* get_linalg_ldl_factor_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"LD", ""}, {"pivots", ""},  {nullptr} };
    static PyTypeObject linalg_ldl_factor_outNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_ldl_factor_out", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&linalg_ldl_factor_outNamedTuple1, &desc);
        linalg_ldl_factor_outNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &linalg_ldl_factor_outNamedTuple1;
}
PyTypeObject* get_linalg_ldl_factor_ex_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"LD", ""}, {"pivots", ""}, {"info", ""},  {nullptr} };
    static PyTypeObject linalg_ldl_factor_exNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_ldl_factor_ex", nullptr, NamedTuple_fields, 3 };
    if (!is_initialized) {
        PyStructSequence_InitType(&linalg_ldl_factor_exNamedTuple, &desc);
        linalg_ldl_factor_exNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &linalg_ldl_factor_exNamedTuple;
}

PyTypeObject* get_linalg_ldl_factor_ex_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"LD", ""}, {"pivots", ""}, {"info", ""},  {nullptr} };
    static PyTypeObject linalg_ldl_factor_ex_outNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_ldl_factor_ex_out", nullptr, NamedTuple_fields, 3 };
    if (!is_initialized) {
        PyStructSequence_InitType(&linalg_ldl_factor_ex_outNamedTuple1, &desc);
        linalg_ldl_factor_ex_outNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &linalg_ldl_factor_ex_outNamedTuple1;
}
PyTypeObject* get_linalg_lstsq_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"solution", ""}, {"residuals", ""}, {"rank", ""}, {"singular_values", ""},  {nullptr} };
    static PyTypeObject linalg_lstsqNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_lstsq", nullptr, NamedTuple_fields, 4 };
    if (!is_initialized) {
        PyStructSequence_InitType(&linalg_lstsqNamedTuple, &desc);
        linalg_lstsqNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &linalg_lstsqNamedTuple;
}

PyTypeObject* get_linalg_lstsq_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"solution", ""}, {"residuals", ""}, {"rank", ""}, {"singular_values", ""},  {nullptr} };
    static PyTypeObject linalg_lstsq_outNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_lstsq_out", nullptr, NamedTuple_fields, 4 };
    if (!is_initialized) {
        PyStructSequence_InitType(&linalg_lstsq_outNamedTuple1, &desc);
        linalg_lstsq_outNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &linalg_lstsq_outNamedTuple1;
}
PyTypeObject* get_linalg_lu_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"P", ""}, {"L", ""}, {"U", ""},  {nullptr} };
    static PyTypeObject linalg_luNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_lu", nullptr, NamedTuple_fields, 3 };
    if (!is_initialized) {
        PyStructSequence_InitType(&linalg_luNamedTuple, &desc);
        linalg_luNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &linalg_luNamedTuple;
}

PyTypeObject* get_linalg_lu_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"P", ""}, {"L", ""}, {"U", ""},  {nullptr} };
    static PyTypeObject linalg_lu_outNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_lu_out", nullptr, NamedTuple_fields, 3 };
    if (!is_initialized) {
        PyStructSequence_InitType(&linalg_lu_outNamedTuple1, &desc);
        linalg_lu_outNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &linalg_lu_outNamedTuple1;
}
PyTypeObject* get_linalg_lu_factor_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"LU", ""}, {"pivots", ""},  {nullptr} };
    static PyTypeObject linalg_lu_factorNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_lu_factor", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&linalg_lu_factorNamedTuple, &desc);
        linalg_lu_factorNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &linalg_lu_factorNamedTuple;
}

PyTypeObject* get_linalg_lu_factor_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"LU", ""}, {"pivots", ""},  {nullptr} };
    static PyTypeObject linalg_lu_factor_outNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_lu_factor_out", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&linalg_lu_factor_outNamedTuple1, &desc);
        linalg_lu_factor_outNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &linalg_lu_factor_outNamedTuple1;
}
PyTypeObject* get_linalg_lu_factor_ex_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"LU", ""}, {"pivots", ""}, {"info", ""},  {nullptr} };
    static PyTypeObject linalg_lu_factor_exNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_lu_factor_ex", nullptr, NamedTuple_fields, 3 };
    if (!is_initialized) {
        PyStructSequence_InitType(&linalg_lu_factor_exNamedTuple, &desc);
        linalg_lu_factor_exNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &linalg_lu_factor_exNamedTuple;
}

PyTypeObject* get_linalg_lu_factor_ex_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"LU", ""}, {"pivots", ""}, {"info", ""},  {nullptr} };
    static PyTypeObject linalg_lu_factor_ex_outNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_lu_factor_ex_out", nullptr, NamedTuple_fields, 3 };
    if (!is_initialized) {
        PyStructSequence_InitType(&linalg_lu_factor_ex_outNamedTuple1, &desc);
        linalg_lu_factor_ex_outNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &linalg_lu_factor_ex_outNamedTuple1;
}
PyTypeObject* get_linalg_qr_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"Q", ""}, {"R", ""},  {nullptr} };
    static PyTypeObject linalg_qrNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_qr", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&linalg_qrNamedTuple, &desc);
        linalg_qrNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &linalg_qrNamedTuple;
}

PyTypeObject* get_linalg_qr_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"Q", ""}, {"R", ""},  {nullptr} };
    static PyTypeObject linalg_qr_outNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_qr_out", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&linalg_qr_outNamedTuple1, &desc);
        linalg_qr_outNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &linalg_qr_outNamedTuple1;
}
PyTypeObject* get_linalg_slogdet_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"sign", ""}, {"logabsdet", ""},  {nullptr} };
    static PyTypeObject linalg_slogdetNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_slogdet", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&linalg_slogdetNamedTuple, &desc);
        linalg_slogdetNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &linalg_slogdetNamedTuple;
}

PyTypeObject* get_linalg_slogdet_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"sign", ""}, {"logabsdet", ""},  {nullptr} };
    static PyTypeObject linalg_slogdet_outNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_slogdet_out", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&linalg_slogdet_outNamedTuple1, &desc);
        linalg_slogdet_outNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &linalg_slogdet_outNamedTuple1;
}
PyTypeObject* get_linalg_solve_ex_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"result", ""}, {"info", ""},  {nullptr} };
    static PyTypeObject linalg_solve_exNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_solve_ex", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&linalg_solve_exNamedTuple, &desc);
        linalg_solve_exNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &linalg_solve_exNamedTuple;
}

PyTypeObject* get_linalg_solve_ex_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"result", ""}, {"info", ""},  {nullptr} };
    static PyTypeObject linalg_solve_ex_outNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_solve_ex_out", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&linalg_solve_ex_outNamedTuple1, &desc);
        linalg_solve_ex_outNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &linalg_solve_ex_outNamedTuple1;
}
PyTypeObject* get_linalg_svd_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"U", ""}, {"S", ""}, {"Vh", ""},  {nullptr} };
    static PyTypeObject linalg_svdNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_svd", nullptr, NamedTuple_fields, 3 };
    if (!is_initialized) {
        PyStructSequence_InitType(&linalg_svdNamedTuple, &desc);
        linalg_svdNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &linalg_svdNamedTuple;
}

PyTypeObject* get_linalg_svd_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"U", ""}, {"S", ""}, {"Vh", ""},  {nullptr} };
    static PyTypeObject linalg_svd_outNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_svd_out", nullptr, NamedTuple_fields, 3 };
    if (!is_initialized) {
        PyStructSequence_InitType(&linalg_svd_outNamedTuple1, &desc);
        linalg_svd_outNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &linalg_svd_outNamedTuple1;
}
PyTypeObject* get_lu_unpack_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"P", ""}, {"L", ""}, {"U", ""},  {nullptr} };
    static PyTypeObject lu_unpackNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.lu_unpack", nullptr, NamedTuple_fields, 3 };
    if (!is_initialized) {
        PyStructSequence_InitType(&lu_unpackNamedTuple, &desc);
        lu_unpackNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &lu_unpackNamedTuple;
}

PyTypeObject* get_lu_unpack_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"P", ""}, {"L", ""}, {"U", ""},  {nullptr} };
    static PyTypeObject lu_unpack_outNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.lu_unpack_out", nullptr, NamedTuple_fields, 3 };
    if (!is_initialized) {
        PyStructSequence_InitType(&lu_unpack_outNamedTuple1, &desc);
        lu_unpack_outNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &lu_unpack_outNamedTuple1;
}
PyTypeObject* get_max_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"values", ""}, {"indices", ""},  {nullptr} };
    static PyTypeObject maxNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.max", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&maxNamedTuple, &desc);
        maxNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &maxNamedTuple;
}

PyTypeObject* get_max_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"values", ""}, {"indices", ""},  {nullptr} };
    static PyTypeObject max_outNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.max_out", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&max_outNamedTuple1, &desc);
        max_outNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &max_outNamedTuple1;
}
PyTypeObject* get_median_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"values", ""}, {"indices", ""},  {nullptr} };
    static PyTypeObject medianNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.median", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&medianNamedTuple, &desc);
        medianNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &medianNamedTuple;
}

PyTypeObject* get_median_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"values", ""}, {"indices", ""},  {nullptr} };
    static PyTypeObject median_outNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.median_out", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&median_outNamedTuple1, &desc);
        median_outNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &median_outNamedTuple1;
}
PyTypeObject* get_min_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"values", ""}, {"indices", ""},  {nullptr} };
    static PyTypeObject minNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.min", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&minNamedTuple, &desc);
        minNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &minNamedTuple;
}

PyTypeObject* get_min_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"values", ""}, {"indices", ""},  {nullptr} };
    static PyTypeObject min_outNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.min_out", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&min_outNamedTuple1, &desc);
        min_outNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &min_outNamedTuple1;
}
PyTypeObject* get_mode_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"values", ""}, {"indices", ""},  {nullptr} };
    static PyTypeObject modeNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.mode", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&modeNamedTuple, &desc);
        modeNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &modeNamedTuple;
}

PyTypeObject* get_mode_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"values", ""}, {"indices", ""},  {nullptr} };
    static PyTypeObject mode_outNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.mode_out", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&mode_outNamedTuple1, &desc);
        mode_outNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &mode_outNamedTuple1;
}
PyTypeObject* get_nanmedian_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"values", ""}, {"indices", ""},  {nullptr} };
    static PyTypeObject nanmedianNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.nanmedian", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&nanmedianNamedTuple, &desc);
        nanmedianNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &nanmedianNamedTuple;
}

PyTypeObject* get_nanmedian_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"values", ""}, {"indices", ""},  {nullptr} };
    static PyTypeObject nanmedian_outNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.nanmedian_out", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&nanmedian_outNamedTuple1, &desc);
        nanmedian_outNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &nanmedian_outNamedTuple1;
}
PyTypeObject* get_qr_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"Q", ""}, {"R", ""},  {nullptr} };
    static PyTypeObject qr_outNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.qr_out", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&qr_outNamedTuple, &desc);
        qr_outNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &qr_outNamedTuple;
}

PyTypeObject* get_qr_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"Q", ""}, {"R", ""},  {nullptr} };
    static PyTypeObject qrNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.qr", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&qrNamedTuple1, &desc);
        qrNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &qrNamedTuple1;
}
PyTypeObject* get_slogdet_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"sign", ""}, {"logabsdet", ""},  {nullptr} };
    static PyTypeObject slogdetNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.slogdet", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&slogdetNamedTuple, &desc);
        slogdetNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &slogdetNamedTuple;
}

PyTypeObject* get_slogdet_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"sign", ""}, {"logabsdet", ""},  {nullptr} };
    static PyTypeObject slogdet_outNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.slogdet_out", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&slogdet_outNamedTuple1, &desc);
        slogdet_outNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &slogdet_outNamedTuple1;
}
PyTypeObject* get_sort_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"values", ""}, {"indices", ""},  {nullptr} };
    static PyTypeObject sort_outNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.sort_out", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&sort_outNamedTuple, &desc);
        sort_outNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &sort_outNamedTuple;
}

PyTypeObject* get_sort_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"values", ""}, {"indices", ""},  {nullptr} };
    static PyTypeObject sortNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.sort", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&sortNamedTuple1, &desc);
        sortNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &sortNamedTuple1;
}
PyTypeObject* get_svd_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"U", ""}, {"S", ""}, {"V", ""},  {nullptr} };
    static PyTypeObject svd_outNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.svd_out", nullptr, NamedTuple_fields, 3 };
    if (!is_initialized) {
        PyStructSequence_InitType(&svd_outNamedTuple, &desc);
        svd_outNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &svd_outNamedTuple;
}

PyTypeObject* get_svd_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"U", ""}, {"S", ""}, {"V", ""},  {nullptr} };
    static PyTypeObject svdNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.svd", nullptr, NamedTuple_fields, 3 };
    if (!is_initialized) {
        PyStructSequence_InitType(&svdNamedTuple1, &desc);
        svdNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &svdNamedTuple1;
}
PyTypeObject* get_topk_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"values", ""}, {"indices", ""},  {nullptr} };
    static PyTypeObject topk_outNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.topk_out", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&topk_outNamedTuple, &desc);
        topk_outNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &topk_outNamedTuple;
}

PyTypeObject* get_topk_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"values", ""}, {"indices", ""},  {nullptr} };
    static PyTypeObject topkNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.topk", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&topkNamedTuple1, &desc);
        topkNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &topkNamedTuple1;
}
PyTypeObject* get_triangular_solve_out_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"solution", ""}, {"cloned_coefficient", ""},  {nullptr} };
    static PyTypeObject triangular_solve_outNamedTuple;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.triangular_solve_out", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&triangular_solve_outNamedTuple, &desc);
        triangular_solve_outNamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &triangular_solve_outNamedTuple;
}

PyTypeObject* get_triangular_solve_structseq() {
    static PyStructSequence_Field NamedTuple_fields[] = { {"solution", ""}, {"cloned_coefficient", ""},  {nullptr} };
    static PyTypeObject triangular_solveNamedTuple1;
    static bool is_initialized = false;
    static PyStructSequence_Desc desc = { "torch.return_types.triangular_solve", nullptr, NamedTuple_fields, 2 };
    if (!is_initialized) {
        PyStructSequence_InitType(&triangular_solveNamedTuple1, &desc);
        triangular_solveNamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
        is_initialized = true;
    }
    return &triangular_solveNamedTuple1;
}

}}}

namespace torch::autograd {

static void addReturnType(
    PyObject* module,
    const char* name,
    PyTypeObject* type) {
  // hold onto the TypeObject for the unlikely case of user
  // deleting or overriding it.
  Py_INCREF(type);
  if (PyModule_AddObject(
          module,
          name,
          (PyObject*)type) != 0) {
    Py_DECREF(type);
    throw python_error();
  }
}

void initReturnTypes(PyObject* module) {
  static struct PyModuleDef def = {
      PyModuleDef_HEAD_INIT, "torch._C._return_types", nullptr, -1, {}};
  PyObject* return_types_module = PyModule_Create(&def);
  if (!return_types_module) {
    throw python_error();
  }

  addReturnType(return_types_module, "_fake_quantize_per_tensor_affine_cachemask_tensor_qparams", generated::get__fake_quantize_per_tensor_affine_cachemask_tensor_qparams_structseq());
  addReturnType(return_types_module, "_fused_moving_avg_obs_fq_helper", generated::get__fused_moving_avg_obs_fq_helper_structseq());
  addReturnType(return_types_module, "_linalg_det", generated::get__linalg_det_structseq());
  addReturnType(return_types_module, "_linalg_det_out", generated::get__linalg_det_out_structseq());
  addReturnType(return_types_module, "_linalg_eigh", generated::get__linalg_eigh_structseq());
  addReturnType(return_types_module, "_linalg_eigh_out", generated::get__linalg_eigh_out_structseq());
  addReturnType(return_types_module, "_linalg_slogdet", generated::get__linalg_slogdet_structseq());
  addReturnType(return_types_module, "_linalg_slogdet_out", generated::get__linalg_slogdet_out_structseq());
  addReturnType(return_types_module, "_linalg_solve_ex", generated::get__linalg_solve_ex_structseq());
  addReturnType(return_types_module, "_linalg_solve_ex_out", generated::get__linalg_solve_ex_out_structseq());
  addReturnType(return_types_module, "_linalg_svd", generated::get__linalg_svd_structseq());
  addReturnType(return_types_module, "_linalg_svd_out", generated::get__linalg_svd_out_structseq());
  addReturnType(return_types_module, "_lu_with_info", generated::get__lu_with_info_structseq());
  addReturnType(return_types_module, "_scaled_dot_product_cudnn_attention", generated::get__scaled_dot_product_cudnn_attention_structseq());
  addReturnType(return_types_module, "_scaled_dot_product_efficient_attention", generated::get__scaled_dot_product_efficient_attention_structseq());
  addReturnType(return_types_module, "_scaled_dot_product_flash_attention", generated::get__scaled_dot_product_flash_attention_structseq());
  addReturnType(return_types_module, "_scaled_dot_product_flash_attention_for_cpu", generated::get__scaled_dot_product_flash_attention_for_cpu_structseq());
  addReturnType(return_types_module, "_unpack_dual", generated::get__unpack_dual_structseq());
  addReturnType(return_types_module, "aminmax", generated::get_aminmax_structseq());
  addReturnType(return_types_module, "aminmax_out", generated::get_aminmax_out_structseq());
  addReturnType(return_types_module, "cummax", generated::get_cummax_structseq());
  addReturnType(return_types_module, "cummax_out", generated::get_cummax_out_structseq());
  addReturnType(return_types_module, "cummin", generated::get_cummin_structseq());
  addReturnType(return_types_module, "cummin_out", generated::get_cummin_out_structseq());
  addReturnType(return_types_module, "frexp", generated::get_frexp_structseq());
  addReturnType(return_types_module, "frexp_out", generated::get_frexp_out_structseq());
  addReturnType(return_types_module, "geqrf_out", generated::get_geqrf_out_structseq());
  addReturnType(return_types_module, "geqrf", generated::get_geqrf_structseq());
  addReturnType(return_types_module, "histogram_out", generated::get_histogram_out_structseq());
  addReturnType(return_types_module, "histogram", generated::get_histogram_structseq());
  addReturnType(return_types_module, "histogramdd", generated::get_histogramdd_structseq());
  addReturnType(return_types_module, "kthvalue", generated::get_kthvalue_structseq());
  addReturnType(return_types_module, "kthvalue_out", generated::get_kthvalue_out_structseq());
  addReturnType(return_types_module, "linalg_cholesky_ex", generated::get_linalg_cholesky_ex_structseq());
  addReturnType(return_types_module, "linalg_cholesky_ex_out", generated::get_linalg_cholesky_ex_out_structseq());
  addReturnType(return_types_module, "linalg_eig", generated::get_linalg_eig_structseq());
  addReturnType(return_types_module, "linalg_eig_out", generated::get_linalg_eig_out_structseq());
  addReturnType(return_types_module, "linalg_eigh", generated::get_linalg_eigh_structseq());
  addReturnType(return_types_module, "linalg_eigh_out", generated::get_linalg_eigh_out_structseq());
  addReturnType(return_types_module, "linalg_inv_ex", generated::get_linalg_inv_ex_structseq());
  addReturnType(return_types_module, "linalg_inv_ex_out", generated::get_linalg_inv_ex_out_structseq());
  addReturnType(return_types_module, "linalg_ldl_factor", generated::get_linalg_ldl_factor_structseq());
  addReturnType(return_types_module, "linalg_ldl_factor_out", generated::get_linalg_ldl_factor_out_structseq());
  addReturnType(return_types_module, "linalg_ldl_factor_ex", generated::get_linalg_ldl_factor_ex_structseq());
  addReturnType(return_types_module, "linalg_ldl_factor_ex_out", generated::get_linalg_ldl_factor_ex_out_structseq());
  addReturnType(return_types_module, "linalg_lstsq", generated::get_linalg_lstsq_structseq());
  addReturnType(return_types_module, "linalg_lstsq_out", generated::get_linalg_lstsq_out_structseq());
  addReturnType(return_types_module, "linalg_lu", generated::get_linalg_lu_structseq());
  addReturnType(return_types_module, "linalg_lu_out", generated::get_linalg_lu_out_structseq());
  addReturnType(return_types_module, "linalg_lu_factor", generated::get_linalg_lu_factor_structseq());
  addReturnType(return_types_module, "linalg_lu_factor_out", generated::get_linalg_lu_factor_out_structseq());
  addReturnType(return_types_module, "linalg_lu_factor_ex", generated::get_linalg_lu_factor_ex_structseq());
  addReturnType(return_types_module, "linalg_lu_factor_ex_out", generated::get_linalg_lu_factor_ex_out_structseq());
  addReturnType(return_types_module, "linalg_qr", generated::get_linalg_qr_structseq());
  addReturnType(return_types_module, "linalg_qr_out", generated::get_linalg_qr_out_structseq());
  addReturnType(return_types_module, "linalg_slogdet", generated::get_linalg_slogdet_structseq());
  addReturnType(return_types_module, "linalg_slogdet_out", generated::get_linalg_slogdet_out_structseq());
  addReturnType(return_types_module, "linalg_solve_ex", generated::get_linalg_solve_ex_structseq());
  addReturnType(return_types_module, "linalg_solve_ex_out", generated::get_linalg_solve_ex_out_structseq());
  addReturnType(return_types_module, "linalg_svd", generated::get_linalg_svd_structseq());
  addReturnType(return_types_module, "linalg_svd_out", generated::get_linalg_svd_out_structseq());
  addReturnType(return_types_module, "lu_unpack", generated::get_lu_unpack_structseq());
  addReturnType(return_types_module, "lu_unpack_out", generated::get_lu_unpack_out_structseq());
  addReturnType(return_types_module, "max", generated::get_max_structseq());
  addReturnType(return_types_module, "max_out", generated::get_max_out_structseq());
  addReturnType(return_types_module, "median", generated::get_median_structseq());
  addReturnType(return_types_module, "median_out", generated::get_median_out_structseq());
  addReturnType(return_types_module, "min", generated::get_min_structseq());
  addReturnType(return_types_module, "min_out", generated::get_min_out_structseq());
  addReturnType(return_types_module, "mode", generated::get_mode_structseq());
  addReturnType(return_types_module, "mode_out", generated::get_mode_out_structseq());
  addReturnType(return_types_module, "nanmedian", generated::get_nanmedian_structseq());
  addReturnType(return_types_module, "nanmedian_out", generated::get_nanmedian_out_structseq());
  addReturnType(return_types_module, "qr_out", generated::get_qr_out_structseq());
  addReturnType(return_types_module, "qr", generated::get_qr_structseq());
  addReturnType(return_types_module, "slogdet", generated::get_slogdet_structseq());
  addReturnType(return_types_module, "slogdet_out", generated::get_slogdet_out_structseq());
  addReturnType(return_types_module, "sort_out", generated::get_sort_out_structseq());
  addReturnType(return_types_module, "sort", generated::get_sort_structseq());
  addReturnType(return_types_module, "svd_out", generated::get_svd_out_structseq());
  addReturnType(return_types_module, "svd", generated::get_svd_structseq());
  addReturnType(return_types_module, "topk_out", generated::get_topk_out_structseq());
  addReturnType(return_types_module, "topk", generated::get_topk_structseq());
  addReturnType(return_types_module, "triangular_solve_out", generated::get_triangular_solve_out_structseq());
  addReturnType(return_types_module, "triangular_solve", generated::get_triangular_solve_structseq());

  // steals a reference to return_types on success
  if (PyModule_AddObject(module, "_return_types", return_types_module) != 0) {
    Py_DECREF(return_types_module);
    throw python_error();
  }
}

} // namespace torch::autograd
