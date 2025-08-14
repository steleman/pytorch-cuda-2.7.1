#include <torch/csrc/autograd/generated/ViewFuncs.h>

// @generated from ../tools/autograd/templates/ViewFuncs.cpp

using at::Tensor;
using at::Scalar;
using at::IntArrayRef;
using at::TensorList;

namespace torch::autograd::generated {

std::vector<c10::SymInt> _ConjViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  return symints;
}

size_t _ConjViewFunc::num_symints() const {
  return static_cast<size_t>(0);
}

void _ConjViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());

}

std::vector<at::Tensor> _ConjViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t _ConjViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void _ConjViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor _ConjViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::_conj::call(input_base);
}

std::unique_ptr<ViewFunc> _ConjViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<_ConjViewFunc>();
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> _IndicesViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  return symints;
}

size_t _IndicesViewFunc::num_symints() const {
  return static_cast<size_t>(0);
}

void _IndicesViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());

}

std::vector<at::Tensor> _IndicesViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t _IndicesViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void _IndicesViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor _IndicesViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::_indices::call(input_base);
}

std::unique_ptr<ViewFunc> _IndicesViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<_IndicesViewFunc>();
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> _NegViewViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  return symints;
}

size_t _NegViewViewFunc::num_symints() const {
  return static_cast<size_t>(0);
}

void _NegViewViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());

}

std::vector<at::Tensor> _NegViewViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t _NegViewViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void _NegViewViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor _NegViewViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::_neg_view::call(input_base);
}

std::unique_ptr<ViewFunc> _NegViewViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<_NegViewViewFunc>();
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> _NestedGetValuesViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  return symints;
}

size_t _NestedGetValuesViewFunc::num_symints() const {
  return static_cast<size_t>(0);
}

void _NestedGetValuesViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());

}

std::vector<at::Tensor> _NestedGetValuesViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t _NestedGetValuesViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void _NestedGetValuesViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor _NestedGetValuesViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::_nested_get_values::call(input_base);
}

std::unique_ptr<ViewFunc> _NestedGetValuesViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<_NestedGetValuesViewFunc>();
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> _NestedViewFromBufferViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  return symints;
}

size_t _NestedViewFromBufferViewFunc::num_symints() const {
  return static_cast<size_t>(0);
}

void _NestedViewFromBufferViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());

}

std::vector<at::Tensor> _NestedViewFromBufferViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  tensors.reserve(1 + 1 + 1);
  tensors.push_back(nested_size);
  tensors.push_back(nested_strides);
  tensors.push_back(offsets);
  return tensors;
}

size_t _NestedViewFromBufferViewFunc::num_tensors() const {
  return static_cast<size_t>(1 + 1 + 1);
}

void _NestedViewFromBufferViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());
  auto i = 0;
  nested_size = tensors[i];
  i += 1;
  nested_strides = tensors[i];
  i += 1;
  offsets = tensors[i];
}

at::Tensor _NestedViewFromBufferViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::_nested_view_from_buffer::call(input_base, nested_size, nested_strides, offsets);
}

std::unique_ptr<ViewFunc> _NestedViewFromBufferViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<_NestedViewFromBufferViewFunc>(nested_size, nested_strides, offsets);
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> _NestedViewFromJaggedViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  return symints;
}

size_t _NestedViewFromJaggedViewFunc::num_symints() const {
  return static_cast<size_t>(0);
}

void _NestedViewFromJaggedViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());

}

std::vector<at::Tensor> _NestedViewFromJaggedViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  tensors.reserve(1 + 1 + (lengths.has_value() ? 1 : 0) + (min_seqlen.has_value() ? 1 : 0) + (max_seqlen.has_value() ? 1 : 0));
  tensors.push_back(offsets);
  tensors.push_back(dummy);
  if(lengths.has_value()) tensors.insert(tensors.end(), *(lengths));
  if(min_seqlen.has_value()) tensors.insert(tensors.end(), *(min_seqlen));
  if(max_seqlen.has_value()) tensors.insert(tensors.end(), *(max_seqlen));
  return tensors;
}

size_t _NestedViewFromJaggedViewFunc::num_tensors() const {
  return static_cast<size_t>(1 + 1 + (lengths.has_value() ? 1 : 0) + (min_seqlen.has_value() ? 1 : 0) + (max_seqlen.has_value() ? 1 : 0));
}

void _NestedViewFromJaggedViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());
  auto i = 0;
  offsets = tensors[i];
  i += 1;
  dummy = tensors[i];
  i += 1;
  if(lengths.has_value()) lengths = tensors[i];
  i += (lengths.has_value() ? 1 : 0);
  if(min_seqlen.has_value()) min_seqlen = tensors[i];
  i += (min_seqlen.has_value() ? 1 : 0);
  if(max_seqlen.has_value()) max_seqlen = tensors[i];
}

at::Tensor _NestedViewFromJaggedViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::_nested_view_from_jagged::call(input_base, offsets, dummy, lengths, ragged_idx, min_seqlen, max_seqlen);
}

std::unique_ptr<ViewFunc> _NestedViewFromJaggedViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<_NestedViewFromJaggedViewFunc>(offsets, dummy, lengths, ragged_idx, min_seqlen, max_seqlen);
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> _ReshapeAliasViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  symints.reserve(size.size() + stride.size());
  symints.insert(symints.end(), size.begin(), size.end());
  symints.insert(symints.end(), stride.begin(), stride.end());
  return symints;
}

size_t _ReshapeAliasViewFunc::num_symints() const {
  return static_cast<size_t>(size.size() + stride.size());
}

void _ReshapeAliasViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());
  auto i = 0;
  std::copy(symints.begin() + i, symints.begin() + i + size.size(), size.begin());
  i += size.size();
  std::copy(symints.begin() + i, symints.begin() + i + stride.size(), stride.begin());
}

std::vector<at::Tensor> _ReshapeAliasViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t _ReshapeAliasViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void _ReshapeAliasViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor _ReshapeAliasViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::_reshape_alias::call(input_base, size, stride);
}

std::unique_ptr<ViewFunc> _ReshapeAliasViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<_ReshapeAliasViewFunc>(size, stride);
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> _TestAutogradMultipleDispatchViewViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  return symints;
}

size_t _TestAutogradMultipleDispatchViewViewFunc::num_symints() const {
  return static_cast<size_t>(0);
}

void _TestAutogradMultipleDispatchViewViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());

}

std::vector<at::Tensor> _TestAutogradMultipleDispatchViewViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t _TestAutogradMultipleDispatchViewViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void _TestAutogradMultipleDispatchViewViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor _TestAutogradMultipleDispatchViewViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::_test_autograd_multiple_dispatch_view::call(input_base);
}

std::unique_ptr<ViewFunc> _TestAutogradMultipleDispatchViewViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<_TestAutogradMultipleDispatchViewViewFunc>();
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> _ValuesViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  return symints;
}

size_t _ValuesViewFunc::num_symints() const {
  return static_cast<size_t>(0);
}

void _ValuesViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());

}

std::vector<at::Tensor> _ValuesViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t _ValuesViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void _ValuesViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor _ValuesViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::_values::call(input_base);
}

std::unique_ptr<ViewFunc> _ValuesViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<_ValuesViewFunc>();
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> AliasViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  return symints;
}

size_t AliasViewFunc::num_symints() const {
  return static_cast<size_t>(0);
}

void AliasViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());

}

std::vector<at::Tensor> AliasViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t AliasViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void AliasViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor AliasViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::alias::call(input_base);
}

std::unique_ptr<ViewFunc> AliasViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<AliasViewFunc>();
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> AsStridedViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  symints.reserve(size.size() + stride.size() + (storage_offset.has_value() ? 1 : 0));
  symints.insert(symints.end(), size.begin(), size.end());
  symints.insert(symints.end(), stride.begin(), stride.end());
  if(storage_offset.has_value()) symints.insert(symints.end(), *(storage_offset));
  return symints;
}

size_t AsStridedViewFunc::num_symints() const {
  return static_cast<size_t>(size.size() + stride.size() + (storage_offset.has_value() ? 1 : 0));
}

void AsStridedViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());
  auto i = 0;
  std::copy(symints.begin() + i, symints.begin() + i + size.size(), size.begin());
  i += size.size();
  std::copy(symints.begin() + i, symints.begin() + i + stride.size(), stride.begin());
  i += stride.size();
  if(storage_offset.has_value()) storage_offset = symints[i];
}

std::vector<at::Tensor> AsStridedViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t AsStridedViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void AsStridedViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor AsStridedViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::as_strided::call(input_base, size, stride, storage_offset);
}

std::unique_ptr<ViewFunc> AsStridedViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<AsStridedViewFunc>(size, stride, storage_offset);
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> CcolIndicesViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  return symints;
}

size_t CcolIndicesViewFunc::num_symints() const {
  return static_cast<size_t>(0);
}

void CcolIndicesViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());

}

std::vector<at::Tensor> CcolIndicesViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t CcolIndicesViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void CcolIndicesViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor CcolIndicesViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::ccol_indices::call(input_base);
}

std::unique_ptr<ViewFunc> CcolIndicesViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<CcolIndicesViewFunc>();
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> ChunkViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  return symints;
}

size_t ChunkViewFunc::num_symints() const {
  return static_cast<size_t>(0);
}

void ChunkViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());

}

std::vector<at::Tensor> ChunkViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t ChunkViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void ChunkViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor ChunkViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::chunk::call(input_base, chunks, dim)[view_idx];
}

std::unique_ptr<ViewFunc> ChunkViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<ChunkViewFunc>(chunks, dim, view_idx);
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> ColIndicesViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  return symints;
}

size_t ColIndicesViewFunc::num_symints() const {
  return static_cast<size_t>(0);
}

void ColIndicesViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());

}

std::vector<at::Tensor> ColIndicesViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t ColIndicesViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void ColIndicesViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor ColIndicesViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::col_indices::call(input_base);
}

std::unique_ptr<ViewFunc> ColIndicesViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<ColIndicesViewFunc>();
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> CrowIndicesViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  return symints;
}

size_t CrowIndicesViewFunc::num_symints() const {
  return static_cast<size_t>(0);
}

void CrowIndicesViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());

}

std::vector<at::Tensor> CrowIndicesViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t CrowIndicesViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void CrowIndicesViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor CrowIndicesViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::crow_indices::call(input_base);
}

std::unique_ptr<ViewFunc> CrowIndicesViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<CrowIndicesViewFunc>();
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> DiagonalViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  return symints;
}

size_t DiagonalViewFunc::num_symints() const {
  return static_cast<size_t>(0);
}

void DiagonalViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());

}

std::vector<at::Tensor> DiagonalViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t DiagonalViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void DiagonalViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor DiagonalViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::diagonal::call(input_base, offset, dim1, dim2);
}

std::unique_ptr<ViewFunc> DiagonalViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<DiagonalViewFunc>(offset, dim1, dim2);
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> ExpandViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  symints.reserve(size.size());
  symints.insert(symints.end(), size.begin(), size.end());
  return symints;
}

size_t ExpandViewFunc::num_symints() const {
  return static_cast<size_t>(size.size());
}

void ExpandViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());
  auto i = 0;
  std::copy(symints.begin() + i, symints.begin() + i + size.size(), size.begin());
}

std::vector<at::Tensor> ExpandViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t ExpandViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void ExpandViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor ExpandViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::expand::call(input_base, size, implicit);
}

std::unique_ptr<ViewFunc> ExpandViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<ExpandViewFunc>(size, implicit);
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> IndicesViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  return symints;
}

size_t IndicesViewFunc::num_symints() const {
  return static_cast<size_t>(0);
}

void IndicesViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());

}

std::vector<at::Tensor> IndicesViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t IndicesViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void IndicesViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor IndicesViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::indices::call(input_base);
}

std::unique_ptr<ViewFunc> IndicesViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<IndicesViewFunc>();
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> NarrowViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  symints.reserve(1 + 1);
  symints.push_back(start);
  symints.push_back(length);
  return symints;
}

size_t NarrowViewFunc::num_symints() const {
  return static_cast<size_t>(1 + 1);
}

void NarrowViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());
  auto i = 0;
  start = symints[i];
  i += 1;
  length = symints[i];
}

std::vector<at::Tensor> NarrowViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t NarrowViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void NarrowViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor NarrowViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::narrow::call(input_base, dim, start, length);
}

std::unique_ptr<ViewFunc> NarrowViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<NarrowViewFunc>(dim, start, length);
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> PermuteViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  return symints;
}

size_t PermuteViewFunc::num_symints() const {
  return static_cast<size_t>(0);
}

void PermuteViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());

}

std::vector<at::Tensor> PermuteViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t PermuteViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void PermuteViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor PermuteViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::permute::call(input_base, dims);
}

std::unique_ptr<ViewFunc> PermuteViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<PermuteViewFunc>(dims);
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> RowIndicesViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  return symints;
}

size_t RowIndicesViewFunc::num_symints() const {
  return static_cast<size_t>(0);
}

void RowIndicesViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());

}

std::vector<at::Tensor> RowIndicesViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t RowIndicesViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void RowIndicesViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor RowIndicesViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::row_indices::call(input_base);
}

std::unique_ptr<ViewFunc> RowIndicesViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<RowIndicesViewFunc>();
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> SelectIntViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  symints.reserve(1);
  symints.push_back(index);
  return symints;
}

size_t SelectIntViewFunc::num_symints() const {
  return static_cast<size_t>(1);
}

void SelectIntViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());
  auto i = 0;
  index = symints[i];
}

std::vector<at::Tensor> SelectIntViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t SelectIntViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void SelectIntViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor SelectIntViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::select_int::call(input_base, dim, index);
}

std::unique_ptr<ViewFunc> SelectIntViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<SelectIntViewFunc>(dim, index);
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> SliceTensorViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  symints.reserve((start.has_value() ? 1 : 0) + (end.has_value() ? 1 : 0) + 1);
  if(start.has_value()) symints.insert(symints.end(), *(start));
  if(end.has_value()) symints.insert(symints.end(), *(end));
  symints.push_back(step);
  return symints;
}

size_t SliceTensorViewFunc::num_symints() const {
  return static_cast<size_t>((start.has_value() ? 1 : 0) + (end.has_value() ? 1 : 0) + 1);
}

void SliceTensorViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());
  auto i = 0;
  if(start.has_value()) start = symints[i];
  i += (start.has_value() ? 1 : 0);
  if(end.has_value()) end = symints[i];
  i += (end.has_value() ? 1 : 0);
  step = symints[i];
}

std::vector<at::Tensor> SliceTensorViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t SliceTensorViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void SliceTensorViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor SliceTensorViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::slice_Tensor::call(input_base, dim, start, end, step);
}

std::unique_ptr<ViewFunc> SliceTensorViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<SliceTensorViewFunc>(dim, start, end, step);
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> SliceInverseViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  symints.reserve((start.has_value() ? 1 : 0) + (end.has_value() ? 1 : 0) + 1);
  if(start.has_value()) symints.insert(symints.end(), *(start));
  if(end.has_value()) symints.insert(symints.end(), *(end));
  symints.push_back(step);
  return symints;
}

size_t SliceInverseViewFunc::num_symints() const {
  return static_cast<size_t>((start.has_value() ? 1 : 0) + (end.has_value() ? 1 : 0) + 1);
}

void SliceInverseViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());
  auto i = 0;
  if(start.has_value()) start = symints[i];
  i += (start.has_value() ? 1 : 0);
  if(end.has_value()) end = symints[i];
  i += (end.has_value() ? 1 : 0);
  step = symints[i];
}

std::vector<at::Tensor> SliceInverseViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  tensors.reserve(1);
  tensors.push_back(src);
  return tensors;
}

size_t SliceInverseViewFunc::num_tensors() const {
  return static_cast<size_t>(1);
}

void SliceInverseViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());
  auto i = 0;
  src = tensors[i];
}

at::Tensor SliceInverseViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::slice_inverse::call(input_base, src, dim, start, end, step);
}

std::unique_ptr<ViewFunc> SliceInverseViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<SliceInverseViewFunc>(src, dim, start, end, step);
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> SplitTensorViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  symints.reserve(1);
  symints.push_back(split_size);
  return symints;
}

size_t SplitTensorViewFunc::num_symints() const {
  return static_cast<size_t>(1);
}

void SplitTensorViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());
  auto i = 0;
  split_size = symints[i];
}

std::vector<at::Tensor> SplitTensorViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t SplitTensorViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void SplitTensorViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor SplitTensorViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::split_Tensor::call(input_base, split_size, dim)[view_idx];
}

std::unique_ptr<ViewFunc> SplitTensorViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<SplitTensorViewFunc>(split_size, dim, view_idx);
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> SplitWithSizesViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  symints.reserve(split_sizes.size());
  symints.insert(symints.end(), split_sizes.begin(), split_sizes.end());
  return symints;
}

size_t SplitWithSizesViewFunc::num_symints() const {
  return static_cast<size_t>(split_sizes.size());
}

void SplitWithSizesViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());
  auto i = 0;
  std::copy(symints.begin() + i, symints.begin() + i + split_sizes.size(), split_sizes.begin());
}

std::vector<at::Tensor> SplitWithSizesViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t SplitWithSizesViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void SplitWithSizesViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor SplitWithSizesViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::split_with_sizes::call(input_base, split_sizes, dim)[view_idx];
}

std::unique_ptr<ViewFunc> SplitWithSizesViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<SplitWithSizesViewFunc>(split_sizes, dim, view_idx);
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> SqueezeViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  return symints;
}

size_t SqueezeViewFunc::num_symints() const {
  return static_cast<size_t>(0);
}

void SqueezeViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());

}

std::vector<at::Tensor> SqueezeViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t SqueezeViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void SqueezeViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor SqueezeViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::squeeze::call(input_base);
}

std::unique_ptr<ViewFunc> SqueezeViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<SqueezeViewFunc>();
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> SqueezeDimViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  return symints;
}

size_t SqueezeDimViewFunc::num_symints() const {
  return static_cast<size_t>(0);
}

void SqueezeDimViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());

}

std::vector<at::Tensor> SqueezeDimViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t SqueezeDimViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void SqueezeDimViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor SqueezeDimViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::squeeze_dim::call(input_base, dim);
}

std::unique_ptr<ViewFunc> SqueezeDimViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<SqueezeDimViewFunc>(dim);
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> SqueezeDimsViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  return symints;
}

size_t SqueezeDimsViewFunc::num_symints() const {
  return static_cast<size_t>(0);
}

void SqueezeDimsViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());

}

std::vector<at::Tensor> SqueezeDimsViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t SqueezeDimsViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void SqueezeDimsViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor SqueezeDimsViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::squeeze_dims::call(input_base, dim);
}

std::unique_ptr<ViewFunc> SqueezeDimsViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<SqueezeDimsViewFunc>(dim);
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> TViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  return symints;
}

size_t TViewFunc::num_symints() const {
  return static_cast<size_t>(0);
}

void TViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());

}

std::vector<at::Tensor> TViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t TViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void TViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor TViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::t::call(input_base);
}

std::unique_ptr<ViewFunc> TViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<TViewFunc>();
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> TransposeIntViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  return symints;
}

size_t TransposeIntViewFunc::num_symints() const {
  return static_cast<size_t>(0);
}

void TransposeIntViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());

}

std::vector<at::Tensor> TransposeIntViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t TransposeIntViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void TransposeIntViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor TransposeIntViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::transpose_int::call(input_base, dim0, dim1);
}

std::unique_ptr<ViewFunc> TransposeIntViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<TransposeIntViewFunc>(dim0, dim1);
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> UnbindIntViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  return symints;
}

size_t UnbindIntViewFunc::num_symints() const {
  return static_cast<size_t>(0);
}

void UnbindIntViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());

}

std::vector<at::Tensor> UnbindIntViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t UnbindIntViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void UnbindIntViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor UnbindIntViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::unbind_int::call(input_base, dim)[view_idx];
}

std::unique_ptr<ViewFunc> UnbindIntViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<UnbindIntViewFunc>(dim, view_idx);
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> UnfoldViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  return symints;
}

size_t UnfoldViewFunc::num_symints() const {
  return static_cast<size_t>(0);
}

void UnfoldViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());

}

std::vector<at::Tensor> UnfoldViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t UnfoldViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void UnfoldViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor UnfoldViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::unfold::call(input_base, dimension, size, step);
}

std::unique_ptr<ViewFunc> UnfoldViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<UnfoldViewFunc>(dimension, size, step);
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> UnsqueezeViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  return symints;
}

size_t UnsqueezeViewFunc::num_symints() const {
  return static_cast<size_t>(0);
}

void UnsqueezeViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());

}

std::vector<at::Tensor> UnsqueezeViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t UnsqueezeViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void UnsqueezeViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor UnsqueezeViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::unsqueeze::call(input_base, dim);
}

std::unique_ptr<ViewFunc> UnsqueezeViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<UnsqueezeViewFunc>(dim);
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> ValuesViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  return symints;
}

size_t ValuesViewFunc::num_symints() const {
  return static_cast<size_t>(0);
}

void ValuesViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());

}

std::vector<at::Tensor> ValuesViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t ValuesViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void ValuesViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor ValuesViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::values::call(input_base);
}

std::unique_ptr<ViewFunc> ValuesViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<ValuesViewFunc>();
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> ViewViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  symints.reserve(size.size());
  symints.insert(symints.end(), size.begin(), size.end());
  return symints;
}

size_t ViewViewFunc::num_symints() const {
  return static_cast<size_t>(size.size());
}

void ViewViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());
  auto i = 0;
  std::copy(symints.begin() + i, symints.begin() + i + size.size(), size.begin());
}

std::vector<at::Tensor> ViewViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t ViewViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void ViewViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor ViewViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::view::call(input_base, size);
}

std::unique_ptr<ViewFunc> ViewViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<ViewViewFunc>(size);
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> ViewDtypeViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  return symints;
}

size_t ViewDtypeViewFunc::num_symints() const {
  return static_cast<size_t>(0);
}

void ViewDtypeViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());

}

std::vector<at::Tensor> ViewDtypeViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t ViewDtypeViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void ViewDtypeViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor ViewDtypeViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::view_dtype::call(input_base, dtype);
}

std::unique_ptr<ViewFunc> ViewDtypeViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<ViewDtypeViewFunc>(dtype);
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> ViewAsComplexViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  return symints;
}

size_t ViewAsComplexViewFunc::num_symints() const {
  return static_cast<size_t>(0);
}

void ViewAsComplexViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());

}

std::vector<at::Tensor> ViewAsComplexViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t ViewAsComplexViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void ViewAsComplexViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor ViewAsComplexViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::view_as_complex::call(input_base);
}

std::unique_ptr<ViewFunc> ViewAsComplexViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<ViewAsComplexViewFunc>();
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

std::vector<c10::SymInt> ViewAsRealViewFunc::get_symints() const {
  ::std::vector<c10::SymInt> symints;
  return symints;
}

size_t ViewAsRealViewFunc::num_symints() const {
  return static_cast<size_t>(0);
}

void ViewAsRealViewFunc::set_symints(std::vector<c10::SymInt> symints) {
  TORCH_INTERNAL_ASSERT(symints.size() == num_symints());

}

std::vector<at::Tensor> ViewAsRealViewFunc::get_tensors() const {
  ::std::vector<at::Tensor> tensors;
  return tensors;
}

size_t ViewAsRealViewFunc::num_tensors() const {
  return static_cast<size_t>(0);
}

void ViewAsRealViewFunc::set_tensors(std::vector<at::Tensor> tensors) {
  TORCH_INTERNAL_ASSERT(tensors.size() == num_tensors());

}

at::Tensor ViewAsRealViewFunc::operator()(const at::Tensor& input_base) const {
  return at::_ops::view_as_real::call(input_base);
}

std::unique_ptr<ViewFunc> ViewAsRealViewFunc::clone_and_set(
    std::optional<std::vector<c10::SymInt>> symints,
    std::optional<std::vector<at::Tensor>> tensors) const {
  auto output = std::make_unique<ViewAsRealViewFunc>();
  if (symints.has_value()) {
    output->set_symints(std::move(*(symints)));
  }
  if (tensors.has_value()) {
    output->set_tensors(std::move(*(tensors)));
  }
  return output;
}

} // namespace torch::autograd::generated
