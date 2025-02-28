/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/runtime/contrib/tachikoma/tachikoma_utils.cc
 */

#include "tachikoma_utils.h"

#include "tvm/runtime/logging.h"

namespace tvm {
namespace runtime {
namespace contrib {

tachikoma::memory::data_type dtype_dl2tachikoma(DLDataType dltype) {
  using dt = tachikoma::memory::data_type;
  dt tachikoma_type = dt::undef;
  if (dltype.code == DataType::TypeCode::kFloat) {
    if (dltype.bits == 16) {
      tachikoma_type = dt::f16;
    } else if (dltype.bits == 32) {
      tachikoma_type = dt::f32;
    }
  } else if (dltype.code == DataType::TypeCode::kBFloat && dltype.bits == 16) {
    tachikoma_type = dt::bf16;
  } else if (dltype.code == DataType::TypeCode::kInt) {
    if (dltype.bits == 8) {
      tachikoma_type = dt::s8;
    } else if (dltype.bits == 32) {
      tachikoma_type = dt::s32;
    }
  } else if (dltype.code == DataType::TypeCode::kUInt && dltype.bits == 8) {
    tachikoma_type = dt::u8;
  }
  if (tachikoma_type == dt::undef) {
    LOG_ERROR << "unsupported datatype: code=" << dltype.code << ", bits=" << dltype.bits;
  }
  return tachikoma_type;
}

tachikoma::memory::dims shape_dl2tachikoma(const std::vector<int64_t>& shape) {
  if (shape.empty()) return {1};  // Tachikoma scalar representation is 1D tensor
  return shape;
}

tachikoma::memory::desc MakePlainDesc(const std::vector<int64_t>& shape, DLDataType dltype) {
  auto tachikoma_shape = shape_dl2tachikoma(shape);
  auto tachikoma_dtype = dtype_dl2tachikoma(dltype);

  auto tachikoma_plain_strides = tachikoma::memory::dims(tachikoma_shape.size(), 1);
  for (int i = tachikoma_shape.size() - 2; i >= 0; i--)
    tachikoma_plain_strides[i] = tachikoma_plain_strides[i + 1] * tachikoma_shape[i + 1];

  return {tachikoma_shape, tachikoma_dtype, tachikoma_plain_strides};
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm