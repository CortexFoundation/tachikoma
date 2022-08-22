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
 * \brief Some Tachikoma specific utility functions
 */

#ifndef TVM_RUNTIME_CONTRIB_TACHIKOMA_TACHIKOMA_UTILS_H_
#define TVM_RUNTIME_CONTRIB_TACHIKOMA_TACHIKOMA_UTILS_H_

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

// TODO(@apeskov): Have to mute warning from tachikoma headers.
//  -Wzero-as-null-pointer-constant and -Wdocumentation-unknown-command
#include <dnnl.hpp>

namespace tachikoma = dnnl;

#include "tvm/runtime/data_type.h"

namespace tvm {
namespace runtime {
namespace contrib {

/*!
 * \brief Convert a DLPack data type to a Tachikoma data type.
 * \param dltype The DLPack data type.
 * \return The corresponding Tachikoma data type.
 */
tachikoma::memory::data_type dtype_dl2tachikoma(DLDataType dltype);

/*!
 * \brief Converter TVM shape to Tachikoma dims
 * \param shape tvm shape
 * \return dims in terms of tachikoma
 */
tachikoma::memory::dims shape_dl2tachikoma(const std::vector<int64_t>& shape);

/*!
 * \brief Construct plain tensor descriptor
 * \param shape provided shape
 * \param dltype provided data type
 * \return resulting plain tensor desc
 */
tachikoma::memory::desc MakePlainDesc(const std::vector<int64_t>& shape, DLDataType dltype);

namespace utils {

/*! \brief Pretty printer util for shape */
inline std::ostream& operator<<(std::ostream& o, const tachikoma::memory::dims& dims) {
  o << "[";
  auto d = dims.begin();
  if (d != dims.end()) o << *d++;
  while (d != dims.end()) o << "," << *d++;
  o << "]";
  return o;
}

/*! \brief Pretty printer util for data type */
inline std::ostream& operator<<(std::ostream& o, const tachikoma::memory::data_type& type) {
  std::string name = "undef";
  switch (type) {
    case tachikoma::memory::data_type::undef:
      name = "undef";
      break;
    case tachikoma::memory::data_type::f32:
      name = "fp32";
      break;
    case tachikoma::memory::data_type::f16:
      name = "fp16";
      break;
    case tachikoma::memory::data_type::bf16:
      name = "bf16";
      break;
    case tachikoma::memory::data_type::s32:
      name = "i32";
      break;
    case tachikoma::memory::data_type::s8:
      name = "i8";
      break;
    case tachikoma::memory::data_type::u8:
      name = "u8";
      break;
  }
  o << name;
  return o;
}

/*! \brief Converter data type template arg to runtime object */
template <typename T>
inline tachikoma::memory::data_type TachikomaDType();

template <>
inline tachikoma::memory::data_type TachikomaDType<int>() {
  return tachikoma::memory::data_type::s32;
}

template <>
inline tachikoma::memory::data_type TachikomaDType<float>() {
  return tachikoma::memory::data_type::f32;
}

template <>
inline tachikoma::memory::data_type TachikomaDType<uint8_t>() {
  return tachikoma::memory::data_type::u8;
}

template <>
inline tachikoma::memory::data_type TachikomaDType<int8_t>() {
  return tachikoma::memory::data_type::s8;
}

}  // namespace utils
}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_TACHIKOMA_TACHIKOMA_UTILS_H_
