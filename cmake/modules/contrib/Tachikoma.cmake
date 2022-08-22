# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

if(IS_DIRECTORY ${USE_TACHIKOMA})
  find_library(EXTERN_LIBRARY_DNNL NAMES Tachikoma dnnl ${USE_TACHIKOMA}/lib/)
  if (EXTERN_LIBRARY_DNNL STREQUAL "EXTERN_LIBRARY_DNNL-NOTFOUND")
    message(WARNING "Cannot find DNNL library at ${USE_TACHIKOMA}.")
  else()
    add_definitions(-DUSE_JSON_RUNTIME=1)
    tvm_file_glob(GLOB TACHIKOMA_RELAY_CONTRIB_SRC src/relay/backend/contrib/tachikoma/*.cc)
    list(APPEND COMPILER_SRCS ${TACHIKOMA_RELAY_CONTRIB_SRC})

    list(APPEND TVM_RUNTIME_LINKER_LIBS ${EXTERN_LIBRARY_DNNL})
    tvm_file_glob(GLOB TACHIKOMA_CONTRIB_SRC src/runtime/contrib/tachikoma/tachikoma_json_runtime.cc
                                        src/runtime/contrib/tachikoma/tachikoma_utils.cc
                                        src/runtime/contrib/tachikoma/tachikoma.cc
                                        src/runtime/contrib/cblas/tachikoma_blas.cc)
    list(APPEND RUNTIME_SRCS ${TACHIKOMA_CONTRIB_SRC})
    message(STATUS "Build with Tachikoma JSON runtime: " ${EXTERN_LIBRARY_DNNL})
  endif()
elseif((USE_TACHIKOMA STREQUAL "ON") OR (USE_TACHIKOMA STREQUAL "JSON"))
  add_definitions(-DUSE_JSON_RUNTIME=1)
  tvm_file_glob(GLOB TACHIKOMA_RELAY_CONTRIB_SRC src/relay/backend/contrib/tachikoma/*.cc)
  list(APPEND COMPILER_SRCS ${TACHIKOMA_RELAY_CONTRIB_SRC})

  find_library(EXTERN_LIBRARY_DNNL Tachikoma)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${EXTERN_LIBRARY_DNNL})
  tvm_file_glob(GLOB TACHIKOMA_CONTRIB_SRC src/runtime/contrib/tachikoma/tachikoma_json_runtime.cc
                                      src/runtime/contrib/tachikoma/tachikoma_utils.cc
                                      src/runtime/contrib/tachikoma/tachikoma.cc
                                      src/runtime/contrib/cblas/tachikoma_blas.cc)
  list(APPEND RUNTIME_SRCS ${TACHIKOMA_CONTRIB_SRC})
  message(STATUS "Build with Tachikoma JSON runtime: " ${EXTERN_LIBRARY_DNNL})
elseif(USE_TACHIKOMA STREQUAL "C_SRC")
  tvm_file_glob(GLOB TACHIKOMA_RELAY_CONTRIB_SRC src/relay/backend/contrib/tachikoma/*.cc)
  list(APPEND COMPILER_SRCS ${TACHIKOMA_RELAY_CONTRIB_SRC})

  find_library(EXTERN_LIBRARY_DNNL Tachikoma)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${EXTERN_LIBRARY_DNNL})
  tvm_file_glob(GLOB TACHIKOMA_CONTRIB_SRC src/runtime/contrib/tachikoma/tachikoma.cc
                                      src/runtime/contrib/cblas/tachikoma_blas.cc)
  list(APPEND RUNTIME_SRCS ${TACHIKOMA_CONTRIB_SRC})
  message(STATUS "Build with Tachikoma C source module: " ${EXTERN_LIBRARY_DNNL})
elseif(USE_TACHIKOMA STREQUAL "OFF")
  # pass
else()
  message(FATAL_ERROR "Invalid option: USE_TACHIKOMA=" ${USE_TACHIKOMA})
endif()

