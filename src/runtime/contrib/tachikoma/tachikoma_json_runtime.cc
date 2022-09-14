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
 * \file src/runtime/contrib/tachikoma/tachikoma_json_runtime.cc
 * \brief A simple JSON runtime for Tachikoma. Based on DNNL
 */

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>
#include <dmlc/io.h>
#include <fstream>

#include <cstddef>
#include <string>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"

// TODO(@liaopeiyuan): Have to mute warning from tachikoma headers.
//  -Wzero-as-null-pointer-constant and -Wdocumentation-unknown-command
#include <dnnl.hpp>

namespace tachikoma = dnnl;

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

class TachikomaJSONRuntime : public JSONRuntimeBase {
  using tag = tachikoma::memory::format_tag;
  using dt = tachikoma::memory::data_type;

 public:
  TachikomaJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                  const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  const char* type_key() const { return "tachikoma_json"; }

  void Init(const Array<NDArray>& consts) override {
    BuildEngine();

    ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";

    // Setup constants entries for weights.
    SetupConstants(consts);
  }

  void Run() override {
    // Fill in the input buffers.
    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      auto eid = EntryID(input_nodes_[i], 0);
      // TODO(@comaniac): Support other data lengths.
      size_t offset_in_bytes = entry_out_mem_[eid].second * 4;
      size_t buffer_size = GetDataSize(*data_entry_[eid]);
      write_to_tachikoma_memory(data_entry_[eid]->data, entry_out_mem_[eid].first, buffer_size,
                           offset_in_bytes);
    }

    // Invoke the engine through intepreting the stream.
    for (size_t i = 0; i < net_.size(); ++i) {
      net_.at(i).execute(stream_, net_args_.at(i));
    }
    stream_.wait();

    // Read output buffers.
    for (size_t i = 0; i < outputs_.size(); ++i) {
      auto eid = EntryID(outputs_[i]);
      size_t offset_in_bytes = entry_out_mem_[eid].second * 4;
      size_t buffer_size = GetDataSize(*data_entry_[eid]);
      read_from_tachikoma_memory(data_entry_[eid]->data, entry_out_mem_[eid].first, buffer_size,
                            offset_in_bytes);
    }

    std::cerr << data_entry_.size() << " vectors in total." << std::endl;
    
    for (size_t vector_id = 0; vector_id < data_entry_.size(); vector_id++) {
      const DLTensor* tensor = data_entry_[vector_id];
      std::string data;
      dmlc::MemoryStringStream writer(&data);
      dmlc::SeekStream* strm = &writer;
      std::string file_name = "/data/tachikoma_results/serialized.ndarray";
      if (tensor != nullptr) {
        SaveDLTensor(strm, tensor);
        std::ofstream fs(file_name, std::ios::out | std::ios::binary);
        ICHECK(!fs.fail()) << "Cannot open " << file_name;
        fs.write(&data[0], data.length());
      }
      std::cerr << (void*) data_entry_[vector_id] << " ";
    }
    std::cerr << std::endl;
    std::cerr << "Run complete." << std::endl;
  }

 private:
  // Build up the engine based on the input graph.
  void BuildEngine() {
    engine_ = tachikoma::engine(tachikoma::engine::kind::cpu, 0);
    stream_ = tachikoma::stream(engine_);

    // Build subgraph engine.
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "kernel") {
        ICHECK_EQ(node.GetOpType(), "kernel");
        auto op_name = node.GetOpName();
        if ("nn.conv2d" == op_name) {
          Conv2d(nid);
        } else if ("tachikoma.conv2d_relu" == op_name) {
          Conv2d(nid, true, false);
        } else if ("tachikoma.conv2d_bias_relu" == op_name) {
          Conv2d(nid, true, true);
        } else if ("nn.dense" == op_name) {
          Dense(nid);
        } else if ("nn.batch_norm" == op_name) {
          BatchNorm(nid);
        } else if ("nn.relu" == op_name) {
          Relu(nid);
        } else if ("add" == op_name) {
          Add(nid);
        } else {
          LOG(FATAL) << "Unsupported op: " << op_name;
        }
      }
    }
  }

  // Bind a JSON graph node entry to a Tachikoma memory.
  tachikoma::memory BindTachikomaMemory(const JSONGraphNodeEntry& entry, tachikoma::memory::desc mem_desc,
                              size_t offset = 0) {
    auto eid = EntryID(entry);
    if (entry_out_mem_.count(eid) == 0) {
      return BindTachikomaMemory(entry, tachikoma::memory(mem_desc, engine_), offset);
    }
    return entry_out_mem_[eid].first;
  }

  // Bind a JSON graph node entry to a given Tachikoma memory.
  tachikoma::memory BindTachikomaMemory(const JSONGraphNodeEntry& entry, tachikoma::memory mem,
                              size_t offset = 0) {
    auto eid = EntryID(entry);
    // Since the Tachikoma memory has been created before calling this function, we assume the entry
    // has not yet been bound to the other Tachikoma memory; otherwise it may have memory leak.
    ICHECK_EQ(entry_out_mem_.count(eid), 0);

    // TODO(@comanic): Support other data types (i.e., int8).
    auto data_node = nodes_[entry.id_];
    auto dltype = data_node.GetOpDataType()[entry.index_];
    ICHECK_EQ(dltype.bits, 32);

    entry_out_mem_[eid] = {mem, offset};
    return entry_out_mem_[eid].first;
  }

  void Conv2d(const size_t& nid, const bool has_relu = false, const bool has_bias = false) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    tachikoma::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    tachikoma::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];
    std::vector<std::string> str_strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> str_padding = node.GetAttr<std::vector<std::string>>("padding");
    tachikoma::memory::dim groups = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);

    tachikoma::memory::dim N = input_shape[0],       // batch size
        IC = input_shape[1],                    // input channels
        IH = input_shape[2],                    // input height
        IW = input_shape[2],                    // input width
        OC = weight_shape[0],                   // output channels
        KH = weight_shape[2],                   // weight height
        KW = weight_shape[3],                   // weight width
        PH_L = std::stoi(str_padding[1]),       // height padding: left
        PH_R = std::stoi(str_padding[3]),       // height padding: right
        PW_L = std::stoi(str_padding[0]),       // width padding: left
        PW_R = std::stoi(str_padding[2]),       // width padding: right
        SH = std::stoi(str_strides[0]),         // height-wise stride
        SW = std::stoi(str_strides[0]),         // weight-wise stride
        OH = (IH - KH + PH_L + PH_R) / SH + 1,  // output height
        OW = (IW - KW + PW_L + PW_R) / SW + 1;  // output width

    // Memory shapes.
    tachikoma::memory::dims src_dims = {N, IC, IH, IW};
    tachikoma::memory::dims weights_dims = {OC, IC, KH, KW};
    if (groups > 1) {
      weights_dims = {groups, 1, IC / groups, KH, KW};
    }
    tachikoma::memory::dims bias_dims = {OC};
    tachikoma::memory::dims dst_dims = {N, OC, OH, OW};
    tachikoma::memory::dims strides_dims = {SH, SW};
    tachikoma::memory::dims padding_dims_l = {PH_L, PW_L};
    tachikoma::memory::dims padding_dims_r = {PH_R, PW_R};

    // Memory descriptions.
    auto conv_src_md = tachikoma::memory::desc(src_dims, dt::f32, tag::any);
    auto conv_weights_md = tachikoma::memory::desc(weights_dims, dt::f32, tag::any);
    auto conv_bias_md = tachikoma::memory::desc(bias_dims, dt::f32, tag::any);
    auto conv_dst_md = tachikoma::memory::desc(dst_dims, dt::f32, tag::nchw);

    // Covn2d description.
    auto conv_desc = tachikoma::convolution_forward::desc(
        tachikoma::prop_kind::forward_inference, tachikoma::algorithm::convolution_direct, conv_src_md,
        conv_weights_md, conv_bias_md, conv_dst_md, strides_dims, padding_dims_l, padding_dims_r);

    // Enable ReLU
    tachikoma::primitive_attr attr;
    if (has_relu) {
      tachikoma::post_ops ops;
      ops.append_eltwise(1.f, tachikoma::algorithm::eltwise_relu, 0.f, 0.f);
      attr.set_post_ops(ops);
    }

    auto conv2d_prim_desc = tachikoma::convolution_forward::primitive_desc(conv_desc, attr, engine_);

    // Push to the network.
    auto conv = tachikoma::convolution_forward(conv2d_prim_desc);
    net_.push_back(conv);

    // Data memory.
    ICHECK_EQ(node.GetAttr<std::vector<std::string>>("data_layout")[0], "NCHW");
    auto conv2d_src_memory = BindTachikomaMemory(data_entry, {src_dims, dt::f32, tag::nchw});

    // Weight memory.
    ICHECK_EQ(node.GetAttr<std::vector<std::string>>("kernel_layout")[0], "OIHW");
    auto conv2d_weights_memory = BindTachikomaMemory(
        weight_entry, {weights_dims, dt::f32, (groups > 1) ? tag::goihw : tag::oihw});

    // Bias memory.
    auto conv2d_bias_memory = tachikoma::memory({bias_dims, dt::f32, tag::x}, engine_);
    if (has_bias) {
      auto bias_entry = node.GetInputs()[2];
      BindTachikomaMemory(bias_entry, conv2d_bias_memory);
    } else {
      float bias[OC] = {0};
      write_to_tachikoma_memory(bias, conv2d_bias_memory, OC * sizeof(float));
    }

    // Output memory.
    JSONGraphNodeEntry out_entry(nid, 0);
    auto conv2d_dst_memory = BindTachikomaMemory(out_entry, conv2d_prim_desc.dst_desc());

    // Bind memory buffers.
    net_args_.push_back({{DNNL_ARG_SRC, conv2d_src_memory},
                         {DNNL_ARG_WEIGHTS, conv2d_weights_memory},
                         {DNNL_ARG_BIAS, conv2d_bias_memory},
                         {DNNL_ARG_DST, conv2d_dst_memory}});
  }

  void Dense(const size_t& nid) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    tachikoma::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    tachikoma::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];

    tachikoma::memory::dim B = input_shape[0],  // batch size
        IC = input_shape[1],               // input channels
        OC = weight_shape[0];              // output channels

    // Memory shapes.
    tachikoma::memory::dims data_dims = {B, IC};
    tachikoma::memory::dims weight_dims = {OC, IC};
    tachikoma::memory::dims bias_dims = {OC};
    tachikoma::memory::dims out_dims = {B, OC};

    // Memory descriptions.
    auto data_md = tachikoma::memory::desc({data_dims, dt::f32, tag::nc});
    auto weight_md = tachikoma::memory::desc({weight_dims, dt::f32, tag::nc});
    auto bias_md = tachikoma::memory::desc({bias_dims, dt::f32, tag::x});
    auto dst_md = tachikoma::memory::desc({out_dims, dt::f32, tag::nc});

    // Dense description.
    auto dense_desc = tachikoma::inner_product_forward::desc(tachikoma::prop_kind::forward_inference, data_md,
                                                        weight_md, bias_md, dst_md);
    auto dense_prim_desc = tachikoma::inner_product_forward::primitive_desc(dense_desc, engine_);

    auto dense = tachikoma::inner_product_forward(dense_prim_desc);
    net_.push_back(dense);

    // Memories.
    auto data_memory = BindTachikomaMemory(data_entry, data_md);
    auto weight_memory = BindTachikomaMemory(weight_entry, weight_md);
    auto bias_memory = tachikoma::memory(bias_md, engine_);
    float bias[OC] = {0};
    write_to_tachikoma_memory(bias, bias_memory, OC * sizeof(float));
    JSONGraphNodeEntry out_entry(nid, 0);
    auto dst_memory = BindTachikomaMemory(out_entry, dense_prim_desc.dst_desc());

    net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                         {DNNL_ARG_WEIGHTS, weight_memory},
                         {DNNL_ARG_BIAS, bias_memory},
                         {DNNL_ARG_DST, dst_memory}});
  }

  void BatchNorm(const size_t& nid) {
    auto node = nodes_[nid];

    auto data_entry = node.GetInputs()[0];
    auto gamma_entry = node.GetInputs()[1];
    auto beta_entry = node.GetInputs()[2];
    auto mean_entry = node.GetInputs()[3];
    auto variance_entry = node.GetInputs()[4];
    tachikoma::memory::dims data_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    tachikoma::memory::dim IC = data_shape[1];
    float epsilon = std::stof(node.GetAttr<std::vector<std::string>>("epsilon")[0]);

    // Memory description.
    tachikoma::memory::desc data_md = GenTachikomaMemDescByShape(data_shape, dt::f32);

    // BN description.
    auto bn_desc = tachikoma::batch_normalization_forward::desc(
        tachikoma::prop_kind::forward_inference, data_md, epsilon,
        tachikoma::normalization_flags::use_global_stats | tachikoma::normalization_flags::use_scale_shift);
    auto bn_prim_desc = tachikoma::batch_normalization_forward::primitive_desc(bn_desc, engine_);
    auto bn = tachikoma::batch_normalization_forward(bn_prim_desc);
    net_.push_back(bn);

    // Memories.
    auto data_memory = BindTachikomaMemory(data_entry, data_md);
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindTachikomaMemory(out_entry, data_md);
    auto mean_memory = BindTachikomaMemory(mean_entry, bn_prim_desc.mean_desc());
    auto variance_memory = BindTachikomaMemory(variance_entry, bn_prim_desc.variance_desc());

    // In Tachikoma, weight is composed of gamma+beta, so we point them to the same Tachikoma memory but
    // assign an offset to beta data for runtime serialization.
    auto weight_memory = BindTachikomaMemory(gamma_entry, bn_prim_desc.weights_desc(), 0);
    BindTachikomaMemory(beta_entry, weight_memory, IC);

    net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                         {DNNL_ARG_DST, out_memory},
                         {DNNL_ARG_SCALE_SHIFT, weight_memory},
                         {DNNL_ARG_MEAN, mean_memory},
                         {DNNL_ARG_VARIANCE, variance_memory}});
  }

  void Relu(const size_t& nid) {
    auto node = nodes_[nid];

    auto data_entry = node.GetInputs()[0];
    tachikoma::memory::dims shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    auto data_md = tachikoma::memory::desc{{shape}, dt::f32, tag::abcd};

    auto relu_desc = tachikoma::eltwise_forward::desc(tachikoma::prop_kind::forward_inference,
                                                 tachikoma::algorithm::eltwise_relu, data_md, 0);
    auto relu_prim_desc = tachikoma::eltwise_forward::primitive_desc(relu_desc, engine_);
    ICHECK(data_md == relu_prim_desc.dst_desc());

    auto relu = tachikoma::eltwise_forward(relu_prim_desc);
    net_.push_back(relu);

    auto data_memory = BindTachikomaMemory(data_entry, data_md);
    auto out_md = tachikoma::memory::desc(shape, dt::f32, tag::abcd);
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindTachikomaMemory(out_entry, out_md);

    net_args_.push_back({{DNNL_ARG_SRC, data_memory}, {DNNL_ARG_DST, out_memory}});
  }

  void Add(const size_t& nid) {
    auto node = nodes_[nid];

    // Memory and compute description.
    std::vector<tachikoma::memory::dims> data_dims;
    std::vector<tachikoma::memory::desc> data_mds;
    std::vector<tachikoma::memory> data_memories;

    ICHECK_EQ(node.GetInputs().size(), 2U);
    for (auto entry : node.GetInputs()) {
      auto data_shape = nodes_[entry.id_].GetOpShape()[entry.index_];
      tachikoma::memory::desc data_md = GenTachikomaMemDescByShape(data_shape, dt::f32);

      data_dims.push_back(data_shape);
      data_mds.push_back(data_md);
      data_memories.push_back(BindTachikomaMemory(entry, data_md));
    }
    ICHECK(data_dims[0] == data_dims[1]);
    auto out_md = data_mds[0];
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindTachikomaMemory(out_entry, out_md);

    auto add_desc =
        tachikoma::binary::desc(tachikoma::algorithm::binary_add, data_mds[0], data_mds[1], out_md);
    auto add_prim_desc = tachikoma::binary::primitive_desc(add_desc, engine_);
    auto add = tachikoma::binary(add_prim_desc);
    net_.push_back(add);

    net_args_.push_back({{DNNL_ARG_SRC_0, data_memories[0]},
                         {DNNL_ARG_SRC_1, data_memories[1]},
                         {DNNL_ARG_DST, out_memory}});
  }

  // Read from Tachikoma memory (+offset) and write to the handle.
  inline void read_from_tachikoma_memory(void* handle, const tachikoma::memory& mem, size_t size,
                                    size_t offset = 0) {
    uint8_t* src = static_cast<uint8_t*>(mem.get_data_handle());
    std::copy(src + offset, src + offset + size, static_cast<uint8_t*>(handle));
  }

  // Read from the handle and write to Tachikoma memory (+offset).
  inline void write_to_tachikoma_memory(void* handle, const tachikoma::memory& mem, size_t size,
                                   size_t offset = 0) {
    uint8_t* dst = static_cast<uint8_t*>(mem.get_data_handle());
    std::copy(reinterpret_cast<uint8_t*>(handle), reinterpret_cast<uint8_t*>(handle) + size,
              dst + offset);
  }

  // Generate Tachikoma memory description and infer the data layout by the given shape.
  inline tachikoma::memory::desc GenTachikomaMemDescByShape(const tachikoma::memory::dims& shape, dt dtype) {
    tachikoma::memory::desc data_md;
    switch (shape.size()) {
      case 2:
        data_md = tachikoma::memory::desc({shape, dtype, tag::ab});
        break;
      case 3:
        data_md = tachikoma::memory::desc({shape, dtype, tag::abc});
        break;
      case 4:
        data_md = tachikoma::memory::desc({shape, dtype, tag::abcd});
        break;
      case 5:
        data_md = tachikoma::memory::desc({shape, dtype, tag::abcde});
        break;
      default:
        LOG(FATAL) << "Unsupported data shape dimension: " << shape.size();
        break;
    }
    return data_md;
  }

  /* The DNNL engine. */
  tachikoma::engine engine_;
  /* The DNNL stream. */
  tachikoma::stream stream_;
  /* The network layers that are represented in tachikoma primitives. */
  std::vector<tachikoma::primitive> net_;
  /* The memory that is consumed by arguments. */
  std::vector<std::unordered_map<int, tachikoma::memory>> net_args_;
  /* The entry ID to its corresponding output memory. */
  std::unordered_map<uint32_t, std::pair<tachikoma::memory, size_t>> entry_out_mem_;
};

runtime::Module TachikomaJSONRuntimeCreate(String symbol_name, String graph_json,
                                      const Array<String>& const_names) {
  auto n = make_object<TachikomaJSONRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.TachikomaJSONRuntimeCreate").set_body_typed(TachikomaJSONRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_tachikoma_json")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<TachikomaJSONRuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
