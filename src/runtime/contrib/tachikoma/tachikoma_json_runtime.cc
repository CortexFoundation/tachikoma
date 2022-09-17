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
#include <regex>
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
  }

  /* Override GetFunction to reimplement Run method */
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) override {
    if (name == "get_symbol") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->symbol_name_; });
    } else if (name == "get_const_vars") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->const_names_; });
    } else if (name == "serialize_computational_trace") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        auto d = this->data_entry_;
        auto n = this->net_args_;
        std::cerr << d.size() << " vectors in total." << std::endl;
        std::cerr << n.size() << " net_args in total." << std::endl;
        for (size_t vector_id = 0; vector_id < d.size(); vector_id++) {
          const DLTensor* tensor = d[vector_id];
          std::string data;
          dmlc::MemoryStringStream writer(&data);
          dmlc::SeekStream* strm = &writer;
          std::string file_name = args[0];
          file_name = file_name + "_" + std::to_string(vector_id);
          if (tensor != nullptr) {
            SaveDLTensor(strm, tensor);
            std::ofstream fs(file_name, std::ios::out | std::ios::binary);
            ICHECK(!fs.fail()) << "Cannot open " << file_name;
            fs.write(&data[0], data.length());
          }
          std::cerr << (void*) d[vector_id] << " ";
        }
        std::cerr << std::endl;
        std::cerr << "Export complete." << std::endl;
      });
    } else if (this->symbol_name_ == name) {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        ICHECK(this->initialized_) << "The module has not been initialized";

        // Bind argument tensors to data entries.
        this->SetInputOutputBuffers(args);
        // Execute the subgraph.
        this->Run();
      });
    } else if ("__init_" + this->symbol_name_ == name) {
      // The function to initialize constant tensors.
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.size(), 1U);
        std::lock_guard<std::mutex> guard(this->initialize_mutex_);
        if (!this->initialized_) {
          this->Init(args[0]);
          this->initialized_ = true;
        }
        *rv = 0;
      });
    } else {
      return PackedFunc(nullptr);
    }
  }

 private:
  // Build up the engine based on the input graph.
  std::map<std::string, tag> layout_dict{
      {"NCW", tag::ncw},       {"OIW", tag::oiw},     {"GOIW", tag::goiw},   {"NCHW", tag::nchw},
      {"OIHW", tag::oihw},     {"GOIHW", tag::goihw}, {"NCDHW", tag::ncdhw}, {"OIDHW", tag::oidhw},
      {"GOIDHW", tag::goidhw}, {"IOHW", tag::iohw},   {"GIOHW", tag::giohw}, {"IODHW", tag::iodhw},
      {"GIODHW", tag::giodhw},
  };

  std::map<std::string, tachikoma::algorithm> elt_name2algo{
      {"abs", tachikoma::algorithm::eltwise_abs},
      {"exp", tachikoma::algorithm::eltwise_exp},
      {"log", tachikoma::algorithm::eltwise_log},
      {"sqrt", tachikoma::algorithm::eltwise_sqrt},
      {"round", tachikoma::algorithm::eltwise_round},
      {"logsumexp", tachikoma::algorithm::eltwise_logsigmoid},
      {"nn.relu", tachikoma::algorithm::eltwise_relu},
      {"nn.leaky_relu", tachikoma::algorithm::eltwise_relu},
      {"tanh", tachikoma::algorithm::eltwise_tanh},
      {"sigmoid", tachikoma::algorithm::eltwise_logistic},
      {"clip", tachikoma::algorithm::eltwise_clip},
  };

  bool ParsingOpName(const std::string op_name, tachikoma::primitive_attr attr) {
    // Define RegExp.
    std::regex bias_add_pat(".*_bias.*");
    std::regex relu_pat(".*_relu.*");
    std::regex tanh_pat(".*_tanh.*");
    std::regex sigmoid_pat(".*_sigmoid.*");

    // Parsing post-ops.
    tachikoma::post_ops ops;
    if (std::regex_match(op_name, relu_pat)) {
      ops.append_eltwise(1.f, tachikoma::algorithm::eltwise_relu, 0.f, 0.f);
    }
    if (std::regex_match(op_name, tanh_pat)) {
      ops.append_eltwise(1.f, tachikoma::algorithm::eltwise_tanh, 0.f, 0.f);
    }
    if (std::regex_match(op_name, sigmoid_pat)) {
      ops.append_eltwise(1.f, tachikoma::algorithm::eltwise_logistic, 0.f, 0.f);
    }
    attr.set_post_ops(ops);

    // Parsing bias_add.
    return std::regex_match(op_name, bias_add_pat) ? true : false;
  }

  tachikoma::memory::dims TransformStr2Dims(std::vector<std::string> strs, std::string str_name) {
    tachikoma::memory::dims out_dims;
    if (str_name == "dilates") {
      std::transform(strs.begin(), strs.end(), std::back_inserter(out_dims),
                     [](const std::string& str) { return std::stoi(str) - 1; });
    } else {
      std::transform(strs.begin(), strs.end(), std::back_inserter(out_dims),
                     [](const std::string& str) { return std::stoi(str); });
    }
    return out_dims;
  }

  void BuildEngine() {
    engine_ = tachikoma::engine(tachikoma::engine::kind::cpu, 0);
    stream_ = tachikoma::stream(engine_);

    std::regex conv_pat(".*conv[1-3]d.*");
    std::regex conv_tranpose_pat(".*conv[1-3]d_transpose.*");
    std::regex dense_pat(".*dense.*");
    std::regex max_pool_pat(".*max_pool[1-3]d");
    std::regex avg_pool_pat(".*avg_pool[1-3]d");

    // Build subgraph engine.
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "kernel") {
        ICHECK_EQ(node.GetOpType(), "kernel");
        auto op_name = node.GetOpName();
        if (std::regex_match(op_name, conv_tranpose_pat)) {
          Deconvolution(nid);
        } else if (std::regex_match(op_name, conv_pat)) {
          Convolution(nid);
        } else if (std::regex_match(op_name, dense_pat)) {
          Dense(nid);
        } else if ("nn.batch_norm" == op_name) {
          BatchNorm(nid);
        } else if (std::regex_match(op_name, max_pool_pat)) {
          Pooling(nid, tachikoma::algorithm::pooling_max);
        } else if (std::regex_match(op_name, avg_pool_pat)) {
          Pooling(nid, tachikoma::algorithm::pooling_avg);
        } else if (elt_name2algo.count(op_name)) {
          Eltwise(nid);
        } else if ("nn.softmax" == op_name) {
          Softmax(nid);
        } else if ("add" == op_name) {
          Binary(nid, tachikoma::algorithm::binary_add);
        } else if ("multiply" == op_name) {
          Binary(nid, tachikoma::algorithm::binary_mul);
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

  void Convolution(const size_t& nid) {
    auto node = nodes_[nid];
    auto op_name = node.GetOpName();
    tachikoma::primitive_attr attr;
    bool has_bias = ParsingOpName(op_name, attr);

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    JSONGraphNodeEntry out_entry(nid, 0);
    tachikoma::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    tachikoma::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];
    tachikoma::memory::dims out_shape = nodes_[out_entry.id_].GetOpShape()[out_entry.index_];
    tachikoma::memory::dim channels =
        node.GetAttr<std::vector<std::string>>("channels")[0] != ""
            ? std::stoi(node.GetAttr<std::vector<std::string>>("channels")[0])
            : out_shape[1];
    std::vector<std::string> str_strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> str_dilates = node.GetAttr<std::vector<std::string>>("dilation");
    std::vector<std::string> str_padding = node.GetAttr<std::vector<std::string>>("padding");
    std::vector<std::string> str_padding_l(str_padding.begin(),
                                           str_padding.begin() + str_padding.size() / 2);
    std::vector<std::string> str_padding_r(str_padding.end() - str_padding.size() / 2,
                                           str_padding.end());
    tachikoma::memory::dim groups = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);
    std::string data_layout = node.GetAttr<std::vector<std::string>>("data_layout")[0];
    std::string kernel_layout = node.GetAttr<std::vector<std::string>>("kernel_layout")[0];

    // Check layout.
    if (layout_dict.find(data_layout) == layout_dict.end() ||
        layout_dict.find(kernel_layout) == layout_dict.end()) {
      LOG(FATAL) << "Unsupported layout for conv: " << data_layout << " " << kernel_layout;
    }

    // Memory shapes.
    tachikoma::memory::dims src_dims = input_shape;       // {N, IC, ID, IH, IW}
    tachikoma::memory::dims weights_dims = weight_shape;  // {OC, IC, KD, KH, KW}
    if (groups > 1) {
      weights_dims = {groups, channels / groups, input_shape[1] / groups};
      weights_dims.insert(weights_dims.end(), weight_shape.begin() + 2, weight_shape.end());
      kernel_layout.insert(0, "G");
    }
    tachikoma::memory::dims bias_dims = {channels};
    tachikoma::memory::dims dst_dims = out_shape;  // {N, OC, OD, OH, OW}
    tachikoma::memory::dims strides_dims = TransformStr2Dims(str_strides, "strides");
    tachikoma::memory::dims dilates_dims = TransformStr2Dims(str_dilates, "dilates");
    tachikoma::memory::dims padding_dims_l = TransformStr2Dims(str_padding_l, "padding");
    tachikoma::memory::dims padding_dims_r = TransformStr2Dims(str_padding_r, "padding");

    // Memory descriptions.
    auto conv_src_md = tachikoma::memory::desc(src_dims, dt::f32, layout_dict[data_layout]);
    auto conv_weights_md = tachikoma::memory::desc(weights_dims, dt::f32, layout_dict[kernel_layout]);
    auto conv_bias_md = tachikoma::memory::desc(bias_dims, dt::f32, tag::any);
    auto conv_dst_md = tachikoma::memory::desc(dst_dims, dt::f32, layout_dict[data_layout]);

    // Covn2d description.
    auto conv_desc =
        has_bias ? tachikoma::convolution_forward::desc(
                       tachikoma::prop_kind::forward_inference, tachikoma::algorithm::convolution_direct,
                       conv_src_md, conv_weights_md, conv_bias_md, conv_dst_md, strides_dims,
                       dilates_dims, padding_dims_l, padding_dims_r)
                 : tachikoma::convolution_forward::desc(tachikoma::prop_kind::forward_inference,
                                                   tachikoma::algorithm::convolution_direct, conv_src_md,
                                                   conv_weights_md, conv_dst_md, strides_dims,
                                                   dilates_dims, padding_dims_l, padding_dims_r);

    // Enable elementwise post-ops.
    auto conv2d_prim_desc = tachikoma::convolution_forward::primitive_desc(conv_desc, attr, engine_);

    // Push to the network.
    auto conv = tachikoma::convolution_forward(conv2d_prim_desc);
    net_.push_back(conv);

    // Data memory.
    auto conv2d_src_memory = BindTachikomaMemory(data_entry, conv_src_md);

    // Weight memory.
    auto conv2d_weights_memory = BindTachikomaMemory(weight_entry, conv_weights_md);

    // Output memory.
    auto conv2d_dst_memory = BindTachikomaMemory(out_entry, conv2d_prim_desc.dst_desc());

    // Bias memory.
    auto conv2d_bias_memory = tachikoma::memory({bias_dims, dt::f32, tag::x}, engine_);
    if (has_bias) {
      auto bias_entry = node.GetInputs()[2];
      BindTachikomaMemory(bias_entry, conv2d_bias_memory);

      // Bind memory buffers.
      net_args_.push_back({{DNNL_ARG_SRC, conv2d_src_memory},
                           {DNNL_ARG_WEIGHTS, conv2d_weights_memory},
                           {DNNL_ARG_BIAS, conv2d_bias_memory},
                           {DNNL_ARG_DST, conv2d_dst_memory}});
    } else {
      // Bind memory buffers.
      net_args_.push_back({{DNNL_ARG_SRC, conv2d_src_memory},
                           {DNNL_ARG_WEIGHTS, conv2d_weights_memory},
                           {DNNL_ARG_DST, conv2d_dst_memory}});
    }
  }

  void Deconvolution(const size_t& nid) {
    auto node = nodes_[nid];
    auto op_name = node.GetOpName();
    tachikoma::primitive_attr attr;
    bool has_bias = ParsingOpName(op_name, attr);

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    JSONGraphNodeEntry out_entry(nid, 0);
    tachikoma::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    tachikoma::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];
    tachikoma::memory::dims out_shape = nodes_[out_entry.id_].GetOpShape()[out_entry.index_];
    tachikoma::memory::dim channels =
        node.GetAttr<std::vector<std::string>>("channels")[0] != ""
            ? std::stoi(node.GetAttr<std::vector<std::string>>("channels")[0])
            : out_shape[1];
    std::vector<std::string> str_strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> str_dilates = node.GetAttr<std::vector<std::string>>("dilation");
    std::vector<std::string> str_padding = node.GetAttr<std::vector<std::string>>("padding");
    std::vector<std::string> str_padding_l(str_padding.begin(),
                                           str_padding.begin() + str_padding.size() / 2);
    std::vector<std::string> str_padding_r(str_padding.end() - str_padding.size() / 2,
                                           str_padding.end());
    tachikoma::memory::dim groups = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);
    std::string data_layout = node.GetAttr<std::vector<std::string>>("data_layout")[0];
    std::string kernel_layout = node.GetAttr<std::vector<std::string>>("kernel_layout")[0];

    // Check layout.
    if (layout_dict.find(data_layout) == layout_dict.end() ||
        layout_dict.find(kernel_layout) == layout_dict.end()) {
      LOG(FATAL) << "Unsupported layout: " << data_layout << " " << kernel_layout;
    }

    // Memory shapes.
    tachikoma::memory::dims src_dims = input_shape;       // {N, IC, ID, IH, IW}
    tachikoma::memory::dims weights_dims = weight_shape;  // {OC, IC, KD, KH, KW}

    // Check weight shape, transform to `OIHW`
    if (weights_dims[0] == src_dims[1] && weights_dims[1] == channels) {
      std::swap(weights_dims[0], weights_dims[1]);
    }
    if (kernel_layout == "OIDHW") {
      kernel_layout = "IODHW";
    }
    if (groups > 1) {
      weights_dims = {groups, channels / groups, input_shape[1] / groups};
      weights_dims.insert(weights_dims.end(), weight_shape.begin() + 2, weight_shape.end());
      kernel_layout.insert(0, "G");
    }
    tachikoma::memory::dims bias_dims = {channels};
    tachikoma::memory::dims dst_dims = out_shape;  // {N, OC, OD, OH, OW}
    tachikoma::memory::dims strides_dims = TransformStr2Dims(str_strides, "strides");
    tachikoma::memory::dims dilates_dims = TransformStr2Dims(str_dilates, "dilates");
    tachikoma::memory::dims padding_dims_l = TransformStr2Dims(str_padding_l, "padding");
    tachikoma::memory::dims padding_dims_r = TransformStr2Dims(str_padding_r, "padding");

    // Memory descriptions.
    auto deconv_src_md = tachikoma::memory::desc(src_dims, dt::f32, layout_dict[data_layout]);
    auto deconv_weights_md = tachikoma::memory::desc(weights_dims, dt::f32, layout_dict[kernel_layout]);
    auto deconv_bias_md = tachikoma::memory::desc(bias_dims, dt::f32, tag::any);
    auto deconv_dst_md = tachikoma::memory::desc(dst_dims, dt::f32, layout_dict[data_layout]);

    // Transposed covn2d description.
    auto deconv_desc =
        has_bias ? tachikoma::deconvolution_forward::desc(
                       tachikoma::prop_kind::forward_inference, tachikoma::algorithm::deconvolution_direct,
                       deconv_src_md, deconv_weights_md, deconv_bias_md, deconv_dst_md,
                       strides_dims, dilates_dims, padding_dims_l, padding_dims_r)
                 : tachikoma::deconvolution_forward::desc(
                       tachikoma::prop_kind::forward_inference, tachikoma::algorithm::deconvolution_direct,
                       deconv_src_md, deconv_weights_md, deconv_dst_md, strides_dims, dilates_dims,
                       padding_dims_l, padding_dims_r);

    // Enable elementwise post-ops.
    auto deconv2d_prim_desc =
        tachikoma::deconvolution_forward::primitive_desc(deconv_desc, attr, engine_);

    // Push to the network.
    auto deconv = tachikoma::deconvolution_forward(deconv2d_prim_desc);
    net_.push_back(deconv);

    // Data memory.
    auto deconv2d_src_memory = BindTachikomaMemory(data_entry, deconv_src_md);

    // Weight memory.
    auto deconv2d_weights_memory = BindTachikomaMemory(weight_entry, deconv_weights_md);

    // Output memory.
    auto deconv2d_dst_memory = BindTachikomaMemory(out_entry, deconv2d_prim_desc.dst_desc());

    // Bias memory.
    auto deconv2d_bias_memory = tachikoma::memory({bias_dims, dt::f32, tag::x}, engine_);
    if (has_bias) {
      auto bias_entry = node.GetInputs()[2];
      BindTachikomaMemory(bias_entry, deconv2d_bias_memory);

      // Bind memory buffers.
      net_args_.push_back({{DNNL_ARG_SRC, deconv2d_src_memory},
                           {DNNL_ARG_WEIGHTS, deconv2d_weights_memory},
                           {DNNL_ARG_BIAS, deconv2d_bias_memory},
                           {DNNL_ARG_DST, deconv2d_dst_memory}});
    } else {
      // Bind memory buffers.
      net_args_.push_back({{DNNL_ARG_SRC, deconv2d_src_memory},
                           {DNNL_ARG_WEIGHTS, deconv2d_weights_memory},
                           {DNNL_ARG_DST, deconv2d_dst_memory}});
    }
  }

  void Dense(const size_t& nid) {
    auto node = nodes_[nid];
    auto op_name = node.GetOpName();
    tachikoma::primitive_attr attr;
    bool has_bias = ParsingOpName(op_name, attr);

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    JSONGraphNodeEntry out_entry(nid, 0);
    tachikoma::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    tachikoma::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];
    tachikoma::memory::dims out_shape = nodes_[out_entry.id_].GetOpShape()[out_entry.index_];
    tachikoma::memory::dim OC = out_shape[1];

    // Memory shapes.
    tachikoma::memory::dims data_dims = input_shape;
    tachikoma::memory::dims weight_dims = weight_shape;
    tachikoma::memory::dims bias_dims = {OC};
    tachikoma::memory::dims out_dims = out_shape;

    // Memory descriptions.
    auto data_md = tachikoma::memory::desc({data_dims, dt::f32, tag::nc});
    auto weight_md = tachikoma::memory::desc({weight_dims, dt::f32, tag::nc});
    auto bias_md = tachikoma::memory::desc({bias_dims, dt::f32, tag::x});
    auto dst_md = tachikoma::memory::desc({out_dims, dt::f32, tag::nc});

    // Dense description.
    auto dense_desc = tachikoma::inner_product_forward::desc(tachikoma::prop_kind::forward_inference, data_md,
                                                        weight_md, bias_md, dst_md);

    // Enable elementwise post-ops.
    auto dense_prim_desc = tachikoma::inner_product_forward::primitive_desc(dense_desc, attr, engine_);

    auto dense = tachikoma::inner_product_forward(dense_prim_desc);
    net_.push_back(dense);

    // Memories.
    auto data_memory = BindTachikomaMemory(data_entry, data_md);
    auto weight_memory = BindTachikomaMemory(weight_entry, weight_md);

    // Bias memory.
    auto bias_memory = tachikoma::memory(bias_md, engine_);
    if (has_bias) {
      auto bias_entry = node.GetInputs()[2];
      BindTachikomaMemory(bias_entry, bias_memory);
    } else {
      float bias[OC] = {0};
      write_to_tachikoma_memory(bias, bias_memory, OC * sizeof(float));
    }

    // Output memory.
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

  void Pooling(const size_t& nid, tachikoma::algorithm algo) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    JSONGraphNodeEntry out_entry(nid, 0);
    tachikoma::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    tachikoma::memory::dims out_shape = nodes_[out_entry.id_].GetOpShape()[out_entry.index_];
    std::vector<std::string> str_kernel = node.GetAttr<std::vector<std::string>>("pool_size");
    std::vector<std::string> str_strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> str_padding = node.GetAttr<std::vector<std::string>>("padding");
    std::vector<std::string> str_padding_l(str_padding.begin(),
                                           str_padding.begin() + str_padding.size() / 2);
    std::vector<std::string> str_padding_r(str_padding.end() - str_padding.size() / 2,
                                           str_padding.end());
    std::vector<std::string> str_dilates = node.GetAttr<std::vector<std::string>>("dilation");
    std::string layout = node.GetAttr<std::vector<std::string>>("layout")[0];

    // Check layout.
    if (layout_dict.find(layout) == layout_dict.end()) {
      LOG(FATAL) << "Unsupported layout for pooling: " << layout;
    }

    // Attributes related to AvgPool
    if (algo == tachikoma::algorithm::pooling_avg) {
      int int_countpad = std::stoi(node.GetAttr<std::vector<std::string>>("count_include_pad")[0]);
      bool count_include_pad = int_countpad != 0 ? true : false;
      algo = count_include_pad ? tachikoma::algorithm::pooling_avg_include_padding
                               : tachikoma::algorithm::pooling_avg_exclude_padding;
    }

    tachikoma::memory::dims src_dims = input_shape;
    tachikoma::memory::dims dst_dims = out_shape;
    tachikoma::memory::dims kernel_dims = TransformStr2Dims(str_kernel, "kernel");
    tachikoma::memory::dims strides_dims = TransformStr2Dims(str_strides, "strides");
    tachikoma::memory::dims dilates_dims = TransformStr2Dims(str_dilates, "dilates");
    tachikoma::memory::dims padding_dims_l = TransformStr2Dims(str_padding_l, "padding");
    tachikoma::memory::dims padding_dims_r = TransformStr2Dims(str_padding_r, "padding");

    // Memory descriptions.
    auto pool_src_md = tachikoma::memory::desc(src_dims, dt::f32, layout_dict[layout]);
    auto pool_dst_md = tachikoma::memory::desc(dst_dims, dt::f32, tag::any);

    // Pooling description.
    auto pool_desc = tachikoma::pooling_forward::desc(tachikoma::prop_kind::forward_inference, algo,
                                                 pool_src_md, pool_dst_md, strides_dims,
                                                 kernel_dims, padding_dims_l, padding_dims_r);

    auto pool_prim_desc = tachikoma::pooling_forward::primitive_desc(pool_desc, engine_, true);
    auto pool = tachikoma::pooling_forward(pool_prim_desc);
    net_.push_back(pool);

    // Memories.
    auto pool2d_src_memory = BindTachikomaMemory(data_entry, pool_src_md);

    auto pool2d_dst_memory = BindTachikomaMemory(out_entry, pool_prim_desc.dst_desc());

    // Bind memory buffers.
    net_args_.push_back({{DNNL_ARG_SRC, pool2d_src_memory}, {DNNL_ARG_DST, pool2d_dst_memory}});
  }

  void Eltwise(const size_t& nid) {
    auto node = nodes_[nid];
    auto op_name = node.GetOpName();
    auto algo = elt_name2algo[op_name];

    auto data_entry = node.GetInputs()[0];
    tachikoma::memory::dims shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    tachikoma::memory::desc data_md = GenTachikomaMemDescByShape(shape, dt::f32);
    float alpha = 0., beta = 0.;
    if (op_name == "clip") {
      alpha = std::stof(node.GetAttr<std::vector<std::string>>("a_min")[0]);
      beta = std::stof(node.GetAttr<std::vector<std::string>>("a_max")[0]);
    } else if (op_name == "nn.leaky_relu") {
      alpha = std::stof(node.GetAttr<std::vector<std::string>>("alpha")[0]);
    }

    auto elt_desc =
        tachikoma::eltwise_forward::desc(tachikoma::prop_kind::forward_inference, algo, data_md, alpha, beta);
    auto elt_prim_desc = tachikoma::eltwise_forward::primitive_desc(elt_desc, engine_);
    ICHECK(data_md == elt_prim_desc.dst_desc());

    auto elt = tachikoma::eltwise_forward(elt_prim_desc);
    net_.push_back(elt);

    auto data_memory = BindTachikomaMemory(data_entry, data_md);
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindTachikomaMemory(out_entry, data_md);

    net_args_.push_back({{DNNL_ARG_SRC, data_memory}, {DNNL_ARG_DST, out_memory}});
  }

  void Softmax(const size_t& nid) {
    auto node = nodes_[nid];

    auto data_entry = node.GetInputs()[0];
    tachikoma::memory::dims shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    int axis = std::stoi(node.GetAttr<std::vector<std::string>>("axis")[0]);
    if (axis < 0) {
      axis = shape.size() + axis;
    }
    tachikoma::memory::desc data_md = GenTachikomaMemDescByShape(shape, dt::f32);

    auto softmax_desc =
        tachikoma::softmax_forward::desc(tachikoma::prop_kind::forward_inference, data_md, axis);
    auto softmax_prim_desc = tachikoma::softmax_forward::primitive_desc(softmax_desc, engine_);
    ICHECK(data_md == softmax_prim_desc.dst_desc());

    auto softmax = tachikoma::softmax_forward(softmax_prim_desc);
    net_.push_back(softmax);

    auto data_memory = BindTachikomaMemory(data_entry, data_md);
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindTachikomaMemory(out_entry, data_md);

    net_args_.push_back({{DNNL_ARG_SRC, data_memory}, {DNNL_ARG_DST, out_memory}});
  }

  void Binary(const size_t& nid, tachikoma::algorithm algo) {
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

    auto binary_desc = tachikoma::binary::desc(algo, data_mds[0], data_mds[1], out_md);
    auto binary_prim_desc = tachikoma::binary::primitive_desc(binary_desc, engine_);
    auto binary = tachikoma::binary(binary_prim_desc);
    net_.push_back(binary);

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

void TachikomaExportModule(runtime::Module mod, const std::string& file_name) {
  tvm::runtime::PackedFunc serializeTrace = mod.GetFunction("serialize_computational_trace", false);
  if (serializeTrace != nullptr) {
    serializeTrace(file_name);
    std::cerr << "Tachikoma module export successful." << std::endl;
  } else {
    std::cerr << "Warning: module is not a Tachikoma module, hence export failed." << std::endl;
  }
}

TVM_REGISTER_GLOBAL("runtime.TachikomaJSONRuntimeCreate").set_body_typed(TachikomaJSONRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_tachikoma_json")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<TachikomaJSONRuntime>);

TVM_REGISTER_GLOBAL("runtime.TachikomaExportModule").set_body_typed(TachikomaExportModule);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
