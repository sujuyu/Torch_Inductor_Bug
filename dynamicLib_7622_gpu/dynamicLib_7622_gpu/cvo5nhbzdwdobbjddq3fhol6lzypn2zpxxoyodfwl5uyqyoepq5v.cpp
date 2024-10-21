#include <torch/csrc/inductor/aoti_runtime/arrayref_tensor.h>
#include <torch/csrc/inductor/aoti_runtime/interface.h>
#include <torch/csrc/inductor/aoti_runtime/model_container.h>
#include <torch/csrc/inductor/aoti_runtime/scalar_to_tensor.h>
#include <torch/csrc/inductor/aoti_runtime/thread_local.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#define CONVERT_EXCEPTION_TO_ERROR_CODE(...)                 \
  try {                                                      \
    __VA_ARGS__                                              \
  } catch (const std::exception& e) {                        \
    std::cerr << "Error: " << e.what() << std::endl;         \
    return AOTI_RUNTIME_FAILURE;                             \
  } catch (...) {                                            \
    std::cerr << "Unknown exception occurred." << std::endl; \
    return AOTI_RUNTIME_FAILURE;                             \
  }                                                          \
  return AOTI_RUNTIME_SUCCESS;

#define AOTI_VECTOR_SIZE_CHECK(actual_size, expected_size, name)  \
  do {                                                            \
    AOTI_RUNTIME_CHECK(                                           \
        actual_size == expected_size,                             \
        "expected " + std::string(name) + " vector size to be " + \
            std::to_string(expected_size) + ", but got " +        \
            std::to_string(actual_size));                         \
  } while (0)

// AOTInductor uses at::addmm_out, which doesn't supports
// arguments that requires gradient. For this reason, we
// enforce no_grad context for run APIs.
//
// A RAII, thread local (!) guard that enables or disables grad mode upon
// construction, and sets it back to the original value upon destruction.
struct AOTINoGradGuard {
  AOTINoGradGuard() : prev_mode(aoti_torch_grad_mode_is_enabled()) {
    aoti_torch_grad_mode_set_enabled(false);
  }
  ~AOTINoGradGuard() {
    aoti_torch_grad_mode_set_enabled(prev_mode);
  }
  bool prev_mode;
};

extern "C" {

AOTIRuntimeError AOTInductorModelContainerCreate(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    bool is_cpu,
    const char* cubin_dir) {
      return AOTInductorModelContainerCreateWithDevice(
        container_handle,
        num_models,
        is_cpu ? "cpu" : "cuda",
        cubin_dir);
}

AOTIRuntimeError AOTInductorModelContainerCreateWithDevice(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    const char* device_str,
    const char* cubin_dir) {
  if (num_models == 0) {
    std::cerr << "Error: num_models must be positive, but got 0" << std::endl;
    return AOTI_RUNTIME_FAILURE;
  }
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    std::optional<std::string> cubin_dir_opt;
    if (cubin_dir != nullptr) {
      cubin_dir_opt.emplace(cubin_dir);
    }
    auto* container = new torch::aot_inductor::AOTInductorModelContainer(
        num_models, std::string(device_str), cubin_dir_opt);
    *container_handle =
        reinterpret_cast<AOTInductorModelContainerHandle>(container);
  })
}

AOTIRuntimeError AOTInductorModelContainerDelete(
    AOTInductorModelContainerHandle container_handle) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* container =
        reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
            container_handle);
    delete container;
  });
}

AOTIRuntimeError AOTInductorModelContainerRun(
    AOTInductorModelContainerHandle container_handle,
    AtenTensorHandle* input_handles, // array of input AtenTensorHandle; handles
                                     // are stolen; the array itself is borrowed
    size_t num_inputs,
    AtenTensorHandle*
        output_handles, // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    size_t num_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  AOTI_VECTOR_SIZE_CHECK(num_inputs, container->num_inputs(), "inputs");
  AOTI_VECTOR_SIZE_CHECK(num_outputs, container->num_outputs(), "outputs");

  auto stream =
      reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    container->run(
        input_handles, output_handles, stream, proxy_executor_handle);
  })
}

AOTIRuntimeError AOTInductorModelContainerGetNumConstants(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_constants) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *num_constants = container->num_constants(); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantName(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    const char** name) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *name = container->constant_name(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantOriginalFQN(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    const char** original_fqn) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *original_fqn = container->constant_original_fqn(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantFromFolded(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    bool* from_folded) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *from_folded = container->constant_from_folded(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantDtype(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    int32_t* dtype) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *dtype = container->constant_dtype(idx); })
}

AOTIRuntimeError AOTInductorModelContainerUpdateConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive,
    bool validate_full_update) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto input_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->update_constant_buffer(
        *input_map, use_inactive, validate_full_update);
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateInactiveConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle) {
  return AOTInductorModelContainerUpdateConstantBuffer(container_handle,
          constant_map_handle,
          /*use_inactive*/ true,
          /*validate_full_update*/ true);
}

AOTIRuntimeError AOTInductorModelContainerRunConstantFolding(
    AOTInductorModelContainerHandle container_handle,
    bool use_inactive,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto stream =
      reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    container->run_const_fold(use_inactive, stream, proxy_executor_handle);
  })
}

AOTIRuntimeError AOTInductorModelContainerSwapConstantBuffer(
    AOTInductorModelContainerHandle container_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->swap_constant_buffer();
  })
}

AOTIRuntimeError AOTInductorModelContainerGetNumInputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_inputs) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_num_inputs = container->num_inputs(); })
}

AOTIRuntimeError AOTInductorModelContainerGetInputName(
    AOTInductorModelContainerHandle container_handle,
    size_t input_idx,
    const char** ret_input_names) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_input_names = container->input_name(input_idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetNumOutputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_outputs) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_num_outputs = container->num_outputs(); })
}

AOTIRuntimeError AOTInductorModelContainerGetOutputName(
    AOTInductorModelContainerHandle container_handle,
    size_t output_idx,
    const char** ret_output_names) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_output_names = container->output_name(output_idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetCallSpec(
    AOTInductorModelContainerHandle container_handle,
    const char** in_spec,
    const char** out_spec) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    *in_spec = container->get_in_spec();
    *out_spec = container->get_out_spec();
  })
}

AOTIRuntimeError AOTInductorModelCreate(
    AOTInductorModelHandle* model_handle,
    AOTInductorConstantMapHandle constant_map_handle){
    CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto constant_map = std::make_shared<torch::aot_inductor::ConstantMap>();
      auto constant_array = std::make_shared<std::vector<torch::aot_inductor::ConstantHandle>>();
      auto input_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);

      auto model = new torch::aot_inductor::AOTInductorModel(
          constant_map,
          constant_array,
          "cpu", // device_str is hardcoded, as AOTInductorModelCreate is only use for CPU models
          ""
      );

      if (input_map) {
        for (auto const& kv : *input_map) {
          constant_map->emplace(kv.first, kv.second);
        }
      } else {
        model->load_constants();
      }

      *model_handle = reinterpret_cast<AOTInductorModelHandle>(model);
    })}

AOTIRuntimeError AOTInductorModelRun(
    AOTInductorModelHandle model_handle,
    AtenTensorHandle* input_handles,
    AtenTensorHandle* output_handles) {
  auto model =
      reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    model->run_impl(
        input_handles,
        output_handles,
        (torch::aot_inductor::DeviceStreamType) nullptr,
        nullptr);
  })
}

AOTIRuntimeError AOTInductorModelDelete(AOTInductorModelHandle model_handle){
    CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto model = reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(
          model_handle);
      delete model;
    })}

AOTIRuntimeError AOTInductorModelGetNumOutputs(
    AOTInductorModelHandle model_handle,
    size_t* ret_num_outputs) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto model = reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
      *ret_num_outputs = model->num_outputs();
  })
}

AOTIRuntimeError AOTInductorModelUpdateConstantsMap(
    AOTInductorModelHandle model_handle,
    AOTInductorConstantMapHandle constant_map_handle) {
  auto model =
      reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto constant_map = std::make_shared<torch::aot_inductor::ConstantMap>();
    auto input_map =
        reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(
            constant_map_handle);

    for (auto const& kv : *input_map) {
      constant_map->emplace(kv.first, kv.second);
    }
    model->update_constants_map(std::move(constant_map));
  })
}

} // extern "C"
// NOTE: Like interface.cpp, this file will be copied into AOTInductor
// generated output. This file is intended to keep implementation
// details separate from the implementation of the AOTI public
// interface. Note also that #includes should go into interface.cpp
// for simplicity of maintenance.

namespace torch {
namespace aot_inductor {
template <typename T>
void convert_output_to_handle(
    const ArrayRefTensor<T>& output,
    AtenTensorHandle& handle) {
  handle = output.expensiveCopyToTensor();
}

template <typename... Ts, std::size_t... Is>
void convert_outputs_to_handles_helper(
    const std::tuple<ArrayRefTensor<Ts>...>& outputs,
    AtenTensorHandle* output_handles,
    std::index_sequence<Is...>) {
  (convert_output_to_handle(std::get<Is>(outputs), output_handles[Is]), ...);
}
template <typename... Ts>
void convert_outputs_to_handles(
    const std::tuple<ArrayRefTensor<Ts>...>& outputs,
    AtenTensorHandle* output_handles) {
  convert_outputs_to_handles_helper(
      outputs, output_handles, std::make_index_sequence<sizeof...(Ts)>());
}

template <typename T>
void convert_handle_to_arrayref_tensor(
    AtenTensorHandle handle,
    ArrayRefTensor<T>& input) {
  void* data_ptr;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr(handle, &data_ptr));
  int64_t dim;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dim(handle, &dim));
  int64_t numel;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_numel(handle, &numel));
  int64_t* sizes;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_sizes(handle, &sizes));
  int64_t* strides;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_strides(handle, &strides));
  int32_t dtype;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(handle, &dtype));
  int32_t device_type;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type(handle, &device_type));
  int32_t device_index;
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_device_index(handle, &device_index));

  input = ArrayRefTensor<T>(
      MiniArrayRef<T>(reinterpret_cast<T*>(data_ptr), numel),
      MiniArrayRef<const int64_t>(sizes, dim),
      MiniArrayRef<const int64_t>(strides, dim),
      device_type,
      device_index);
}

template <typename... Ts, std::size_t... Is>
void convert_handles_to_inputs_helper(
    AtenTensorHandle* input_handles,
    std::tuple<ArrayRefTensor<Ts>...>& inputs,
    std::index_sequence<Is...>) {
  (convert_handle_to_arrayref_tensor(input_handles[Is], std::get<Is>(inputs)),
   ...);
}

template <typename... Ts>
void convert_handles_to_inputs(
    AtenTensorHandle* input_handles,
    std::tuple<ArrayRefTensor<Ts>...>& inputs) {
  convert_handles_to_inputs_helper(
      input_handles, inputs, std::make_index_sequence<sizeof...(Ts)>());
}

template <typename T>
void assert_numel(const ArrayRefTensor<T>& tensor, int64_t numel) {
  if (tensor.numel() != numel) {
    std::stringstream err;
    err << "incorrect numel for input tensor. expected " << numel << ", got " << tensor.numel();
    throw std::runtime_error(err.str());
  }
}
} // namespace aot_inductor
} // namespace torch

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/BinaryOps.h>
#include <torch/csrc/inductor/aoti_runtime/utils.h>
#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>
#include <torch/csrc/inductor/inductor_ops.h>
#include <torch/types.h>
#include <ATen/ops/bernoulli_native.h>

#define reinterpret_tensor torch::inductor::_reinterpret_tensor
#define alloc_from_pool torch::inductor::_alloc_from_pool
#include <c10/util/generic_math.h>

[[maybe_unused]] static int64_t align(int64_t nbytes) {
  return (nbytes + 64 - 1) & -64;
}
#include <filesystem>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/EmptyTensor.h>

#define CUDA_DRIVER_CHECK(EXPR)                    \
do {                                               \
    CUresult code = EXPR;                          \
    const char *msg;                               \
    cuGetErrorString(code, &msg);                  \
    if (code != CUDA_SUCCESS) {                    \
        throw std::runtime_error(                  \
            std::string("CUDA driver error: ") +   \
            std::string(msg));                     \
    }                                              \
} while (0);

namespace {

struct Grid {
    Grid(uint32_t x, uint32_t y, uint32_t z)
      : grid_x(x), grid_y(y), grid_z(z) {}
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t grid_z;

    bool is_non_zero() {
        return grid_x > 0 && grid_y > 0 && grid_z > 0;
    }
};

}  // anonymous namespace

static inline CUfunction loadKernel(
        std::string filePath,
        const std::string &funcName,
        uint32_t sharedMemBytes,
        const std::optional<std::string> &cubinDir = std::nullopt) {
    if (cubinDir) {
        std::filesystem::path p1{*cubinDir};
        std::filesystem::path p2{filePath};
        filePath = (p1 / p2.filename()).string();
    }

    CUmodule mod;
    CUfunction func;
    CUDA_DRIVER_CHECK(cuModuleLoad(&mod, filePath.c_str()));
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&func, mod, funcName.c_str()));
    if (sharedMemBytes > 0) {
        CUDA_DRIVER_CHECK(cuFuncSetAttribute(
            func,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            sharedMemBytes
        ))
    }
    return func;
}

static inline void launchKernel(
        CUfunction func,
        uint32_t gridX,
        uint32_t gridY,
        uint32_t gridZ,
        uint32_t numWarps,
        uint32_t sharedMemBytes,
        void* args[],
        cudaStream_t stream) {
    CUDA_DRIVER_CHECK(cuLaunchKernel(
        func, gridX, gridY, gridZ, 32*numWarps, 1, 1, sharedMemBytes, stream, args, nullptr
    ));
}
namespace torch {
namespace aot_inductor {

namespace {
class AOTInductorModelKernels : public AOTInductorModelKernelsBase {
  public:
    CUfunction triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17{nullptr};
    CUfunction triton_per_fused__softmax_div_8{nullptr};
    CUfunction triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19{nullptr};
    CUfunction triton_per_fused__to_copy_logical_or_lt_sum_4{nullptr};
    CUfunction triton_poi_fused_scalar_tensor_where_3{nullptr};
    CUfunction triton_poi_fused_cat_15{nullptr};
    CUfunction triton_poi_fused_clone_7{nullptr};
    CUfunction triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18{nullptr};
    CUfunction triton_per_fused_mul_pow_sum_6{nullptr};
    CUfunction triton_poi_fused_leaky_relu_5{nullptr};
    CUfunction triton_poi_fused_scalar_tensor_where_1{nullptr};
    CUfunction triton_poi_fused_cat_13{nullptr};
    CUfunction triton_poi_fused_leaky_relu_10{nullptr};
    CUfunction triton_poi_fused_cat_12{nullptr};
    CUfunction triton_per_fused_mul_pow_sum_11{nullptr};
    CUfunction triton_poi_fused_cat_20{nullptr};
    CUfunction triton_poi_fused_cat_16{nullptr};
    CUfunction triton_per_fused_native_layer_norm_2{nullptr};
    CUfunction triton_poi_fused_clone_9{nullptr};
    CUfunction triton_poi_fused_cat_14{nullptr};
    CUfunction triton_per_fused_native_layer_norm_0{nullptr};
};
}  // namespace

AOTInductorModel::AOTInductorModel(std::shared_ptr<ConstantMap> constants_map,
                                   std::shared_ptr<std::vector<ConstantHandle>> constants_array,
                                   const std::string& device_str,
                                   std::optional<std::string> cubin_dir)
    : AOTInductorModelBase(9, 4, 106, device_str, cubin_dir) {
    inputs_info_[0].name = "arg122_1";
    inputs_info_[1].name = "arg123_1";
    inputs_info_[2].name = "arg124_1";
    inputs_info_[3].name = "arg125_1";
    inputs_info_[4].name = "arg126_1";
    inputs_info_[5].name = "arg127_1";
    inputs_info_[6].name = "arg128_1";
    inputs_info_[7].name = "arg129_1";
    inputs_info_[8].name = "arg130_1";
    constants_info_[0].name = "L__self___user_model_feedforwards_item_clk_seq_fc1_linear_weight";
    constants_info_[0].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[0].offset = 0;
    constants_info_[0].data_size = 123904;
    constants_info_[0].from_folded = false;
    constants_info_[0].shape = {176, 176};
    constants_info_[0].stride = {176, 1};
    constants_info_[0].original_fqn = "user_model.feedforwards.item_clk_seq.fc1.linear.weight";
    constants_info_[1].name = "L__self___user_model_feedforwards_item_clk_seq_fc1_linear_bias";
    constants_info_[1].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[1].offset = 0;
    constants_info_[1].data_size = 704;
    constants_info_[1].from_folded = false;
    constants_info_[1].shape = {176};
    constants_info_[1].stride = {1};
    constants_info_[1].original_fqn = "user_model.feedforwards.item_clk_seq.fc1.linear.bias";
    constants_info_[2].name = "L__self___user_model_feedforwards_item_clk_seq_fc2_linear_weight";
    constants_info_[2].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[2].offset = 0;
    constants_info_[2].data_size = 123904;
    constants_info_[2].from_folded = false;
    constants_info_[2].shape = {176, 176};
    constants_info_[2].stride = {176, 1};
    constants_info_[2].original_fqn = "user_model.feedforwards.item_clk_seq.fc2.linear.weight";
    constants_info_[3].name = "L__self___user_model_feedforwards_item_clk_seq_fc2_linear_bias";
    constants_info_[3].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[3].offset = 0;
    constants_info_[3].data_size = 704;
    constants_info_[3].from_folded = false;
    constants_info_[3].shape = {176};
    constants_info_[3].stride = {1};
    constants_info_[3].original_fqn = "user_model.feedforwards.item_clk_seq.fc2.linear.bias";
    constants_info_[4].name = "L__self___user_model_feedforwards_query_seq_fc1_linear_weight";
    constants_info_[4].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[4].offset = 0;
    constants_info_[4].data_size = 118336;
    constants_info_[4].from_folded = false;
    constants_info_[4].shape = {172, 172};
    constants_info_[4].stride = {172, 1};
    constants_info_[4].original_fqn = "user_model.feedforwards.query_seq.fc1.linear.weight";
    constants_info_[5].name = "L__self___user_model_feedforwards_query_seq_fc1_linear_bias";
    constants_info_[5].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[5].offset = 0;
    constants_info_[5].data_size = 688;
    constants_info_[5].from_folded = false;
    constants_info_[5].shape = {172};
    constants_info_[5].stride = {1};
    constants_info_[5].original_fqn = "user_model.feedforwards.query_seq.fc1.linear.bias";
    constants_info_[6].name = "L__self___user_model_feedforwards_query_seq_fc2_linear_weight";
    constants_info_[6].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[6].offset = 0;
    constants_info_[6].data_size = 118336;
    constants_info_[6].from_folded = false;
    constants_info_[6].shape = {172, 172};
    constants_info_[6].stride = {172, 1};
    constants_info_[6].original_fqn = "user_model.feedforwards.query_seq.fc2.linear.weight";
    constants_info_[7].name = "L__self___user_model_feedforwards_query_seq_fc2_linear_bias";
    constants_info_[7].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[7].offset = 0;
    constants_info_[7].data_size = 688;
    constants_info_[7].from_folded = false;
    constants_info_[7].shape = {172};
    constants_info_[7].stride = {1};
    constants_info_[7].original_fqn = "user_model.feedforwards.query_seq.fc2.linear.bias";
    constants_info_[8].name = "L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_q_linear_weight";
    constants_info_[8].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[8].offset = 0;
    constants_info_[8].data_size = 79872;
    constants_info_[8].from_folded = false;
    constants_info_[8].shape = {128, 156};
    constants_info_[8].stride = {156, 1};
    constants_info_[8].original_fqn = "user_model.multi_h_attens.item_clk_seq_ta.proj_q.linear.weight";
    constants_info_[9].name = "L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_q_linear_bias";
    constants_info_[9].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[9].offset = 0;
    constants_info_[9].data_size = 512;
    constants_info_[9].from_folded = false;
    constants_info_[9].shape = {128};
    constants_info_[9].stride = {1};
    constants_info_[9].original_fqn = "user_model.multi_h_attens.item_clk_seq_ta.proj_q.linear.bias";
    constants_info_[10].name = "L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_k_linear_weight";
    constants_info_[10].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[10].offset = 0;
    constants_info_[10].data_size = 90112;
    constants_info_[10].from_folded = false;
    constants_info_[10].shape = {128, 176};
    constants_info_[10].stride = {176, 1};
    constants_info_[10].original_fqn = "user_model.multi_h_attens.item_clk_seq_ta.proj_k.linear.weight";
    constants_info_[11].name = "L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_k_linear_bias";
    constants_info_[11].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[11].offset = 0;
    constants_info_[11].data_size = 512;
    constants_info_[11].from_folded = false;
    constants_info_[11].shape = {128};
    constants_info_[11].stride = {1};
    constants_info_[11].original_fqn = "user_model.multi_h_attens.item_clk_seq_ta.proj_k.linear.bias";
    constants_info_[12].name = "L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_v_linear_weight";
    constants_info_[12].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[12].offset = 0;
    constants_info_[12].data_size = 90112;
    constants_info_[12].from_folded = false;
    constants_info_[12].shape = {128, 176};
    constants_info_[12].stride = {176, 1};
    constants_info_[12].original_fqn = "user_model.multi_h_attens.item_clk_seq_ta.proj_v.linear.weight";
    constants_info_[13].name = "L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_v_linear_bias";
    constants_info_[13].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[13].offset = 0;
    constants_info_[13].data_size = 512;
    constants_info_[13].from_folded = false;
    constants_info_[13].shape = {128};
    constants_info_[13].stride = {1};
    constants_info_[13].original_fqn = "user_model.multi_h_attens.item_clk_seq_ta.proj_v.linear.bias";
    constants_info_[14].name = "L__self___user_model_multi_h_attens_item_clk_seq_ta_q_layer_norm_weight";
    constants_info_[14].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[14].offset = 0;
    constants_info_[14].data_size = 64;
    constants_info_[14].from_folded = false;
    constants_info_[14].shape = {16};
    constants_info_[14].stride = {1};
    constants_info_[14].original_fqn = "user_model.multi_h_attens.item_clk_seq_ta.q_layer_norm.weight";
    constants_info_[15].name = "L__self___user_model_multi_h_attens_item_clk_seq_ta_q_layer_norm_bias";
    constants_info_[15].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[15].offset = 0;
    constants_info_[15].data_size = 64;
    constants_info_[15].from_folded = false;
    constants_info_[15].shape = {16};
    constants_info_[15].stride = {1};
    constants_info_[15].original_fqn = "user_model.multi_h_attens.item_clk_seq_ta.q_layer_norm.bias";
    constants_info_[16].name = "L__self___user_model_multi_h_attens_item_clk_seq_ta_k_layer_norm_weight";
    constants_info_[16].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[16].offset = 0;
    constants_info_[16].data_size = 64;
    constants_info_[16].from_folded = false;
    constants_info_[16].shape = {16};
    constants_info_[16].stride = {1};
    constants_info_[16].original_fqn = "user_model.multi_h_attens.item_clk_seq_ta.k_layer_norm.weight";
    constants_info_[17].name = "L__self___user_model_multi_h_attens_item_clk_seq_ta_k_layer_norm_bias";
    constants_info_[17].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[17].offset = 0;
    constants_info_[17].data_size = 64;
    constants_info_[17].from_folded = false;
    constants_info_[17].shape = {16};
    constants_info_[17].stride = {1};
    constants_info_[17].original_fqn = "user_model.multi_h_attens.item_clk_seq_ta.k_layer_norm.bias";
    constants_info_[18].name = "L__self___user_model_multi_h_attens_query_seq_ta_proj_q_linear_weight";
    constants_info_[18].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[18].offset = 0;
    constants_info_[18].data_size = 79872;
    constants_info_[18].from_folded = false;
    constants_info_[18].shape = {128, 156};
    constants_info_[18].stride = {156, 1};
    constants_info_[18].original_fqn = "user_model.multi_h_attens.query_seq_ta.proj_q.linear.weight";
    constants_info_[19].name = "L__self___user_model_multi_h_attens_query_seq_ta_proj_q_linear_bias";
    constants_info_[19].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[19].offset = 0;
    constants_info_[19].data_size = 512;
    constants_info_[19].from_folded = false;
    constants_info_[19].shape = {128};
    constants_info_[19].stride = {1};
    constants_info_[19].original_fqn = "user_model.multi_h_attens.query_seq_ta.proj_q.linear.bias";
    constants_info_[20].name = "L__self___user_model_multi_h_attens_query_seq_ta_proj_k_linear_weight";
    constants_info_[20].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[20].offset = 0;
    constants_info_[20].data_size = 88064;
    constants_info_[20].from_folded = false;
    constants_info_[20].shape = {128, 172};
    constants_info_[20].stride = {172, 1};
    constants_info_[20].original_fqn = "user_model.multi_h_attens.query_seq_ta.proj_k.linear.weight";
    constants_info_[21].name = "L__self___user_model_multi_h_attens_query_seq_ta_proj_k_linear_bias";
    constants_info_[21].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[21].offset = 0;
    constants_info_[21].data_size = 512;
    constants_info_[21].from_folded = false;
    constants_info_[21].shape = {128};
    constants_info_[21].stride = {1};
    constants_info_[21].original_fqn = "user_model.multi_h_attens.query_seq_ta.proj_k.linear.bias";
    constants_info_[22].name = "L__self___user_model_multi_h_attens_query_seq_ta_proj_v_linear_weight";
    constants_info_[22].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[22].offset = 0;
    constants_info_[22].data_size = 88064;
    constants_info_[22].from_folded = false;
    constants_info_[22].shape = {128, 172};
    constants_info_[22].stride = {172, 1};
    constants_info_[22].original_fqn = "user_model.multi_h_attens.query_seq_ta.proj_v.linear.weight";
    constants_info_[23].name = "L__self___user_model_multi_h_attens_query_seq_ta_proj_v_linear_bias";
    constants_info_[23].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[23].offset = 0;
    constants_info_[23].data_size = 512;
    constants_info_[23].from_folded = false;
    constants_info_[23].shape = {128};
    constants_info_[23].stride = {1};
    constants_info_[23].original_fqn = "user_model.multi_h_attens.query_seq_ta.proj_v.linear.bias";
    constants_info_[24].name = "L__self___user_model_multi_h_attens_query_seq_ta_q_layer_norm_weight";
    constants_info_[24].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[24].offset = 0;
    constants_info_[24].data_size = 64;
    constants_info_[24].from_folded = false;
    constants_info_[24].shape = {16};
    constants_info_[24].stride = {1};
    constants_info_[24].original_fqn = "user_model.multi_h_attens.query_seq_ta.q_layer_norm.weight";
    constants_info_[25].name = "L__self___user_model_multi_h_attens_query_seq_ta_q_layer_norm_bias";
    constants_info_[25].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[25].offset = 0;
    constants_info_[25].data_size = 64;
    constants_info_[25].from_folded = false;
    constants_info_[25].shape = {16};
    constants_info_[25].stride = {1};
    constants_info_[25].original_fqn = "user_model.multi_h_attens.query_seq_ta.q_layer_norm.bias";
    constants_info_[26].name = "L__self___user_model_multi_h_attens_query_seq_ta_k_layer_norm_weight";
    constants_info_[26].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[26].offset = 0;
    constants_info_[26].data_size = 64;
    constants_info_[26].from_folded = false;
    constants_info_[26].shape = {16};
    constants_info_[26].stride = {1};
    constants_info_[26].original_fqn = "user_model.multi_h_attens.query_seq_ta.k_layer_norm.weight";
    constants_info_[27].name = "L__self___user_model_multi_h_attens_query_seq_ta_k_layer_norm_bias";
    constants_info_[27].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[27].offset = 0;
    constants_info_[27].data_size = 64;
    constants_info_[27].from_folded = false;
    constants_info_[27].shape = {16};
    constants_info_[27].stride = {1};
    constants_info_[27].original_fqn = "user_model.multi_h_attens.query_seq_ta.k_layer_norm.bias";
    constants_info_[28].name = "getattr_L__self___user_model_ctr_net_tower_layers___0___fc_weight";
    constants_info_[28].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[28].offset = 0;
    constants_info_[28].data_size = 3833856;
    constants_info_[28].from_folded = false;
    constants_info_[28].shape = {512, 1872};
    constants_info_[28].stride = {1872, 1};
    constants_info_[28].original_fqn = "user_model.ctr_net.tower_layers.0.fc.weight";
    constants_info_[29].name = "getattr_L__self___user_model_ctr_net_tower_layers___0___norm_weight";
    constants_info_[29].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[29].offset = 0;
    constants_info_[29].data_size = 2048;
    constants_info_[29].from_folded = false;
    constants_info_[29].shape = {512};
    constants_info_[29].stride = {1};
    constants_info_[29].original_fqn = "user_model.ctr_net.tower_layers.0.norm.weight";
    constants_info_[30].name = "getattr_L__self___user_model_ctr_net_tower_layers___0___norm_bias";
    constants_info_[30].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[30].offset = 0;
    constants_info_[30].data_size = 2048;
    constants_info_[30].from_folded = false;
    constants_info_[30].shape = {512};
    constants_info_[30].stride = {1};
    constants_info_[30].original_fqn = "user_model.ctr_net.tower_layers.0.norm.bias";
    constants_info_[31].name = "getattr_L__self___user_model_ctr_net_tower_layers___1___fc_weight";
    constants_info_[31].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[31].offset = 0;
    constants_info_[31].data_size = 524288;
    constants_info_[31].from_folded = false;
    constants_info_[31].shape = {256, 512};
    constants_info_[31].stride = {512, 1};
    constants_info_[31].original_fqn = "user_model.ctr_net.tower_layers.1.fc.weight";
    constants_info_[32].name = "getattr_L__self___user_model_ctr_net_tower_layers___1___norm_weight";
    constants_info_[32].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[32].offset = 0;
    constants_info_[32].data_size = 1024;
    constants_info_[32].from_folded = false;
    constants_info_[32].shape = {256};
    constants_info_[32].stride = {1};
    constants_info_[32].original_fqn = "user_model.ctr_net.tower_layers.1.norm.weight";
    constants_info_[33].name = "getattr_L__self___user_model_ctr_net_tower_layers___1___norm_bias";
    constants_info_[33].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[33].offset = 0;
    constants_info_[33].data_size = 1024;
    constants_info_[33].from_folded = false;
    constants_info_[33].shape = {256};
    constants_info_[33].stride = {1};
    constants_info_[33].original_fqn = "user_model.ctr_net.tower_layers.1.norm.bias";
    constants_info_[34].name = "getattr_L__self___user_model_ctr_net_tower_layers___2___fc_weight";
    constants_info_[34].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[34].offset = 0;
    constants_info_[34].data_size = 131072;
    constants_info_[34].from_folded = false;
    constants_info_[34].shape = {128, 256};
    constants_info_[34].stride = {256, 1};
    constants_info_[34].original_fqn = "user_model.ctr_net.tower_layers.2.fc.weight";
    constants_info_[35].name = "getattr_L__self___user_model_ctr_net_tower_layers___2___norm_weight";
    constants_info_[35].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[35].offset = 0;
    constants_info_[35].data_size = 512;
    constants_info_[35].from_folded = false;
    constants_info_[35].shape = {128};
    constants_info_[35].stride = {1};
    constants_info_[35].original_fqn = "user_model.ctr_net.tower_layers.2.norm.weight";
    constants_info_[36].name = "getattr_L__self___user_model_ctr_net_tower_layers___2___norm_bias";
    constants_info_[36].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[36].offset = 0;
    constants_info_[36].data_size = 512;
    constants_info_[36].from_folded = false;
    constants_info_[36].shape = {128};
    constants_info_[36].stride = {1};
    constants_info_[36].original_fqn = "user_model.ctr_net.tower_layers.2.norm.bias";
    constants_info_[37].name = "getattr_L__self___user_model_click_net_tower_layers___0___fc_weight";
    constants_info_[37].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[37].offset = 0;
    constants_info_[37].data_size = 3833856;
    constants_info_[37].from_folded = false;
    constants_info_[37].shape = {512, 1872};
    constants_info_[37].stride = {1872, 1};
    constants_info_[37].original_fqn = "user_model.click_net.tower_layers.0.fc.weight";
    constants_info_[38].name = "getattr_L__self___user_model_click_net_tower_layers___0___norm_weight";
    constants_info_[38].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[38].offset = 0;
    constants_info_[38].data_size = 2048;
    constants_info_[38].from_folded = false;
    constants_info_[38].shape = {512};
    constants_info_[38].stride = {1};
    constants_info_[38].original_fqn = "user_model.click_net.tower_layers.0.norm.weight";
    constants_info_[39].name = "getattr_L__self___user_model_click_net_tower_layers___0___norm_bias";
    constants_info_[39].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[39].offset = 0;
    constants_info_[39].data_size = 2048;
    constants_info_[39].from_folded = false;
    constants_info_[39].shape = {512};
    constants_info_[39].stride = {1};
    constants_info_[39].original_fqn = "user_model.click_net.tower_layers.0.norm.bias";
    constants_info_[40].name = "getattr_L__self___user_model_click_net_tower_layers___1___fc_weight";
    constants_info_[40].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[40].offset = 0;
    constants_info_[40].data_size = 524288;
    constants_info_[40].from_folded = false;
    constants_info_[40].shape = {256, 512};
    constants_info_[40].stride = {512, 1};
    constants_info_[40].original_fqn = "user_model.click_net.tower_layers.1.fc.weight";
    constants_info_[41].name = "getattr_L__self___user_model_click_net_tower_layers___1___norm_weight";
    constants_info_[41].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[41].offset = 0;
    constants_info_[41].data_size = 1024;
    constants_info_[41].from_folded = false;
    constants_info_[41].shape = {256};
    constants_info_[41].stride = {1};
    constants_info_[41].original_fqn = "user_model.click_net.tower_layers.1.norm.weight";
    constants_info_[42].name = "getattr_L__self___user_model_click_net_tower_layers___1___norm_bias";
    constants_info_[42].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[42].offset = 0;
    constants_info_[42].data_size = 1024;
    constants_info_[42].from_folded = false;
    constants_info_[42].shape = {256};
    constants_info_[42].stride = {1};
    constants_info_[42].original_fqn = "user_model.click_net.tower_layers.1.norm.bias";
    constants_info_[43].name = "getattr_L__self___user_model_click_net_tower_layers___2___fc_weight";
    constants_info_[43].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[43].offset = 0;
    constants_info_[43].data_size = 131072;
    constants_info_[43].from_folded = false;
    constants_info_[43].shape = {128, 256};
    constants_info_[43].stride = {256, 1};
    constants_info_[43].original_fqn = "user_model.click_net.tower_layers.2.fc.weight";
    constants_info_[44].name = "getattr_L__self___user_model_click_net_tower_layers___2___norm_weight";
    constants_info_[44].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[44].offset = 0;
    constants_info_[44].data_size = 512;
    constants_info_[44].from_folded = false;
    constants_info_[44].shape = {128};
    constants_info_[44].stride = {1};
    constants_info_[44].original_fqn = "user_model.click_net.tower_layers.2.norm.weight";
    constants_info_[45].name = "getattr_L__self___user_model_click_net_tower_layers___2___norm_bias";
    constants_info_[45].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[45].offset = 0;
    constants_info_[45].data_size = 512;
    constants_info_[45].from_folded = false;
    constants_info_[45].shape = {128};
    constants_info_[45].stride = {1};
    constants_info_[45].original_fqn = "user_model.click_net.tower_layers.2.norm.bias";
    constants_info_[46].name = "getattr_L__self___user_model_page_net_tower_layers___0___fc_weight";
    constants_info_[46].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[46].offset = 0;
    constants_info_[46].data_size = 3833856;
    constants_info_[46].from_folded = false;
    constants_info_[46].shape = {512, 1872};
    constants_info_[46].stride = {1872, 1};
    constants_info_[46].original_fqn = "user_model.page_net.tower_layers.0.fc.weight";
    constants_info_[47].name = "getattr_L__self___user_model_page_net_tower_layers___0___norm_weight";
    constants_info_[47].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[47].offset = 0;
    constants_info_[47].data_size = 2048;
    constants_info_[47].from_folded = false;
    constants_info_[47].shape = {512};
    constants_info_[47].stride = {1};
    constants_info_[47].original_fqn = "user_model.page_net.tower_layers.0.norm.weight";
    constants_info_[48].name = "getattr_L__self___user_model_page_net_tower_layers___0___norm_bias";
    constants_info_[48].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[48].offset = 0;
    constants_info_[48].data_size = 2048;
    constants_info_[48].from_folded = false;
    constants_info_[48].shape = {512};
    constants_info_[48].stride = {1};
    constants_info_[48].original_fqn = "user_model.page_net.tower_layers.0.norm.bias";
    constants_info_[49].name = "getattr_L__self___user_model_page_net_tower_layers___1___fc_weight";
    constants_info_[49].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[49].offset = 0;
    constants_info_[49].data_size = 524288;
    constants_info_[49].from_folded = false;
    constants_info_[49].shape = {256, 512};
    constants_info_[49].stride = {512, 1};
    constants_info_[49].original_fqn = "user_model.page_net.tower_layers.1.fc.weight";
    constants_info_[50].name = "getattr_L__self___user_model_page_net_tower_layers___1___norm_weight";
    constants_info_[50].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[50].offset = 0;
    constants_info_[50].data_size = 1024;
    constants_info_[50].from_folded = false;
    constants_info_[50].shape = {256};
    constants_info_[50].stride = {1};
    constants_info_[50].original_fqn = "user_model.page_net.tower_layers.1.norm.weight";
    constants_info_[51].name = "getattr_L__self___user_model_page_net_tower_layers___1___norm_bias";
    constants_info_[51].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[51].offset = 0;
    constants_info_[51].data_size = 1024;
    constants_info_[51].from_folded = false;
    constants_info_[51].shape = {256};
    constants_info_[51].stride = {1};
    constants_info_[51].original_fqn = "user_model.page_net.tower_layers.1.norm.bias";
    constants_info_[52].name = "getattr_L__self___user_model_page_net_tower_layers___2___fc_weight";
    constants_info_[52].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[52].offset = 0;
    constants_info_[52].data_size = 131072;
    constants_info_[52].from_folded = false;
    constants_info_[52].shape = {128, 256};
    constants_info_[52].stride = {256, 1};
    constants_info_[52].original_fqn = "user_model.page_net.tower_layers.2.fc.weight";
    constants_info_[53].name = "getattr_L__self___user_model_page_net_tower_layers___2___norm_weight";
    constants_info_[53].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[53].offset = 0;
    constants_info_[53].data_size = 512;
    constants_info_[53].from_folded = false;
    constants_info_[53].shape = {128};
    constants_info_[53].stride = {1};
    constants_info_[53].original_fqn = "user_model.page_net.tower_layers.2.norm.weight";
    constants_info_[54].name = "getattr_L__self___user_model_page_net_tower_layers___2___norm_bias";
    constants_info_[54].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[54].offset = 0;
    constants_info_[54].data_size = 512;
    constants_info_[54].from_folded = false;
    constants_info_[54].shape = {128};
    constants_info_[54].stride = {1};
    constants_info_[54].original_fqn = "user_model.page_net.tower_layers.2.norm.bias";
    constants_info_[55].name = "getattr_L__self___user_model_pay_net_tower_layers___0___fc_weight";
    constants_info_[55].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[55].offset = 0;
    constants_info_[55].data_size = 3833856;
    constants_info_[55].from_folded = false;
    constants_info_[55].shape = {512, 1872};
    constants_info_[55].stride = {1872, 1};
    constants_info_[55].original_fqn = "user_model.pay_net.tower_layers.0.fc.weight";
    constants_info_[56].name = "getattr_L__self___user_model_pay_net_tower_layers___0___norm_weight";
    constants_info_[56].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[56].offset = 0;
    constants_info_[56].data_size = 2048;
    constants_info_[56].from_folded = false;
    constants_info_[56].shape = {512};
    constants_info_[56].stride = {1};
    constants_info_[56].original_fqn = "user_model.pay_net.tower_layers.0.norm.weight";
    constants_info_[57].name = "getattr_L__self___user_model_pay_net_tower_layers___0___norm_bias";
    constants_info_[57].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[57].offset = 0;
    constants_info_[57].data_size = 2048;
    constants_info_[57].from_folded = false;
    constants_info_[57].shape = {512};
    constants_info_[57].stride = {1};
    constants_info_[57].original_fqn = "user_model.pay_net.tower_layers.0.norm.bias";
    constants_info_[58].name = "getattr_L__self___user_model_pay_net_tower_layers___1___fc_weight";
    constants_info_[58].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[58].offset = 0;
    constants_info_[58].data_size = 524288;
    constants_info_[58].from_folded = false;
    constants_info_[58].shape = {256, 512};
    constants_info_[58].stride = {512, 1};
    constants_info_[58].original_fqn = "user_model.pay_net.tower_layers.1.fc.weight";
    constants_info_[59].name = "getattr_L__self___user_model_pay_net_tower_layers___1___norm_weight";
    constants_info_[59].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[59].offset = 0;
    constants_info_[59].data_size = 1024;
    constants_info_[59].from_folded = false;
    constants_info_[59].shape = {256};
    constants_info_[59].stride = {1};
    constants_info_[59].original_fqn = "user_model.pay_net.tower_layers.1.norm.weight";
    constants_info_[60].name = "getattr_L__self___user_model_pay_net_tower_layers___1___norm_bias";
    constants_info_[60].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[60].offset = 0;
    constants_info_[60].data_size = 1024;
    constants_info_[60].from_folded = false;
    constants_info_[60].shape = {256};
    constants_info_[60].stride = {1};
    constants_info_[60].original_fqn = "user_model.pay_net.tower_layers.1.norm.bias";
    constants_info_[61].name = "getattr_L__self___user_model_pay_net_tower_layers___2___fc_weight";
    constants_info_[61].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[61].offset = 0;
    constants_info_[61].data_size = 131072;
    constants_info_[61].from_folded = false;
    constants_info_[61].shape = {128, 256};
    constants_info_[61].stride = {256, 1};
    constants_info_[61].original_fqn = "user_model.pay_net.tower_layers.2.fc.weight";
    constants_info_[62].name = "getattr_L__self___user_model_pay_net_tower_layers___2___norm_weight";
    constants_info_[62].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[62].offset = 0;
    constants_info_[62].data_size = 512;
    constants_info_[62].from_folded = false;
    constants_info_[62].shape = {128};
    constants_info_[62].stride = {1};
    constants_info_[62].original_fqn = "user_model.pay_net.tower_layers.2.norm.weight";
    constants_info_[63].name = "getattr_L__self___user_model_pay_net_tower_layers___2___norm_bias";
    constants_info_[63].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[63].offset = 0;
    constants_info_[63].data_size = 512;
    constants_info_[63].from_folded = false;
    constants_info_[63].shape = {128};
    constants_info_[63].stride = {1};
    constants_info_[63].original_fqn = "user_model.pay_net.tower_layers.2.norm.bias";
    constants_info_[64].name = "getattr_L__self___user_model_bias_net_tower_layers___0___fc_weight";
    constants_info_[64].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[64].offset = 0;
    constants_info_[64].data_size = 581632;
    constants_info_[64].from_folded = false;
    constants_info_[64].shape = {256, 568};
    constants_info_[64].stride = {568, 1};
    constants_info_[64].original_fqn = "user_model.bias_net.tower_layers.0.fc.weight";
    constants_info_[65].name = "getattr_L__self___user_model_bias_net_tower_layers___0___norm_weight";
    constants_info_[65].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[65].offset = 0;
    constants_info_[65].data_size = 1024;
    constants_info_[65].from_folded = false;
    constants_info_[65].shape = {256};
    constants_info_[65].stride = {1};
    constants_info_[65].original_fqn = "user_model.bias_net.tower_layers.0.norm.weight";
    constants_info_[66].name = "getattr_L__self___user_model_bias_net_tower_layers___0___norm_bias";
    constants_info_[66].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[66].offset = 0;
    constants_info_[66].data_size = 1024;
    constants_info_[66].from_folded = false;
    constants_info_[66].shape = {256};
    constants_info_[66].stride = {1};
    constants_info_[66].original_fqn = "user_model.bias_net.tower_layers.0.norm.bias";
    constants_info_[67].name = "getattr_L__self___user_model_bias_net_tower_layers___1___fc_weight";
    constants_info_[67].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[67].offset = 0;
    constants_info_[67].data_size = 131072;
    constants_info_[67].from_folded = false;
    constants_info_[67].shape = {128, 256};
    constants_info_[67].stride = {256, 1};
    constants_info_[67].original_fqn = "user_model.bias_net.tower_layers.1.fc.weight";
    constants_info_[68].name = "getattr_L__self___user_model_bias_net_tower_layers___1___norm_weight";
    constants_info_[68].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[68].offset = 0;
    constants_info_[68].data_size = 512;
    constants_info_[68].from_folded = false;
    constants_info_[68].shape = {128};
    constants_info_[68].stride = {1};
    constants_info_[68].original_fqn = "user_model.bias_net.tower_layers.1.norm.weight";
    constants_info_[69].name = "getattr_L__self___user_model_bias_net_tower_layers___1___norm_bias";
    constants_info_[69].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[69].offset = 0;
    constants_info_[69].data_size = 512;
    constants_info_[69].from_folded = false;
    constants_info_[69].shape = {128};
    constants_info_[69].stride = {1};
    constants_info_[69].original_fqn = "user_model.bias_net.tower_layers.1.norm.bias";
    constants_info_[70].name = "L__self___user_model_ctr_head_weight";
    constants_info_[70].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[70].offset = 0;
    constants_info_[70].data_size = 512;
    constants_info_[70].from_folded = false;
    constants_info_[70].shape = {1, 128};
    constants_info_[70].stride = {128, 1};
    constants_info_[70].original_fqn = "user_model.ctr_head.weight";
    constants_info_[71].name = "L__self___user_model_click_head_weight";
    constants_info_[71].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[71].offset = 0;
    constants_info_[71].data_size = 512;
    constants_info_[71].from_folded = false;
    constants_info_[71].shape = {1, 128};
    constants_info_[71].stride = {128, 1};
    constants_info_[71].original_fqn = "user_model.click_head.weight";
    constants_info_[72].name = "L__self___user_model_page_head_weight";
    constants_info_[72].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[72].offset = 0;
    constants_info_[72].data_size = 512;
    constants_info_[72].from_folded = false;
    constants_info_[72].shape = {1, 128};
    constants_info_[72].stride = {128, 1};
    constants_info_[72].original_fqn = "user_model.page_head.weight";
    constants_info_[73].name = "L__self___user_model_pay_head_weight";
    constants_info_[73].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[73].offset = 0;
    constants_info_[73].data_size = 512;
    constants_info_[73].from_folded = false;
    constants_info_[73].shape = {1, 128};
    constants_info_[73].stride = {128, 1};
    constants_info_[73].original_fqn = "user_model.pay_head.weight";
    constants_info_[74].name = "L__self___user_model_ctr_bias_head_weight";
    constants_info_[74].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[74].offset = 0;
    constants_info_[74].data_size = 512;
    constants_info_[74].from_folded = false;
    constants_info_[74].shape = {1, 128};
    constants_info_[74].stride = {128, 1};
    constants_info_[74].original_fqn = "user_model.ctr_bias_head.weight";
    constants_info_[75].name = "L__self___user_model_click_bias_head_weight";
    constants_info_[75].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[75].offset = 0;
    constants_info_[75].data_size = 512;
    constants_info_[75].from_folded = false;
    constants_info_[75].shape = {1, 128};
    constants_info_[75].stride = {128, 1};
    constants_info_[75].original_fqn = "user_model.click_bias_head.weight";
    constants_info_[76].name = "L__self___user_model_page_bias_head_weight";
    constants_info_[76].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[76].offset = 0;
    constants_info_[76].data_size = 512;
    constants_info_[76].from_folded = false;
    constants_info_[76].shape = {1, 128};
    constants_info_[76].stride = {128, 1};
    constants_info_[76].original_fqn = "user_model.page_bias_head.weight";
    constants_info_[77].name = "L__self___user_model_pay_bias_head_weight";
    constants_info_[77].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[77].offset = 0;
    constants_info_[77].data_size = 512;
    constants_info_[77].from_folded = false;
    constants_info_[77].shape = {1, 128};
    constants_info_[77].stride = {128, 1};
    constants_info_[77].original_fqn = "user_model.pay_bias_head.weight";
    constants_info_[78].name = "getattr_L__self___user_model_ctr_net_tower_layers___0___norm_running_mean";
    constants_info_[78].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[78].offset = 0;
    constants_info_[78].data_size = 2048;
    constants_info_[78].from_folded = false;
    constants_info_[78].shape = {512};
    constants_info_[78].stride = {1};
    constants_info_[78].original_fqn = "user_model.ctr_net.tower_layers.0.norm.running_mean";
    constants_info_[79].name = "getattr_L__self___user_model_ctr_net_tower_layers___0___norm_running_var";
    constants_info_[79].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[79].offset = 0;
    constants_info_[79].data_size = 2048;
    constants_info_[79].from_folded = false;
    constants_info_[79].shape = {512};
    constants_info_[79].stride = {1};
    constants_info_[79].original_fqn = "user_model.ctr_net.tower_layers.0.norm.running_var";
    constants_info_[80].name = "getattr_L__self___user_model_ctr_net_tower_layers___1___norm_running_mean";
    constants_info_[80].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[80].offset = 0;
    constants_info_[80].data_size = 1024;
    constants_info_[80].from_folded = false;
    constants_info_[80].shape = {256};
    constants_info_[80].stride = {1};
    constants_info_[80].original_fqn = "user_model.ctr_net.tower_layers.1.norm.running_mean";
    constants_info_[81].name = "getattr_L__self___user_model_ctr_net_tower_layers___1___norm_running_var";
    constants_info_[81].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[81].offset = 0;
    constants_info_[81].data_size = 1024;
    constants_info_[81].from_folded = false;
    constants_info_[81].shape = {256};
    constants_info_[81].stride = {1};
    constants_info_[81].original_fqn = "user_model.ctr_net.tower_layers.1.norm.running_var";
    constants_info_[82].name = "getattr_L__self___user_model_ctr_net_tower_layers___2___norm_running_mean";
    constants_info_[82].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[82].offset = 0;
    constants_info_[82].data_size = 512;
    constants_info_[82].from_folded = false;
    constants_info_[82].shape = {128};
    constants_info_[82].stride = {1};
    constants_info_[82].original_fqn = "user_model.ctr_net.tower_layers.2.norm.running_mean";
    constants_info_[83].name = "getattr_L__self___user_model_ctr_net_tower_layers___2___norm_running_var";
    constants_info_[83].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[83].offset = 0;
    constants_info_[83].data_size = 512;
    constants_info_[83].from_folded = false;
    constants_info_[83].shape = {128};
    constants_info_[83].stride = {1};
    constants_info_[83].original_fqn = "user_model.ctr_net.tower_layers.2.norm.running_var";
    constants_info_[84].name = "getattr_L__self___user_model_click_net_tower_layers___0___norm_running_mean";
    constants_info_[84].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[84].offset = 0;
    constants_info_[84].data_size = 2048;
    constants_info_[84].from_folded = false;
    constants_info_[84].shape = {512};
    constants_info_[84].stride = {1};
    constants_info_[84].original_fqn = "user_model.click_net.tower_layers.0.norm.running_mean";
    constants_info_[85].name = "getattr_L__self___user_model_click_net_tower_layers___0___norm_running_var";
    constants_info_[85].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[85].offset = 0;
    constants_info_[85].data_size = 2048;
    constants_info_[85].from_folded = false;
    constants_info_[85].shape = {512};
    constants_info_[85].stride = {1};
    constants_info_[85].original_fqn = "user_model.click_net.tower_layers.0.norm.running_var";
    constants_info_[86].name = "getattr_L__self___user_model_click_net_tower_layers___1___norm_running_mean";
    constants_info_[86].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[86].offset = 0;
    constants_info_[86].data_size = 1024;
    constants_info_[86].from_folded = false;
    constants_info_[86].shape = {256};
    constants_info_[86].stride = {1};
    constants_info_[86].original_fqn = "user_model.click_net.tower_layers.1.norm.running_mean";
    constants_info_[87].name = "getattr_L__self___user_model_click_net_tower_layers___1___norm_running_var";
    constants_info_[87].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[87].offset = 0;
    constants_info_[87].data_size = 1024;
    constants_info_[87].from_folded = false;
    constants_info_[87].shape = {256};
    constants_info_[87].stride = {1};
    constants_info_[87].original_fqn = "user_model.click_net.tower_layers.1.norm.running_var";
    constants_info_[88].name = "getattr_L__self___user_model_click_net_tower_layers___2___norm_running_mean";
    constants_info_[88].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[88].offset = 0;
    constants_info_[88].data_size = 512;
    constants_info_[88].from_folded = false;
    constants_info_[88].shape = {128};
    constants_info_[88].stride = {1};
    constants_info_[88].original_fqn = "user_model.click_net.tower_layers.2.norm.running_mean";
    constants_info_[89].name = "getattr_L__self___user_model_click_net_tower_layers___2___norm_running_var";
    constants_info_[89].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[89].offset = 0;
    constants_info_[89].data_size = 512;
    constants_info_[89].from_folded = false;
    constants_info_[89].shape = {128};
    constants_info_[89].stride = {1};
    constants_info_[89].original_fqn = "user_model.click_net.tower_layers.2.norm.running_var";
    constants_info_[90].name = "getattr_L__self___user_model_page_net_tower_layers___0___norm_running_mean";
    constants_info_[90].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[90].offset = 0;
    constants_info_[90].data_size = 2048;
    constants_info_[90].from_folded = false;
    constants_info_[90].shape = {512};
    constants_info_[90].stride = {1};
    constants_info_[90].original_fqn = "user_model.page_net.tower_layers.0.norm.running_mean";
    constants_info_[91].name = "getattr_L__self___user_model_page_net_tower_layers___0___norm_running_var";
    constants_info_[91].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[91].offset = 0;
    constants_info_[91].data_size = 2048;
    constants_info_[91].from_folded = false;
    constants_info_[91].shape = {512};
    constants_info_[91].stride = {1};
    constants_info_[91].original_fqn = "user_model.page_net.tower_layers.0.norm.running_var";
    constants_info_[92].name = "getattr_L__self___user_model_page_net_tower_layers___1___norm_running_mean";
    constants_info_[92].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[92].offset = 0;
    constants_info_[92].data_size = 1024;
    constants_info_[92].from_folded = false;
    constants_info_[92].shape = {256};
    constants_info_[92].stride = {1};
    constants_info_[92].original_fqn = "user_model.page_net.tower_layers.1.norm.running_mean";
    constants_info_[93].name = "getattr_L__self___user_model_page_net_tower_layers___1___norm_running_var";
    constants_info_[93].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[93].offset = 0;
    constants_info_[93].data_size = 1024;
    constants_info_[93].from_folded = false;
    constants_info_[93].shape = {256};
    constants_info_[93].stride = {1};
    constants_info_[93].original_fqn = "user_model.page_net.tower_layers.1.norm.running_var";
    constants_info_[94].name = "getattr_L__self___user_model_page_net_tower_layers___2___norm_running_mean";
    constants_info_[94].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[94].offset = 0;
    constants_info_[94].data_size = 512;
    constants_info_[94].from_folded = false;
    constants_info_[94].shape = {128};
    constants_info_[94].stride = {1};
    constants_info_[94].original_fqn = "user_model.page_net.tower_layers.2.norm.running_mean";
    constants_info_[95].name = "getattr_L__self___user_model_page_net_tower_layers___2___norm_running_var";
    constants_info_[95].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[95].offset = 0;
    constants_info_[95].data_size = 512;
    constants_info_[95].from_folded = false;
    constants_info_[95].shape = {128};
    constants_info_[95].stride = {1};
    constants_info_[95].original_fqn = "user_model.page_net.tower_layers.2.norm.running_var";
    constants_info_[96].name = "getattr_L__self___user_model_pay_net_tower_layers___0___norm_running_mean";
    constants_info_[96].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[96].offset = 0;
    constants_info_[96].data_size = 2048;
    constants_info_[96].from_folded = false;
    constants_info_[96].shape = {512};
    constants_info_[96].stride = {1};
    constants_info_[96].original_fqn = "user_model.pay_net.tower_layers.0.norm.running_mean";
    constants_info_[97].name = "getattr_L__self___user_model_pay_net_tower_layers___0___norm_running_var";
    constants_info_[97].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[97].offset = 0;
    constants_info_[97].data_size = 2048;
    constants_info_[97].from_folded = false;
    constants_info_[97].shape = {512};
    constants_info_[97].stride = {1};
    constants_info_[97].original_fqn = "user_model.pay_net.tower_layers.0.norm.running_var";
    constants_info_[98].name = "getattr_L__self___user_model_pay_net_tower_layers___1___norm_running_mean";
    constants_info_[98].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[98].offset = 0;
    constants_info_[98].data_size = 1024;
    constants_info_[98].from_folded = false;
    constants_info_[98].shape = {256};
    constants_info_[98].stride = {1};
    constants_info_[98].original_fqn = "user_model.pay_net.tower_layers.1.norm.running_mean";
    constants_info_[99].name = "getattr_L__self___user_model_pay_net_tower_layers___1___norm_running_var";
    constants_info_[99].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[99].offset = 0;
    constants_info_[99].data_size = 1024;
    constants_info_[99].from_folded = false;
    constants_info_[99].shape = {256};
    constants_info_[99].stride = {1};
    constants_info_[99].original_fqn = "user_model.pay_net.tower_layers.1.norm.running_var";
    constants_info_[100].name = "getattr_L__self___user_model_pay_net_tower_layers___2___norm_running_mean";
    constants_info_[100].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[100].offset = 0;
    constants_info_[100].data_size = 512;
    constants_info_[100].from_folded = false;
    constants_info_[100].shape = {128};
    constants_info_[100].stride = {1};
    constants_info_[100].original_fqn = "user_model.pay_net.tower_layers.2.norm.running_mean";
    constants_info_[101].name = "getattr_L__self___user_model_pay_net_tower_layers___2___norm_running_var";
    constants_info_[101].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[101].offset = 0;
    constants_info_[101].data_size = 512;
    constants_info_[101].from_folded = false;
    constants_info_[101].shape = {128};
    constants_info_[101].stride = {1};
    constants_info_[101].original_fqn = "user_model.pay_net.tower_layers.2.norm.running_var";
    constants_info_[102].name = "getattr_L__self___user_model_bias_net_tower_layers___0___norm_running_mean";
    constants_info_[102].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[102].offset = 0;
    constants_info_[102].data_size = 1024;
    constants_info_[102].from_folded = false;
    constants_info_[102].shape = {256};
    constants_info_[102].stride = {1};
    constants_info_[102].original_fqn = "user_model.bias_net.tower_layers.0.norm.running_mean";
    constants_info_[103].name = "getattr_L__self___user_model_bias_net_tower_layers___0___norm_running_var";
    constants_info_[103].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[103].offset = 0;
    constants_info_[103].data_size = 1024;
    constants_info_[103].from_folded = false;
    constants_info_[103].shape = {256};
    constants_info_[103].stride = {1};
    constants_info_[103].original_fqn = "user_model.bias_net.tower_layers.0.norm.running_var";
    constants_info_[104].name = "getattr_L__self___user_model_bias_net_tower_layers___1___norm_running_mean";
    constants_info_[104].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[104].offset = 0;
    constants_info_[104].data_size = 512;
    constants_info_[104].from_folded = false;
    constants_info_[104].shape = {128};
    constants_info_[104].stride = {1};
    constants_info_[104].original_fqn = "user_model.bias_net.tower_layers.1.norm.running_mean";
    constants_info_[105].name = "getattr_L__self___user_model_bias_net_tower_layers___1___norm_running_var";
    constants_info_[105].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[105].offset = 0;
    constants_info_[105].data_size = 512;
    constants_info_[105].from_folded = false;
    constants_info_[105].shape = {128};
    constants_info_[105].stride = {1};
    constants_info_[105].original_fqn = "user_model.bias_net.tower_layers.1.norm.running_var";
    update_constants_map(std::move(constants_map));
    update_constants_array(std::move(constants_array));
    in_spec_ = "[1, {\"type\": \"builtins.tuple\", \"context\": \"null\", \"children_spec\": [{\"type\": \"builtins.tuple\", \"context\": \"null\", \"children_spec\": [{\"type\": null, \"context\": null, \"children_spec\": []}, {\"type\": null, \"context\": null, \"children_spec\": []}, {\"type\": null, \"context\": null, \"children_spec\": []}, {\"type\": null, \"context\": null, \"children_spec\": []}, {\"type\": null, \"context\": null, \"children_spec\": []}, {\"type\": null, \"context\": null, \"children_spec\": []}, {\"type\": null, \"context\": null, \"children_spec\": []}, {\"type\": null, \"context\": null, \"children_spec\": []}, {\"type\": null, \"context\": null, \"children_spec\": []}]}, {\"type\": \"builtins.dict\", \"context\": \"[]\", \"children_spec\": []}]}]";
    out_spec_ = "[1, {\"type\": \"builtins.tuple\", \"context\": \"null\", \"children_spec\": [{\"type\": null, \"context\": null, \"children_spec\": []}, {\"type\": null, \"context\": null, \"children_spec\": []}, {\"type\": null, \"context\": null, \"children_spec\": []}, {\"type\": null, \"context\": null, \"children_spec\": []}]}]";
    outputs_info_[0].name = "output0";
    outputs_info_[1].name = "output1";
    outputs_info_[2].name = "output2";
    outputs_info_[3].name = "output3";
    this->kernels_ = std::make_unique<AOTInductorModelKernels>();
}

std::unordered_map<std::string, AtenTensorHandle> AOTInductorModel::const_run_impl(
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor,
    bool initialization
) {

    if (!initialization) {
        std::cerr << "[WARNING] Calling constant_folding in model, but compiled with config: "
                  << "aot_inductor.use_runtime_constant_folding=False\n";
    }
    return {};
}

void AOTInductorModel::_const_run_impl(
    std::vector<AtenTensorHandle>& output_handles,
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor
) {}

void AOTInductorModel::run_impl(
    AtenTensorHandle*
        input_handles, // array of input AtenTensorHandle; handles
                        // are stolen; the array itself is borrowed
    AtenTensorHandle*
        output_handles, // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor
) {

    auto inputs = alloc_tensors_by_stealing_from_handles(input_handles, 9);
    auto arg122_1 = std::move(inputs[0]);
    auto arg123_1 = std::move(inputs[1]);
    auto arg124_1 = std::move(inputs[2]);
    auto arg125_1 = std::move(inputs[3]);
    auto arg126_1 = std::move(inputs[4]);
    auto arg127_1 = std::move(inputs[5]);
    auto arg128_1 = std::move(inputs[6]);
    auto arg129_1 = std::move(inputs[7]);
    auto arg130_1 = std::move(inputs[8]);
    auto L__self___user_model_feedforwards_item_clk_seq_fc1_linear_weight = *tensor_handle_to_tensor_pointer(constants_->at(0));
    auto L__self___user_model_feedforwards_item_clk_seq_fc1_linear_bias = *tensor_handle_to_tensor_pointer(constants_->at(1));
    auto L__self___user_model_feedforwards_item_clk_seq_fc2_linear_weight = *tensor_handle_to_tensor_pointer(constants_->at(2));
    auto L__self___user_model_feedforwards_item_clk_seq_fc2_linear_bias = *tensor_handle_to_tensor_pointer(constants_->at(3));
    auto L__self___user_model_feedforwards_query_seq_fc1_linear_weight = *tensor_handle_to_tensor_pointer(constants_->at(4));
    auto L__self___user_model_feedforwards_query_seq_fc1_linear_bias = *tensor_handle_to_tensor_pointer(constants_->at(5));
    auto L__self___user_model_feedforwards_query_seq_fc2_linear_weight = *tensor_handle_to_tensor_pointer(constants_->at(6));
    auto L__self___user_model_feedforwards_query_seq_fc2_linear_bias = *tensor_handle_to_tensor_pointer(constants_->at(7));
    auto L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_q_linear_weight = *tensor_handle_to_tensor_pointer(constants_->at(8));
    auto L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_q_linear_bias = *tensor_handle_to_tensor_pointer(constants_->at(9));
    auto L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_k_linear_weight = *tensor_handle_to_tensor_pointer(constants_->at(10));
    auto L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_k_linear_bias = *tensor_handle_to_tensor_pointer(constants_->at(11));
    auto L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_v_linear_weight = *tensor_handle_to_tensor_pointer(constants_->at(12));
    auto L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_v_linear_bias = *tensor_handle_to_tensor_pointer(constants_->at(13));
    auto L__self___user_model_multi_h_attens_item_clk_seq_ta_q_layer_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(14));
    auto L__self___user_model_multi_h_attens_item_clk_seq_ta_q_layer_norm_bias = *tensor_handle_to_tensor_pointer(constants_->at(15));
    auto L__self___user_model_multi_h_attens_item_clk_seq_ta_k_layer_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(16));
    auto L__self___user_model_multi_h_attens_item_clk_seq_ta_k_layer_norm_bias = *tensor_handle_to_tensor_pointer(constants_->at(17));
    auto L__self___user_model_multi_h_attens_query_seq_ta_proj_q_linear_weight = *tensor_handle_to_tensor_pointer(constants_->at(18));
    auto L__self___user_model_multi_h_attens_query_seq_ta_proj_q_linear_bias = *tensor_handle_to_tensor_pointer(constants_->at(19));
    auto L__self___user_model_multi_h_attens_query_seq_ta_proj_k_linear_weight = *tensor_handle_to_tensor_pointer(constants_->at(20));
    auto L__self___user_model_multi_h_attens_query_seq_ta_proj_k_linear_bias = *tensor_handle_to_tensor_pointer(constants_->at(21));
    auto L__self___user_model_multi_h_attens_query_seq_ta_proj_v_linear_weight = *tensor_handle_to_tensor_pointer(constants_->at(22));
    auto L__self___user_model_multi_h_attens_query_seq_ta_proj_v_linear_bias = *tensor_handle_to_tensor_pointer(constants_->at(23));
    auto L__self___user_model_multi_h_attens_query_seq_ta_q_layer_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(24));
    auto L__self___user_model_multi_h_attens_query_seq_ta_q_layer_norm_bias = *tensor_handle_to_tensor_pointer(constants_->at(25));
    auto L__self___user_model_multi_h_attens_query_seq_ta_k_layer_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(26));
    auto L__self___user_model_multi_h_attens_query_seq_ta_k_layer_norm_bias = *tensor_handle_to_tensor_pointer(constants_->at(27));
    auto getattr_L__self___user_model_ctr_net_tower_layers___0___fc_weight = *tensor_handle_to_tensor_pointer(constants_->at(28));
    auto getattr_L__self___user_model_ctr_net_tower_layers___0___norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(29));
    auto getattr_L__self___user_model_ctr_net_tower_layers___0___norm_bias = *tensor_handle_to_tensor_pointer(constants_->at(30));
    auto getattr_L__self___user_model_ctr_net_tower_layers___1___fc_weight = *tensor_handle_to_tensor_pointer(constants_->at(31));
    auto getattr_L__self___user_model_ctr_net_tower_layers___1___norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(32));
    auto getattr_L__self___user_model_ctr_net_tower_layers___1___norm_bias = *tensor_handle_to_tensor_pointer(constants_->at(33));
    auto getattr_L__self___user_model_ctr_net_tower_layers___2___fc_weight = *tensor_handle_to_tensor_pointer(constants_->at(34));
    auto getattr_L__self___user_model_ctr_net_tower_layers___2___norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(35));
    auto getattr_L__self___user_model_ctr_net_tower_layers___2___norm_bias = *tensor_handle_to_tensor_pointer(constants_->at(36));
    auto getattr_L__self___user_model_click_net_tower_layers___0___fc_weight = *tensor_handle_to_tensor_pointer(constants_->at(37));
    auto getattr_L__self___user_model_click_net_tower_layers___0___norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(38));
    auto getattr_L__self___user_model_click_net_tower_layers___0___norm_bias = *tensor_handle_to_tensor_pointer(constants_->at(39));
    auto getattr_L__self___user_model_click_net_tower_layers___1___fc_weight = *tensor_handle_to_tensor_pointer(constants_->at(40));
    auto getattr_L__self___user_model_click_net_tower_layers___1___norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(41));
    auto getattr_L__self___user_model_click_net_tower_layers___1___norm_bias = *tensor_handle_to_tensor_pointer(constants_->at(42));
    auto getattr_L__self___user_model_click_net_tower_layers___2___fc_weight = *tensor_handle_to_tensor_pointer(constants_->at(43));
    auto getattr_L__self___user_model_click_net_tower_layers___2___norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(44));
    auto getattr_L__self___user_model_click_net_tower_layers___2___norm_bias = *tensor_handle_to_tensor_pointer(constants_->at(45));
    auto getattr_L__self___user_model_page_net_tower_layers___0___fc_weight = *tensor_handle_to_tensor_pointer(constants_->at(46));
    auto getattr_L__self___user_model_page_net_tower_layers___0___norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(47));
    auto getattr_L__self___user_model_page_net_tower_layers___0___norm_bias = *tensor_handle_to_tensor_pointer(constants_->at(48));
    auto getattr_L__self___user_model_page_net_tower_layers___1___fc_weight = *tensor_handle_to_tensor_pointer(constants_->at(49));
    auto getattr_L__self___user_model_page_net_tower_layers___1___norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(50));
    auto getattr_L__self___user_model_page_net_tower_layers___1___norm_bias = *tensor_handle_to_tensor_pointer(constants_->at(51));
    auto getattr_L__self___user_model_page_net_tower_layers___2___fc_weight = *tensor_handle_to_tensor_pointer(constants_->at(52));
    auto getattr_L__self___user_model_page_net_tower_layers___2___norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(53));
    auto getattr_L__self___user_model_page_net_tower_layers___2___norm_bias = *tensor_handle_to_tensor_pointer(constants_->at(54));
    auto getattr_L__self___user_model_pay_net_tower_layers___0___fc_weight = *tensor_handle_to_tensor_pointer(constants_->at(55));
    auto getattr_L__self___user_model_pay_net_tower_layers___0___norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(56));
    auto getattr_L__self___user_model_pay_net_tower_layers___0___norm_bias = *tensor_handle_to_tensor_pointer(constants_->at(57));
    auto getattr_L__self___user_model_pay_net_tower_layers___1___fc_weight = *tensor_handle_to_tensor_pointer(constants_->at(58));
    auto getattr_L__self___user_model_pay_net_tower_layers___1___norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(59));
    auto getattr_L__self___user_model_pay_net_tower_layers___1___norm_bias = *tensor_handle_to_tensor_pointer(constants_->at(60));
    auto getattr_L__self___user_model_pay_net_tower_layers___2___fc_weight = *tensor_handle_to_tensor_pointer(constants_->at(61));
    auto getattr_L__self___user_model_pay_net_tower_layers___2___norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(62));
    auto getattr_L__self___user_model_pay_net_tower_layers___2___norm_bias = *tensor_handle_to_tensor_pointer(constants_->at(63));
    auto getattr_L__self___user_model_bias_net_tower_layers___0___fc_weight = *tensor_handle_to_tensor_pointer(constants_->at(64));
    auto getattr_L__self___user_model_bias_net_tower_layers___0___norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(65));
    auto getattr_L__self___user_model_bias_net_tower_layers___0___norm_bias = *tensor_handle_to_tensor_pointer(constants_->at(66));
    auto getattr_L__self___user_model_bias_net_tower_layers___1___fc_weight = *tensor_handle_to_tensor_pointer(constants_->at(67));
    auto getattr_L__self___user_model_bias_net_tower_layers___1___norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(68));
    auto getattr_L__self___user_model_bias_net_tower_layers___1___norm_bias = *tensor_handle_to_tensor_pointer(constants_->at(69));
    auto L__self___user_model_ctr_head_weight = *tensor_handle_to_tensor_pointer(constants_->at(70));
    auto L__self___user_model_click_head_weight = *tensor_handle_to_tensor_pointer(constants_->at(71));
    auto L__self___user_model_page_head_weight = *tensor_handle_to_tensor_pointer(constants_->at(72));
    auto L__self___user_model_pay_head_weight = *tensor_handle_to_tensor_pointer(constants_->at(73));
    auto L__self___user_model_ctr_bias_head_weight = *tensor_handle_to_tensor_pointer(constants_->at(74));
    auto L__self___user_model_click_bias_head_weight = *tensor_handle_to_tensor_pointer(constants_->at(75));
    auto L__self___user_model_page_bias_head_weight = *tensor_handle_to_tensor_pointer(constants_->at(76));
    auto L__self___user_model_pay_bias_head_weight = *tensor_handle_to_tensor_pointer(constants_->at(77));
    auto getattr_L__self___user_model_ctr_net_tower_layers___0___norm_running_mean = *tensor_handle_to_tensor_pointer(constants_->at(78));
    auto getattr_L__self___user_model_ctr_net_tower_layers___0___norm_running_var = *tensor_handle_to_tensor_pointer(constants_->at(79));
    auto getattr_L__self___user_model_ctr_net_tower_layers___1___norm_running_mean = *tensor_handle_to_tensor_pointer(constants_->at(80));
    auto getattr_L__self___user_model_ctr_net_tower_layers___1___norm_running_var = *tensor_handle_to_tensor_pointer(constants_->at(81));
    auto getattr_L__self___user_model_ctr_net_tower_layers___2___norm_running_mean = *tensor_handle_to_tensor_pointer(constants_->at(82));
    auto getattr_L__self___user_model_ctr_net_tower_layers___2___norm_running_var = *tensor_handle_to_tensor_pointer(constants_->at(83));
    auto getattr_L__self___user_model_click_net_tower_layers___0___norm_running_mean = *tensor_handle_to_tensor_pointer(constants_->at(84));
    auto getattr_L__self___user_model_click_net_tower_layers___0___norm_running_var = *tensor_handle_to_tensor_pointer(constants_->at(85));
    auto getattr_L__self___user_model_click_net_tower_layers___1___norm_running_mean = *tensor_handle_to_tensor_pointer(constants_->at(86));
    auto getattr_L__self___user_model_click_net_tower_layers___1___norm_running_var = *tensor_handle_to_tensor_pointer(constants_->at(87));
    auto getattr_L__self___user_model_click_net_tower_layers___2___norm_running_mean = *tensor_handle_to_tensor_pointer(constants_->at(88));
    auto getattr_L__self___user_model_click_net_tower_layers___2___norm_running_var = *tensor_handle_to_tensor_pointer(constants_->at(89));
    auto getattr_L__self___user_model_page_net_tower_layers___0___norm_running_mean = *tensor_handle_to_tensor_pointer(constants_->at(90));
    auto getattr_L__self___user_model_page_net_tower_layers___0___norm_running_var = *tensor_handle_to_tensor_pointer(constants_->at(91));
    auto getattr_L__self___user_model_page_net_tower_layers___1___norm_running_mean = *tensor_handle_to_tensor_pointer(constants_->at(92));
    auto getattr_L__self___user_model_page_net_tower_layers___1___norm_running_var = *tensor_handle_to_tensor_pointer(constants_->at(93));
    auto getattr_L__self___user_model_page_net_tower_layers___2___norm_running_mean = *tensor_handle_to_tensor_pointer(constants_->at(94));
    auto getattr_L__self___user_model_page_net_tower_layers___2___norm_running_var = *tensor_handle_to_tensor_pointer(constants_->at(95));
    auto getattr_L__self___user_model_pay_net_tower_layers___0___norm_running_mean = *tensor_handle_to_tensor_pointer(constants_->at(96));
    auto getattr_L__self___user_model_pay_net_tower_layers___0___norm_running_var = *tensor_handle_to_tensor_pointer(constants_->at(97));
    auto getattr_L__self___user_model_pay_net_tower_layers___1___norm_running_mean = *tensor_handle_to_tensor_pointer(constants_->at(98));
    auto getattr_L__self___user_model_pay_net_tower_layers___1___norm_running_var = *tensor_handle_to_tensor_pointer(constants_->at(99));
    auto getattr_L__self___user_model_pay_net_tower_layers___2___norm_running_mean = *tensor_handle_to_tensor_pointer(constants_->at(100));
    auto getattr_L__self___user_model_pay_net_tower_layers___2___norm_running_var = *tensor_handle_to_tensor_pointer(constants_->at(101));
    auto getattr_L__self___user_model_bias_net_tower_layers___0___norm_running_mean = *tensor_handle_to_tensor_pointer(constants_->at(102));
    auto getattr_L__self___user_model_bias_net_tower_layers___0___norm_running_var = *tensor_handle_to_tensor_pointer(constants_->at(103));
    auto getattr_L__self___user_model_bias_net_tower_layers___1___norm_running_mean = *tensor_handle_to_tensor_pointer(constants_->at(104));
    auto getattr_L__self___user_model_bias_net_tower_layers___1___norm_running_var = *tensor_handle_to_tensor_pointer(constants_->at(105));
    auto arg122_1_size = arg122_1.sizes();
    auto s0 = arg122_1_size[0];
    inputs.clear();
    auto& kernels = static_cast<AOTInductorModelKernels&>(*this->kernels_.get());

    at::cuda::CUDAStreamGuard stream_guard(at::cuda::getStreamFromExternal(stream, this->device_idx_));
    at::Tensor buf0 = at::detail::empty_strided_cuda({s0, 128L}, {128L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [user_model_multi_h_attens_item_clk_seq_ta_proj_q_linear], Original ATen: [aten.addmm]
    at::addmm_out(buf0, reinterpret_tensor(L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_q_linear_bias, {s0, 128L}, {0L, 1L}, 0L), reinterpret_tensor(arg128_1, {s0, 156L}, {156L, 1L}, 0L), reinterpret_tensor(L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_q_linear_weight, {156L, 128L}, {1L, 156L}, 0L), 1L, 1L);
    at::Tensor buf24 = at::detail::empty_strided_cuda({s0, 1L, 8L, 16L}, {128L, 128L, 16L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [user_model_multi_h_attens_item_clk_seq_ta_q_layer_norm], Original ATen: [aten.native_layer_norm]
    auto triton_per_fused_native_layer_norm_0_xnumel = 8L*s0;
    if (kernels.triton_per_fused_native_layer_norm_0 == nullptr) {
        kernels.triton_per_fused_native_layer_norm_0 = loadKernel("/home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/dynamicLib_7622_gpu/cjibfre4wxzmfwmnbgeqe6n3wtdygvxmamdglv7w6l6uk4cx5jny.cubin", "triton__0d1d2d3d4e5de", 0, this->cubin_dir_);
    }
    CUdeviceptr var_0 = reinterpret_cast<CUdeviceptr>(buf0.data_ptr());
    CUdeviceptr var_1 = reinterpret_cast<CUdeviceptr>(L__self___user_model_multi_h_attens_item_clk_seq_ta_q_layer_norm_weight.data_ptr());
    CUdeviceptr var_2 = reinterpret_cast<CUdeviceptr>(L__self___user_model_multi_h_attens_item_clk_seq_ta_q_layer_norm_bias.data_ptr());
    CUdeviceptr var_3 = reinterpret_cast<CUdeviceptr>(buf24.data_ptr());
    auto var_4 = triton_per_fused_native_layer_norm_0_xnumel;
    auto var_5 = 16;
    void* kernel_args_var_0[] = {&var_0, &var_1, &var_2, &var_3, &var_4, &var_5};
    Grid triton_per_fused_native_layer_norm_0_grid_0 = Grid(s0, 1L, 1L);
    if (triton_per_fused_native_layer_norm_0_grid_0.is_non_zero()) {
    launchKernel(kernels.triton_per_fused_native_layer_norm_0, triton_per_fused_native_layer_norm_0_grid_0.grid_x, triton_per_fused_native_layer_norm_0_grid_0.grid_y, triton_per_fused_native_layer_norm_0_grid_0.grid_z, 2, 0, kernel_args_var_0, stream);
    }
    at::Tensor buf4 = at::detail::empty_strided_cuda({50L*s0, 176L}, {176L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [where], Original ATen: [aten.scalar_tensor, aten.where]
    auto triton_poi_fused_scalar_tensor_where_1_xnumel = 8800L*s0;
    if (kernels.triton_poi_fused_scalar_tensor_where_1 == nullptr) {
        kernels.triton_poi_fused_scalar_tensor_where_1 = loadKernel("/home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/dynamicLib_7622_gpu/cfkhqwq2nuupmkhvxgasjdfyilkwsd6ck23dhqnddwayfy5uzswq.cubin", "triton__0d1d2d3de", 0, this->cubin_dir_);
    }
    CUdeviceptr var_6 = reinterpret_cast<CUdeviceptr>(arg124_1.data_ptr());
    CUdeviceptr var_7 = reinterpret_cast<CUdeviceptr>(arg123_1.data_ptr());
    CUdeviceptr var_8 = reinterpret_cast<CUdeviceptr>(buf4.data_ptr());
    auto var_9 = triton_poi_fused_scalar_tensor_where_1_xnumel;
    void* kernel_args_var_1[] = {&var_6, &var_7, &var_8, &var_9};
    Grid triton_poi_fused_scalar_tensor_where_1_grid_1 = Grid(((511L + (8800L*s0))/512L), 1L, 1L);
    if (triton_poi_fused_scalar_tensor_where_1_grid_1.is_non_zero()) {
    launchKernel(kernels.triton_poi_fused_scalar_tensor_where_1, triton_poi_fused_scalar_tensor_where_1_grid_1.grid_x, triton_poi_fused_scalar_tensor_where_1_grid_1.grid_y, triton_poi_fused_scalar_tensor_where_1_grid_1.grid_z, 8, 0, kernel_args_var_1, stream);
    }
    arg123_1.reset();
    at::Tensor buf5 = at::detail::empty_strided_cuda({50L*s0, 128L}, {128L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [user_model_multi_h_attens_item_clk_seq_ta_proj_k_linear], Original ATen: [aten.addmm]
    at::addmm_out(buf5, reinterpret_tensor(L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_k_linear_bias, {50L*s0, 128L}, {0L, 1L}, 0L), reinterpret_tensor(buf4, {50L*s0, 176L}, {176L, 1L}, 0L), reinterpret_tensor(L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_k_linear_weight, {176L, 128L}, {1L, 176L}, 0L), 1L, 1L);
    at::Tensor buf6 = at::detail::empty_strided_cuda({s0, 50L, 8L, 1L}, {400L, 8L, 1L, 400L*s0}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf7 = at::detail::empty_strided_cuda({s0, 50L, 8L, 1L}, {400L, 8L, 1L, 400L*s0}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [user_model_multi_h_attens_item_clk_seq_ta_k_layer_norm], Original ATen: [aten.native_layer_norm]
    auto triton_per_fused_native_layer_norm_2_xnumel = 400L*s0;
    if (kernels.triton_per_fused_native_layer_norm_2 == nullptr) {
        kernels.triton_per_fused_native_layer_norm_2 = loadKernel("/home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/dynamicLib_7622_gpu/cunkkosgypax7p2fxdqvwe2avvt3vxtrh42idsqolqqn6z2uathy.cubin", "triton__0d1d2d3de4de", 32, this->cubin_dir_);
    }
    CUdeviceptr var_10 = reinterpret_cast<CUdeviceptr>(buf5.data_ptr());
    CUdeviceptr var_11 = reinterpret_cast<CUdeviceptr>(buf6.data_ptr());
    CUdeviceptr var_12 = reinterpret_cast<CUdeviceptr>(buf7.data_ptr());
    auto var_13 = triton_per_fused_native_layer_norm_2_xnumel;
    auto var_14 = 16;
    void* kernel_args_var_2[] = {&var_10, &var_11, &var_12, &var_13, &var_14};
    Grid triton_per_fused_native_layer_norm_2_grid_2 = Grid(50L*s0, 1L, 1L);
    if (triton_per_fused_native_layer_norm_2_grid_2.is_non_zero()) {
    launchKernel(kernels.triton_per_fused_native_layer_norm_2, triton_per_fused_native_layer_norm_2_grid_2.grid_x, triton_per_fused_native_layer_norm_2_grid_2.grid_y, triton_per_fused_native_layer_norm_2_grid_2.grid_z, 2, 32, kernel_args_var_2, stream);
    }
    decltype(auto) buf9 = buf0; buf0.reset();;  // reuse
    // Source Nodes: [user_model_multi_h_attens_query_seq_ta_proj_q_linear], Original ATen: [aten.addmm]
    at::addmm_out(buf9, reinterpret_tensor(L__self___user_model_multi_h_attens_query_seq_ta_proj_q_linear_bias, {s0, 128L}, {0L, 1L}, 0L), reinterpret_tensor(arg128_1, {s0, 156L}, {156L, 1L}, 0L), reinterpret_tensor(L__self___user_model_multi_h_attens_query_seq_ta_proj_q_linear_weight, {156L, 128L}, {1L, 156L}, 0L), 1L, 1L);
    at::Tensor buf39 = at::detail::empty_strided_cuda({s0, 1L, 8L, 16L}, {128L, 128L, 16L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [user_model_multi_h_attens_query_seq_ta_q_layer_norm], Original ATen: [aten.native_layer_norm]
    triton_per_fused_native_layer_norm_0_xnumel = 8L*s0;
    CUdeviceptr var_15 = reinterpret_cast<CUdeviceptr>(buf9.data_ptr());
    CUdeviceptr var_16 = reinterpret_cast<CUdeviceptr>(L__self___user_model_multi_h_attens_query_seq_ta_q_layer_norm_weight.data_ptr());
    CUdeviceptr var_17 = reinterpret_cast<CUdeviceptr>(L__self___user_model_multi_h_attens_query_seq_ta_q_layer_norm_bias.data_ptr());
    CUdeviceptr var_18 = reinterpret_cast<CUdeviceptr>(buf39.data_ptr());
    auto var_19 = triton_per_fused_native_layer_norm_0_xnumel;
    auto var_20 = 16;
    void* kernel_args_var_3[] = {&var_15, &var_16, &var_17, &var_18, &var_19, &var_20};
    Grid triton_per_fused_native_layer_norm_0_grid_3 = Grid(s0, 1L, 1L);
    if (triton_per_fused_native_layer_norm_0_grid_3.is_non_zero()) {
    launchKernel(kernels.triton_per_fused_native_layer_norm_0, triton_per_fused_native_layer_norm_0_grid_3.grid_x, triton_per_fused_native_layer_norm_0_grid_3.grid_y, triton_per_fused_native_layer_norm_0_grid_3.grid_z, 2, 0, kernel_args_var_3, stream);
    }
    buf9.reset();
    at::Tensor buf13 = at::detail::empty_strided_cuda({50L*s0, 172L}, {172L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [where_1], Original ATen: [aten.scalar_tensor, aten.where]
    auto triton_poi_fused_scalar_tensor_where_3_xnumel = 8600L*s0;
    if (kernels.triton_poi_fused_scalar_tensor_where_3 == nullptr) {
        kernels.triton_poi_fused_scalar_tensor_where_3 = loadKernel("/home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/dynamicLib_7622_gpu/cvrlc6mvrjvrnka7zaiihzy36to6j52f6refqkgikxzthpazhmmo.cubin", "triton__0d1d2d3e", 0, this->cubin_dir_);
    }
    CUdeviceptr var_21 = reinterpret_cast<CUdeviceptr>(arg127_1.data_ptr());
    CUdeviceptr var_22 = reinterpret_cast<CUdeviceptr>(arg126_1.data_ptr());
    CUdeviceptr var_23 = reinterpret_cast<CUdeviceptr>(buf13.data_ptr());
    auto var_24 = triton_poi_fused_scalar_tensor_where_3_xnumel;
    void* kernel_args_var_4[] = {&var_21, &var_22, &var_23, &var_24};
    Grid triton_poi_fused_scalar_tensor_where_3_grid_4 = Grid(((511L + (8600L*s0))/512L), 1L, 1L);
    if (triton_poi_fused_scalar_tensor_where_3_grid_4.is_non_zero()) {
    launchKernel(kernels.triton_poi_fused_scalar_tensor_where_3, triton_poi_fused_scalar_tensor_where_3_grid_4.grid_x, triton_poi_fused_scalar_tensor_where_3_grid_4.grid_y, triton_poi_fused_scalar_tensor_where_3_grid_4.grid_z, 8, 0, kernel_args_var_4, stream);
    }
    arg126_1.reset();
    at::Tensor buf14 = at::detail::empty_strided_cuda({50L*s0, 128L}, {128L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [user_model_multi_h_attens_query_seq_ta_proj_k_linear], Original ATen: [aten.addmm]
    at::addmm_out(buf14, reinterpret_tensor(L__self___user_model_multi_h_attens_query_seq_ta_proj_k_linear_bias, {50L*s0, 128L}, {0L, 1L}, 0L), reinterpret_tensor(buf13, {50L*s0, 172L}, {172L, 1L}, 0L), reinterpret_tensor(L__self___user_model_multi_h_attens_query_seq_ta_proj_k_linear_weight, {172L, 128L}, {1L, 172L}, 0L), 1L, 1L);
    at::Tensor buf15 = at::detail::empty_strided_cuda({s0, 50L, 8L, 1L}, {400L, 8L, 1L, 400L*s0}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf16 = at::detail::empty_strided_cuda({s0, 50L, 8L, 1L}, {400L, 8L, 1L, 400L*s0}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [user_model_multi_h_attens_query_seq_ta_k_layer_norm], Original ATen: [aten.native_layer_norm]
    triton_per_fused_native_layer_norm_2_xnumel = 400L*s0;
    CUdeviceptr var_25 = reinterpret_cast<CUdeviceptr>(buf14.data_ptr());
    CUdeviceptr var_26 = reinterpret_cast<CUdeviceptr>(buf15.data_ptr());
    CUdeviceptr var_27 = reinterpret_cast<CUdeviceptr>(buf16.data_ptr());
    auto var_28 = triton_per_fused_native_layer_norm_2_xnumel;
    auto var_29 = 16;
    void* kernel_args_var_5[] = {&var_25, &var_26, &var_27, &var_28, &var_29};
    Grid triton_per_fused_native_layer_norm_2_grid_5 = Grid(50L*s0, 1L, 1L);
    if (triton_per_fused_native_layer_norm_2_grid_5.is_non_zero()) {
    launchKernel(kernels.triton_per_fused_native_layer_norm_2, triton_per_fused_native_layer_norm_2_grid_5.grid_x, triton_per_fused_native_layer_norm_2_grid_5.grid_y, triton_per_fused_native_layer_norm_2_grid_5.grid_z, 2, 32, kernel_args_var_5, stream);
    }
    at::Tensor buf18 = at::detail::empty_strided_cuda({s0, 1L}, {1L, s0}, at::kFloat, c10::DeviceType::CUDA);
    at::Tensor buf23 = at::detail::empty_strided_cuda({s0, 1L}, {1L, s0}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [logical_or, lt, sum_1, sum_3, to, to_2], Original ATen: [aten._to_copy, aten.logical_or, aten.lt, aten.sum]
    if (kernels.triton_per_fused__to_copy_logical_or_lt_sum_4 == nullptr) {
        kernels.triton_per_fused__to_copy_logical_or_lt_sum_4 = loadKernel("/home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/dynamicLib_7622_gpu/cs3rnshoyxv3jdwkgb3orgelizikfo3ntbpelzu4tmtn6zi2to3r.cubin", "triton__0d1d2d34", 8, this->cubin_dir_);
    }
    CUdeviceptr var_30 = reinterpret_cast<CUdeviceptr>(arg124_1.data_ptr());
    CUdeviceptr var_31 = reinterpret_cast<CUdeviceptr>(buf18.data_ptr());
    CUdeviceptr var_32 = reinterpret_cast<CUdeviceptr>(buf23.data_ptr());
    auto var_33 = s0;
    auto var_34 = 50;
    void* kernel_args_var_6[] = {&var_30, &var_31, &var_32, &var_33, &var_34};
    Grid triton_per_fused__to_copy_logical_or_lt_sum_4_grid_6 = Grid(s0, 1L, 1L);
    if (triton_per_fused__to_copy_logical_or_lt_sum_4_grid_6.is_non_zero()) {
    launchKernel(kernels.triton_per_fused__to_copy_logical_or_lt_sum_4, triton_per_fused__to_copy_logical_or_lt_sum_4_grid_6.grid_x, triton_per_fused__to_copy_logical_or_lt_sum_4_grid_6.grid_y, triton_per_fused__to_copy_logical_or_lt_sum_4_grid_6.grid_z, 2, 8, kernel_args_var_6, stream);
    }
    at::Tensor buf19 = at::detail::empty_strided_cuda({50L*s0, 176L}, {176L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [], Original ATen: []
    at::mm_out(buf19, reinterpret_tensor(buf4, {50L*s0, 176L}, {176L, 1L}, 0L), reinterpret_tensor(L__self___user_model_feedforwards_item_clk_seq_fc1_linear_weight, {176L, 176L}, {1L, 176L}, 0L));
    decltype(auto) buf20 = reinterpret_tensor(buf19, {s0, 50L, 176L}, {8800L, 176L, 1L}, 0L); buf19.reset();  // reuse
    // Source Nodes: [leaky_relu], Original ATen: [aten.leaky_relu]
    auto triton_poi_fused_leaky_relu_5_xnumel = 8800L*s0;
    if (kernels.triton_poi_fused_leaky_relu_5 == nullptr) {
        kernels.triton_poi_fused_leaky_relu_5 = loadKernel("/home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/dynamicLib_7622_gpu/cum7pvp55zxxcogvaqndd7zqihjkdsog4pjm4fedanid25b3jbm5.cubin", "triton__0d1d2de", 0, this->cubin_dir_);
    }
    CUdeviceptr var_35 = reinterpret_cast<CUdeviceptr>(buf20.data_ptr());
    CUdeviceptr var_36 = reinterpret_cast<CUdeviceptr>(L__self___user_model_feedforwards_item_clk_seq_fc1_linear_bias.data_ptr());
    auto var_37 = triton_poi_fused_leaky_relu_5_xnumel;
    void* kernel_args_var_7[] = {&var_35, &var_36, &var_37};
    Grid triton_poi_fused_leaky_relu_5_grid_7 = Grid(((511L + (8800L*s0))/512L), 1L, 1L);
    if (triton_poi_fused_leaky_relu_5_grid_7.is_non_zero()) {
    launchKernel(kernels.triton_poi_fused_leaky_relu_5, triton_poi_fused_leaky_relu_5_grid_7.grid_x, triton_poi_fused_leaky_relu_5_grid_7.grid_y, triton_poi_fused_leaky_relu_5_grid_7.grid_z, 8, 0, kernel_args_var_7, stream);
    }
    at::Tensor buf21 = at::detail::empty_strided_cuda({50L*s0, 176L}, {176L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [], Original ATen: []
    at::mm_out(buf21, reinterpret_tensor(buf20, {50L*s0, 176L}, {176L, 1L}, 0L), reinterpret_tensor(L__self___user_model_feedforwards_item_clk_seq_fc2_linear_weight, {176L, 176L}, {1L, 176L}, 0L));
    buf20.reset();
    at::Tensor buf56 = at::detail::empty_strided_cuda({s0, 1872L}, {1872L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    decltype(auto) buf51 = reinterpret_tensor(buf56, {s0, 176L}, {1872L, 1L}, 1112L);  // alias
    // Source Nodes: [mul, pow_1, sum_4], Original ATen: [aten.mul, aten.pow, aten.sum]
    auto triton_per_fused_mul_pow_sum_6_xnumel = 176L*s0;
    if (kernels.triton_per_fused_mul_pow_sum_6 == nullptr) {
        kernels.triton_per_fused_mul_pow_sum_6 = loadKernel("/home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/dynamicLib_7622_gpu/cbkden7m47hznzfqzzmk6qapfxj5vnj2xcsncuokhmalbl6hom7y.cubin", "triton__0d1d2d3d4d5d67de8", 1024, this->cubin_dir_);
    }
    CUdeviceptr var_38 = reinterpret_cast<CUdeviceptr>(arg124_1.data_ptr());
    CUdeviceptr var_39 = reinterpret_cast<CUdeviceptr>(buf18.data_ptr());
    CUdeviceptr var_40 = reinterpret_cast<CUdeviceptr>(buf21.data_ptr());
    CUdeviceptr var_41 = reinterpret_cast<CUdeviceptr>(L__self___user_model_feedforwards_item_clk_seq_fc2_linear_bias.data_ptr());
    CUdeviceptr var_42 = reinterpret_cast<CUdeviceptr>(buf4.data_ptr());
    CUdeviceptr var_43 = reinterpret_cast<CUdeviceptr>(buf23.data_ptr());
    CUdeviceptr var_44 = reinterpret_cast<CUdeviceptr>(buf51.data_ptr());
    auto var_45 = triton_per_fused_mul_pow_sum_6_xnumel;
    auto var_46 = 50;
    void* kernel_args_var_8[] = {&var_38, &var_39, &var_40, &var_41, &var_42, &var_43, &var_44, &var_45, &var_46};
    Grid triton_per_fused_mul_pow_sum_6_grid_8 = Grid(((31L + (176L*s0))/32L), 1L, 1L);
    if (triton_per_fused_mul_pow_sum_6_grid_8.is_non_zero()) {
    launchKernel(kernels.triton_per_fused_mul_pow_sum_6, triton_per_fused_mul_pow_sum_6_grid_8.grid_x, triton_per_fused_mul_pow_sum_6_grid_8.grid_y, triton_per_fused_mul_pow_sum_6_grid_8.grid_z, 8, 1024, kernel_args_var_8, stream);
    }
    arg124_1.reset();
    buf21.reset();
    at::Tensor buf25 = at::detail::empty_strided_cuda({s0, 8L, 16L, 50L}, {6400L, 800L, 50L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [matmul], Original ATen: [aten.clone]
    auto triton_poi_fused_clone_7_ynumel = 128L*s0;
    if (kernels.triton_poi_fused_clone_7 == nullptr) {
        kernels.triton_poi_fused_clone_7 = loadKernel("/home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/dynamicLib_7622_gpu/c67e3herfmhdzoayhqtkhpsog4gor3koprmbqvlfeh4bolc56pyq.cubin", "triton__0d1d2d3d4d5d6de7", 4224, this->cubin_dir_);
    }
    CUdeviceptr var_47 = reinterpret_cast<CUdeviceptr>(buf5.data_ptr());
    CUdeviceptr var_48 = reinterpret_cast<CUdeviceptr>(buf6.data_ptr());
    CUdeviceptr var_49 = reinterpret_cast<CUdeviceptr>(buf7.data_ptr());
    CUdeviceptr var_50 = reinterpret_cast<CUdeviceptr>(L__self___user_model_multi_h_attens_item_clk_seq_ta_k_layer_norm_weight.data_ptr());
    CUdeviceptr var_51 = reinterpret_cast<CUdeviceptr>(L__self___user_model_multi_h_attens_item_clk_seq_ta_k_layer_norm_bias.data_ptr());
    CUdeviceptr var_52 = reinterpret_cast<CUdeviceptr>(buf25.data_ptr());
    auto var_53 = triton_poi_fused_clone_7_ynumel;
    auto var_54 = 50;
    void* kernel_args_var_9[] = {&var_47, &var_48, &var_49, &var_50, &var_51, &var_52, &var_53, &var_54};
    Grid triton_poi_fused_clone_7_grid_9 = Grid(2L, std::floor(4L*s0*(1.0/((65534L + (4L*s0))/65535L))), ((65534L + (4L*s0))/65535L));
    if (triton_poi_fused_clone_7_grid_9.is_non_zero()) {
    launchKernel(kernels.triton_poi_fused_clone_7, triton_poi_fused_clone_7_grid_9.grid_x, triton_poi_fused_clone_7_grid_9.grid_y, triton_poi_fused_clone_7_grid_9.grid_z, 4, 4224, kernel_args_var_9, stream);
    }
    decltype(auto) buf26 = reinterpret_tensor(buf7, {8L*s0, 1L, 50L}, {50L, 50L, 1L}, 0L); buf7.reset();  // reuse
    // Source Nodes: [matmul], Original ATen: [aten.bmm]
    at::bmm_out(buf26, reinterpret_tensor(buf24, {8L*s0, 1L, 16L}, {16L, 0L, 1L}, 0L), reinterpret_tensor(buf25, {8L*s0, 16L, 50L}, {800L, 50L, 1L}, 0L));
    decltype(auto) buf30 = reinterpret_tensor(buf6, {s0, 8L, 1L, 50L}, {400L, 50L, 50L, 1L}, 0L); buf6.reset();  // reuse
    // Source Nodes: [softmax, truediv], Original ATen: [aten._softmax, aten.div]
    auto triton_per_fused__softmax_div_8_xnumel = 8L*s0;
    if (kernels.triton_per_fused__softmax_div_8 == nullptr) {
        kernels.triton_per_fused__softmax_div_8 = loadKernel("/home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/dynamicLib_7622_gpu/c2ekcqvvrgwlu4s32fl423btq6sadvn44bvknzf3opnf2pcmjuat.cubin", "triton__0d1d2e3", 0, this->cubin_dir_);
    }
    CUdeviceptr var_55 = reinterpret_cast<CUdeviceptr>(buf26.data_ptr());
    CUdeviceptr var_56 = reinterpret_cast<CUdeviceptr>(buf30.data_ptr());
    auto var_57 = triton_per_fused__softmax_div_8_xnumel;
    auto var_58 = 50;
    void* kernel_args_var_10[] = {&var_55, &var_56, &var_57, &var_58};
    Grid triton_per_fused__softmax_div_8_grid_10 = Grid(((31L + (8L*s0))/32L), 1L, 1L);
    if (triton_per_fused__softmax_div_8_grid_10.is_non_zero()) {
    launchKernel(kernels.triton_per_fused__softmax_div_8, triton_per_fused__softmax_div_8_grid_10.grid_x, triton_per_fused__softmax_div_8_grid_10.grid_y, triton_per_fused__softmax_div_8_grid_10.grid_z, 8, 0, kernel_args_var_10, stream);
    }
    buf26.reset();
    decltype(auto) buf29 = reinterpret_tensor(buf25, {50L*s0, 128L}, {128L, 1L}, 0L); buf25.reset();  // reuse
    // Source Nodes: [], Original ATen: []
    at::mm_out(buf29, reinterpret_tensor(buf4, {50L*s0, 176L}, {176L, 1L}, 0L), reinterpret_tensor(L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_v_linear_weight, {176L, 128L}, {1L, 176L}, 0L));
    buf4.reset();
    decltype(auto) buf31 = reinterpret_tensor(buf5, {s0, 8L, 50L, 16L}, {6400L, 800L, 16L, 1L}, 0L); buf5.reset();  // reuse
    // Source Nodes: [matmul_1], Original ATen: [aten.clone]
    auto triton_poi_fused_clone_9_xnumel = 6400L*s0;
    if (kernels.triton_poi_fused_clone_9 == nullptr) {
        kernels.triton_poi_fused_clone_9 = loadKernel("/home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/dynamicLib_7622_gpu/cjsrwm3766d43kca5anwpfktwrtycxdt2plwnu6int52kfvxw2m5.cubin", "triton__0d1d2d3de", 0, this->cubin_dir_);
    }
    CUdeviceptr var_59 = reinterpret_cast<CUdeviceptr>(buf29.data_ptr());
    CUdeviceptr var_60 = reinterpret_cast<CUdeviceptr>(L__self___user_model_multi_h_attens_item_clk_seq_ta_proj_v_linear_bias.data_ptr());
    CUdeviceptr var_61 = reinterpret_cast<CUdeviceptr>(buf31.data_ptr());
    auto var_62 = triton_poi_fused_clone_9_xnumel;
    void* kernel_args_var_11[] = {&var_59, &var_60, &var_61, &var_62};
    Grid triton_poi_fused_clone_9_grid_11 = Grid(((511L + (6400L*s0))/512L), 1L, 1L);
    if (triton_poi_fused_clone_9_grid_11.is_non_zero()) {
    launchKernel(kernels.triton_poi_fused_clone_9, triton_poi_fused_clone_9_grid_11.grid_x, triton_poi_fused_clone_9_grid_11.grid_y, triton_poi_fused_clone_9_grid_11.grid_z, 8, 0, kernel_args_var_11, stream);
    }
    buf29.reset();
    decltype(auto) buf32 = reinterpret_tensor(buf24, {8L*s0, 1L, 16L}, {16L, 16L, 1L}, 0L); buf24.reset();  // reuse
    // Source Nodes: [matmul_1], Original ATen: [aten.bmm]
    at::bmm_out(buf32, reinterpret_tensor(buf30, {8L*s0, 1L, 50L}, {50L, 0L, 1L}, 0L), reinterpret_tensor(buf31, {8L*s0, 50L, 16L}, {800L, 16L, 1L}, 0L));
    buf30.reset();
    decltype(auto) buf33 = buf23; buf23.reset();;  // reuse
    decltype(auto) buf38 = buf18; buf18.reset();;  // reuse
    // Source Nodes: [logical_or_1, lt_1, sum_2, sum_5, to_1, to_3], Original ATen: [aten._to_copy, aten.logical_or, aten.lt, aten.sum]
    CUdeviceptr var_63 = reinterpret_cast<CUdeviceptr>(arg127_1.data_ptr());
    CUdeviceptr var_64 = reinterpret_cast<CUdeviceptr>(buf33.data_ptr());
    CUdeviceptr var_65 = reinterpret_cast<CUdeviceptr>(buf38.data_ptr());
    auto var_66 = s0;
    auto var_67 = 50;
    void* kernel_args_var_12[] = {&var_63, &var_64, &var_65, &var_66, &var_67};
    Grid triton_per_fused__to_copy_logical_or_lt_sum_4_grid_12 = Grid(s0, 1L, 1L);
    if (triton_per_fused__to_copy_logical_or_lt_sum_4_grid_12.is_non_zero()) {
    launchKernel(kernels.triton_per_fused__to_copy_logical_or_lt_sum_4, triton_per_fused__to_copy_logical_or_lt_sum_4_grid_12.grid_x, triton_per_fused__to_copy_logical_or_lt_sum_4_grid_12.grid_y, triton_per_fused__to_copy_logical_or_lt_sum_4_grid_12.grid_z, 2, 8, kernel_args_var_12, stream);
    }
    at::Tensor buf34 = at::detail::empty_strided_cuda({50L*s0, 172L}, {172L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [], Original ATen: []
    at::mm_out(buf34, reinterpret_tensor(buf13, {50L*s0, 172L}, {172L, 1L}, 0L), reinterpret_tensor(L__self___user_model_feedforwards_query_seq_fc1_linear_weight, {172L, 172L}, {1L, 172L}, 0L));
    decltype(auto) buf35 = reinterpret_tensor(buf34, {s0, 50L, 172L}, {8600L, 172L, 1L}, 0L); buf34.reset();  // reuse
    // Source Nodes: [leaky_relu_1], Original ATen: [aten.leaky_relu]
    auto triton_poi_fused_leaky_relu_10_xnumel = 8600L*s0;
    if (kernels.triton_poi_fused_leaky_relu_10 == nullptr) {
        kernels.triton_poi_fused_leaky_relu_10 = loadKernel("/home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/dynamicLib_7622_gpu/cywkwmay3cvstqrgp6im7v4lpdfiprl3o5jmwhpvrwaxnwuxymtv.cubin", "triton__0d1d2e", 0, this->cubin_dir_);
    }
    CUdeviceptr var_68 = reinterpret_cast<CUdeviceptr>(buf35.data_ptr());
    CUdeviceptr var_69 = reinterpret_cast<CUdeviceptr>(L__self___user_model_feedforwards_query_seq_fc1_linear_bias.data_ptr());
    auto var_70 = triton_poi_fused_leaky_relu_10_xnumel;
    void* kernel_args_var_13[] = {&var_68, &var_69, &var_70};
    Grid triton_poi_fused_leaky_relu_10_grid_13 = Grid(((511L + (8600L*s0))/512L), 1L, 1L);
    if (triton_poi_fused_leaky_relu_10_grid_13.is_non_zero()) {
    launchKernel(kernels.triton_poi_fused_leaky_relu_10, triton_poi_fused_leaky_relu_10_grid_13.grid_x, triton_poi_fused_leaky_relu_10_grid_13.grid_y, triton_poi_fused_leaky_relu_10_grid_13.grid_z, 8, 0, kernel_args_var_13, stream);
    }
    at::Tensor buf36 = at::detail::empty_strided_cuda({50L*s0, 172L}, {172L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [], Original ATen: []
    at::mm_out(buf36, reinterpret_tensor(buf35, {50L*s0, 172L}, {172L, 1L}, 0L), reinterpret_tensor(L__self___user_model_feedforwards_query_seq_fc2_linear_weight, {172L, 172L}, {1L, 172L}, 0L));
    buf35.reset();
    decltype(auto) buf53 = reinterpret_tensor(buf56, {s0, 172L}, {1872L, 1L}, 1416L);  // alias
    // Source Nodes: [mul_1, pow_2, sum_6], Original ATen: [aten.mul, aten.pow, aten.sum]
    auto triton_per_fused_mul_pow_sum_11_xnumel = 172L*s0;
    if (kernels.triton_per_fused_mul_pow_sum_11 == nullptr) {
        kernels.triton_per_fused_mul_pow_sum_11 = loadKernel("/home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/dynamicLib_7622_gpu/cexvui6fxogvowvju3sicyqh32hjwdduf6c6dxku4wcs67hbbdwu.cubin", "triton__0d1d2d3d4d5d678", 128, this->cubin_dir_);
    }
    CUdeviceptr var_71 = reinterpret_cast<CUdeviceptr>(arg127_1.data_ptr());
    CUdeviceptr var_72 = reinterpret_cast<CUdeviceptr>(buf33.data_ptr());
    CUdeviceptr var_73 = reinterpret_cast<CUdeviceptr>(buf36.data_ptr());
    CUdeviceptr var_74 = reinterpret_cast<CUdeviceptr>(L__self___user_model_feedforwards_query_seq_fc2_linear_bias.data_ptr());
    CUdeviceptr var_75 = reinterpret_cast<CUdeviceptr>(buf13.data_ptr());
    CUdeviceptr var_76 = reinterpret_cast<CUdeviceptr>(buf38.data_ptr());
    CUdeviceptr var_77 = reinterpret_cast<CUdeviceptr>(buf53.data_ptr());
    auto var_78 = triton_per_fused_mul_pow_sum_11_xnumel;
    auto var_79 = 50;
    void* kernel_args_var_14[] = {&var_71, &var_72, &var_73, &var_74, &var_75, &var_76, &var_77, &var_78, &var_79};
    Grid triton_per_fused_mul_pow_sum_11_grid_14 = Grid(((7L + (172L*s0))/8L), 1L, 1L);
    if (triton_per_fused_mul_pow_sum_11_grid_14.is_non_zero()) {
    launchKernel(kernels.triton_per_fused_mul_pow_sum_11, triton_per_fused_mul_pow_sum_11_grid_14.grid_x, triton_per_fused_mul_pow_sum_11_grid_14.grid_y, triton_per_fused_mul_pow_sum_11_grid_14.grid_z, 4, 128, kernel_args_var_14, stream);
    }
    arg127_1.reset();
    buf36.reset();
    decltype(auto) buf40 = reinterpret_tensor(buf31, {s0, 8L, 16L, 50L}, {6400L, 800L, 50L, 1L}, 0L); buf31.reset();  // reuse
    // Source Nodes: [matmul_2], Original ATen: [aten.clone]
    triton_poi_fused_clone_7_ynumel = 128L*s0;
    CUdeviceptr var_80 = reinterpret_cast<CUdeviceptr>(buf14.data_ptr());
    CUdeviceptr var_81 = reinterpret_cast<CUdeviceptr>(buf15.data_ptr());
    CUdeviceptr var_82 = reinterpret_cast<CUdeviceptr>(buf16.data_ptr());
    CUdeviceptr var_83 = reinterpret_cast<CUdeviceptr>(L__self___user_model_multi_h_attens_query_seq_ta_k_layer_norm_weight.data_ptr());
    CUdeviceptr var_84 = reinterpret_cast<CUdeviceptr>(L__self___user_model_multi_h_attens_query_seq_ta_k_layer_norm_bias.data_ptr());
    CUdeviceptr var_85 = reinterpret_cast<CUdeviceptr>(buf40.data_ptr());
    auto var_86 = triton_poi_fused_clone_7_ynumel;
    auto var_87 = 50;
    void* kernel_args_var_15[] = {&var_80, &var_81, &var_82, &var_83, &var_84, &var_85, &var_86, &var_87};
    Grid triton_poi_fused_clone_7_grid_15 = Grid(2L, std::floor(4L*s0*(1.0/((65534L + (4L*s0))/65535L))), ((65534L + (4L*s0))/65535L));
    if (triton_poi_fused_clone_7_grid_15.is_non_zero()) {
    launchKernel(kernels.triton_poi_fused_clone_7, triton_poi_fused_clone_7_grid_15.grid_x, triton_poi_fused_clone_7_grid_15.grid_y, triton_poi_fused_clone_7_grid_15.grid_z, 4, 4224, kernel_args_var_15, stream);
    }
    decltype(auto) buf41 = reinterpret_tensor(buf16, {8L*s0, 1L, 50L}, {50L, 50L, 1L}, 0L); buf16.reset();  // reuse
    // Source Nodes: [matmul_2], Original ATen: [aten.bmm]
    at::bmm_out(buf41, reinterpret_tensor(buf39, {8L*s0, 1L, 16L}, {16L, 0L, 1L}, 0L), reinterpret_tensor(buf40, {8L*s0, 16L, 50L}, {800L, 50L, 1L}, 0L));
    decltype(auto) buf45 = reinterpret_tensor(buf15, {s0, 8L, 1L, 50L}, {400L, 50L, 50L, 1L}, 0L); buf15.reset();  // reuse
    // Source Nodes: [softmax_1, truediv_1], Original ATen: [aten._softmax, aten.div]
    triton_per_fused__softmax_div_8_xnumel = 8L*s0;
    CUdeviceptr var_88 = reinterpret_cast<CUdeviceptr>(buf41.data_ptr());
    CUdeviceptr var_89 = reinterpret_cast<CUdeviceptr>(buf45.data_ptr());
    auto var_90 = triton_per_fused__softmax_div_8_xnumel;
    auto var_91 = 50;
    void* kernel_args_var_16[] = {&var_88, &var_89, &var_90, &var_91};
    Grid triton_per_fused__softmax_div_8_grid_16 = Grid(((31L + (8L*s0))/32L), 1L, 1L);
    if (triton_per_fused__softmax_div_8_grid_16.is_non_zero()) {
    launchKernel(kernels.triton_per_fused__softmax_div_8, triton_per_fused__softmax_div_8_grid_16.grid_x, triton_per_fused__softmax_div_8_grid_16.grid_y, triton_per_fused__softmax_div_8_grid_16.grid_z, 8, 0, kernel_args_var_16, stream);
    }
    buf41.reset();
    decltype(auto) buf44 = reinterpret_tensor(buf40, {50L*s0, 128L}, {128L, 1L}, 0L); buf40.reset();  // reuse
    // Source Nodes: [], Original ATen: []
    at::mm_out(buf44, reinterpret_tensor(buf13, {50L*s0, 172L}, {172L, 1L}, 0L), reinterpret_tensor(L__self___user_model_multi_h_attens_query_seq_ta_proj_v_linear_weight, {172L, 128L}, {1L, 172L}, 0L));
    buf13.reset();
    decltype(auto) buf46 = reinterpret_tensor(buf14, {s0, 8L, 50L, 16L}, {6400L, 800L, 16L, 1L}, 0L); buf14.reset();  // reuse
    // Source Nodes: [matmul_3], Original ATen: [aten.clone]
    triton_poi_fused_clone_9_xnumel = 6400L*s0;
    CUdeviceptr var_92 = reinterpret_cast<CUdeviceptr>(buf44.data_ptr());
    CUdeviceptr var_93 = reinterpret_cast<CUdeviceptr>(L__self___user_model_multi_h_attens_query_seq_ta_proj_v_linear_bias.data_ptr());
    CUdeviceptr var_94 = reinterpret_cast<CUdeviceptr>(buf46.data_ptr());
    auto var_95 = triton_poi_fused_clone_9_xnumel;
    void* kernel_args_var_17[] = {&var_92, &var_93, &var_94, &var_95};
    Grid triton_poi_fused_clone_9_grid_17 = Grid(((511L + (6400L*s0))/512L), 1L, 1L);
    if (triton_poi_fused_clone_9_grid_17.is_non_zero()) {
    launchKernel(kernels.triton_poi_fused_clone_9, triton_poi_fused_clone_9_grid_17.grid_x, triton_poi_fused_clone_9_grid_17.grid_y, triton_poi_fused_clone_9_grid_17.grid_z, 8, 0, kernel_args_var_17, stream);
    }
    buf44.reset();
    decltype(auto) buf47 = reinterpret_tensor(buf39, {8L*s0, 1L, 16L}, {16L, 16L, 1L}, 0L); buf39.reset();  // reuse
    // Source Nodes: [matmul_3], Original ATen: [aten.bmm]
    at::bmm_out(buf47, reinterpret_tensor(buf45, {8L*s0, 1L, 50L}, {50L, 0L, 1L}, 0L), reinterpret_tensor(buf46, {8L*s0, 50L, 16L}, {800L, 16L, 1L}, 0L));
    buf45.reset();
    buf46.reset();
    decltype(auto) buf48 = reinterpret_tensor(buf56, {s0, 204L}, {1872L, 1L}, 0L);  // alias
    // Source Nodes: [cat], Original ATen: [aten.cat]
    auto triton_poi_fused_cat_12_xnumel = 204L*s0;
    if (kernels.triton_poi_fused_cat_12 == nullptr) {
        kernels.triton_poi_fused_cat_12 = loadKernel("/home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/dynamicLib_7622_gpu/c5n7simaui7uubouyfhllqmz2a74mhngeupm5j3cuktiyxutnwxt.cubin", "triton__0d1d2", 0, this->cubin_dir_);
    }
    CUdeviceptr var_96 = reinterpret_cast<CUdeviceptr>(arg130_1.data_ptr());
    CUdeviceptr var_97 = reinterpret_cast<CUdeviceptr>(buf48.data_ptr());
    auto var_98 = triton_poi_fused_cat_12_xnumel;
    void* kernel_args_var_18[] = {&var_96, &var_97, &var_98};
    Grid triton_poi_fused_cat_12_grid_18 = Grid(((511L + (204L*s0))/512L), 1L, 1L);
    if (triton_poi_fused_cat_12_grid_18.is_non_zero()) {
    launchKernel(kernels.triton_poi_fused_cat_12, triton_poi_fused_cat_12_grid_18.grid_x, triton_poi_fused_cat_12_grid_18.grid_y, triton_poi_fused_cat_12_grid_18.grid_z, 4, 0, kernel_args_var_18, stream);
    }
    decltype(auto) buf49 = reinterpret_tensor(buf56, {s0, 220L}, {1872L, 1L}, 204L);  // alias
    // Source Nodes: [cat], Original ATen: [aten.cat]
    auto triton_poi_fused_cat_13_xnumel = 220L*s0;
    if (kernels.triton_poi_fused_cat_13 == nullptr) {
        kernels.triton_poi_fused_cat_13 = loadKernel("/home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/dynamicLib_7622_gpu/cmk5lg25gtm45ejuo65jdp32zggmopx363wbxbdqkn4venraerrt.cubin", "triton__0d12", 2048, this->cubin_dir_);
    }
    CUdeviceptr var_99 = reinterpret_cast<CUdeviceptr>(arg125_1.data_ptr());
    CUdeviceptr var_100 = reinterpret_cast<CUdeviceptr>(buf49.data_ptr());
    auto var_101 = triton_poi_fused_cat_13_xnumel;
    void* kernel_args_var_19[] = {&var_99, &var_100, &var_101};
    Grid triton_poi_fused_cat_13_grid_19 = Grid(((1023L + (220L*s0))/1024L), 1L, 1L);
    if (triton_poi_fused_cat_13_grid_19.is_non_zero()) {
    launchKernel(kernels.triton_poi_fused_cat_13, triton_poi_fused_cat_13_grid_19.grid_x, triton_poi_fused_cat_13_grid_19.grid_y, triton_poi_fused_cat_13_grid_19.grid_z, 4, 2048, kernel_args_var_19, stream);
    }
    arg125_1.reset();
    decltype(auto) buf50 = reinterpret_tensor(buf56, {s0, 688L}, {1872L, 1L}, 424L);  // alias
    // Source Nodes: [cat], Original ATen: [aten.cat]
    auto triton_poi_fused_cat_14_xnumel = 688L*s0;
    if (kernels.triton_poi_fused_cat_14 == nullptr) {
        kernels.triton_poi_fused_cat_14 = loadKernel("/home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/dynamicLib_7622_gpu/czknzayxkk75s236gtm4jiv5p6prdwd4fs3lot4wyvvho7zebbrq.cubin", "triton__0d12de", 2048, this->cubin_dir_);
    }
    CUdeviceptr var_102 = reinterpret_cast<CUdeviceptr>(arg122_1.data_ptr());
    CUdeviceptr var_103 = reinterpret_cast<CUdeviceptr>(buf50.data_ptr());
    auto var_104 = triton_poi_fused_cat_14_xnumel;
    void* kernel_args_var_20[] = {&var_102, &var_103, &var_104};
    Grid triton_poi_fused_cat_14_grid_20 = Grid(((1023L + (688L*s0))/1024L), 1L, 1L);
    if (triton_poi_fused_cat_14_grid_20.is_non_zero()) {
    launchKernel(kernels.triton_poi_fused_cat_14, triton_poi_fused_cat_14_grid_20.grid_x, triton_poi_fused_cat_14_grid_20.grid_y, triton_poi_fused_cat_14_grid_20.grid_z, 4, 2048, kernel_args_var_20, stream);
    }
    arg122_1.reset();
    decltype(auto) buf52 = reinterpret_tensor(buf56, {s0, 128L}, {1872L, 1L}, 1288L);  // alias
    // Source Nodes: [cat], Original ATen: [aten.cat]
    auto triton_poi_fused_cat_15_xnumel = 128L*s0;
    if (kernels.triton_poi_fused_cat_15 == nullptr) {
        kernels.triton_poi_fused_cat_15 = loadKernel("/home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/dynamicLib_7622_gpu/ccaeuvb5jtl3kcpojt7bkonfmcmfzzfcwiwxeigfi6md4bycisxs.cubin", "triton__0d12de", 2048, this->cubin_dir_);
    }
    CUdeviceptr var_105 = reinterpret_cast<CUdeviceptr>(buf32.data_ptr());
    CUdeviceptr var_106 = reinterpret_cast<CUdeviceptr>(buf52.data_ptr());
    auto var_107 = triton_poi_fused_cat_15_xnumel;
    void* kernel_args_var_21[] = {&var_105, &var_106, &var_107};
    Grid triton_poi_fused_cat_15_grid_21 = Grid(((511L + (128L*s0))/512L), 1L, 1L);
    if (triton_poi_fused_cat_15_grid_21.is_non_zero()) {
    launchKernel(kernels.triton_poi_fused_cat_15, triton_poi_fused_cat_15_grid_21.grid_x, triton_poi_fused_cat_15_grid_21.grid_y, triton_poi_fused_cat_15_grid_21.grid_z, 4, 2048, kernel_args_var_21, stream);
    }
    decltype(auto) buf54 = reinterpret_tensor(buf56, {s0, 128L}, {1872L, 1L}, 1588L);  // alias
    // Source Nodes: [cat], Original ATen: [aten.cat]
    triton_poi_fused_cat_15_xnumel = 128L*s0;
    CUdeviceptr var_108 = reinterpret_cast<CUdeviceptr>(buf47.data_ptr());
    CUdeviceptr var_109 = reinterpret_cast<CUdeviceptr>(buf54.data_ptr());
    auto var_110 = triton_poi_fused_cat_15_xnumel;
    void* kernel_args_var_22[] = {&var_108, &var_109, &var_110};
    Grid triton_poi_fused_cat_15_grid_22 = Grid(((511L + (128L*s0))/512L), 1L, 1L);
    if (triton_poi_fused_cat_15_grid_22.is_non_zero()) {
    launchKernel(kernels.triton_poi_fused_cat_15, triton_poi_fused_cat_15_grid_22.grid_x, triton_poi_fused_cat_15_grid_22.grid_y, triton_poi_fused_cat_15_grid_22.grid_z, 4, 2048, kernel_args_var_22, stream);
    }
    decltype(auto) buf55 = reinterpret_tensor(buf56, {s0, 156L}, {1872L, 1L}, 1716L);  // alias
    // Source Nodes: [cat], Original ATen: [aten.cat]
    auto triton_poi_fused_cat_16_xnumel = 156L*s0;
    if (kernels.triton_poi_fused_cat_16 == nullptr) {
        kernels.triton_poi_fused_cat_16 = loadKernel("/home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/dynamicLib_7622_gpu/carcpa4kgovjm7caxtm3vse4zvfpdduuthq74bfduvnrxxh5edw3.cubin", "triton__0d12", 2048, this->cubin_dir_);
    }
    CUdeviceptr var_111 = reinterpret_cast<CUdeviceptr>(arg128_1.data_ptr());
    CUdeviceptr var_112 = reinterpret_cast<CUdeviceptr>(buf55.data_ptr());
    auto var_113 = triton_poi_fused_cat_16_xnumel;
    void* kernel_args_var_23[] = {&var_111, &var_112, &var_113};
    Grid triton_poi_fused_cat_16_grid_23 = Grid(((511L + (156L*s0))/512L), 1L, 1L);
    if (triton_poi_fused_cat_16_grid_23.is_non_zero()) {
    launchKernel(kernels.triton_poi_fused_cat_16, triton_poi_fused_cat_16_grid_23.grid_x, triton_poi_fused_cat_16_grid_23.grid_y, triton_poi_fused_cat_16_grid_23.grid_z, 4, 2048, kernel_args_var_23, stream);
    }
    arg128_1.reset();
    buf48.reset();
    buf49.reset();
    buf50.reset();
    buf52.reset();
    buf54.reset();
    buf55.reset();
    at::Tensor buf57 = at::detail::empty_strided_cuda({s0, 512L}, {512L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [user_model_ctr_net_tower_layers_0_fc], Original ATen: [aten.mm]
    at::mm_out(buf57, buf56, reinterpret_tensor(getattr_L__self___user_model_ctr_net_tower_layers___0___fc_weight, {1872L, 512L}, {1L, 1872L}, 0L));
    decltype(auto) buf58 = buf57; buf57.reset();;  // reuse
    decltype(auto) buf59 = buf58; buf58.reset();;  // reuse
    // Source Nodes: [user_model_ctr_net_tower_layers_0_act, user_model_ctr_net_tower_layers_0_norm], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
    auto triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_xnumel = 512L*s0;
    if (kernels.triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17 == nullptr) {
        kernels.triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17 = loadKernel("/home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/dynamicLib_7622_gpu/cjvkfvzjntd3ek7fz5pbacpc2abqjvfefexsntsqy4zatlj4pwcj.cubin", "triton__0d1d2d3d4d5de", 0, this->cubin_dir_);
    }
    CUdeviceptr var_114 = reinterpret_cast<CUdeviceptr>(buf59.data_ptr());
    CUdeviceptr var_115 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_ctr_net_tower_layers___0___norm_running_mean.data_ptr());
    CUdeviceptr var_116 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_ctr_net_tower_layers___0___norm_running_var.data_ptr());
    CUdeviceptr var_117 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_ctr_net_tower_layers___0___norm_weight.data_ptr());
    CUdeviceptr var_118 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_ctr_net_tower_layers___0___norm_bias.data_ptr());
    auto var_119 = triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_xnumel;
    void* kernel_args_var_24[] = {&var_114, &var_115, &var_116, &var_117, &var_118, &var_119};
    Grid triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_grid_24 = Grid(s0, 1L, 1L);
    if (triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_grid_24.is_non_zero()) {
    launchKernel(kernels.triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_grid_24.grid_x, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_grid_24.grid_y, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_grid_24.grid_z, 8, 0, kernel_args_var_24, stream);
    }
    at::Tensor buf60 = at::detail::empty_strided_cuda({s0, 256L}, {256L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [user_model_ctr_net_tower_layers_0_act, user_model_ctr_net_tower_layers_1_fc], Original ATen: [aten.leaky_relu, aten.mm]
    at::mm_out(buf60, buf59, reinterpret_tensor(getattr_L__self___user_model_ctr_net_tower_layers___1___fc_weight, {512L, 256L}, {1L, 512L}, 0L));
    decltype(auto) buf61 = buf60; buf60.reset();;  // reuse
    decltype(auto) buf62 = buf61; buf61.reset();;  // reuse
    // Source Nodes: [user_model_ctr_net_tower_layers_1_act, user_model_ctr_net_tower_layers_1_norm], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
    auto triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_xnumel = 256L*s0;
    if (kernels.triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18 == nullptr) {
        kernels.triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18 = loadKernel("/home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/dynamicLib_7622_gpu/ch6xye7v77wq3v3tay4vuw6gizw6btto6ibzuuxlqlqeizoq6vz2.cubin", "triton__0d1d2d3d4d5de", 0, this->cubin_dir_);
    }
    CUdeviceptr var_120 = reinterpret_cast<CUdeviceptr>(buf62.data_ptr());
    CUdeviceptr var_121 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_ctr_net_tower_layers___1___norm_running_mean.data_ptr());
    CUdeviceptr var_122 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_ctr_net_tower_layers___1___norm_running_var.data_ptr());
    CUdeviceptr var_123 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_ctr_net_tower_layers___1___norm_weight.data_ptr());
    CUdeviceptr var_124 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_ctr_net_tower_layers___1___norm_bias.data_ptr());
    auto var_125 = triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_xnumel;
    void* kernel_args_var_25[] = {&var_120, &var_121, &var_122, &var_123, &var_124, &var_125};
    Grid triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_grid_25 = Grid(((1023L + (256L*s0))/1024L), 1L, 1L);
    if (triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_grid_25.is_non_zero()) {
    launchKernel(kernels.triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_grid_25.grid_x, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_grid_25.grid_y, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_grid_25.grid_z, 4, 0, kernel_args_var_25, stream);
    }
    decltype(auto) buf63 = reinterpret_tensor(buf47, {s0, 128L}, {128L, 1L}, 0L); buf47.reset();  // reuse
    // Source Nodes: [user_model_ctr_net_tower_layers_1_act, user_model_ctr_net_tower_layers_2_fc], Original ATen: [aten.leaky_relu, aten.mm]
    at::mm_out(buf63, buf62, reinterpret_tensor(getattr_L__self___user_model_ctr_net_tower_layers___2___fc_weight, {256L, 128L}, {1L, 256L}, 0L));
    decltype(auto) buf64 = buf63; buf63.reset();;  // reuse
    decltype(auto) buf71 = buf64; buf64.reset();;  // reuse
    // Source Nodes: [user_model_ctr_net_tower_layers_2_act, user_model_ctr_net_tower_layers_2_norm], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
    auto triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel = 128L*s0;
    if (kernels.triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19 == nullptr) {
        kernels.triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19 = loadKernel("/home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/dynamicLib_7622_gpu/cu2cr23ksmz3rmv2n2p7wrydis2z25ocounpeplkq4jbq2rzpqtc.cubin", "triton__0d1d2d3d4d5de", 0, this->cubin_dir_);
    }
    CUdeviceptr var_126 = reinterpret_cast<CUdeviceptr>(buf71.data_ptr());
    CUdeviceptr var_127 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_ctr_net_tower_layers___2___norm_running_mean.data_ptr());
    CUdeviceptr var_128 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_ctr_net_tower_layers___2___norm_running_var.data_ptr());
    CUdeviceptr var_129 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_ctr_net_tower_layers___2___norm_weight.data_ptr());
    CUdeviceptr var_130 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_ctr_net_tower_layers___2___norm_bias.data_ptr());
    auto var_131 = triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel;
    void* kernel_args_var_26[] = {&var_126, &var_127, &var_128, &var_129, &var_130, &var_131};
    Grid triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_grid_26 = Grid(((511L + (128L*s0))/512L), 1L, 1L);
    if (triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_grid_26.is_non_zero()) {
    launchKernel(kernels.triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_grid_26.grid_x, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_grid_26.grid_y, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_grid_26.grid_z, 4, 0, kernel_args_var_26, stream);
    }
    at::Tensor buf65 = at::detail::empty_strided_cuda({s0, 568L}, {568L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [cat_1], Original ATen: [aten.cat]
    auto triton_poi_fused_cat_20_xnumel = 568L*s0;
    if (kernels.triton_poi_fused_cat_20 == nullptr) {
        kernels.triton_poi_fused_cat_20 = loadKernel("/home/admin/zy429782/fx_experiments/torch_aot_tool/dynamicLib_7622_gpu/dynamicLib_7622_gpu/cbvolgpdwredcubeo3n2a3tlc6dnlmahs4wj5vrbt3ojkqbv7ofr.cubin", "triton__0d1d234d5e", 0, this->cubin_dir_);
    }
    CUdeviceptr var_132 = reinterpret_cast<CUdeviceptr>(arg130_1.data_ptr());
    CUdeviceptr var_133 = reinterpret_cast<CUdeviceptr>(arg129_1.data_ptr());
    CUdeviceptr var_134 = reinterpret_cast<CUdeviceptr>(buf51.data_ptr());
    CUdeviceptr var_135 = reinterpret_cast<CUdeviceptr>(buf53.data_ptr());
    CUdeviceptr var_136 = reinterpret_cast<CUdeviceptr>(buf65.data_ptr());
    auto var_137 = triton_poi_fused_cat_20_xnumel;
    void* kernel_args_var_27[] = {&var_132, &var_133, &var_134, &var_135, &var_136, &var_137};
    Grid triton_poi_fused_cat_20_grid_27 = Grid(((1023L + (568L*s0))/1024L), 1L, 1L);
    if (triton_poi_fused_cat_20_grid_27.is_non_zero()) {
    launchKernel(kernels.triton_poi_fused_cat_20, triton_poi_fused_cat_20_grid_27.grid_x, triton_poi_fused_cat_20_grid_27.grid_y, triton_poi_fused_cat_20_grid_27.grid_z, 4, 0, kernel_args_var_27, stream);
    }
    arg129_1.reset();
    arg130_1.reset();
    buf51.reset();
    buf53.reset();
    decltype(auto) buf66 = buf62; buf62.reset();;  // reuse
    // Source Nodes: [cat_1, user_model_bias_net_tower_layers_0_fc], Original ATen: [aten.cat, aten.mm]
    at::mm_out(buf66, buf65, reinterpret_tensor(getattr_L__self___user_model_bias_net_tower_layers___0___fc_weight, {568L, 256L}, {1L, 568L}, 0L));
    buf65.reset();
    decltype(auto) buf67 = buf66; buf66.reset();;  // reuse
    decltype(auto) buf68 = buf67; buf67.reset();;  // reuse
    // Source Nodes: [user_model_bias_net_tower_layers_0_act, user_model_bias_net_tower_layers_0_norm], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
    triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_xnumel = 256L*s0;
    CUdeviceptr var_138 = reinterpret_cast<CUdeviceptr>(buf68.data_ptr());
    CUdeviceptr var_139 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_bias_net_tower_layers___0___norm_running_mean.data_ptr());
    CUdeviceptr var_140 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_bias_net_tower_layers___0___norm_running_var.data_ptr());
    CUdeviceptr var_141 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_bias_net_tower_layers___0___norm_weight.data_ptr());
    CUdeviceptr var_142 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_bias_net_tower_layers___0___norm_bias.data_ptr());
    auto var_143 = triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_xnumel;
    void* kernel_args_var_28[] = {&var_138, &var_139, &var_140, &var_141, &var_142, &var_143};
    Grid triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_grid_28 = Grid(((1023L + (256L*s0))/1024L), 1L, 1L);
    if (triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_grid_28.is_non_zero()) {
    launchKernel(kernels.triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_grid_28.grid_x, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_grid_28.grid_y, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_grid_28.grid_z, 4, 0, kernel_args_var_28, stream);
    }
    decltype(auto) buf69 = reinterpret_tensor(buf32, {s0, 128L}, {128L, 1L}, 0L); buf32.reset();  // reuse
    // Source Nodes: [user_model_bias_net_tower_layers_0_act, user_model_bias_net_tower_layers_1_fc], Original ATen: [aten.leaky_relu, aten.mm]
    at::mm_out(buf69, buf68, reinterpret_tensor(getattr_L__self___user_model_bias_net_tower_layers___1___fc_weight, {256L, 128L}, {1L, 256L}, 0L));
    decltype(auto) buf70 = buf69; buf69.reset();;  // reuse
    decltype(auto) buf72 = buf70; buf70.reset();;  // reuse
    // Source Nodes: [user_model_bias_net_tower_layers_1_act, user_model_bias_net_tower_layers_1_norm], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
    triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel = 128L*s0;
    CUdeviceptr var_144 = reinterpret_cast<CUdeviceptr>(buf72.data_ptr());
    CUdeviceptr var_145 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_bias_net_tower_layers___1___norm_running_mean.data_ptr());
    CUdeviceptr var_146 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_bias_net_tower_layers___1___norm_running_var.data_ptr());
    CUdeviceptr var_147 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_bias_net_tower_layers___1___norm_weight.data_ptr());
    CUdeviceptr var_148 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_bias_net_tower_layers___1___norm_bias.data_ptr());
    auto var_149 = triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel;
    void* kernel_args_var_29[] = {&var_144, &var_145, &var_146, &var_147, &var_148, &var_149};
    Grid triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_grid_29 = Grid(((511L + (128L*s0))/512L), 1L, 1L);
    if (triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_grid_29.is_non_zero()) {
    launchKernel(kernels.triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_grid_29.grid_x, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_grid_29.grid_y, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_grid_29.grid_z, 4, 0, kernel_args_var_29, stream);
    }
    decltype(auto) buf73 = reinterpret_tensor(buf38, {s0, 1L}, {1L, 1L}, 0L); buf38.reset();  // reuse
    // Source Nodes: [add_2, user_model_bias_net_tower_layers_1_act, user_model_ctr_net_tower_layers_2_act], Original ATen: [aten.add, aten.leaky_relu]
    torch::inductor::_mm_plus_mm(buf71, reinterpret_tensor(L__self___user_model_ctr_head_weight, {128L, 1L}, {1L, 128L}, 0L), buf72, reinterpret_tensor(L__self___user_model_ctr_bias_head_weight, {128L, 1L}, {1L, 128L}, 0L), buf73);
    decltype(auto) buf74 = buf59; buf59.reset();;  // reuse
    // Source Nodes: [user_model_click_net_tower_layers_0_fc], Original ATen: [aten.mm]
    at::mm_out(buf74, buf56, reinterpret_tensor(getattr_L__self___user_model_click_net_tower_layers___0___fc_weight, {1872L, 512L}, {1L, 1872L}, 0L));
    decltype(auto) buf75 = buf74; buf74.reset();;  // reuse
    decltype(auto) buf76 = buf75; buf75.reset();;  // reuse
    // Source Nodes: [user_model_click_net_tower_layers_0_act, user_model_click_net_tower_layers_0_norm], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
    triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_xnumel = 512L*s0;
    CUdeviceptr var_150 = reinterpret_cast<CUdeviceptr>(buf76.data_ptr());
    CUdeviceptr var_151 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_click_net_tower_layers___0___norm_running_mean.data_ptr());
    CUdeviceptr var_152 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_click_net_tower_layers___0___norm_running_var.data_ptr());
    CUdeviceptr var_153 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_click_net_tower_layers___0___norm_weight.data_ptr());
    CUdeviceptr var_154 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_click_net_tower_layers___0___norm_bias.data_ptr());
    auto var_155 = triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_xnumel;
    void* kernel_args_var_30[] = {&var_150, &var_151, &var_152, &var_153, &var_154, &var_155};
    Grid triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_grid_30 = Grid(s0, 1L, 1L);
    if (triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_grid_30.is_non_zero()) {
    launchKernel(kernels.triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_grid_30.grid_x, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_grid_30.grid_y, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_grid_30.grid_z, 8, 0, kernel_args_var_30, stream);
    }
    decltype(auto) buf77 = buf68; buf68.reset();;  // reuse
    // Source Nodes: [user_model_click_net_tower_layers_0_act, user_model_click_net_tower_layers_1_fc], Original ATen: [aten.leaky_relu, aten.mm]
    at::mm_out(buf77, buf76, reinterpret_tensor(getattr_L__self___user_model_click_net_tower_layers___1___fc_weight, {512L, 256L}, {1L, 512L}, 0L));
    decltype(auto) buf78 = buf77; buf77.reset();;  // reuse
    decltype(auto) buf79 = buf78; buf78.reset();;  // reuse
    // Source Nodes: [user_model_click_net_tower_layers_1_act, user_model_click_net_tower_layers_1_norm], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
    triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_xnumel = 256L*s0;
    CUdeviceptr var_156 = reinterpret_cast<CUdeviceptr>(buf79.data_ptr());
    CUdeviceptr var_157 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_click_net_tower_layers___1___norm_running_mean.data_ptr());
    CUdeviceptr var_158 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_click_net_tower_layers___1___norm_running_var.data_ptr());
    CUdeviceptr var_159 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_click_net_tower_layers___1___norm_weight.data_ptr());
    CUdeviceptr var_160 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_click_net_tower_layers___1___norm_bias.data_ptr());
    auto var_161 = triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_xnumel;
    void* kernel_args_var_31[] = {&var_156, &var_157, &var_158, &var_159, &var_160, &var_161};
    Grid triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_grid_31 = Grid(((1023L + (256L*s0))/1024L), 1L, 1L);
    if (triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_grid_31.is_non_zero()) {
    launchKernel(kernels.triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_grid_31.grid_x, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_grid_31.grid_y, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_grid_31.grid_z, 4, 0, kernel_args_var_31, stream);
    }
    decltype(auto) buf80 = buf71; buf71.reset();;  // reuse
    // Source Nodes: [user_model_click_net_tower_layers_1_act, user_model_click_net_tower_layers_2_fc], Original ATen: [aten.leaky_relu, aten.mm]
    at::mm_out(buf80, buf79, reinterpret_tensor(getattr_L__self___user_model_click_net_tower_layers___2___fc_weight, {256L, 128L}, {1L, 256L}, 0L));
    decltype(auto) buf81 = buf80; buf80.reset();;  // reuse
    decltype(auto) buf82 = buf81; buf81.reset();;  // reuse
    // Source Nodes: [user_model_click_net_tower_layers_2_act, user_model_click_net_tower_layers_2_norm], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
    triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel = 128L*s0;
    CUdeviceptr var_162 = reinterpret_cast<CUdeviceptr>(buf82.data_ptr());
    CUdeviceptr var_163 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_click_net_tower_layers___2___norm_running_mean.data_ptr());
    CUdeviceptr var_164 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_click_net_tower_layers___2___norm_running_var.data_ptr());
    CUdeviceptr var_165 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_click_net_tower_layers___2___norm_weight.data_ptr());
    CUdeviceptr var_166 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_click_net_tower_layers___2___norm_bias.data_ptr());
    auto var_167 = triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel;
    void* kernel_args_var_32[] = {&var_162, &var_163, &var_164, &var_165, &var_166, &var_167};
    Grid triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_grid_32 = Grid(((511L + (128L*s0))/512L), 1L, 1L);
    if (triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_grid_32.is_non_zero()) {
    launchKernel(kernels.triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_grid_32.grid_x, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_grid_32.grid_y, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_grid_32.grid_z, 4, 0, kernel_args_var_32, stream);
    }
    decltype(auto) buf83 = reinterpret_tensor(buf33, {s0, 1L}, {1L, 1L}, 0L); buf33.reset();  // reuse
    // Source Nodes: [add_3, user_model_click_net_tower_layers_2_act], Original ATen: [aten.add, aten.leaky_relu]
    torch::inductor::_mm_plus_mm(buf82, reinterpret_tensor(L__self___user_model_click_head_weight, {128L, 1L}, {1L, 128L}, 0L), buf72, reinterpret_tensor(L__self___user_model_click_bias_head_weight, {128L, 1L}, {1L, 128L}, 0L), buf83);
    decltype(auto) buf84 = buf76; buf76.reset();;  // reuse
    // Source Nodes: [user_model_page_net_tower_layers_0_fc], Original ATen: [aten.mm]
    at::mm_out(buf84, buf56, reinterpret_tensor(getattr_L__self___user_model_page_net_tower_layers___0___fc_weight, {1872L, 512L}, {1L, 1872L}, 0L));
    decltype(auto) buf85 = buf84; buf84.reset();;  // reuse
    decltype(auto) buf86 = buf85; buf85.reset();;  // reuse
    // Source Nodes: [user_model_page_net_tower_layers_0_act, user_model_page_net_tower_layers_0_norm], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
    triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_xnumel = 512L*s0;
    CUdeviceptr var_168 = reinterpret_cast<CUdeviceptr>(buf86.data_ptr());
    CUdeviceptr var_169 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_page_net_tower_layers___0___norm_running_mean.data_ptr());
    CUdeviceptr var_170 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_page_net_tower_layers___0___norm_running_var.data_ptr());
    CUdeviceptr var_171 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_page_net_tower_layers___0___norm_weight.data_ptr());
    CUdeviceptr var_172 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_page_net_tower_layers___0___norm_bias.data_ptr());
    auto var_173 = triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_xnumel;
    void* kernel_args_var_33[] = {&var_168, &var_169, &var_170, &var_171, &var_172, &var_173};
    Grid triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_grid_33 = Grid(s0, 1L, 1L);
    if (triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_grid_33.is_non_zero()) {
    launchKernel(kernels.triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_grid_33.grid_x, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_grid_33.grid_y, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_grid_33.grid_z, 8, 0, kernel_args_var_33, stream);
    }
    decltype(auto) buf87 = buf79; buf79.reset();;  // reuse
    // Source Nodes: [user_model_page_net_tower_layers_0_act, user_model_page_net_tower_layers_1_fc], Original ATen: [aten.leaky_relu, aten.mm]
    at::mm_out(buf87, buf86, reinterpret_tensor(getattr_L__self___user_model_page_net_tower_layers___1___fc_weight, {512L, 256L}, {1L, 512L}, 0L));
    decltype(auto) buf88 = buf87; buf87.reset();;  // reuse
    decltype(auto) buf89 = buf88; buf88.reset();;  // reuse
    // Source Nodes: [user_model_page_net_tower_layers_1_act, user_model_page_net_tower_layers_1_norm], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
    triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_xnumel = 256L*s0;
    CUdeviceptr var_174 = reinterpret_cast<CUdeviceptr>(buf89.data_ptr());
    CUdeviceptr var_175 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_page_net_tower_layers___1___norm_running_mean.data_ptr());
    CUdeviceptr var_176 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_page_net_tower_layers___1___norm_running_var.data_ptr());
    CUdeviceptr var_177 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_page_net_tower_layers___1___norm_weight.data_ptr());
    CUdeviceptr var_178 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_page_net_tower_layers___1___norm_bias.data_ptr());
    auto var_179 = triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_xnumel;
    void* kernel_args_var_34[] = {&var_174, &var_175, &var_176, &var_177, &var_178, &var_179};
    Grid triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_grid_34 = Grid(((1023L + (256L*s0))/1024L), 1L, 1L);
    if (triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_grid_34.is_non_zero()) {
    launchKernel(kernels.triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_grid_34.grid_x, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_grid_34.grid_y, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_grid_34.grid_z, 4, 0, kernel_args_var_34, stream);
    }
    decltype(auto) buf90 = buf82; buf82.reset();;  // reuse
    // Source Nodes: [user_model_page_net_tower_layers_1_act, user_model_page_net_tower_layers_2_fc], Original ATen: [aten.leaky_relu, aten.mm]
    at::mm_out(buf90, buf89, reinterpret_tensor(getattr_L__self___user_model_page_net_tower_layers___2___fc_weight, {256L, 128L}, {1L, 256L}, 0L));
    decltype(auto) buf91 = buf90; buf90.reset();;  // reuse
    decltype(auto) buf92 = buf91; buf91.reset();;  // reuse
    // Source Nodes: [user_model_page_net_tower_layers_2_act, user_model_page_net_tower_layers_2_norm], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
    triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel = 128L*s0;
    CUdeviceptr var_180 = reinterpret_cast<CUdeviceptr>(buf92.data_ptr());
    CUdeviceptr var_181 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_page_net_tower_layers___2___norm_running_mean.data_ptr());
    CUdeviceptr var_182 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_page_net_tower_layers___2___norm_running_var.data_ptr());
    CUdeviceptr var_183 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_page_net_tower_layers___2___norm_weight.data_ptr());
    CUdeviceptr var_184 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_page_net_tower_layers___2___norm_bias.data_ptr());
    auto var_185 = triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel;
    void* kernel_args_var_35[] = {&var_180, &var_181, &var_182, &var_183, &var_184, &var_185};
    Grid triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_grid_35 = Grid(((511L + (128L*s0))/512L), 1L, 1L);
    if (triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_grid_35.is_non_zero()) {
    launchKernel(kernels.triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_grid_35.grid_x, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_grid_35.grid_y, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_grid_35.grid_z, 4, 0, kernel_args_var_35, stream);
    }
    at::Tensor buf93 = at::detail::empty_strided_cuda({s0, 1L}, {1L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [add_4, user_model_page_net_tower_layers_2_act], Original ATen: [aten.add, aten.leaky_relu]
    torch::inductor::_mm_plus_mm(buf92, reinterpret_tensor(L__self___user_model_page_head_weight, {128L, 1L}, {1L, 128L}, 0L), buf72, reinterpret_tensor(L__self___user_model_page_bias_head_weight, {128L, 1L}, {1L, 128L}, 0L), buf93);
    decltype(auto) buf94 = buf86; buf86.reset();;  // reuse
    // Source Nodes: [user_model_pay_net_tower_layers_0_fc], Original ATen: [aten.mm]
    at::mm_out(buf94, buf56, reinterpret_tensor(getattr_L__self___user_model_pay_net_tower_layers___0___fc_weight, {1872L, 512L}, {1L, 1872L}, 0L));
    buf56.reset();
    decltype(auto) buf95 = buf94; buf94.reset();;  // reuse
    decltype(auto) buf96 = buf95; buf95.reset();;  // reuse
    // Source Nodes: [user_model_pay_net_tower_layers_0_act, user_model_pay_net_tower_layers_0_norm], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
    triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_xnumel = 512L*s0;
    CUdeviceptr var_186 = reinterpret_cast<CUdeviceptr>(buf96.data_ptr());
    CUdeviceptr var_187 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_pay_net_tower_layers___0___norm_running_mean.data_ptr());
    CUdeviceptr var_188 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_pay_net_tower_layers___0___norm_running_var.data_ptr());
    CUdeviceptr var_189 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_pay_net_tower_layers___0___norm_weight.data_ptr());
    CUdeviceptr var_190 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_pay_net_tower_layers___0___norm_bias.data_ptr());
    auto var_191 = triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_xnumel;
    void* kernel_args_var_36[] = {&var_186, &var_187, &var_188, &var_189, &var_190, &var_191};
    Grid triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_grid_36 = Grid(s0, 1L, 1L);
    if (triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_grid_36.is_non_zero()) {
    launchKernel(kernels.triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_grid_36.grid_x, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_grid_36.grid_y, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_17_grid_36.grid_z, 8, 0, kernel_args_var_36, stream);
    }
    decltype(auto) buf97 = buf89; buf89.reset();;  // reuse
    // Source Nodes: [user_model_pay_net_tower_layers_0_act, user_model_pay_net_tower_layers_1_fc], Original ATen: [aten.leaky_relu, aten.mm]
    at::mm_out(buf97, buf96, reinterpret_tensor(getattr_L__self___user_model_pay_net_tower_layers___1___fc_weight, {512L, 256L}, {1L, 512L}, 0L));
    buf96.reset();
    decltype(auto) buf98 = buf97; buf97.reset();;  // reuse
    decltype(auto) buf99 = buf98; buf98.reset();;  // reuse
    // Source Nodes: [user_model_pay_net_tower_layers_1_act, user_model_pay_net_tower_layers_1_norm], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
    triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_xnumel = 256L*s0;
    CUdeviceptr var_192 = reinterpret_cast<CUdeviceptr>(buf99.data_ptr());
    CUdeviceptr var_193 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_pay_net_tower_layers___1___norm_running_mean.data_ptr());
    CUdeviceptr var_194 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_pay_net_tower_layers___1___norm_running_var.data_ptr());
    CUdeviceptr var_195 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_pay_net_tower_layers___1___norm_weight.data_ptr());
    CUdeviceptr var_196 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_pay_net_tower_layers___1___norm_bias.data_ptr());
    auto var_197 = triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_xnumel;
    void* kernel_args_var_37[] = {&var_192, &var_193, &var_194, &var_195, &var_196, &var_197};
    Grid triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_grid_37 = Grid(((1023L + (256L*s0))/1024L), 1L, 1L);
    if (triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_grid_37.is_non_zero()) {
    launchKernel(kernels.triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_grid_37.grid_x, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_grid_37.grid_y, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_18_grid_37.grid_z, 4, 0, kernel_args_var_37, stream);
    }
    decltype(auto) buf100 = buf92; buf92.reset();;  // reuse
    // Source Nodes: [user_model_pay_net_tower_layers_1_act, user_model_pay_net_tower_layers_2_fc], Original ATen: [aten.leaky_relu, aten.mm]
    at::mm_out(buf100, buf99, reinterpret_tensor(getattr_L__self___user_model_pay_net_tower_layers___2___fc_weight, {256L, 128L}, {1L, 256L}, 0L));
    buf99.reset();
    decltype(auto) buf101 = buf100; buf100.reset();;  // reuse
    decltype(auto) buf102 = buf101; buf101.reset();;  // reuse
    // Source Nodes: [user_model_pay_net_tower_layers_2_act, user_model_pay_net_tower_layers_2_norm], Original ATen: [aten._native_batch_norm_legit_no_training, aten.leaky_relu]
    triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel = 128L*s0;
    CUdeviceptr var_198 = reinterpret_cast<CUdeviceptr>(buf102.data_ptr());
    CUdeviceptr var_199 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_pay_net_tower_layers___2___norm_running_mean.data_ptr());
    CUdeviceptr var_200 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_pay_net_tower_layers___2___norm_running_var.data_ptr());
    CUdeviceptr var_201 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_pay_net_tower_layers___2___norm_weight.data_ptr());
    CUdeviceptr var_202 = reinterpret_cast<CUdeviceptr>(getattr_L__self___user_model_pay_net_tower_layers___2___norm_bias.data_ptr());
    auto var_203 = triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_xnumel;
    void* kernel_args_var_38[] = {&var_198, &var_199, &var_200, &var_201, &var_202, &var_203};
    Grid triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_grid_38 = Grid(((511L + (128L*s0))/512L), 1L, 1L);
    if (triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_grid_38.is_non_zero()) {
    launchKernel(kernels.triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_grid_38.grid_x, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_grid_38.grid_y, triton_poi_fused__native_batch_norm_legit_no_training_leaky_relu_19_grid_38.grid_z, 4, 0, kernel_args_var_38, stream);
    }
    at::Tensor buf103 = at::detail::empty_strided_cuda({s0, 1L}, {1L, 1L}, at::kFloat, c10::DeviceType::CUDA);
    // Source Nodes: [add_5, user_model_pay_net_tower_layers_2_act], Original ATen: [aten.add, aten.leaky_relu]
    torch::inductor::_mm_plus_mm(buf102, reinterpret_tensor(L__self___user_model_pay_head_weight, {128L, 1L}, {1L, 128L}, 0L), buf72, reinterpret_tensor(L__self___user_model_pay_bias_head_weight, {128L, 1L}, {1L, 128L}, 0L), buf103);
    buf102.reset();
    buf72.reset();
    output_handles[0] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf73));
    output_handles[1] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf83));
    output_handles[2] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf93));
    output_handles[3] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(buf103));
} // AOTInductorModel::run_impl
} // namespace aot_inductor
} // namespace torch
