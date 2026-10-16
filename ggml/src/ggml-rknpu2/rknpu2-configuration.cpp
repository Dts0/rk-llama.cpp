#include "ggml-impl.h"

#include "rknpu2-configuration.h"

#include <algorithm>
#include <arm_neon.h>
#include <cstdlib>
#include <sstream>

namespace {
    bool trace_enabled() {
        static const bool enabled = std::getenv("RKNPU_TRACE") != nullptr;
        return enabled;
    }

    // Function for parsing ENV variable
    std::vector<std::string> split_string(const std::string& str, char delimiter) {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(str);
        while (std::getline(tokenStream, token, delimiter)) {
            if(!token.empty()) tokens.push_back(token);
        }
        return tokens;
    }

    // Function for parsing complex integer lists
    std::vector<int> parse_int_list(const std::string& str) {
        std::vector<int> result;
        if (str.empty()) return result;
        std::stringstream ss(str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            if (token.empty()) continue;
            auto dash_pos = token.find('-');
            if (dash_pos != std::string::npos) {
                int start = std::strtol(token.substr(0, dash_pos).c_str(), nullptr, 10);
                int end = std::strtol(token.substr(dash_pos + 1).c_str(), nullptr, 10);
                for (int i = start; i <= end; ++i) result.push_back(i);
            } else {
                result.push_back(std::strtol(token.c_str(), nullptr, 10));
            }
        }
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());
        return result;
    }
} // anonymous namespace

namespace rknpu2_configuration {

Rknpu2ConfigManager& Rknpu2ConfigManager::get_instance() {
    static Rknpu2ConfigManager instance;
    return instance;
}

const std::vector<std::string>* Rknpu2DeviceConfig::get_active_pattern(int tensor_type) const {
    auto it = default_patterns.find(tensor_type);
    if (it == default_patterns.end()) {
        return nullptr;
    }

    if (use_custom_pattern && !custom_hybrid_pattern.empty()) {
        return &custom_hybrid_pattern;
    }

    return &it->second;
}

const Rknpu2HardwarePipeline* Rknpu2DeviceConfig::resolve_op_support(const struct ggml_tensor* w_tensor) const {
    if (!w_tensor) return nullptr;

    auto find_pipeline = [this](const std::string& name) -> const Rknpu2HardwarePipeline* {
        for (const auto& pipe : hardware_pipelines) {
            if (pipe.pipeline_name == name) return &pipe;
        }
        return nullptr;
    };

    // Retrieve active quantization pattern based on tensor type (or custom ENV variable)
    const std::vector<std::string>* pattern_ptr = get_active_pattern((int)w_tensor->type);

    // If no pattern is registered for this type and no global override exists, reject operation
    if (!pattern_ptr || pattern_ptr->empty()) {
        return nullptr;
    }

    const auto& pattern = *pattern_ptr;

    // Acquiring the lock on the pattern mutex for thread-safe tensor tracking
    std::lock_guard<std::mutex> lock(*pattern_mutex);

    // Retrieving the unique tensor name
    std::string name = w_tensor->name;
    if (name.empty()) {
        name = "ptr_" + std::to_string(reinterpret_cast<uintptr_t>(w_tensor));
    }

    // Assigning the next sequence number if this tensor is seen for the first time
    if (tensor_sequence_map.find(name) == tensor_sequence_map.end()) {
        tensor_sequence_map[name] = global_tensor_counter++;
    }

    // Selecting the pipeline cyclically based on the defined pattern
    int seq_id = tensor_sequence_map[name];
    size_t pattern_idx = seq_id % pattern.size();

    const std::string& selected_pipeline = pattern[pattern_idx];
    const auto* pipeline = find_pipeline(selected_pipeline);

    // K256 is an opt-in Q4_K-only pipeline. Keep other tensor types on their
    // default pipeline (notably Q6_K -> W8A16_STANDARD) when RKNPU_HYBRID
    // selects W4A16_Q4K_K256 globally.
    if (pipeline && pipeline->use_q4k_k256 && w_tensor->type != GGML_TYPE_Q4_K) {
        auto default_it = default_patterns.find((int)w_tensor->type);
        if (default_it != default_patterns.end() && !default_it->second.empty()) {
            const auto & fallback_pattern = default_it->second;
            const auto * fallback = find_pipeline(fallback_pattern[seq_id % fallback_pattern.size()]);
            if (fallback) {
                pipeline = fallback;
            }
        }
    }

    // Explicit FFN INT4 hybrid: Q4_K FFN uses the platform's per-channel
    // INT4 Hadamard pipeline; other tensors use the platform's INT8 path.
    // RK3576 supports FP16 activations (W4A16/W8A16); RK3588 supports only
    // INT4/INT8 activations for these quantized matmuls (W4A4/W8A8).
    if (pipeline && pipeline->use_q4k_ffn_int4_hybrid) {
        const bool is_rk3588 = device_name == "RK3588";
        const char * int4_pipeline = is_rk3588 ? "W4A4_HADAMARD" : "W4A16_HADAMARD";
        const char * int8_pipeline = is_rk3588 ? "W8A8_STANDARD" : "W8A16_STANDARD";
        if (w_tensor->type != GGML_TYPE_Q4_K) {
            const auto * fallback = find_pipeline(int8_pipeline);
            if (fallback) pipeline = fallback;
        } else {
            const std::string tensor_name = w_tensor->name[0] ? w_tensor->name : "";
            const bool is_ffn = tensor_name.find(".ffn_") != std::string::npos;
            const auto * mapped = find_pipeline(is_ffn ? int4_pipeline : int8_pipeline);
            if (mapped) pipeline = mapped;
        }
    }

    // Q4_K staged INT8 is a Q4_K-only opt-in. Keep other tensor types on
    // their default pipelines (notably Q6_K -> W8A16_STANDARD).
    if (pipeline && pipeline->use_q4k_staged_int8 && w_tensor->type != GGML_TYPE_Q4_K) {
        auto default_it = default_patterns.find((int)w_tensor->type);
        if (default_it != default_patterns.end() && !default_it->second.empty()) {
            const auto & fallback_pattern = default_it->second;
            const auto * fallback = find_pipeline(fallback_pattern[seq_id % fallback_pattern.size()]);
            if (fallback) pipeline = fallback;
        }
    }

    // Selective staging: keep small Q4_K tensors on persistent W8A16 and only
    // stage large tensors where conversion can be amortized. The threshold is
    // configurable for experiments; default is 8M elements (e.g. 2048x4096).
    if (pipeline && pipeline->use_q4k_staged_int8 && w_tensor->type == GGML_TYPE_Q4_K) {
        static const int64_t staged_min_elems = []() -> int64_t {
            const char * env = std::getenv("RKNPU_STAGED_MIN_ELEMS");
            if (env != nullptr) {
                const long long v = std::strtoll(env, nullptr, 10);
                if (v >= 0) return (int64_t) v;
            }
            return (int64_t) 8 * 1024 * 1024;
        }();
        const int64_t elems = (int64_t) w_tensor->ne[0] * (int64_t) w_tensor->ne[1];
        if (elems < staged_min_elems) {
            const auto * fallback = find_pipeline("W8A16_STANDARD");
            if (fallback) pipeline = fallback;
        }
    }

    if (pipeline && pipeline->use_q4k_hybrid) {
        if (w_tensor->type != GGML_TYPE_Q4_K) {
            auto default_it = default_patterns.find((int)w_tensor->type);
            if (default_it != default_patterns.end() && !default_it->second.empty()) {
                const auto & fallback_pattern = default_it->second;
                const auto * fallback = find_pipeline(fallback_pattern[seq_id % fallback_pattern.size()]);
                if (fallback) pipeline = fallback;
            }
        } else {
            const std::string tensor_name = w_tensor->name[0] ? w_tensor->name : "";
            const bool is_ffn = tensor_name.find(".ffn_") != std::string::npos;
            const char * selected = is_ffn ? "W8A16_STANDARD" : "W4A16_STANDARD";
            const auto * mapped = find_pipeline(selected);
            if (mapped) pipeline = mapped;
        }
    }

    // If no hardware pipeline exists with this name, reject operation
    if (!pipeline) {
        if (trace_enabled()) {
            fprintf(stderr,
                    "RKNPU trace: pipeline '%s' not found on device '%s' for tensor '%s' (type=%d)\n",
                    selected_pipeline.c_str(),
                    device_name.c_str(),
                    w_tensor->name[0] ? w_tensor->name : "<unnamed>",
                    (int)w_tensor->type);
        }
        return nullptr;
    }

    return pipeline;
}

Rknpu2ConfigManager::Rknpu2ConfigManager() {
    const char* env_device = std::getenv("RKNPU_DEVICE");
    const std::string target_device = env_device ? env_device : "RK3588";

    // Reading custom hybrid pattern ENV variable
    const char* env_pattern = std::getenv("RKNPU_HYBRID");
    bool use_custom_pattern = false;
    std::vector<std::string> custom_pattern;

    if (env_pattern != nullptr) {
        custom_pattern = split_string(env_pattern, ',');
        use_custom_pattern = true;
    }

    // Reading specific active cores ENV variable
    const char* env_cores = std::getenv("RKNPU_CORES");
    std::vector<int> custom_cores;
    if (env_cores != nullptr) {
        custom_cores = parse_int_list(env_cores);
    }

    // --- Define RK3588 Configuration ---
    Rknpu2DeviceConfig rk3588_config;
    rk3588_config.device_name = "RK3588";
    rk3588_config.active_cores = custom_cores.empty() ? std::vector<int>{0, 1, 2} : custom_cores;
    rk3588_config.max_k_limit = 8192;
    rk3588_config.hardware_pipelines = {
        {
            /* .pipeline_name = */ "W16A16_STANDARD",
            /* .npu_type_a    = */ NPU_TYPE_FP16,
            /* .npu_type_b    = */ NPU_TYPE_FP16,
            /* .npu_type_c    = */ NPU_TYPE_FP32,
            /* .mm_type       = */ RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32,
            /* .k_align       = */ 32,
            /* .n_align       = */ 16,
            /* .effective_k   = */ 0,
            /* .use_hadamard  = */ false,
            /* .use_lowrank   = */ false
        },
        {
            /* .pipeline_name = */ "W16A16_HADAMARD",
            /* .npu_type_a    = */ NPU_TYPE_FP16,
            /* .npu_type_b    = */ NPU_TYPE_FP16,
            /* .npu_type_c    = */ NPU_TYPE_FP32,
            /* .mm_type       = */ RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32,
            /* .k_align       = */ 32,
            /* .n_align       = */ 16,
            /* .effective_k   = */ 0,
            /* .use_hadamard  = */ true,
            /* .use_lowrank   = */ false
        },
        {
            /* .pipeline_name = */ "W8A8_STANDARD",
            /* .npu_type_a    = */ NPU_TYPE_INT8,
            /* .npu_type_b    = */ NPU_TYPE_INT8,
            /* .npu_type_c    = */ NPU_TYPE_INT32,
            /* .mm_type       = */ RKNN_INT8_MM_INT8_TO_INT32,
            /* .k_align       = */ 32,
            /* .n_align       = */ 32,
            /* .effective_k   = */ 0,
            /* .use_hadamard  = */ false,
            /* .use_lowrank   = */ false
        },
        {
            /* .pipeline_name = */ "W8A8_HADAMARD",
            /* .npu_type_a    = */ NPU_TYPE_INT8,
            /* .npu_type_b    = */ NPU_TYPE_INT8,
            /* .npu_type_c    = */ NPU_TYPE_INT32,
            /* .mm_type       = */ RKNN_INT8_MM_INT8_TO_INT32,
            /* .k_align       = */ 32,
            /* .n_align       = */ 32,
            /* .effective_k   = */ 0,
            /* .use_hadamard  = */ true,
            /* .use_lowrank   = */ false
        },
        {
            /* .pipeline_name = */ "W8A8_LOWRANK",
            /* .npu_type_a    = */ NPU_TYPE_INT8,
            /* .npu_type_b    = */ NPU_TYPE_INT8,
            /* .npu_type_c    = */ NPU_TYPE_INT32,
            /* .mm_type       = */ RKNN_INT8_MM_INT8_TO_INT32,
            /* .k_align       = */ 32,
            /* .n_align       = */ 32,
            /* .effective_k   = */ 0,
            /* .use_hadamard  = */ false,
            /* .use_lowrank   = */ true
        },
        {
            /* .pipeline_name = */ "W8A8_HADAMARD_LOWRANK",
            /* .npu_type_a    = */ NPU_TYPE_INT8,
            /* .npu_type_b    = */ NPU_TYPE_INT8,
            /* .npu_type_c    = */ NPU_TYPE_INT32,
            /* .mm_type       = */ RKNN_INT8_MM_INT8_TO_INT32,
            /* .k_align       = */ 32,
            /* .n_align       = */ 32,
            /* .effective_k   = */ 0,
            /* .use_hadamard  = */ true,
            /* .use_lowrank   = */ true
        },
        {
            /* .pipeline_name = */ "W4A4_STANDARD",
            /* .npu_type_a    = */ NPU_TYPE_INT4,
            /* .npu_type_b    = */ NPU_TYPE_INT4,
            /* .npu_type_c    = */ NPU_TYPE_INT16,
            /* .mm_type       = */ RKNN_INT4_MM_INT4_TO_INT16,
            /* .k_align       = */ 32,
            /* .n_align       = */ 64,
            /* .effective_k   = */ 0,
            /* .use_hadamard  = */ false,
            /* .use_lowrank   = */ false
        },
        {
            /* .pipeline_name = */ "W4A4_HADAMARD",
            /* .npu_type_a    = */ NPU_TYPE_INT4,
            /* .npu_type_b    = */ NPU_TYPE_INT4,
            /* .npu_type_c    = */ NPU_TYPE_INT16,
            /* .mm_type       = */ RKNN_INT4_MM_INT4_TO_INT16,
            /* .k_align       = */ 32,
            /* .n_align       = */ 64,
            /* .effective_k   = */ 0,
            /* .use_hadamard  = */ true,
            /* .use_lowrank   = */ false
        },
        {
            /* .pipeline_name = */ "W4A4_LOWRANK",
            /* .npu_type_a    = */ NPU_TYPE_INT4,
            /* .npu_type_b    = */ NPU_TYPE_INT4,
            /* .npu_type_c    = */ NPU_TYPE_INT16,
            /* .mm_type       = */ RKNN_INT4_MM_INT4_TO_INT16,
            /* .k_align       = */ 32,
            /* .n_align       = */ 64,
            /* .effective_k   = */ 0,
            /* .use_hadamard  = */ false,
            /* .use_lowrank   = */ true
        },
        {
            /* .pipeline_name = */ "W4A4_HADAMARD_LOWRANK",
            /* .npu_type_a    = */ NPU_TYPE_INT4,
            /* .npu_type_b    = */ NPU_TYPE_INT4,
            /* .npu_type_c    = */ NPU_TYPE_INT16,
            /* .mm_type       = */ RKNN_INT4_MM_INT4_TO_INT16,
            /* .k_align       = */ 32,
            /* .n_align       = */ 64,
            /* .effective_k   = */ 0,
            /* .use_hadamard  = */ true,
            /* .use_lowrank   = */ true
        },
        {
            /* .pipeline_name = */ "W4A4_Q4K_FFN_HYBRID",
            /* .npu_type_a    = */ NPU_TYPE_INT4,
            /* .npu_type_b    = */ NPU_TYPE_INT4,
            /* .npu_type_c    = */ NPU_TYPE_INT16,
            /* .mm_type       = */ RKNN_INT4_MM_INT4_TO_INT16,
            /* .k_align       = */ 32,
            /* .n_align       = */ 64,
            /* .effective_k   = */ 0,
            /* .use_hadamard  = */ true,
            /* .use_lowrank   = */ false,
            /* .use_q4k_block32_asym = */ false,
            /* .use_q4k_k256 = */ false,
            /* .use_q4k_hybrid = */ false,
            /* .npu_type_b_secondary = */ NPU_TYPE_INT4,
            /* .mm_type_secondary = */ RKNN_INT4_MM_INT4_TO_INT16,
            /* .use_q4k_staged_int8 = */ false,
            /* .use_q4k_ffn_int4_hybrid = */ true
        }
    };

    // Assigning custom variables
    rk3588_config.use_custom_pattern = use_custom_pattern;
    rk3588_config.custom_hybrid_pattern = custom_pattern;

    // Defining default quantization sequences for each supported ggml_type
    rk3588_config.default_patterns[(int)GGML_TYPE_F16]  = {"W16A16_STANDARD"};
    rk3588_config.default_patterns[(int)GGML_TYPE_Q8_0] = {"W8A8_STANDARD"};
    rk3588_config.default_patterns[(int)GGML_TYPE_Q6_K] = {"W8A8_STANDARD"};
    rk3588_config.default_patterns[(int)GGML_TYPE_Q4_0] = {"W4A4_HADAMARD"};
    rk3588_config.default_patterns[(int)GGML_TYPE_Q4_K] = {"W8A8_STANDARD"};

    device_configs["RK3588"] = rk3588_config;

    // --- Define RK3576 Configuration ---
    Rknpu2DeviceConfig rk3576_config;
    rk3576_config.device_name = "RK3576";
    rk3576_config.active_cores = custom_cores.empty() ? std::vector<int>{0, 1} : custom_cores;
    rk3576_config.max_k_limit = 4096;
    rk3576_config.hardware_pipelines = {
        {
            /* .pipeline_name = */ "W16A16_STANDARD",
            /* .npu_type_a    = */ NPU_TYPE_FP16,
            /* .npu_type_b    = */ NPU_TYPE_FP16,
            /* .npu_type_c    = */ NPU_TYPE_FP32,
            /* .mm_type       = */ RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32,
            /* .k_align       = */ 32,
            /* .n_align       = */ 16,
            /* .effective_k   = */ 0,
            /* .use_hadamard  = */ false,
            /* .use_lowrank   = */ false
        },
        {
            /* .pipeline_name = */ "W16A16_HADAMARD",
            /* .npu_type_a    = */ NPU_TYPE_FP16,
            /* .npu_type_b    = */ NPU_TYPE_FP16,
            /* .npu_type_c    = */ NPU_TYPE_FP32,
            /* .mm_type       = */ RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32,
            /* .k_align       = */ 32,
            /* .n_align       = */ 16,
            /* .effective_k   = */ 0,
            /* .use_hadamard  = */ true,
            /* .use_lowrank   = */ false
        },
        {
            /* .pipeline_name = */ "W8A8_STANDARD",
            /* .npu_type_a    = */ NPU_TYPE_INT8,
            /* .npu_type_b    = */ NPU_TYPE_INT8,
            /* .npu_type_c    = */ NPU_TYPE_INT32,
            /* .mm_type       = */ RKNN_INT8_MM_INT8_TO_INT32,
            /* .k_align       = */ 32,
            /* .n_align       = */ 32,
            /* .effective_k   = */ 0,
            /* .use_hadamard  = */ false,
            /* .use_lowrank   = */ false
        },
        {
            /* .pipeline_name = */ "W8A8_HADAMARD",
            /* .npu_type_a    = */ NPU_TYPE_INT8,
            /* .npu_type_b    = */ NPU_TYPE_INT8,
            /* .npu_type_c    = */ NPU_TYPE_INT32,
            /* .mm_type       = */ RKNN_INT8_MM_INT8_TO_INT32,
            /* .k_align       = */ 32,
            /* .n_align       = */ 32,
            /* .effective_k   = */ 0,
            /* .use_hadamard  = */ true,
            /* .use_lowrank   = */ false
        },
        {
            /* .pipeline_name = */ "W8A16_STANDARD",
            /* .npu_type_a    = */ NPU_TYPE_FP16,
            /* .npu_type_b    = */ NPU_TYPE_INT8,
            /* .npu_type_c    = */ NPU_TYPE_FP32,
            /* .mm_type       = */ RKNN_FLOAT16_MM_INT8_TO_FLOAT32,
            /* .k_align       = */ 32,
            /* .n_align       = */ 16,
            /* .effective_k   = */ 0,
            /* .use_hadamard  = */ false,
            /* .use_lowrank   = */ false
        },
        {
            /* .pipeline_name = */ "W8A16_HADAMARD",
            /* .npu_type_a    = */ NPU_TYPE_FP16,
            /* .npu_type_b    = */ NPU_TYPE_INT8,
            /* .npu_type_c    = */ NPU_TYPE_FP32,
            /* .mm_type       = */ RKNN_FLOAT16_MM_INT8_TO_FLOAT32,
            /* .k_align       = */ 32,
            /* .n_align       = */ 16,
            /* .effective_k   = */ 0,
            /* .use_hadamard  = */ true,
            /* .use_lowrank   = */ false
        },
        {
            /* .pipeline_name = */ "W4A4_STANDARD",
            /* .npu_type_a    = */ NPU_TYPE_INT4,
            /* .npu_type_b    = */ NPU_TYPE_INT4,
            /* .npu_type_c    = */ NPU_TYPE_INT16,
            /* .mm_type       = */ RKNN_INT4_MM_INT4_TO_INT16,
            /* .k_align       = */ 32,
            /* .n_align       = */ 64,
            /* .effective_k   = */ 0,
            /* .use_hadamard  = */ false,
            /* .use_lowrank   = */ false
        },
        {
            /* .pipeline_name = */ "W4A4_HADAMARD",
            /* .npu_type_a    = */ NPU_TYPE_INT4,
            /* .npu_type_b    = */ NPU_TYPE_INT4,
            /* .npu_type_c    = */ NPU_TYPE_INT16,
            /* .mm_type       = */ RKNN_INT4_MM_INT4_TO_INT16,
            /* .k_align       = */ 32,
            /* .n_align       = */ 64,
            /* .effective_k   = */ 0,
            /* .use_hadamard  = */ true,
            /* .use_lowrank   = */ false
        },
        {
            /* .pipeline_name = */ "W4A16_STANDARD",
            /* .npu_type_a    = */ NPU_TYPE_FP16,
            /* .npu_type_b    = */ NPU_TYPE_INT4,
            /* .npu_type_c    = */ NPU_TYPE_FP32,
            /* .mm_type       = */ RKNN_FLOAT16_MM_INT4_TO_FLOAT32,
            /* .k_align       = */ 32,
            /* .n_align       = */ 16,
            /* .effective_k   = */ 32,
            /* .use_hadamard  = */ false,
            /* .use_lowrank   = */ false,
            /* .use_q4k_block32_asym = */ true
        },
        {
            /* .pipeline_name = */ "W4A16_HADAMARD",
            /* .npu_type_a    = */ NPU_TYPE_FP16,
            /* .npu_type_b    = */ NPU_TYPE_INT4,
            /* .npu_type_c    = */ NPU_TYPE_FP32,
            /* .mm_type       = */ RKNN_FLOAT16_MM_INT4_TO_FLOAT32,
            /* .k_align       = */ 32,
            /* .n_align       = */ 16,
            /* .effective_k   = */ 0,
            /* .use_hadamard  = */ true,
            /* .use_lowrank   = */ false
        },
        {
            /* .pipeline_name = */ "W4A16_Q4K_K256",
            /* .npu_type_a    = */ NPU_TYPE_FP16,
            /* .npu_type_b    = */ NPU_TYPE_INT8,
            /* .npu_type_c    = */ NPU_TYPE_FP32,
            /* .mm_type       = */ RKNN_FLOAT16_MM_INT8_TO_FLOAT32,
            /* .k_align       = */ 256,
            /* .n_align       = */ 16,
            /* .effective_k   = */ 4096,
            /* .use_hadamard  = */ false,
            /* .use_lowrank   = */ false,
            /* .use_q4k_block32_asym = */ false,
            /* .use_q4k_k256 = */ true,
            /* .use_q4k_hybrid = */ false,
            /* .npu_type_b_secondary = */ NPU_TYPE_INT4,
            /* .mm_type_secondary = */ RKNN_FLOAT16_MM_INT4_TO_FLOAT32
        },
        {
            /* .pipeline_name = */ "W4A16_Q4K_HYBRID",
            /* .npu_type_a    = */ NPU_TYPE_FP16,
            /* .npu_type_b    = */ NPU_TYPE_INT4,
            /* .npu_type_c    = */ NPU_TYPE_FP32,
            /* .mm_type       = */ RKNN_FLOAT16_MM_INT4_TO_FLOAT32,
            /* .k_align       = */ 32,
            /* .n_align       = */ 16,
            /* .effective_k   = */ 0,
            /* .use_hadamard  = */ false,
            /* .use_lowrank   = */ false,
            /* .use_q4k_block32_asym = */ false,
            /* .use_q4k_k256 = */ false,
            /* .use_q4k_hybrid = */ true,
            /* .npu_type_b_secondary = */ NPU_TYPE_INT4,
            /* .mm_type_secondary = */ RKNN_FLOAT16_MM_INT4_TO_FLOAT32
        },
        {
            /* .pipeline_name = */ "W4A16_Q4K_STAGED_INT8",
            /* .npu_type_a    = */ NPU_TYPE_FP16,
            /* .npu_type_b    = */ NPU_TYPE_INT8,
            /* .npu_type_c    = */ NPU_TYPE_FP32,
            /* .mm_type       = */ RKNN_FLOAT16_MM_INT8_TO_FLOAT32,
            /* .k_align       = */ 32,
            /* .n_align       = */ 16,
            /* .effective_k   = */ 0,
            /* .use_hadamard  = */ false,
            /* .use_lowrank   = */ false,
            /* .use_q4k_block32_asym = */ false,
            /* .use_q4k_k256 = */ false,
            /* .use_q4k_hybrid = */ false,
            /* .npu_type_b_secondary = */ NPU_TYPE_INT4,
            /* .mm_type_secondary = */ RKNN_FLOAT16_MM_INT4_TO_FLOAT32,
            /* .use_q4k_staged_int8 = */ true
        },
        {
            /* .pipeline_name = */ "W4A16_Q4K_FFN_INT4_HYBRID",
            /* .npu_type_a    = */ NPU_TYPE_FP16,
            /* .npu_type_b    = */ NPU_TYPE_INT4,
            /* .npu_type_c    = */ NPU_TYPE_FP32,
            /* .mm_type       = */ RKNN_FLOAT16_MM_INT4_TO_FLOAT32,
            /* .k_align       = */ 32,
            /* .n_align       = */ 16,
            /* .effective_k   = */ 0,
            /* .use_hadamard  = */ false,
            /* .use_lowrank   = */ false,
            /* .use_q4k_block32_asym = */ false,
            /* .use_q4k_k256 = */ false,
            /* .use_q4k_hybrid = */ false,
            /* .npu_type_b_secondary = */ NPU_TYPE_INT4,
            /* .mm_type_secondary = */ RKNN_FLOAT16_MM_INT4_TO_FLOAT32,
            /* .use_q4k_staged_int8 = */ false,
            /* .use_q4k_ffn_int4_hybrid = */ true
        }
    };

    rk3576_config.use_custom_pattern = use_custom_pattern;
    rk3576_config.custom_hybrid_pattern = custom_pattern;

    rk3576_config.default_patterns[(int)GGML_TYPE_F16]  = {"W16A16_STANDARD"};
    rk3576_config.default_patterns[(int)GGML_TYPE_Q8_0] = {"W8A16_STANDARD"};
    rk3576_config.default_patterns[(int)GGML_TYPE_Q6_K] = {"W8A16_STANDARD"};
    rk3576_config.default_patterns[(int)GGML_TYPE_Q4_0] = {"W4A16_HADAMARD"};
    rk3576_config.default_patterns[(int)GGML_TYPE_Q4_K] = {"W4A16_Q4K_FFN_INT4_HYBRID"};

    device_configs["RK3576"] = rk3576_config;

    // --- Define RK3566 Configuration (Placeholder) ---
    // Rknpu2DeviceConfig rk3566_config;
    // ... fill config for RK3566 ...
    // device_configs["RK3566"] = rk3566_config;

    if (!select_device(target_device) && !device_configs.empty()) {
        select_device(device_configs.begin()->first);
    }

    if (trace_enabled() && current_config != nullptr) {
        fprintf(stderr,
                "RKNPU trace: selected device '%s' (cores=%zu, max_k=%d)\n",
                current_config->device_name.c_str(),
                current_config->active_cores.size(),
                current_config->max_k_limit);
    }
}

bool Rknpu2ConfigManager::select_device(const std::string& device_name) {
    auto it = device_configs.find(device_name);
    if (it != device_configs.end()) {
        current_config = &it->second;
        return true;
    }
    return false;
}

const Rknpu2DeviceConfig& Rknpu2ConfigManager::get_current_config() const {
    GGML_ASSERT(current_config != nullptr && "No device configuration selected or available.");
    return *current_config;
}

} // namespace rknpu2_configuration
