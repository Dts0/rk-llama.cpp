#include "ggml-rknpu2.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

#include "rknpu2-quantization.h"
#include "rknpu2-calibration.h"
#include "rknpu2-configuration.h"
#include "rknpu2-weight-cache.h"

#include <rknn_api.h>
#include <rknn_matmul_api.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#include <omp.h>

#include <cassert>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include <tuple>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <random>
#include <limits>
#include <sys/mman.h>
#include <sys/resource.h>
#include <atomic>
#include <type_traits>
#include <sstream>

#ifdef GGML_RKNPU2_DEBUG
#include <chrono>
#endif

#define UNUSED(x) (void)(x)

namespace {
bool env_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && strcmp(value, "0") != 0;
}

bool trace_enabled() {
    static const bool enabled = std::getenv("RKNPU_TRACE") != nullptr;
    return enabled;
}

bool execution_plan_cache_enabled() {
    static const bool enabled = env_flag_enabled("RKNPU_EXECUTION_PLAN_CACHE");
    return enabled;
}

std::once_flag trace_pipeline_once;
std::once_flag trace_run_once;
std::once_flag trace_exact_once;
std::once_flag trace_k256_once;
std::once_flag trace_staged_once;
std::once_flag trace_ffn_int4_once;
} // namespace

static float fp32_abs_max(const float * src, int n) {
    float amax = 0.0f;
#ifdef __ARM_NEON
    int i = 0;
    float32x4_t maxv = vdupq_n_f32(0.0f);
    for (; i + 15 < n; i += 16) {
        maxv = vmaxq_f32(maxv, vabsq_f32(vld1q_f32(src + i)));
        maxv = vmaxq_f32(maxv, vabsq_f32(vld1q_f32(src + i + 4)));
        maxv = vmaxq_f32(maxv, vabsq_f32(vld1q_f32(src + i + 8)));
        maxv = vmaxq_f32(maxv, vabsq_f32(vld1q_f32(src + i + 12)));
    }
    for (; i + 3 < n; i += 4) {
        maxv = vmaxq_f32(maxv, vabsq_f32(vld1q_f32(src + i)));
    }
    amax = vmaxvq_f32(maxv);
    for (; i < n; ++i) {
        amax = std::max(amax, std::abs(src[i]));
    }
#else
    for (int i = 0; i < n; ++i) {
        amax = std::max(amax, std::abs(src[i]));
    }
#endif
    return amax;
}

// --- IOMMU Domain Manager ---

// Helper function for parsing complex integer lists
static std::vector<int32_t> parse_domain_list(const std::string& str) {
    std::vector<int32_t> result;
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

struct IOMMUDomainManager {
    std::mutex mutex;

    // Max domain size for assigning
    const size_t max_domain_size = ((size_t) std::numeric_limits<int32_t>::max() - 65536);

    // Storage for domains and their sizes
    std::unordered_map<int32_t, size_t> domain_sizes;
    std::unordered_map<int32_t, rknn_matmul_ctx> allocator_contexts;

    // Allowed domain IDs defined by the user
    std::vector<int32_t> allowed_domains;

    IOMMUDomainManager() {
        // Read restricted domains from ENV variable
        const char* env_domains = std::getenv("RKNPU_DOMAINS");
        if (env_domains != nullptr) {
            allowed_domains = parse_domain_list(env_domains);

            if (!allowed_domains.empty()) {
                fprintf(stderr, "\n"
                    "RKNPU WARNING: Custom IOMMU domains detected via RKNPU_DOMAINS.\n"
                    "Due to Rockchip library limitations, concurrent execution of\n"
                    "multiple processes accessing the NPU simultaneously WILL LEAD\n"
                    "to a SYSTEM KERNEL PANIC and WILL FREEZE YOUR OPERATING SYSTEM.\n"
                    "Execute models SEQUENTIALLY if using multiple independent processes.\n");
            }
        }
    }

    // Function for assigning the domain for the tensor of given size
    int32_t assign_domain_memory(size_t size) {
        std::lock_guard<std::mutex> lock(mutex);

        // Allocate strictly within the allowed domains
        if (!allowed_domains.empty()) {
            for (int32_t d : allowed_domains) {
                if (domain_sizes[d] + size <= max_domain_size) {
                    domain_sizes[d] += size;
                    ensure_allocator_context(d);
                    return d;
                }
            }

            fprintf(stderr, "RKNPU ERROR: Out of memory in allowed IOMMU domains!\n");
            assert(false);
            return -1;
        // Allocate dynamically
        } else {
            for (int32_t i = 0; i <= 15; ++i) {
                if (domain_sizes[i] + size <= max_domain_size) {
                    domain_sizes[i] += size;
                    ensure_allocator_context(i);
                    return i;
                }
            }
            fprintf(stderr, "RKNPU ERROR: Out of memory in all IOMMU domains!\n");
            assert(false);
            return -1;
        }
    }

    // Function for releasing the given size of the domain memory
    void release_domain_memory(int32_t domain_id, size_t size) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = domain_sizes.find(domain_id);
        if (it != domain_sizes.end()) {
            if (it->second >= size) {
                it->second -= size;
            } else {
                it->second = 0;
            }
        }
    }

    // Function for getting a new dummy context in the required domain
    rknn_matmul_ctx get_allocator_context(int32_t domain_id) {
        std::lock_guard<std::mutex> lock(mutex);
        ensure_allocator_context(domain_id);
        return allocator_contexts[domain_id];
    }

private:
    // Function for ensuring a dummy context existence in the required domain
    void ensure_allocator_context(int32_t domain_id) {
        if (allocator_contexts.find(domain_id) == allocator_contexts.end()) {
            rknn_matmul_info info;
            memset(&info, 0, sizeof(info));
            info.M = 32; info.K = 32; info.N = 32;
            info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
            info.iommu_domain_id = domain_id;

            rknn_matmul_io_attr io_attr;
            rknn_matmul_ctx ctx = 0;
            rknn_matmul_create(&ctx, &info, &io_attr);
            allocator_contexts[domain_id] = ctx;
        }
    }
};
static IOMMUDomainManager g_domain_manager;

static const std::string & get_rknn_runtime_version(rknn_matmul_ctx ctx) {
    static std::once_flag once;
    static std::string result;
    std::call_once(once, [ctx]() {
        rknn_sdk_version version = {};
        if (rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(version)) == RKNN_SUCC) {
            result = std::string(version.api_version) + "|" + version.drv_version;
        } else {
            result = "unknown";
        }
    });
    return result;
}

// Macro for RKNN API calls
#define RKNN_CHECK(stmt, msg)                                           \
    do {                                                                \
        int ret = (stmt);                                               \
        if (ret < 0) {                                                  \
            fprintf(stderr,"RKNN error %d at %s:%d: %s\n", ret,         \
                __FILE__, __LINE__, msg);                               \
            assert(false);                                              \
        }                                                               \
    } while (0)

// --- Hashers ---

// Function for hash combinations
template <class T>
inline void hash_combine(std::size_t& seed, const T& v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Hasher for std::pair
struct PairHasher {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        std::size_t seed = 0;
        hash_combine(seed, p.first);
        hash_combine(seed, p.second);
        return seed;
    }
};

// Hasher for std::tuple
struct TupleHasher {
    template <typename... Ts>
    std::size_t operator()(const std::tuple<Ts...>& t) const {
        std::size_t seed = 0;
        std::apply([&](const auto&... args) {
            (hash_combine(seed, args), ...);
        }, t);
        return seed;
    }
};

// --- Segmenters ---

// Matrix segment information for N dimension
struct MatrixSegmentN {
    int offset_n;
    int size_n;
    int core_id;
};

// Matrix segment information for K dimension
struct MatrixSegmentK {
    int offset_k;
    int size_k;
};

// Split B-matrix into N-segments for cores
static std::vector<MatrixSegmentN> compute_n_segments(int N, const std::vector<int>& active_cores, int alignment) {
    std::vector<MatrixSegmentN> segments;
    int num_cores = active_cores.size();

    if (num_cores == 0) return segments;

    int base_segment_size = (N / num_cores / alignment) * alignment;
    int remaining = N - (base_segment_size * num_cores);

    int offset = 0;
    for (int i = 0; i < num_cores; i++) {
        MatrixSegmentN seg;
        seg.offset_n = offset;
        seg.size_n = base_segment_size;
        seg.core_id = active_cores[i];

        if (i < remaining / alignment) {
            seg.size_n += alignment;
        }

        offset += seg.size_n;
        segments.push_back(seg);
    }
    return segments;
}

// Split B-matrix into K-segments for hardware limit
static std::vector<MatrixSegmentK> compute_k_segments(int K_op, int k_limit, int alignment) {
    std::vector<MatrixSegmentK> segments;

    if (k_limit <= 0 || K_op <= k_limit) {
        segments.push_back({0, K_op});
        return segments;
    }

    int k_limit_aligned = (k_limit / alignment) * alignment;
    int offset = 0;
    while (offset < K_op) {
        int size = std::min(k_limit_aligned, K_op - offset);
        segments.push_back({offset, size});
        offset += size;
    }
    return segments;
}

// --- Structs ---

// low-Rank Approximation factors
struct LowRankSegmentFactors {
    std::vector<float> v; // K-dimension factors
    std::vector<float> u; // N-dimension factors
};

static std::atomic<uint64_t> buffer_instance_counter { 0 };
static std::atomic<uint64_t> native_allocation_counter { 0 };

// RKNN buffer context
struct ggml_backend_rknpu_buffer_context {
    uint64_t instance_id = ++buffer_instance_counter;
    std::unordered_map<const ggml_tensor *, uint64_t> tensor_versions;
    void* virtual_base;
    size_t total_size;
    std::string name;

    // RKNN buffers allocations for each tensor
    struct TensorAllocation {
        rknn_tensor_mem* mem = nullptr;
        size_t size = 0;
        int32_t iommu_domain_id = 0;
        uint64_t instance_id = 0;
    };
    std::unordered_map<size_t, TensorAllocation> tensor_allocs;

    // Per-channel scaling factors for each K segment of quantized weights
    std::unordered_map<const struct ggml_tensor *, std::vector<float>> quantized_tensor_scales;

    // Per-channel zero points for exact per-block asymmetric INT4 weights
    std::unordered_map<const struct ggml_tensor *, std::vector<float>> quantized_tensor_zeros;

    // Q4_K K256 decomposition metadata: d per 256-element superblock and z per 32-element group.
    std::unordered_map<const struct ggml_tensor *, std::vector<float>> q4k_k256_d;
    std::unordered_map<const struct ggml_tensor *, std::vector<float>> q4k_k256_z;

    // Per-tensor random sign vector for Hadamard Transform
    std::unordered_map<const struct ggml_tensor *, std::vector<float>> hadamard_s_vectors;

    // Per-tensor low-Rank Approximation factors
    std::unordered_map<const struct ggml_tensor *, std::vector<LowRankSegmentFactors>> lowrank_segment_factors;

    std::mutex mutex;

    // Function for the allocation of a RKNN buffer for the individual tensor
    TensorAllocation get_tensor_allocation(size_t tensor_offset, size_t size) {
        std::lock_guard<std::mutex> lock(mutex);

        // Trying to find an existing buffer
        auto it = tensor_allocs.find(tensor_offset);
        if (it != tensor_allocs.end()) {
            if (it->second.size < size) {
                rknn_matmul_ctx old_ctx = g_domain_manager.get_allocator_context(it->second.iommu_domain_id);
                rknn_destroy_mem(old_ctx, it->second.mem);
                g_domain_manager.release_domain_memory(it->second.iommu_domain_id, it->second.size);

                it->second.iommu_domain_id = g_domain_manager.assign_domain_memory(size);
                rknn_matmul_ctx new_ctx = g_domain_manager.get_allocator_context(it->second.iommu_domain_id);
                it->second.mem = rknn_create_mem(new_ctx, size);
                GGML_ASSERT(it->second.mem != nullptr && "Failed to resize tensor memory via RKNN API");
                it->second.size = size;
                it->second.instance_id = ++native_allocation_counter;
            }
            return it->second;
        }

        // Acquiring a domain for allocation
        int32_t domain_id = g_domain_manager.assign_domain_memory(size);
        rknn_matmul_ctx alloc_ctx = g_domain_manager.get_allocator_context(domain_id);

        // Allocating a new buffer for the tensor
        TensorAllocation alloc;
        alloc.mem = rknn_create_mem(alloc_ctx, size);
        alloc.size = size;
        alloc.iommu_domain_id = domain_id;
        alloc.instance_id = ++native_allocation_counter;

        GGML_ASSERT(alloc.mem != nullptr && "Failed to allocate tensor memory via RKNN API");
        tensor_allocs[tensor_offset] = alloc;

        return alloc;
    }
};


// RKNN matmul operation context
struct rknpu_matmul_context {
    rknn_matmul_info info;
    rknn_matmul_io_attr io_attr;
    rknn_matmul_ctx ctx = 0;

    bool b_bound = false;
    std::shared_ptr<rknn_tensor_mem> mem_B;
    rknn_tensor_mem * mem_A_bound = nullptr;
    rknn_tensor_mem * mem_C_bound = nullptr;

    rknpu_matmul_context(int M, int K, int N, rknn_matmul_type type, int32_t domain_id) {
        memset(&info, 0, sizeof(info));
        info.M = M;
        info.K = K;
        info.N = N;
        info.type = type;
        info.B_layout = RKNN_MM_LAYOUT_NATIVE;
        info.AC_layout = RKNN_MM_LAYOUT_NORM;
        info.iommu_domain_id = domain_id;

        int ret = rknn_matmul_create(&ctx, &info, &io_attr);
        if (ret < 0) ctx = 0;
    }

    ~rknpu_matmul_context() {
        mem_B.reset();

        if (ctx != 0) {
            rknn_matmul_destroy(ctx);
        }
    }
};

struct rknpu_k_execution_plan {
    MatrixSegmentK k_segment;
    std::vector<std::shared_ptr<rknpu_matmul_context>> matmul_ctxs;
    std::vector<size_t> b_offsets;
    std::vector<std::shared_ptr<rknn_tensor_mem>> mem_B_segments;
    std::vector<std::shared_ptr<rknn_tensor_mem>> mem_A_segments;
    std::vector<std::shared_ptr<rknn_tensor_mem>> mem_C_segments;

    // Q4_K K256 secondary (INT4 high-part) plane.
    std::vector<std::shared_ptr<rknpu_matmul_context>> matmul_ctxs_hi;
    std::vector<size_t> b_offsets_lo;
    std::vector<size_t> b_offsets_hi;
    std::vector<std::shared_ptr<rknn_tensor_mem>> mem_B_segments_hi;
    std::vector<std::shared_ptr<rknn_tensor_mem>> mem_A_segments_hi;
    std::vector<std::shared_ptr<rknn_tensor_mem>> mem_C_segments_hi;
};

struct rknpu_execution_plan {
    const rknpu2_configuration::Rknpu2HardwarePipeline * pipeline = nullptr;
    uint64_t buffer_instance_id = 0;
    uint64_t tensor_version = 0;
    int M_op = 0;
    int K = 0;
    int N = 0;
    int K_op = 0;
    std::vector<MatrixSegmentN> active_n_segments;
    std::vector<float> scales_B_grid;
    std::vector<float> zeros_B_grid;
    std::vector<float> k256_d_grid;
    std::vector<float> k256_z_grid;
    std::vector<float> hadamard_s_vector;
    std::vector<LowRankSegmentFactors> lowrank_factors;
    std::vector<rknpu_k_execution_plan> k_plans;
};

// Backend main context
struct ggml_backend_rknpu_context {
    std::string name;
    std::mutex mutex;

    // RKNN matmul contexts cache (tensor_id, offset, M, K, N, core_id, type, domain_id, allocation_id)
    std::unordered_map<std::tuple<uintptr_t, size_t, int, int, int, int, int, int, uint64_t>, std::shared_ptr<rknpu_matmul_context>, TupleHasher> matmul_ctx_cache;

    // Shared matmul contexts for exact per-block INT4 (M, K, N, core_id, type, domain_id)
    std::unordered_map<std::tuple<int, int, int, int, int, int>, std::shared_ptr<rknpu_matmul_context>, TupleHasher> shared_matmul_ctx_cache;

    // A-matrices cache (M, K, n_offset, npu_type_a, domain_id)
    std::unordered_map<std::tuple<int, int, int, int, int>, std::shared_ptr<rknn_tensor_mem>, TupleHasher> a_buffer_cache;

    // C-matrices cache (M, N, core_id, npu_type_c, domain_id)
    std::unordered_map<std::tuple<int, int, int, int, int>, std::shared_ptr<rknn_tensor_mem>, TupleHasher> c_buffer_cache;

    // K256 A/C/B buffer caches. Keys include matmul type so LO/HI never alias.
    std::unordered_map<std::tuple<int, int, int, int, int>, std::shared_ptr<rknn_tensor_mem>, TupleHasher> k256_a_buffer_cache;
    std::unordered_map<std::tuple<int, int, int, int, int>, std::shared_ptr<rknn_tensor_mem>, TupleHasher> k256_c_buffer_cache;
    std::unordered_map<std::tuple<uintptr_t, int, int, int, int>, std::shared_ptr<rknn_tensor_mem>, TupleHasher> k256_b_buffer_cache;

    // Shared stage-INT8 B scratch buffers (K_seg, N_seg, core, matmul_type, domain).
    std::unordered_map<std::tuple<int, int, int, int, int>, std::shared_ptr<rknn_tensor_mem>, TupleHasher> staged_b_buffer_cache;

    // Prepared operation state (weight tensor, M)
    std::unordered_map<std::pair<uintptr_t, int>, std::shared_ptr<rknpu_execution_plan>, PairHasher> execution_plan_cache;

    static rknn_core_mask core_mask_for(int core_id) {
        switch (core_id) {
            case 0: return RKNN_NPU_CORE_0;
            case 1: return RKNN_NPU_CORE_1;
            case 2: return RKNN_NPU_CORE_2;
            default: return RKNN_NPU_CORE_AUTO;
        }
    }

    std::shared_ptr<rknpu_matmul_context> get_matmul_ctx(uintptr_t tensor_id, size_t offset, int M, int K, int N, int core_id, rknn_matmul_type type, int32_t domain_id, uint64_t allocation_id) {
        std::lock_guard<std::mutex> lock(mutex);

        auto key = std::make_tuple(tensor_id, offset, M, K, N, core_id, (int)type, (int)domain_id, allocation_id);
        auto it = matmul_ctx_cache.find(key);
        if (it != matmul_ctx_cache.end()) {
            return it->second;
        }

        auto ctx = std::make_shared<rknpu_matmul_context>(M, K, N, type, domain_id);
        if (ctx->ctx == 0) {
            return nullptr;
        }

        const rknn_core_mask core_mask = core_mask_for(core_id);
        const int ret = rknn_matmul_set_core_mask(ctx->ctx, core_mask);
        if (ret != RKNN_SUCC) {
            // Handle error
        }

        matmul_ctx_cache[key] = ctx;
        return ctx;
    }

    std::shared_ptr<rknpu_matmul_context> get_shared_matmul_ctx(int M, int K, int N, int core_id, rknn_matmul_type type, int32_t domain_id) {
        std::lock_guard<std::mutex> lock(mutex);

        auto key = std::make_tuple(M, K, N, core_id, (int) type, (int) domain_id);
        auto it = shared_matmul_ctx_cache.find(key);
        if (it != shared_matmul_ctx_cache.end()) {
            return it->second;
        }

        auto ctx = std::make_shared<rknpu_matmul_context>(M, K, N, type, domain_id);
        if (ctx->ctx == 0) {
            return nullptr;
        }

        const rknn_core_mask core_mask = core_mask_for(core_id);
        const int ret = rknn_matmul_set_core_mask(ctx->ctx, core_mask);
        if (ret != RKNN_SUCC) {
            // Handle error
        }

        shared_matmul_ctx_cache[key] = ctx;
        return ctx;
    }
};


//
// Backend
//

static const char * ggml_backend_rknpu_name(ggml_backend_t backend) {
    UNUSED(backend);
    return "RKNPU";
}

static void ggml_backend_rknpu_free(ggml_backend_t backend) {
    ggml_backend_rknpu_context * ctx = (ggml_backend_rknpu_context *)backend->context;
    delete ctx;
    delete backend;
}

// Function for acquiring a pointer for tensor data
static void * get_tensor_real_ptr(const struct ggml_tensor * tensor) {
    return tensor ? tensor->data : nullptr;
}

// Function for getting buffer from cache or creating new one
template <typename CacheKeyType>
static std::shared_ptr<rknn_tensor_mem> get_tensor_buffer(
    ggml_backend_rknpu_context* backend_ctx,
    rknn_matmul_ctx matmul_ctx,
    size_t size,
    const CacheKeyType& key,
    std::unordered_map<CacheKeyType, std::shared_ptr<rknn_tensor_mem>, TupleHasher>& cache
) {
    std::lock_guard<std::mutex> lock(backend_ctx->mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
        if (it->second->size >= size) {
            return it->second;
        }
    }

    rknn_tensor_mem* mem = rknn_create_mem(matmul_ctx, size);
    if (!mem) { return nullptr; }

    auto deleter = [matmul_ctx](rknn_tensor_mem* m) {
        if (m != 0) {
            rknn_destroy_mem(matmul_ctx, m);
        }
    };

    std::shared_ptr<rknn_tensor_mem> mem_shared(mem, deleter);
    cache[key] = mem_shared;
    return mem_shared;
}

template <typename CacheKeyType>
static std::shared_ptr<rknn_tensor_mem> get_fd_tensor_buffer(
    ggml_backend_rknpu_context* backend_ctx,
    rknn_matmul_ctx matmul_ctx,
    int tensor_fd,
    void * tensor_address,
    size_t tensor_size,
    size_t offset,
    const CacheKeyType& key,
    std::unordered_map<CacheKeyType, std::shared_ptr<rknn_tensor_mem>, TupleHasher>& cache
) {
    std::lock_guard<std::mutex> lock(backend_ctx->mutex);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    rknn_tensor_mem * mem = rknn_create_mem_from_fd(matmul_ctx, tensor_fd, tensor_address, tensor_size, offset);
    if (mem == nullptr) {
        return nullptr;
    }
    auto deleter = [matmul_ctx](rknn_tensor_mem * value) {
        if (value != nullptr) {
            rknn_destroy_mem(matmul_ctx, value);
        }
    };
    std::shared_ptr<rknn_tensor_mem> mem_shared(mem, deleter);
    cache[key] = mem_shared;
    return mem_shared;
}

static size_t get_tensor_packed_size(const struct ggml_tensor * tensor);
static void ggml_backend_rknpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                                const void * data, size_t offset, size_t size);
static void stage_q4k_int8_segment(
    const struct ggml_tensor * tensor,
    const void * raw_data,
    int K, int N, int K_op,
    const MatrixSegmentK & k_seg,
    const MatrixSegmentN & n_seg,
    const std::vector<float> & scales_grid,
    size_t k_idx,
    rknn_tensor_mem * mem_B,
    const rknpu2_configuration::Rknpu2HardwarePipeline * pipeline);
static void stage_q4k_int8_segment_fused(
    const struct ggml_tensor * tensor,
    const void * raw_data,
    int K, int N,
    const MatrixSegmentK & k_seg,
    const MatrixSegmentN & n_seg,
    const std::vector<float> & scales_grid,
    size_t k_idx,
    rknn_tensor_mem * mem_B);

static std::shared_ptr<rknpu_execution_plan> get_execution_plan(
    ggml_backend_rknpu_context * backend_ctx,
    const struct ggml_tensor * src0,
    int M_op)
{
    auto * source_buffer_ctx = (ggml_backend_rknpu_buffer_context *) src0->buffer->context;
    uint64_t tensor_version = 0;
    {
        std::lock_guard<std::mutex> lock(source_buffer_ctx->mutex);
        auto version = source_buffer_ctx->tensor_versions.find(src0);
        if (version != source_buffer_ctx->tensor_versions.end()) {
            tensor_version = version->second;
        }
    }
    if (tensor_version == 0) {
        // The no-mmap loader writes host buffers directly, bypassing set_tensor.
        ggml_backend_rknpu_buffer_set_tensor(src0->buffer, const_cast<ggml_tensor *>(src0), src0->data, 0, ggml_nbytes(src0));
        std::lock_guard<std::mutex> lock(source_buffer_ctx->mutex);
        auto version = source_buffer_ctx->tensor_versions.find(src0);
        if (version == source_buffer_ctx->tensor_versions.end()) {
            return nullptr;
        }
        tensor_version = version->second;
    }
    const auto cache_key = std::make_pair((uintptr_t) src0, M_op);
    if (execution_plan_cache_enabled()) {
        std::lock_guard<std::mutex> lock(backend_ctx->mutex);
        auto it = backend_ctx->execution_plan_cache.find(cache_key);
        if (it != backend_ctx->execution_plan_cache.end()) {
            if (it->second->buffer_instance_id == source_buffer_ctx->instance_id && it->second->tensor_version == tensor_version) {
                return it->second;
            }
            backend_ctx->execution_plan_cache.erase(it);
        }
    }

    const auto & config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    const auto * pipeline = config.resolve_op_support(src0);
    if (pipeline == nullptr) {
        return nullptr;
    }

    auto plan = std::make_shared<rknpu_execution_plan>();
    plan->pipeline = pipeline;
    plan->buffer_instance_id = source_buffer_ctx->instance_id;
    plan->tensor_version = tensor_version;
    plan->M_op = M_op;
    plan->K = (int) src0->ne[0];
    plan->N = (int) src0->ne[1];
    plan->K_op = pipeline->use_hadamard ? rknpu2_calibration::next_power_of_two(plan->K) : plan->K;

    int k_limit = config.max_k_limit;
    if (pipeline->effective_k > 0) {
        k_limit = k_limit > 0 ? std::min(k_limit, pipeline->effective_k) : pipeline->effective_k;
    }

    const auto k_segments = compute_k_segments(plan->K_op, k_limit, pipeline->k_align);
    const auto n_segments = compute_n_segments(plan->N, config.active_cores, pipeline->n_align);
    for (const auto & segment : n_segments) {
        if (segment.size_n > 0) {
            plan->active_n_segments.push_back(segment);
        }
    }
    if (plan->active_n_segments.empty()) {
        return nullptr;
    }

    auto * buffer_ctx = source_buffer_ctx;
    const bool staged_q4k = pipeline->use_q4k_staged_int8 && src0->type == GGML_TYPE_Q4_K;
    int tensor_fd = -1;
    int32_t domain_id = 0;
    void * tensor_address = nullptr;
    size_t tensor_size = 0;
    uint64_t allocation_id = 0;
    if (staged_q4k) {
        tensor_address = src0->data;
        tensor_size = ggml_nbytes(src0);
        std::lock_guard<std::mutex> lock(buffer_ctx->mutex);
        auto scales = buffer_ctx->quantized_tensor_scales.find(src0);
        GGML_ASSERT(scales != buffer_ctx->quantized_tensor_scales.end() && "Staged INT8 scales not found");
        plan->scales_B_grid = scales->second;
    } else {
        const size_t tensor_offset = (uintptr_t) src0->data - (uintptr_t) buffer_ctx->virtual_base;
        std::lock_guard<std::mutex> lock(buffer_ctx->mutex);
        auto allocation = buffer_ctx->tensor_allocs.find(tensor_offset);
        GGML_ASSERT(allocation != buffer_ctx->tensor_allocs.end() && "B-matrix RKNN buffer not found");
        tensor_fd = allocation->second.mem->fd;
        tensor_address = allocation->second.mem->virt_addr;
        tensor_size = allocation->second.size;
        allocation_id = allocation->second.instance_id;
        domain_id = allocation->second.iommu_domain_id;

        if (pipeline->use_hadamard) {
            auto signs = buffer_ctx->hadamard_s_vectors.find(src0);
            GGML_ASSERT(signs != buffer_ctx->hadamard_s_vectors.end() && "Hadamard 's' vector not found");
            plan->hadamard_s_vector = signs->second;
        }
        if (!pipeline->use_q4k_k256 &&
            (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8 || pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4)) {
            auto scales = buffer_ctx->quantized_tensor_scales.find(src0);
            GGML_ASSERT(scales != buffer_ctx->quantized_tensor_scales.end() && "Quantized scales grid not found");
            plan->scales_B_grid = scales->second;
        }
        if (pipeline->use_q4k_block32_asym) {
            auto zeros = buffer_ctx->quantized_tensor_zeros.find(src0);
            GGML_ASSERT(zeros != buffer_ctx->quantized_tensor_zeros.end() && "Asymmetric zero grid not found");
            plan->zeros_B_grid = zeros->second;
        }
        if (pipeline->use_q4k_k256) {
            auto d_grid = buffer_ctx->q4k_k256_d.find(src0);
            auto z_grid = buffer_ctx->q4k_k256_z.find(src0);
            GGML_ASSERT(d_grid != buffer_ctx->q4k_k256_d.end() && "K256 d grid not found");
            GGML_ASSERT(z_grid != buffer_ctx->q4k_k256_z.end() && "K256 z grid not found");
            plan->k256_d_grid = d_grid->second;
            plan->k256_z_grid = z_grid->second;
        }
        if (pipeline->use_lowrank) {
            auto factors = buffer_ctx->lowrank_segment_factors.find(src0);
            GGML_ASSERT(factors != buffer_ctx->lowrank_segment_factors.end() && "Low-Rank factors not found");
            plan->lowrank_factors = factors->second;
        }
    }

    if (pipeline->use_q4k_k256) {
        plan->k_plans.reserve(k_segments.size());
        for (const auto & k_segment : k_segments) {
            GGML_ASSERT(k_segment.size_k % 256 == 0);
            rknpu_k_execution_plan k_plan;
            k_plan.k_segment = k_segment;
            const int sb_count = k_segment.size_k / 256;
            k_plan.matmul_ctxs.reserve(plan->active_n_segments.size());
            k_plan.matmul_ctxs_hi.reserve(plan->active_n_segments.size());
            k_plan.mem_B_segments.reserve(plan->active_n_segments.size());
            k_plan.mem_B_segments_hi.reserve(plan->active_n_segments.size());
            k_plan.mem_A_segments.reserve(plan->active_n_segments.size());
            k_plan.mem_A_segments_hi.reserve(plan->active_n_segments.size());
            k_plan.mem_C_segments.reserve(plan->active_n_segments.size());
            k_plan.mem_C_segments_hi.reserve(plan->active_n_segments.size());
            k_plan.b_offsets_lo.reserve(plan->active_n_segments.size() * sb_count);
            k_plan.b_offsets_hi.reserve(plan->active_n_segments.size() * sb_count);

            for (const auto & n_segment : plan->active_n_segments) {
                auto lo_ctx = backend_ctx->get_shared_matmul_ctx(
                    M_op, 256, n_segment.size_n, n_segment.core_id,
                    pipeline->mm_type, domain_id);
                auto hi_ctx = backend_ctx->get_shared_matmul_ctx(
                    M_op, 256, n_segment.size_n, n_segment.core_id,
                    pipeline->mm_type_secondary, domain_id);
                if (!lo_ctx || !hi_ctx || lo_ctx->ctx == 0 || hi_ctx->ctx == 0) {
                    return nullptr;
                }

                auto b_key_lo = std::make_tuple((uintptr_t) tensor_address, n_segment.size_n,
                                                 n_segment.core_id, (int) pipeline->mm_type, (int) domain_id);
                auto b_key_hi = std::make_tuple((uintptr_t) tensor_address, n_segment.size_n,
                                                 n_segment.core_id, (int) pipeline->mm_type_secondary, (int) domain_id);
                auto lo_b = get_fd_tensor_buffer(backend_ctx, lo_ctx->ctx, tensor_fd, tensor_address,
                                                 tensor_size, 0, b_key_lo, backend_ctx->k256_b_buffer_cache);
                auto hi_b = get_fd_tensor_buffer(backend_ctx, hi_ctx->ctx, tensor_fd, tensor_address,
                                                 tensor_size, 0, b_key_hi, backend_ctx->k256_b_buffer_cache);
                if (!lo_b || !hi_b) {
                    return nullptr;
                }
                lo_b->size = (uint32_t) n_segment.size_n * 256;
                hi_b->size = (uint32_t) n_segment.size_n * 128;
                RKNN_CHECK(rknn_matmul_set_io_mem(lo_ctx->ctx, lo_b.get(), &lo_ctx->io_attr.B), "set K256 LO B segment");
                RKNN_CHECK(rknn_matmul_set_io_mem(hi_ctx->ctx, hi_b.get(), &hi_ctx->io_attr.B), "set K256 HI B segment");
                k_plan.mem_B_segments.push_back(lo_b);
                k_plan.mem_B_segments_hi.push_back(hi_b);

                auto a_key_lo = std::make_tuple(M_op, n_segment.size_n, n_segment.core_id,
                                                (int) pipeline->mm_type, (int) domain_id);
                auto a_key_hi = std::make_tuple(M_op, n_segment.size_n, n_segment.core_id,
                                                (int) pipeline->mm_type_secondary, (int) domain_id);
                auto c_key_lo = a_key_lo;
                auto c_key_hi = a_key_hi;
                auto lo_a = get_tensor_buffer(backend_ctx, lo_ctx->ctx, lo_ctx->io_attr.A.size,
                                              a_key_lo, backend_ctx->k256_a_buffer_cache);
                auto hi_a = get_tensor_buffer(backend_ctx, hi_ctx->ctx, hi_ctx->io_attr.A.size,
                                              a_key_hi, backend_ctx->k256_a_buffer_cache);
                auto lo_c = get_tensor_buffer(backend_ctx, lo_ctx->ctx, lo_ctx->io_attr.C.size,
                                              c_key_lo, backend_ctx->k256_c_buffer_cache);
                auto hi_c = get_tensor_buffer(backend_ctx, hi_ctx->ctx, hi_ctx->io_attr.C.size,
                                              c_key_hi, backend_ctx->k256_c_buffer_cache);
                if (!lo_a || !hi_a || !lo_c || !hi_c) {
                    return nullptr;
                }
                RKNN_CHECK(rknn_matmul_set_io_mem(lo_ctx->ctx, lo_a.get(), &lo_ctx->io_attr.A), "set K256 LO A");
                RKNN_CHECK(rknn_matmul_set_io_mem(hi_ctx->ctx, hi_a.get(), &hi_ctx->io_attr.A), "set K256 HI A");
                RKNN_CHECK(rknn_matmul_set_io_mem(lo_ctx->ctx, lo_c.get(), &lo_ctx->io_attr.C), "set K256 LO C");
                RKNN_CHECK(rknn_matmul_set_io_mem(hi_ctx->ctx, hi_c.get(), &hi_ctx->io_attr.C), "set K256 HI C");
                k_plan.matmul_ctxs.push_back(lo_ctx);
                k_plan.matmul_ctxs_hi.push_back(hi_ctx);
                k_plan.mem_A_segments.push_back(lo_a);
                k_plan.mem_A_segments_hi.push_back(hi_a);
                k_plan.mem_C_segments.push_back(lo_c);
                k_plan.mem_C_segments_hi.push_back(hi_c);

                for (int sb = 0; sb < sb_count; ++sb) {
                    const size_t lo_off = (size_t) n_segment.offset_n * plan->K
                        + (size_t) (k_segment.offset_k + sb * 256) * n_segment.size_n;
                    const size_t hi_off = (size_t) plan->N * plan->K
                        + (size_t) n_segment.offset_n * (plan->K / 2)
                        + (size_t) (k_segment.offset_k + sb * 256) * (n_segment.size_n / 2);
                    k_plan.b_offsets_lo.push_back(lo_off);
                    k_plan.b_offsets_hi.push_back(hi_off);
                }
            }
            plan->k_plans.push_back(std::move(k_plan));
        }
        return plan;
    }

    size_t current_offset = 0;
    const bool exact_q4k = pipeline->use_q4k_block32_asym;
    std::vector<std::shared_ptr<rknn_tensor_mem>> shared_b_mems;
    if (exact_q4k) {
        shared_b_mems.reserve(plan->active_n_segments.size());
        const int K_seg = k_segments.empty() ? 0 : k_segments[0].size_k;
        for (const auto & n_segment : plan->active_n_segments) {
            auto matmul_ctx = backend_ctx->get_shared_matmul_ctx(
                M_op, K_seg, n_segment.size_n, n_segment.core_id, pipeline->mm_type, domain_id);
            if (matmul_ctx == nullptr || matmul_ctx->ctx == 0) {
                return nullptr;
            }
            rknn_tensor_mem * mem = rknn_create_mem_from_fd(
                matmul_ctx->ctx, tensor_fd, tensor_address, tensor_size, 0);
            if (mem == nullptr) {
                return nullptr;
            }
            auto deleter = [ctx = matmul_ctx->ctx](rknn_tensor_mem * value) {
                if (value != nullptr) {
                    rknn_destroy_mem(ctx, value);
                }
            };
            shared_b_mems.emplace_back(mem, deleter);
        }
    }

    plan->k_plans.reserve(k_segments.size());
    for (const auto & k_segment : k_segments) {
        rknpu_k_execution_plan k_plan;
        k_plan.k_segment = k_segment;
        k_plan.matmul_ctxs.reserve(plan->active_n_segments.size());
        if (exact_q4k) {
            k_plan.b_offsets.reserve(plan->active_n_segments.size());
            k_plan.mem_B_segments = shared_b_mems;
        } else if (staged_q4k) {
            k_plan.mem_B_segments.reserve(plan->active_n_segments.size());
        }

        for (const auto & n_segment : plan->active_n_segments) {
            const size_t offset = current_offset;
            std::shared_ptr<rknpu_matmul_context> matmul_ctx;
            if (exact_q4k) {
                matmul_ctx = backend_ctx->get_shared_matmul_ctx(
                    M_op, k_segment.size_k, n_segment.size_n,
                    n_segment.core_id, pipeline->mm_type, domain_id);
            } else if (staged_q4k) {
                matmul_ctx = backend_ctx->get_shared_matmul_ctx(
                    M_op, k_segment.size_k, n_segment.size_n,
                    n_segment.core_id, pipeline->mm_type, domain_id);
            } else {
                matmul_ctx = backend_ctx->get_matmul_ctx(
                    (uintptr_t) tensor_address, offset, M_op, k_segment.size_k, n_segment.size_n,
                    n_segment.core_id, pipeline->mm_type, domain_id, allocation_id);
            }
            if (matmul_ctx == nullptr || matmul_ctx->ctx == 0) {
                return nullptr;
            }

            const size_t elements = (size_t) k_segment.size_k * n_segment.size_n;
            const size_t packed_bytes = pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_FP16 ? elements * 2 :
                                        pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8 ? elements : elements / 2;
            GGML_ASSERT(matmul_ctx->io_attr.B.size == packed_bytes);
            if (!staged_q4k) {
                GGML_ASSERT(offset <= tensor_size && packed_bytes <= tensor_size - offset);
            }

            if (exact_q4k) {
                k_plan.b_offsets.push_back(offset);
                const size_t mem_index = k_plan.b_offsets.size() - 1;
                GGML_ASSERT(mem_index < k_plan.mem_B_segments.size());
                k_plan.mem_B_segments[mem_index]->size = matmul_ctx->io_attr.B.size;
            } else if (staged_q4k) {
                auto b_key = std::make_tuple(k_segment.size_k, n_segment.size_n, n_segment.core_id,
                                             (int) pipeline->mm_type, domain_id);
                auto mem_B = get_tensor_buffer(backend_ctx, matmul_ctx->ctx, packed_bytes, b_key,
                                               backend_ctx->staged_b_buffer_cache);
                if (mem_B == nullptr) {
                    return nullptr;
                }
                RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctx->ctx, mem_B.get(), &matmul_ctx->io_attr.B), "set staged B segment");
                k_plan.mem_B_segments.push_back(mem_B);
            } else if (!matmul_ctx->b_bound) {
                rknn_tensor_mem * mem = rknn_create_mem_from_fd(
                    matmul_ctx->ctx, tensor_fd, tensor_address, matmul_ctx->io_attr.B.size, offset);
                if (mem == nullptr) {
                    return nullptr;
                }
                auto deleter = [ctx = matmul_ctx->ctx](rknn_tensor_mem * value) {
                    if (value != nullptr) {
                        rknn_destroy_mem(ctx, value);
                    }
                };
                matmul_ctx->mem_B = std::shared_ptr<rknn_tensor_mem>(mem, deleter);
                RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctx->ctx, matmul_ctx->mem_B.get(), &matmul_ctx->io_attr.B), "set_io_mem B segment");
                matmul_ctx->b_bound = true;
            }

            if (!staged_q4k) {
                current_offset += matmul_ctx->io_attr.B.size;
            }
            k_plan.matmul_ctxs.push_back(std::move(matmul_ctx));
        }

        const size_t num_a_matrices = pipeline->use_lowrank ? plan->active_n_segments.size() : 1;
        k_plan.mem_A_segments.resize(num_a_matrices);
        for (size_t i = 0; i < num_a_matrices; ++i) {
            auto & matmul_ctx = k_plan.matmul_ctxs[i];
            auto a_key = std::make_tuple(M_op, k_segment.size_k, pipeline->use_lowrank ? plan->active_n_segments[i].offset_n : 0,
                                        (int) pipeline->npu_type_a, domain_id);
            k_plan.mem_A_segments[i] = get_tensor_buffer(backend_ctx, matmul_ctx->ctx, matmul_ctx->io_attr.A.size,
                                                        a_key, backend_ctx->a_buffer_cache);
            if (k_plan.mem_A_segments[i] == nullptr) {
                return nullptr;
            }
        }

        k_plan.mem_C_segments.resize(plan->active_n_segments.size());
        for (size_t i = 0; i < plan->active_n_segments.size(); ++i) {
            auto & matmul_ctx = k_plan.matmul_ctxs[i];
            auto & mem_A = k_plan.mem_A_segments[pipeline->use_lowrank ? i : 0];
            if (matmul_ctx->mem_A_bound != mem_A.get()) {
                RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctx->ctx, mem_A.get(), &matmul_ctx->io_attr.A), "set_io_mem A for core");
                matmul_ctx->mem_A_bound = mem_A.get();
            }

            const auto & n_segment = plan->active_n_segments[i];
            auto c_key = std::make_tuple(M_op, n_segment.size_n, n_segment.core_id, (int) pipeline->npu_type_c, domain_id);
            k_plan.mem_C_segments[i] = get_tensor_buffer(backend_ctx, matmul_ctx->ctx, matmul_ctx->io_attr.C.size,
                                                         c_key, backend_ctx->c_buffer_cache);
            if (k_plan.mem_C_segments[i] == nullptr) {
                return nullptr;
            }
            if (matmul_ctx->mem_C_bound != k_plan.mem_C_segments[i].get()) {
                RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctx->ctx, k_plan.mem_C_segments[i].get(), &matmul_ctx->io_attr.C), "set_io_mem C");
                matmul_ctx->mem_C_bound = k_plan.mem_C_segments[i].get();
            }
        }

        plan->k_plans.push_back(std::move(k_plan));
    }

    GGML_ASSERT(current_offset == get_tensor_packed_size(src0) && "Packed B-matrix size mismatch");
    if (execution_plan_cache_enabled()) {
        std::lock_guard<std::mutex> lock(backend_ctx->mutex);
        auto result = backend_ctx->execution_plan_cache.emplace(cache_key, plan);
        result.first->second = plan;
        return plan;
    }
    return plan;
}

template <typename T>
static void collect_rknpu_row(const T * src, float * dst, int count, const float * weight_scales,
                              const float * weight_zeros, float activation_scale, float zero_sum, bool accumulate) {
    int n = 0;
#ifdef __ARM_NEON
    const float32x4_t activation_vec = vdupq_n_f32(activation_scale);
    for (; n + 3 < count; n += 4) {
        float32x4_t values;
        if constexpr (std::is_same_v<T, float>) {
            values = vld1q_f32(src + n);
        } else if constexpr (std::is_same_v<T, int32_t>) {
            values = vcvtq_f32_s32(vld1q_s32(src + n));
        } else {
            values = vcvtq_f32_s32(vmovl_s16(vld1_s16(src + n)));
        }
        const float32x4_t scales = weight_scales ? vmulq_f32(vld1q_f32(weight_scales + n), activation_vec) : activation_vec;
        float32x4_t result = vmulq_f32(values, scales);
        if (weight_zeros) {
            result = vmlaq_n_f32(result, vld1q_f32(weight_zeros + n), zero_sum);
        }
        vst1q_f32(dst + n, accumulate ? vaddq_f32(vld1q_f32(dst + n), result) : result);
    }
#endif
    for (; n < count; ++n) {
        float result = (float) src[n] * (activation_scale * (weight_scales ? weight_scales[n] : 1.0f));
        if (weight_zeros) {
            result += weight_zeros[n] * zero_sum;
        }
        if (accumulate) {
            dst[n] += result;
        } else {
            dst[n] = result;
        }
    }
}

static enum ggml_status ggml_backend_rknpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph* cgraph) {
    auto* backend_ctx = (ggml_backend_rknpu_context*)backend->context;

#ifdef GGML_RKNPU2_DEBUG
    static int64_t t_setup = 0, t_prepA = 0, t_prepC = 0, t_npu = 0, t_collect = 0;
    static int call_count = 0;
    int64_t node_t_setup = 0, node_t_prepA = 0, node_t_prepC = 0, node_t_npu = 0, node_t_collect = 0;
#endif

    for (int node_i = 0; node_i < cgraph->n_nodes; node_i++) {
        struct ggml_tensor* node = cgraph->nodes[node_i];
        if (node->op != GGML_OP_MUL_MAT) continue;

        const struct ggml_tensor* src0 = node->src[0]; // Weights      :  (K x N)
        const struct ggml_tensor* src1 = node->src[1]; // Activations  :  (M x K)
        struct ggml_tensor* dst = node;

        const int M = (int)ggml_nrows(src1);
        if (M == 0 || src0->ne[0] == 0 || src0->ne[1] == 0) {
            continue;
        }

        const int M_op = M > 1 ? rknpu2_calibration::next_power_of_two(M) : M;
#ifdef GGML_RKNPU2_DEBUG
        auto t1_start = std::chrono::high_resolution_clock::now();
#endif
        auto plan = get_execution_plan(backend_ctx, src0, M_op);
        if (plan == nullptr) {
            return GGML_STATUS_FAILED;
        }
#ifdef GGML_RKNPU2_DEBUG
        auto plan_ready = std::chrono::high_resolution_clock::now();
        node_t_setup += std::chrono::duration_cast<std::chrono::microseconds>(plan_ready - t1_start).count();
#endif

        const auto * pipeline = plan->pipeline;
        const bool is_hadamard = pipeline->use_hadamard;
        const bool is_lowrank = pipeline->use_lowrank;
        const auto & lowrank_factors = plan->lowrank_factors;
        const int K = plan->K;
        const int N = plan->N;
        const int K_op = plan->K_op;
        const auto & active_n_segments = plan->active_n_segments;
        const auto & scales_B_grid = plan->scales_B_grid;
        const auto & zeros_B_grid = plan->zeros_B_grid;
        const auto & s_vec = plan->hadamard_s_vector;
        const size_t num_active_segments = active_n_segments.size();
        float * dst_data = (float *) get_tensor_real_ptr(dst);

        if (trace_enabled()) {
            std::call_once(trace_pipeline_once, [&]() {
                fprintf(stderr,
                        "RKNPU trace: tensor '%s' type=%d uses pipeline '%s' (k_align=%d, n_align=%d, hadamard=%d)\n",
                        src0->name[0] ? src0->name : "<unnamed>",
                        (int) src0->type,
                        pipeline->pipeline_name.c_str(),
                        pipeline->k_align,
                        pipeline->n_align,
                        pipeline->use_hadamard ? 1 : 0);
            });
            if (pipeline->use_q4k_block32_asym) {
                std::call_once(trace_exact_once, [&]() {
                    fprintf(stderr,
                            "RKNPU trace: tensor '%s' type=%d uses exact Q4_K per-32 asymmetric INT4 (k_align=%d, n_align=%d)\n",
                            src0->name[0] ? src0->name : "<unnamed>",
                            (int) src0->type,
                            pipeline->k_align,
                            pipeline->n_align);
                });
            }
            if (pipeline->use_q4k_k256) {
                std::call_once(trace_k256_once, [&]() {
                    fprintf(stderr,
                            "RKNPU trace: tensor '%s' type=%d uses W4A16_Q4K_K256 (k_align=%d, n_align=%d)\n",
                            src0->name[0] ? src0->name : "<unnamed>",
                            (int) src0->type,
                            pipeline->k_align,
                            pipeline->n_align);
                });
            }
            if (pipeline->use_q4k_staged_int8) {
                std::call_once(trace_staged_once, [&]() {
                    fprintf(stderr,
                            "RKNPU trace: tensor '%s' type=%d uses W4A16_Q4K_STAGED_INT8 (k_align=%d, n_align=%d)\n",
                            src0->name[0] ? src0->name : "<unnamed>",
                            (int) src0->type,
                            pipeline->k_align,
                            pipeline->n_align);
                });
            }
            if (pipeline->use_hadamard && src0->type == GGML_TYPE_Q4_K) {
                std::call_once(trace_ffn_int4_once, [&]() {
                    fprintf(stderr,
                            "RKNPU trace: tensor '%s' type=%d uses %s FFN INT4 hybrid\n",
                            src0->name[0] ? src0->name : "<unnamed>",
                            (int) src0->type,
                            pipeline->pipeline_name.c_str());
                });
            }
        }

        for (size_t k_idx = 0; k_idx < plan->k_plans.size(); ++k_idx) {
#ifdef GGML_RKNPU2_DEBUG
            auto t2_start = std::chrono::high_resolution_clock::now();
#endif
            auto & k_plan = plan->k_plans[k_idx];
            const auto & k_seg = k_plan.k_segment;
            const int K_seg_op = k_seg.size_k;
            auto & matmul_ctxs = k_plan.matmul_ctxs;
            auto & mem_A_segments = k_plan.mem_A_segments;
            const size_t num_a_matrices = mem_A_segments.size();
            auto & mem_C_segments = k_plan.mem_C_segments;

            if (pipeline->use_q4k_k256) {
                const int sb_count = K_seg_op / 256;
                const int groups_per_sb = 8;
                const float * x = (const float *) get_tensor_real_ptr(src1);
                const int row_stride = (int) (src1->nb[1] / sizeof(float));

                for (int sb = 0; sb < sb_count; ++sb) {
                    const int k0 = k_seg.offset_k + sb * 256;
                    std::vector<uint16_t> a_fp16((size_t) M * 256);
                    #pragma omp parallel for
                    for (int m = 0; m < M; ++m) {
                        const float * src_row = x + (size_t) m * row_stride;
                        rknpu2_quantization::convert_fp32_to_fp16(src_row + k0, a_fp16.data() + (size_t) m * 256, 256);
                    }

                    for (size_t i = 0; i < num_active_segments; ++i) {
                        auto & lo_a = k_plan.mem_A_segments[i];
                        auto & hi_a = k_plan.mem_A_segments_hi[i];
                        memcpy(lo_a->virt_addr, a_fp16.data(), (size_t) M * 256 * sizeof(uint16_t));
                        memcpy(hi_a->virt_addr, a_fp16.data(), (size_t) M * 256 * sizeof(uint16_t));
                        if (M < M_op) {
                            memset((uint8_t *) lo_a->virt_addr + (size_t) M * 256 * sizeof(uint16_t), 0,
                                   (size_t) (M_op - M) * 256 * sizeof(uint16_t));
                            memset((uint8_t *) hi_a->virt_addr + (size_t) M * 256 * sizeof(uint16_t), 0,
                                   (size_t) (M_op - M) * 256 * sizeof(uint16_t));
                        }
                    }

                    std::vector<float> sum_a((size_t) M * groups_per_sb, 0.0f);
                    #pragma omp parallel for
                    for (int m = 0; m < M; ++m) {
                        for (int j = 0; j < groups_per_sb; ++j) {
                            float sum = 0.0f;
                            for (int l = 0; l < 32; ++l) {
                                sum += ggml_fp16_to_fp32(a_fp16[(size_t) m * 256 + j * 32 + l]);
                            }
                            sum_a[(size_t) m * groups_per_sb + j] = sum;
                        }
                    }

                    #pragma omp parallel for num_threads(num_active_segments)
                    for (size_t i = 0; i < num_active_segments; ++i) {
                        auto & lo_ctx = k_plan.matmul_ctxs[i];
                        auto & hi_ctx = k_plan.matmul_ctxs_hi[i];
                        auto & lo_b = k_plan.mem_B_segments[i];
                        auto & hi_b = k_plan.mem_B_segments_hi[i];
                        const auto & n_segment = active_n_segments[i];
                        const size_t boff = (size_t) i * sb_count + sb;

                        lo_b->offset = (int32_t) k_plan.b_offsets_lo[boff];
                        lo_b->size = (uint32_t) n_segment.size_n * 256;
                        hi_b->offset = (int32_t) k_plan.b_offsets_hi[boff];
                        hi_b->size = (uint32_t) n_segment.size_n * 128;
                        RKNN_CHECK(rknn_matmul_set_io_mem(lo_ctx->ctx, lo_b.get(), &lo_ctx->io_attr.B), "set K256 LO B");
                        RKNN_CHECK(rknn_matmul_set_io_mem(hi_ctx->ctx, hi_b.get(), &hi_ctx->io_attr.B), "set K256 HI B");
                        // Contexts are shared across K segments. Rebind A/C on every
                        // segment so the last plan created cannot steal an earlier
                        // segment's buffers.
                        RKNN_CHECK(rknn_matmul_set_io_mem(lo_ctx->ctx, k_plan.mem_A_segments[i].get(), &lo_ctx->io_attr.A), "set K256 LO A");
                        RKNN_CHECK(rknn_matmul_set_io_mem(hi_ctx->ctx, k_plan.mem_A_segments_hi[i].get(), &hi_ctx->io_attr.A), "set K256 HI A");
                        RKNN_CHECK(rknn_matmul_set_io_mem(lo_ctx->ctx, k_plan.mem_C_segments[i].get(), &lo_ctx->io_attr.C), "set K256 LO C");
                        RKNN_CHECK(rknn_matmul_set_io_mem(hi_ctx->ctx, k_plan.mem_C_segments_hi[i].get(), &hi_ctx->io_attr.C), "set K256 HI C");

                        RKNN_CHECK(rknn_mem_sync(lo_ctx->ctx, k_plan.mem_A_segments[i].get(), RKNN_MEMORY_SYNC_TO_DEVICE), "sync K256 LO A");
                        RKNN_CHECK(rknn_mem_sync(hi_ctx->ctx, k_plan.mem_A_segments_hi[i].get(), RKNN_MEMORY_SYNC_TO_DEVICE), "sync K256 HI A");
                        RKNN_CHECK(rknn_mem_sync(lo_ctx->ctx, lo_b.get(), RKNN_MEMORY_SYNC_TO_DEVICE), "sync K256 LO B");
                        RKNN_CHECK(rknn_mem_sync(hi_ctx->ctx, hi_b.get(), RKNN_MEMORY_SYNC_TO_DEVICE), "sync K256 HI B");
                        RKNN_CHECK(rknn_matmul_run(lo_ctx->ctx), "run K256 LO");
                        RKNN_CHECK(rknn_matmul_run(hi_ctx->ctx), "run K256 HI");
                        RKNN_CHECK(rknn_mem_sync(lo_ctx->ctx, k_plan.mem_C_segments[i].get(), RKNN_MEMORY_SYNC_FROM_DEVICE), "sync K256 LO C");
                        RKNN_CHECK(rknn_mem_sync(hi_ctx->ctx, k_plan.mem_C_segments_hi[i].get(), RKNN_MEMORY_SYNC_FROM_DEVICE), "sync K256 HI C");
                    }

                    #pragma omp parallel for
                    for (int m = 0; m < M; ++m) {
                        for (size_t i = 0; i < num_active_segments; ++i) {
                            const auto & n_segment = active_n_segments[i];
                            const float * lo_c = (const float *) k_plan.mem_C_segments[i]->virt_addr;
                            const float * hi_c = (const float *) k_plan.mem_C_segments_hi[i]->virt_addr;
                            for (int n = 0; n < n_segment.size_n; ++n) {
                                const int global_n = n_segment.offset_n + n;
                                const int global_sb = k0 / 256;
                                double zsum = 0.0;
                                for (int j = 0; j < groups_per_sb; ++j) {
                                    const int global_g = k0 / 32 + j;
                                    zsum += (double) plan->k256_z_grid[(size_t) global_n * (K / 32) + global_g]
                                          * (double) sum_a[(size_t) m * groups_per_sb + j];
                                }
                                const double d = plan->k256_d_grid[(size_t) global_n * (K / 256) + global_sb];
                                const double value = d * ((double) lo_c[(size_t) m * n_segment.size_n + n]
                                                      + 128.0 * (double) hi_c[(size_t) m * n_segment.size_n + n]) + zsum;
                                float * out = dst_data + (size_t) m * N + global_n;
                                if (k_idx == 0 && sb == 0) {
                                    *out = (float) value;
                                } else {
                                    *out += (float) value;
                                }

                            }
                        }
                    }
                }

                continue;
            }

            const bool quantized_A = pipeline->npu_type_a != rknpu2_configuration::NPU_TYPE_FP16;
            std::vector<std::vector<float>> scales_A_per_segment(num_a_matrices);
            if (quantized_A) {
                for (auto & scales : scales_A_per_segment) {
                    scales.resize(M, 1.0f);
                }
            }
            std::vector<float> sum_A(M, 0.0f);
            {
                const float * x = (const float *) get_tensor_real_ptr(src1);
                const int row_stride = (int) (src1->nb[1] / sizeof(float));
                auto prepare_row = [&](int m) {
                    const float * src_row = x + (size_t) m * row_stride;
                    thread_local std::vector<float> signed_row;
                    thread_local std::vector<float> hadamard_row;
                    thread_local std::vector<float> ready_row;
                    const float * segment = src_row + k_seg.offset_k;
                    if (is_hadamard) {
                        signed_row.resize(K);
                        hadamard_row.resize(K_op);
                        for (int k = 0; k < K; ++k) {
                            signed_row[k] = src_row[k] * s_vec[k];
                        }
                        rknpu2_calibration::hadamard_transform(hadamard_row.data(), signed_row.data(), K, K_op);
                        segment = hadamard_row.data() + k_seg.offset_k;
                    }
                    for (size_t idx = 0; idx < num_a_matrices; ++idx) {
                        const float * ready = segment;
                        if (is_lowrank) {
                            ready_row.resize(K_seg_op);
                            const auto & factors = lowrank_factors[k_idx * num_active_segments + idx];
                            for (int k = 0; k < K_seg_op; ++k) {
                                ready_row[k] = segment[k] * factors.v[k];
                            }
                            ready = ready_row.data();
                        }
                        void * dst_base = mem_A_segments[idx]->virt_addr;
                        if (pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_FP16) {
                            auto * dst_row = (uint16_t *) dst_base + (size_t) m * K_seg_op;
                            rknpu2_quantization::convert_fp32_to_fp16(ready, dst_row, K_seg_op);
                            if (!zeros_B_grid.empty()) {
                                float sum = 0.0f;
                                for (int k = 0; k < K_seg_op; ++k) {
                                    sum += ggml_fp16_to_fp32(dst_row[k]);
                                }
                                sum_A[m] = sum;
                            }
                        } else {
                            const float divisor = pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_INT8 ? 127.0f : 7.0f;
                            float & scale = scales_A_per_segment[idx][m];
                            scale = fp32_abs_max(ready, K_seg_op) / divisor;
                            if (pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_INT8) {
                                auto * dst_row = (int8_t *) dst_base + (size_t) m * K_seg_op;
                                rknpu2_quantization::quantize_fp32_to_int8(ready, dst_row, K_seg_op, &scale, 1);
                            } else {
                                auto * dst_row = (uint8_t *) dst_base + (size_t) m * (K_seg_op / 2);
                                rknpu2_quantization::quantize_fp32_to_int4_packed(ready, dst_row, K_seg_op, &scale, 1);
                            }
                        }
                    }
                };
                if (M >= 8) {
                    #pragma omp parallel for
                    for (int m = 0; m < M; ++m) {
                        prepare_row(m);
                    }
                } else {
                    for (int m = 0; m < M; ++m) {
                        prepare_row(m);
                    }
                }
                const size_t row_bytes = (size_t) K_seg_op * (pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_FP16 ? 2 : 1) /
                                         (pipeline->npu_type_a == rknpu2_configuration::NPU_TYPE_INT4 ? 2 : 1);
                for (size_t idx = 0; idx < num_a_matrices; ++idx) {
                    auto & mem_A = mem_A_segments[idx];
                    if (M < M_op) {
                        memset((uint8_t *) mem_A->virt_addr + (size_t) M * row_bytes, 0, (size_t) (M_op - M) * row_bytes);
                    }
                    RKNN_CHECK(rknn_mem_sync(matmul_ctxs[idx]->ctx, mem_A.get(), RKNN_MEMORY_SYNC_TO_DEVICE), "sync A TO_DEVICE");
                }
            }

            if (pipeline->use_q4k_staged_int8) {
                for (size_t idx = 0; idx < num_active_segments; ++idx) {
                    auto & mem_B = k_plan.mem_B_segments[idx];
                    stage_q4k_int8_segment_fused(src0, src0->data, K, N, k_seg, active_n_segments[idx],
                                                scales_B_grid, k_idx, mem_B.get());
                    RKNN_CHECK(rknn_mem_sync(matmul_ctxs[idx]->ctx, mem_B.get(), RKNN_MEMORY_SYNC_TO_DEVICE), "sync staged B TO_DEVICE");
                    RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctxs[idx]->ctx, mem_B.get(), &matmul_ctxs[idx]->io_attr.B), "set staged B at run");
                }
            }

#ifdef GGML_RKNPU2_DEBUG
            auto t3_start = std::chrono::high_resolution_clock::now();
            node_t_prepA += std::chrono::duration_cast<std::chrono::microseconds>(t3_start - t2_start).count();
            auto t4_start = std::chrono::high_resolution_clock::now();
            node_t_prepC += std::chrono::duration_cast<std::chrono::microseconds>(t4_start - t3_start).count();
#endif

            {
                std::atomic<bool> run_failed { false };
                #pragma omp parallel for num_threads(num_active_segments)
                for (size_t idx = 0; idx < num_active_segments; idx++) {
                    auto & matmul_ctx = matmul_ctxs[idx];
                    auto & mem_A = mem_A_segments[is_lowrank ? idx : 0];
                    if (matmul_ctx->mem_A_bound != mem_A.get()) {
                        RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctx->ctx, mem_A.get(), &matmul_ctx->io_attr.A), "set_io_mem A at run");
                        matmul_ctx->mem_A_bound = mem_A.get();
                    }

                    auto & mem_C = mem_C_segments[idx];
                    if (matmul_ctx->mem_C_bound != mem_C.get()) {
                        RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctx->ctx, mem_C.get(), &matmul_ctx->io_attr.C), "set_io_mem C at run");
                        matmul_ctx->mem_C_bound = mem_C.get();
                    }

                    if (pipeline->use_q4k_block32_asym) {
                        auto & mem_B = k_plan.mem_B_segments[idx];
                        mem_B->offset = (int32_t) k_plan.b_offsets[idx];
                        RKNN_CHECK(rknn_matmul_set_io_mem(matmul_ctx->ctx, mem_B.get(), &matmul_ctx->io_attr.B), "set exact B segment");
                    }

                    int ret = rknn_matmul_run(matmul_ctx->ctx);
                    if (trace_enabled()) {
                        std::call_once(trace_run_once, [ret]() {
                            fprintf(stderr, "RKNPU trace: rknn_matmul_run returned %d\n", ret);
                        });
                    }
                    if (ret != RKNN_SUCC) {
                        run_failed = true;
                    }
                }
                if (run_failed) {
                    return GGML_STATUS_FAILED;
                }
            }

#ifdef GGML_RKNPU2_DEBUG
            auto t5_start = std::chrono::high_resolution_clock::now();
            node_t_npu += std::chrono::duration_cast<std::chrono::microseconds>(t5_start - t4_start).count();
#endif
            {
                #pragma omp parallel for num_threads(num_active_segments)
                for (size_t idx = 0; idx < num_active_segments; idx++) {
                    RKNN_CHECK(rknn_mem_sync(matmul_ctxs[idx]->ctx, mem_C_segments[idx].get(), RKNN_MEMORY_SYNC_FROM_DEVICE), "sync C FROM_DEVICE");
                }

                const float hadamard_divisor = pipeline->use_hadamard ? (float)K_op : 1.0f;

                auto collect_row = [&](int m) {
                    for (size_t idx = 0; idx < num_active_segments; ++idx) {
                        const auto & segment = active_n_segments[idx];
                        const auto & scales_A = scales_A_per_segment[is_lowrank ? idx : 0];
                        const float scale_A = scales_A.empty() ? 1.0f : scales_A[m];
                        const float * scales_B = is_lowrank ? lowrank_factors[k_idx * num_active_segments + idx].u.data() :
                                                (scales_B_grid.empty() ? nullptr : scales_B_grid.data() + k_idx * N + segment.offset_n);
                        const float * zeros_B = is_lowrank ? nullptr :
                                                (zeros_B_grid.empty() ? nullptr : zeros_B_grid.data() + k_idx * N + segment.offset_n);
                        const float zero_sum = zeros_B ? sum_A[m] : 0.0f;
                        float * output = dst_data + (size_t) m * N + segment.offset_n;
                        const size_t row_offset = (size_t) m * segment.size_n;
                        switch (pipeline->npu_type_c) {
                            case rknpu2_configuration::NPU_TYPE_FP32:
                                collect_rknpu_row((const float *) mem_C_segments[idx]->virt_addr + row_offset, output,
                                                 segment.size_n, scales_B, zeros_B, scale_A / hadamard_divisor, zero_sum, k_idx != 0);
                                break;
                            case rknpu2_configuration::NPU_TYPE_INT32:
                                collect_rknpu_row((const int32_t *) mem_C_segments[idx]->virt_addr + row_offset, output,
                                                 segment.size_n, scales_B, nullptr, scale_A / hadamard_divisor, 0.0f, k_idx != 0);
                                break;
                            case rknpu2_configuration::NPU_TYPE_INT16:
                                collect_rknpu_row((const int16_t *) mem_C_segments[idx]->virt_addr + row_offset, output,
                                                 segment.size_n, scales_B, nullptr, scale_A / hadamard_divisor, 0.0f, k_idx != 0);
                                break;
                            default:
                                GGML_ABORT("Unsupported RKNPU output type");
                        }
                    }
                };

                if (M >= 8) {
                    #pragma omp parallel for
                    for (int m = 0; m < M; m++) {
                        collect_row(m);
                    }
                } else {
                    for (int m = 0; m < M; m++) {
                        collect_row(m);
                    }
                }
            }
#ifdef GGML_RKNPU2_DEBUG
            auto t6_start = std::chrono::high_resolution_clock::now();
            node_t_collect += std::chrono::duration_cast<std::chrono::microseconds>(t6_start - t5_start).count();
#endif
        }
    }

#ifdef GGML_RKNPU2_DEBUG
    t_setup += node_t_setup; t_prepA += node_t_prepA; t_prepC += node_t_prepC;
    t_npu += node_t_npu; t_collect += node_t_collect;
    call_count++;

    if (call_count > 0 && call_count % 50 == 0) {
        int64_t total = t_setup + t_prepA + t_prepC + t_npu + t_collect;
        if (total > 0) {
            int64_t avg_total = total / call_count;
            fprintf(stderr, "\nRKNPU2 profile (%d calls, avg %lld us/call):\n", call_count, (long long)avg_total);
            fprintf(stderr, "  Setup (ctx+B) : %6.1f%%  %lld us\n", 100.0*t_setup/total, (long long)t_setup/call_count);
            fprintf(stderr, "  Prep A (quant): %6.1f%%  %lld us\n", 100.0*t_prepA/total, (long long)t_prepA/call_count);
            fprintf(stderr, "  Prep C         : %6.1f%%  %lld us\n", 100.0*t_prepC/total, (long long)t_prepC/call_count);
            fprintf(stderr, "  NPU run        : %6.1f%%  %lld us\n", 100.0*t_npu/total, (long long)t_npu/call_count);
            fprintf(stderr, "  Collect (deq)  : %6.1f%%  %lld us\n", 100.0*t_collect/total, (long long)t_collect/call_count);
        }
    }
#endif

    return GGML_STATUS_SUCCESS;
}


//
// Buffer
//

// Function for calculating a real tensor size for the NPU
static size_t get_tensor_packed_size(const struct ggml_tensor * tensor) {
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    const auto* pipeline = config.resolve_op_support(tensor);

    if (pipeline) {
        if (pipeline->use_q4k_staged_int8) {
            return 0;
        }
        if (tensor->ne[0] <= 0 || tensor->ne[0] > INT32_MAX || tensor->ne[1] <= 0 || tensor->ne[1] > INT32_MAX ||
            tensor->ne[2] != 1 || tensor->ne[3] != 1 || (pipeline->use_hadamard && tensor->ne[0] > (1 << 30))) {
            return 0;
        }
        const int K = (int)tensor->ne[0];
        const int N = (int)tensor->ne[1];

        // Tensor too small for NPU alignment: skip NPU packing
        if (K % pipeline->k_align != 0 || N % pipeline->n_align != 0) {
            return 0;
        }

        const int K_op = pipeline->use_hadamard ? rknpu2_calibration::next_power_of_two(K) : K;

        if (pipeline->use_q4k_k256) {
            if (tensor->type != GGML_TYPE_Q4_K || K % 256 != 0) {
                return 0;
            }
            return (size_t) N * K + (size_t) N * K / 2;
        }

        int k_limit = config.max_k_limit;
        if (pipeline->effective_k > 0) {
            k_limit = (k_limit > 0) ? std::min(k_limit, pipeline->effective_k) : pipeline->effective_k;
        }

        auto k_segments = compute_k_segments(K_op, k_limit, pipeline->k_align);
        auto n_segments = compute_n_segments(N, config.active_cores, pipeline->n_align);

        size_t total_size = 0;
        for (const auto& k_seg : k_segments) {
            for (const auto& seg : n_segments) {
                if (seg.size_n > 0) {
                    if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4) {
                        total_size += (size_t)seg.size_n * k_seg.size_k / 2;
                    } else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8) {
                        total_size += (size_t)seg.size_n * k_seg.size_k;
                    } else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_FP16) {
                        total_size += (size_t)seg.size_n * k_seg.size_k * 2;
                    }
                }
            }
        }
        return total_size;
    }
    return ggml_nbytes(tensor);
}

static void ggml_backend_rknpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rknpu_buffer_context * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;

    // Freeing an every individual RKNN buffer using the allocator context
    for (auto& pair : ctx->tensor_allocs) {
        if (pair.second.mem) {
            rknn_matmul_ctx alloc_ctx = g_domain_manager.get_allocator_context(pair.second.iommu_domain_id);
            rknn_destroy_mem(alloc_ctx, pair.second.mem);
            g_domain_manager.release_domain_memory(pair.second.iommu_domain_id, pair.second.size);
        }
    }

    // Freeing the virtual memory block
    munmap(ctx->virtual_base, ctx->total_size);

    delete ctx;
}

static void * ggml_backend_rknpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rknpu_buffer_context * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
    return ctx->virtual_base;
}

static enum ggml_status ggml_backend_rknpu_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    auto * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;

    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    const auto* pipeline = config.resolve_op_support(tensor);

    // Initialize tensor only if it is supported by the pipeline and has valid NPU dimensions
    if (pipeline) {
        size_t size = get_tensor_packed_size(tensor);
        if (size > 0) {
            size_t offset = (uintptr_t)tensor->data - (uintptr_t)ctx->virtual_base;
            ctx->get_tensor_allocation(offset, size);
        }
    }

    return GGML_STATUS_SUCCESS;
}

// Function for dequantizing a single row from GGUF format to FP32
static void dequantize_row(
    const struct ggml_tensor * tensor,
    const void * raw_data,
    int n, int K,
    float * row_out)
{
    if (tensor->type == GGML_TYPE_F32) {
        const float* src = (const float*)raw_data;
        memcpy(row_out, src + (size_t)n * K, K * sizeof(float));
    } else if (tensor->type == GGML_TYPE_F16) {
        const ggml_fp16_t* src = (const ggml_fp16_t*)raw_data;
        const ggml_fp16_t* src_row = src + (size_t)n * K;
        for (int k = 0; k < K; ++k) row_out[k] = ggml_fp16_to_fp32(src_row[k]);
    } else if (tensor->type == GGML_TYPE_Q8_0) {
        const block_q8_0* src = (const block_q8_0*)raw_data;
        dequantize_row_q8_0(src + (size_t)n * (K / QK8_0), row_out, K);
    } else if (tensor->type == GGML_TYPE_Q6_K) {
        const block_q6_K* src = (const block_q6_K*)raw_data;
        dequantize_row_q6_K(src + (size_t)n * (K / QK_K), row_out, K);
    } else if (tensor->type == GGML_TYPE_Q4_0) {
        const block_q4_0* src = (const block_q4_0*)raw_data;
        dequantize_row_q4_0(src + (size_t)n * (K / QK4_0), row_out, K);
    } else if (tensor->type == GGML_TYPE_Q4_K) {
        const block_q4_K* src = (const block_q4_K*)raw_data;
        dequantize_row_q4_K(src + (size_t)n * (K / QK_K), row_out, K);
    } else {
        GGML_ASSERT(false && "Unsupported weight type for NPU pipeline");
    }
}

// Decode the 6-bit scale/min pair of one 32-element Q4_K sub-block.
static inline void rknpu_get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Extract one 32-element Q4_K sub-block exactly.
// The stored q is mapped to signed int4 q' = q - 8.
// The affine reconstruction is x = scale * q' + zero.
static void extract_q4k_asym_segment(
    std::vector<uint8_t> & out_q,
    std::vector<float> & scales,
    std::vector<float> & zeros,
    const struct ggml_tensor * tensor,
    const void * raw_data,
    int K, int N,
    const MatrixSegmentK & k_seg,
    const MatrixSegmentN & n_seg)
{
    GGML_ASSERT(tensor->type == GGML_TYPE_Q4_K);
    GGML_ASSERT(k_seg.size_k == 32 && k_seg.offset_k % 32 == 0);

    const int K_seg = k_seg.size_k;
    const int N_seg = n_seg.size_n;
    out_q.assign((size_t) N_seg * (K_seg / 2), 0);
    scales.assign(N_seg, 1.0f);
    zeros.assign(N_seg, 0.0f);

    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_K, K);
    const int block_idx = k_seg.offset_k / QK_K;
    const int sub = (k_seg.offset_k % QK_K) / 32;

    #pragma omp parallel for
    for (int i = 0; i < N_seg; ++i) {
        const int global_n = n_seg.offset_n + i;
        if (global_n >= N) {
            continue;
        }

        const uint8_t * row_ptr = (const uint8_t *) raw_data + (size_t) global_n * row_bytes;
        const block_q4_K * blk = (const block_q4_K *) row_ptr + block_idx;

        uint8_t sc = 0;
        uint8_t m = 0;
        rknpu_get_scale_min_k4(sub, blk->scales, &sc, &m);

        const float d = ggml_fp16_to_fp32(blk->d);
        const float dmin = ggml_fp16_to_fp32(blk->dmin);
        const float scale = d * sc;
        scales[i] = scale;
        zeros[i] = 8.0f * scale - dmin * m;

        const uint8_t * qsrc = blk->qs + (sub / 2) * 32;
        uint8_t * qdst = out_q.data() + (size_t) i * (K_seg / 2);
        for (int l = 0; l < 32; ++l) {
            const uint8_t qraw = (sub & 1) ? (qsrc[l] >> 4) : (qsrc[l] & 0xF);
            const uint8_t qs = (uint8_t) (qraw - 8) & 0x0F;
            if ((l & 1) == 0) {
                qdst[l / 2] |= qs;
            } else {
                qdst[l / 2] |= (uint8_t) (qs << 4);
            }
        }
    }
}

// Extract one Q4_K K=256 segment for the INT8+INT4 decomposition pipeline.
// P_L is row-major INT8, P_H is row-major packed INT4, d is per superblock,
// and z is per 32-element group.
static void extract_q4k_k256_segment(
    std::vector<uint8_t> & out_lo,
    std::vector<uint8_t> & out_hi,
    std::vector<float> & out_d,
    std::vector<float> & out_z,
    const struct ggml_tensor * tensor,
    const void * raw_data,
    int K, int N,
    const MatrixSegmentK & k_seg,
    const MatrixSegmentN & n_seg)
{
    GGML_ASSERT(tensor->type == GGML_TYPE_Q4_K);
    GGML_ASSERT(k_seg.size_k % 256 == 0 && k_seg.offset_k % 256 == 0);

    const int K_seg = k_seg.size_k;
    const int N_seg = n_seg.size_n;
    const int sb_count = K_seg / 256;
    const int group_count = K_seg / 32;

    out_lo.assign((size_t) N_seg * K_seg, 0);
    out_hi.assign((size_t) N_seg * (K_seg / 2), 0);
    out_d.assign((size_t) N_seg * sb_count, 0.0f);
    out_z.assign((size_t) N_seg * group_count, 0.0f);

    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_K, K);

    #pragma omp parallel for
    for (int i = 0; i < N_seg; ++i) {
        const int global_n = n_seg.offset_n + i;
        if (global_n >= N) {
            continue;
        }

        const uint8_t * row_ptr = (const uint8_t *) raw_data + (size_t) global_n * row_bytes;
        uint8_t * lo_row = out_lo.data() + (size_t) i * K_seg;
        uint8_t * hi_row = out_hi.data() + (size_t) i * (K_seg / 2);
        float * d_row = out_d.data() + (size_t) i * sb_count;
        float * z_row = out_z.data() + (size_t) i * group_count;

        for (int sb = 0; sb < sb_count; ++sb) {
            const int global_sb = k_seg.offset_k / 256 + sb;
            const block_q4_K * blk = (const block_q4_K *) row_ptr + global_sb;
            const float d = ggml_fp16_to_fp32(blk->d);
            const float dmin = ggml_fp16_to_fp32(blk->dmin);
            d_row[sb] = d;

            for (int j = 0; j < 8; ++j) {
                uint8_t sc = 0;
                uint8_t m = 0;
                rknpu_get_scale_min_k4(j, blk->scales, &sc, &m);
                z_row[sb * 8 + j] = 8.0f * d * (float) sc - dmin * (float) m;

                const uint8_t * qsrc = blk->qs + (j / 2) * 32;
                for (int l = 0; l < 32; ++l) {
                    const int qraw = (j & 1) ? (qsrc[l] >> 4) : (qsrc[l] & 0xF);
                    const int p = (int) sc * (qraw - 8);
                    const int hi = (p + 64 >= 0) ? (p + 64) / 128 : -((-(p + 64) + 127) / 128);
                    const int lo = p - 128 * hi;
                    const int kk = sb * 256 + j * 32 + l;
                    lo_row[kk] = (uint8_t) (int8_t) lo;
                    const uint8_t hv = (uint8_t) (hi & 0xF);
                    if ((kk & 1) == 0) hi_row[kk / 2] |= hv;
                    else              hi_row[kk / 2] |= (uint8_t) (hv << 4);
                }
            }
        }
    }
}

// Function for extracting a specific tensor segment and converting it to FP32
static void dequantize_tensor_segment(
    std::vector<float>& out_segment,
    const struct ggml_tensor * tensor,
    ggml_backend_rknpu_buffer_context * ctx,
    const void * raw_data,
    int K, int N, int K_op,
    const MatrixSegmentK & k_seg,
    const MatrixSegmentN & n_seg,
    bool use_hadamard)
{
    size_t seg_elements = (size_t)n_seg.size_n * k_seg.size_k;
    out_segment.resize(seg_elements);

    std::vector<float> s_vec;
    if (use_hadamard) {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        s_vec = ctx->hadamard_s_vectors[tensor];
    }

    #pragma omp parallel for
    for (int i = 0; i < n_seg.size_n; ++i) {
        int global_n = n_seg.offset_n + i;

        if (global_n < N) {
            std::vector<float> row_raw(K);
            std::vector<float> row_processed(K_op, 0.0f);

            dequantize_row(tensor, raw_data, global_n, K, row_raw.data());

            if (use_hadamard) {
                std::vector<float> signed_row(K);
                for (int k = 0; k < K; ++k) signed_row[k] = row_raw[k] * s_vec[k];
                rknpu2_calibration::hadamard_transform(row_processed.data(), signed_row.data(), K, K_op);
            } else {
                memcpy(row_processed.data(), row_raw.data(), K * sizeof(float));
            }

            memcpy(&out_segment[i * k_seg.size_k], &row_processed[k_seg.offset_k], k_seg.size_k * sizeof(float));
        } else {
            memset(&out_segment[i * k_seg.size_k], 0, k_seg.size_k * sizeof(float));
        }
    }
}

// Function for quantizing the FP32 segment to the target NPU format
static void quantize_tensor_segment(
    const std::vector<float>& fp32_segment,
    std::vector<uint8_t>& out_quantized,
    const MatrixSegmentK & k_seg,
    const MatrixSegmentN & n_seg,
    const std::vector<float>& scales,
    rknpu2_configuration::Rknpu2NpuType npu_type)
{
    const int K_seg = k_seg.size_k;
    const int N_seg = n_seg.size_n;
    const size_t seg_elements = (size_t)N_seg * K_seg;

    const bool is_per_element = (scales.size() == seg_elements);

    if (npu_type == rknpu2_configuration::NPU_TYPE_FP16) {
        out_quantized.resize(seg_elements * 2);
        rknpu2_quantization::convert_fp32_to_fp16(
            fp32_segment.data(),
            (uint16_t*)out_quantized.data(),
            seg_elements);
    }
    else if (npu_type == rknpu2_configuration::NPU_TYPE_INT8) {
        out_quantized.resize(seg_elements);
        int8_t* dst = (int8_t*)out_quantized.data();

        #pragma omp parallel for
        for (int i = 0; i < N_seg; ++i) {
            const float* src_row = fp32_segment.data() + (size_t)i * K_seg;
            int8_t* dst_row = dst + (size_t)i * K_seg;

            if (is_per_element) {
                rknpu2_quantization::quantize_fp32_to_int8(src_row, dst_row, K_seg, scales.data() + (size_t)i * K_seg, K_seg);
            } else {
                rknpu2_quantization::quantize_fp32_to_int8(src_row, dst_row, K_seg, &scales[i], 1);
            }
        }
    }
    else if (npu_type == rknpu2_configuration::NPU_TYPE_INT4) {
        out_quantized.resize(seg_elements / 2);
        uint8_t* dst = out_quantized.data();

        #pragma omp parallel for
        for (int i = 0; i < N_seg; ++i) {
            const float* src_row = fp32_segment.data() + (size_t)i * K_seg;
            uint8_t* dst_row = dst + (size_t)i * (K_seg / 2);

            if (is_per_element) {
                rknpu2_quantization::quantize_fp32_to_int4_packed(src_row, dst_row, K_seg, scales.data() + (size_t)i * K_seg, K_seg);
            } else {
                rknpu2_quantization::quantize_fp32_to_int4_packed(src_row, dst_row, K_seg, &scales[i], 1);
            }
        }
    }
}

// Function for packing
static void pack_native(
    uint8_t* dst, const uint8_t* src,
    int K_total, int k_offset, int k_segment, int k_align,
    int N_total, int n_offset, int n_segment, int n_align,
    int element_bits)
{
    UNUSED(N_total);

    GGML_ASSERT(k_segment % k_align == 0 && "k_segment must be aligned to k_align");
    GGML_ASSERT(n_segment % n_align == 0 && "n_segment must be aligned to n_align");

    const size_t k_sub_bytes     = (size_t)k_align * element_bits / 8;
    const size_t src_row_bytes  = (size_t)K_total * element_bits / 8;
    const size_t n_blocks       = n_segment / n_align;
    const size_t k_blocks       = k_segment / k_align;
    const size_t kblock_stride  = (size_t)n_align * k_sub_bytes;
    const size_t nblock_stride  = k_blocks * kblock_stride;

    for (size_t ni = 0; ni < n_blocks; ++ni) {
        for (size_t ki = 0; ki < k_blocks; ++ki) {
            uint8_t* dst_tile = dst + ni * nblock_stride + ki * kblock_stride;

            for (int nn = 0; nn < n_align; ++nn) {
                const size_t n_global = (size_t)n_offset + ni * n_align + nn;
                const size_t k_start  = (size_t)k_offset + ki * k_align;

                const uint8_t* src_ptr = src + n_global * src_row_bytes
                                             + k_start * element_bits / 8;
                uint8_t* dst_ptr = dst_tile + nn * k_sub_bytes;

                size_t off = 0;
                for (; off + 16 <= k_sub_bytes; off += 16) {
                    vst1q_u8(dst_ptr + off, vld1q_u8(src_ptr + off));
                }
                for (; off < k_sub_bytes; ++off) {
                    dst_ptr[off] = src_ptr[off];
                }
            }
        }
    }
}

// Function for packing the quantized segment into the native NPU layout and writing to DMA
static size_t pack_tensor_segment(
    const std::vector<uint8_t>& quantized_segment,
    uint8_t * dst_dma_ptr,
    const MatrixSegmentK & k_seg,
    const MatrixSegmentN & n_seg,
    const rknpu2_configuration::Rknpu2HardwarePipeline * pipeline)
{
    int element_bits = 0;
    size_t segment_packed_size = 0;

    if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_FP16) {
        element_bits = 16;
        segment_packed_size = (size_t)n_seg.size_n * k_seg.size_k * 2;
    } else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8) {
        element_bits = 8;
        segment_packed_size = (size_t)n_seg.size_n * k_seg.size_k;
    } else if (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4) {
        element_bits = 4;
        segment_packed_size = (size_t)n_seg.size_n * k_seg.size_k / 2;
    }

    pack_native(dst_dma_ptr, quantized_segment.data(),
                k_seg.size_k, 0, k_seg.size_k, pipeline->k_align,
                n_seg.size_n, 0, n_seg.size_n, pipeline->n_align,
                element_bits);

    return segment_packed_size;
}

// Fill a shared native INT8 B scratch buffer for one Q4_K (K segment, N segment).
// The source GGUF Q4_K bytes stay resident; only this bounded scratch is materialized.
static void stage_q4k_int8_segment(
    const struct ggml_tensor * tensor,
    const void * raw_data,
    int K,
    int N,
    int K_op,
    const MatrixSegmentK & k_seg,
    const MatrixSegmentN & n_seg,
    const std::vector<float> & scales_grid,
    size_t k_idx,
    rknn_tensor_mem * mem_B,
    const rknpu2_configuration::Rknpu2HardwarePipeline * pipeline)
{
    std::vector<float> fp32_segment;
    std::vector<uint8_t> int8_segment;
    dequantize_tensor_segment(fp32_segment, tensor, nullptr, raw_data, K, N, K_op, k_seg, n_seg, false);

    std::vector<float> segment_scales((size_t) n_seg.size_n);
    for (int i = 0; i < n_seg.size_n; ++i) {
        segment_scales[i] = scales_grid[k_idx * (size_t) N + n_seg.offset_n + i];
    }
    quantize_tensor_segment(fp32_segment, int8_segment, k_seg, n_seg, segment_scales,
                            rknpu2_configuration::NPU_TYPE_INT8);
    pack_tensor_segment(int8_segment, (uint8_t *) mem_B->virt_addr, k_seg, n_seg, pipeline);
}

// Fused Q4_K -> INT8 converter that writes the NPU native layout directly.
// It avoids materializing the full FP32 segment and the row-major INT8 intermediate.
static void stage_q4k_int8_segment_fused(
    const struct ggml_tensor * tensor,
    const void * raw_data,
    int K,
    int N,
    const MatrixSegmentK & k_seg,
    const MatrixSegmentN & n_seg,
    const std::vector<float> & scales_grid,
    size_t k_idx,
    rknn_tensor_mem * mem_B)
{
    UNUSED(tensor);
    GGML_ASSERT(k_seg.offset_k % 256 == 0);
    GGML_ASSERT(k_seg.size_k % 32 == 0);
    GGML_ASSERT(n_seg.offset_n % 16 == 0);
    GGML_ASSERT(n_seg.size_n % 16 == 0);

    const uint8_t * raw = (const uint8_t *) raw_data;
    const size_t row_bytes = (size_t) (K / 256) * sizeof(block_q4_K);
    const int k_blocks = k_seg.size_k / 32;
    const int n_blocks = n_seg.size_n / 16;
    const size_t kblock_stride = 16 * 32;
    const size_t nblock_stride = (size_t) k_blocks * kblock_stride;
    uint8_t * dst = (uint8_t *) mem_B->virt_addr;

    #pragma omp parallel for collapse(2)
    for (int nb = 0; nb < n_blocks; ++nb) {
        for (int kb = 0; kb < k_blocks; ++kb) {
            const int local_k = kb * 32;
            const int global_group = (k_seg.offset_k + local_k) / 32;
            const int sb = global_group / 8;
            const int j = global_group % 8;
            const bool high_nibble = (j & 1) != 0;
            const int q_offset = (j / 2) * 32;

            for (int ni = 0; ni < 16; ++ni) {
                const int n_global = n_seg.offset_n + nb * 16 + ni;
                const block_q4_K * blk = (const block_q4_K *) (raw
                    + (size_t) n_global * row_bytes
                    + (size_t) sb * sizeof(block_q4_K));
                uint8_t sc = 0, m = 0;
                rknpu_get_scale_min_k4(j, blk->scales, &sc, &m);
                const float d    = ggml_fp16_to_fp32(blk->d) * (float) sc;
                const float dmin = ggml_fp16_to_fp32(blk->dmin) * (float) m;
                const float inv_scale = 1.0f / scales_grid[k_idx * (size_t) N + n_global];
                const uint8_t * q = blk->qs + q_offset;
                uint8_t * out = dst + (size_t) nb * nblock_stride
                    + (size_t) kb * kblock_stride + (size_t) ni * 32;
                for (int l = 0; l < 32; ++l) {
                    const int qv = high_nibble ? (q[l] >> 4) : (q[l] & 0x0F);
                    int iv = (int) std::lrintf((d * (float) qv - dmin) * inv_scale);
                    iv = std::max(-127, std::min(127, iv));
                    out[l] = (int8_t) iv;
                }
            }
        }
    }
}

static rknpu2_weight_cache::Descriptor build_weight_cache_descriptor(
    const struct ggml_tensor * tensor,
    const rknpu2_configuration::Rknpu2DeviceConfig & config,
    const rknpu2_configuration::Rknpu2HardwarePipeline * pipeline,
    const std::vector<MatrixSegmentK> & k_segments,
    const std::vector<MatrixSegmentN> & n_segments,
    rknpu2_weight_cache::Hash128 source_hash,
    size_t source_size,
    size_t packed_size,
    const std::string & runtime_version,
    const std::vector<float> & signs)
{
    rknpu2_weight_cache::Descriptor descriptor;
    descriptor.source_hash = source_hash;
    descriptor.source_size = source_size;
    descriptor.packed_size = packed_size;
    descriptor.tensor_name = tensor->name;
    descriptor.tensor_type = tensor->type;
    for (size_t i = 0; i < descriptor.ne.size(); ++i) {
        descriptor.ne[i] = tensor->ne[i];
    }

    descriptor.device_name = config.device_name;
    descriptor.runtime_version = runtime_version;
    descriptor.pipeline_name = pipeline->pipeline_name;
    descriptor.max_k_limit = config.max_k_limit;
    descriptor.npu_type_a = pipeline->npu_type_a;
    descriptor.npu_type_b = pipeline->npu_type_b;
    descriptor.npu_type_c = pipeline->npu_type_c;
    descriptor.matmul_type = pipeline->mm_type;
    descriptor.k_align = pipeline->k_align;
    descriptor.n_align = pipeline->n_align;
    descriptor.effective_k = pipeline->effective_k;
    descriptor.use_hadamard = pipeline->use_hadamard ? 1 : 0;
    const bool is_quantized = pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4 ||
                              pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8;
    descriptor.use_lowrank = pipeline->use_lowrank && is_quantized ? 1 : 0;
    descriptor.scale_scheme = descriptor.use_lowrank ? rknpu2_weight_cache::SCALE_LOWRANK :
                              (is_quantized ? rknpu2_weight_cache::SCALE_PER_CHANNEL_ABSMAX : rknpu2_weight_cache::SCALE_NONE);
    descriptor.sign_count = signs.size();
    descriptor.seed_scheme = pipeline->use_hadamard ? rknpu2_weight_cache::SEED_TENSOR_NAME_STD_HASH_MT19937 : rknpu2_weight_cache::SEED_NONE;
    if (pipeline->use_hadamard) {
        descriptor.hadamard_sign_hash = rknpu2_weight_cache::hash_source(signs.data(), signs.size() * sizeof(float));
    }

    descriptor.active_cores.reserve(config.active_cores.size());
    for (int core : config.active_cores) {
        descriptor.active_cores.push_back(core);
    }

    size_t total_k = 0;
    descriptor.k_segments.reserve(k_segments.size());
    for (const auto & segment : k_segments) {
        descriptor.k_segments.push_back({ segment.offset_k, segment.size_k });
        total_k += segment.size_k;
    }

    size_t active_n_count = 0;
    size_t total_n = 0;
    descriptor.n_segments.reserve(n_segments.size());
    for (const auto & segment : n_segments) {
        descriptor.n_segments.push_back({ segment.offset_n, segment.size_n, segment.core_id });
        if (segment.size_n > 0) {
            ++active_n_count;
            total_n += segment.size_n;
        }
    }

    if (descriptor.use_lowrank) {
        descriptor.lowrank_u_count = k_segments.size() * total_n;
        descriptor.lowrank_v_count = total_k * active_n_count;
    } else if (is_quantized) {
        descriptor.scale_count = k_segments.size() * (size_t) tensor->ne[1];
    }
    return descriptor;
}

static void ggml_backend_rknpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * ctx = (ggml_backend_rknpu_buffer_context *) buffer->context;
    const size_t tensor_size = ggml_nbytes(tensor);
    GGML_ASSERT(offset <= tensor_size && size <= tensor_size - offset);

    // Keep the GGUF host copy separate from the native NPU allocation.
    memmove((uint8_t *) tensor->data + offset, data, size);
    if (size == 0) {
        return;
    }

    const auto & config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();
    const auto * pipeline = config.resolve_op_support(tensor);
    if (!pipeline) {
        return;
    }

    bool already_packed = false;
    {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        auto it = ctx->tensor_versions.find(tensor);
        already_packed = it != ctx->tensor_versions.end() && it->second > 0;
    }
    if (offset + size != tensor_size && !already_packed) {
        return;
    }

    const bool staged_q4k = pipeline->use_q4k_staged_int8 && tensor->type == GGML_TYPE_Q4_K;
    if (staged_q4k) {
        if (offset + size != tensor_size) {
            return;
        }
        const int K = (int) tensor->ne[0];
        const int N = (int) tensor->ne[1];
        int k_limit = config.max_k_limit;
        if (pipeline->effective_k > 0) {
            k_limit = k_limit > 0 ? std::min(k_limit, pipeline->effective_k) : pipeline->effective_k;
        }
        const auto k_segments = compute_k_segments(K, k_limit, pipeline->k_align);
        std::vector<float> scales(k_segments.size() * (size_t) N, 1.0f);
        #pragma omp parallel for
        for (int n = 0; n < N; ++n) {
            std::vector<float> row(K);
            dequantize_row(tensor, tensor->data, n, K, row.data());
            for (size_t k_idx = 0; k_idx < k_segments.size(); ++k_idx) {
                const auto & k_seg = k_segments[k_idx];
                float amax = 0.0f;
                for (int k = k_seg.offset_k; k < k_seg.offset_k + k_seg.size_k; ++k) {
                    amax = std::max(amax, std::abs(row[k]));
                }
                scales[k_idx * (size_t) N + n] = amax == 0.0f ? 1.0f : amax / 127.0f;
            }
        }
        std::lock_guard<std::mutex> lock(ctx->mutex);
        ctx->quantized_tensor_scales[tensor] = std::move(scales);
        ctx->quantized_tensor_zeros.erase(tensor);
        ++ctx->tensor_versions[tensor];
        return;
    }

    const size_t required_size = get_tensor_packed_size(tensor);
    if (required_size == 0) {
        return;
    }

    const int K = (int) tensor->ne[0];
    const int N = (int) tensor->ne[1];
    const int K_op = pipeline->use_hadamard ? rknpu2_calibration::next_power_of_two(K) : K;
    const bool is_lowrank = pipeline->use_lowrank &&
        (pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT8 ||
         pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4);
    const bool use_q4k_exact = pipeline->use_q4k_block32_asym &&
        tensor->type == GGML_TYPE_Q4_K && !pipeline->use_hadamard && !is_lowrank;
    const bool use_q4k_k256 = pipeline->use_q4k_k256 &&
        tensor->type == GGML_TYPE_Q4_K && !pipeline->use_hadamard && !is_lowrank;

    int k_limit = config.max_k_limit;
    if (pipeline->effective_k > 0) {
        k_limit = (k_limit > 0) ? std::min(k_limit, pipeline->effective_k) : pipeline->effective_k;
    }
    const auto k_segments = compute_k_segments(K_op, k_limit, pipeline->k_align);
    const auto n_segments = compute_n_segments(N, config.active_cores, pipeline->n_align);
    size_t active_n_count = 0;
    for (const auto & segment : n_segments) {
        active_n_count += segment.size_n > 0;
    }

    const size_t tensor_offset_in_virtual = (uintptr_t) tensor->data - (uintptr_t) ctx->virtual_base;
    const auto alloc = ctx->get_tensor_allocation(tensor_offset_in_virtual, required_size);
    auto * tensor_dma_ptr = (uint8_t *) alloc.mem->virt_addr;
    const rknn_matmul_ctx sync_ctx = g_domain_manager.get_allocator_context(alloc.iommu_domain_id);

    std::vector<float> signs;
    if (pipeline->use_hadamard) {
        signs.resize(K_op, 1.0f);
        std::string tensor_name = tensor->name;
        std::mt19937 gen(static_cast<uint32_t>(std::hash<std::string>{}(tensor_name)));
        std::uniform_int_distribution<int> distrib(0, 1);
        for (int k = 0; k < K_op; ++k) {
            signs[k] = (distrib(gen) == 0) ? -1.0f : 1.0f;
        }
    }

    if (use_q4k_k256) {
        std::vector<float> k256_d((size_t) N * (K / 256), 0.0f);
        std::vector<float> k256_z((size_t) N * (K / 32), 0.0f);
        std::vector<uint8_t> seg_lo;
        std::vector<uint8_t> seg_hi;
        std::vector<float> seg_d;
        std::vector<float> seg_z;

        for (size_t k_idx = 0; k_idx < k_segments.size(); ++k_idx) {
            const auto & k_seg = k_segments[k_idx];
            GGML_ASSERT(k_seg.size_k % 256 == 0);
            for (const auto & n_seg : n_segments) {
                if (n_seg.size_n == 0) {
                    continue;
                }
                extract_q4k_k256_segment(seg_lo, seg_hi, seg_d, seg_z,
                                         tensor, tensor->data, K, N, k_seg, n_seg);

                const int N_seg = n_seg.size_n;
                const int K_seg = k_seg.size_k;
                for (int sb = 0; sb < K_seg / 256; ++sb) {
                    std::vector<uint8_t> lo_pack((size_t) N_seg * 256);
                    std::vector<uint8_t> hi_pack((size_t) N_seg * 128);
                    for (int i = 0; i < N_seg; ++i) {
                        memcpy(lo_pack.data() + (size_t) i * 256,
                               seg_lo.data() + (size_t) i * K_seg + (size_t) sb * 256, 256);
                        memcpy(hi_pack.data() + (size_t) i * 128,
                               seg_hi.data() + (size_t) i * (K_seg / 2) + (size_t) sb * 128, 128);
                    }

                    uint8_t * dst_lo = tensor_dma_ptr
                        + (size_t) n_seg.offset_n * K
                        + (size_t) (k_seg.offset_k + sb * 256) * N_seg;
                    uint8_t * dst_hi = tensor_dma_ptr
                        + (size_t) N * K
                        + (size_t) n_seg.offset_n * (K / 2)
                        + (size_t) (k_seg.offset_k + sb * 256) * (N_seg / 2);
                    pack_native(dst_lo, lo_pack.data(), 256, 0, 256, 32, N_seg, 0, N_seg, 16, 8);
                    pack_native(dst_hi, hi_pack.data(), 256, 0, 256, 32, N_seg, 0, N_seg, 16, 4);
                }

                for (int i = 0; i < N_seg; ++i) {
                    const int global_n = n_seg.offset_n + i;
                    for (int sb = 0; sb < K_seg / 256; ++sb) {
                        k256_d[(size_t) global_n * (K / 256) + k_seg.offset_k / 256 + sb] =
                            seg_d[(size_t) i * (K_seg / 256) + sb];
                    }
                    for (int g = 0; g < K_seg / 32; ++g) {
                        k256_z[(size_t) global_n * (K / 32) + k_seg.offset_k / 32 + g] =
                            seg_z[(size_t) i * (K_seg / 32) + g];
                    }
                }
            }
        }

        RKNN_CHECK(rknn_mem_sync(sync_ctx, alloc.mem, RKNN_MEMORY_SYNC_TO_DEVICE), "sync K256 B TO_DEVICE");
        std::lock_guard<std::mutex> lock(ctx->mutex);
        ctx->q4k_k256_d[tensor] = std::move(k256_d);
        ctx->q4k_k256_z[tensor] = std::move(k256_z);
        ctx->quantized_tensor_scales.erase(tensor);
        ctx->quantized_tensor_zeros.erase(tensor);
        ctx->lowrank_segment_factors.erase(tensor);
        ctx->hadamard_s_vectors.erase(tensor);
        ++ctx->tensor_versions[tensor];
        return;
    }

    const bool use_weight_cache = rknpu2_weight_cache::enabled() && !use_q4k_exact;
    rknpu2_weight_cache::Descriptor cache_descriptor;
    if (use_weight_cache) {
        const auto source_hash = rknpu2_weight_cache::hash_source(tensor->data, tensor_size);
        cache_descriptor = build_weight_cache_descriptor(tensor, config, pipeline, k_segments, n_segments, source_hash,
                                                         tensor_size, required_size, get_rknn_runtime_version(sync_ctx), signs);
        std::vector<float> cached_scales;
        std::vector<float> cached_signs;
        std::vector<float> cached_u;
        std::vector<float> cached_v;
        if (rknpu2_weight_cache::load(cache_descriptor, tensor_dma_ptr, cached_scales, cached_signs, cached_u, cached_v)) {
            std::vector<LowRankSegmentFactors> cached_factors;
            if (is_lowrank) {
                cached_factors.resize(k_segments.size() * active_n_count);
                size_t u_offset = 0;
                size_t v_offset = 0;
                for (size_t k_idx = 0; k_idx < k_segments.size(); ++k_idx) {
                    size_t n_idx = 0;
                    for (const auto & n_seg : n_segments) {
                        if (n_seg.size_n == 0) {
                            continue;
                        }
                        auto & factors = cached_factors[k_idx * active_n_count + n_idx++];
                        factors.u.assign(cached_u.begin() + u_offset, cached_u.begin() + u_offset + n_seg.size_n);
                        factors.v.assign(cached_v.begin() + v_offset, cached_v.begin() + v_offset + k_segments[k_idx].size_k);
                        u_offset += n_seg.size_n;
                        v_offset += k_segments[k_idx].size_k;
                    }
                }
                GGML_ASSERT(u_offset == cached_u.size() && v_offset == cached_v.size());
            }
            RKNN_CHECK(rknn_mem_sync(sync_ctx, alloc.mem, RKNN_MEMORY_SYNC_TO_DEVICE), "sync cached B TO_DEVICE");
            std::lock_guard<std::mutex> lock(ctx->mutex);
            ctx->quantized_tensor_scales[tensor] = std::move(cached_scales);
            if (pipeline->use_hadamard) {
                ctx->hadamard_s_vectors[tensor] = std::move(cached_signs);
            } else {
                ctx->hadamard_s_vectors.erase(tensor);
            }
            if (is_lowrank) {
                ctx->lowrank_segment_factors[tensor] = std::move(cached_factors);
            } else {
                ctx->lowrank_segment_factors.erase(tensor);
            }
            ++ctx->tensor_versions[tensor];
            return;
        }
    }

    if (pipeline->use_hadamard) {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        ctx->hadamard_s_vectors[tensor] = signs;
    }

    std::vector<float> seg_fp32;
    std::vector<uint8_t> seg_npu;
    std::vector<float> segment_scales;
    std::vector<float> segment_zeros;
    uint8_t * current_write_ptr = tensor_dma_ptr;
    std::vector<float> per_channel_scales;
    std::vector<float> per_channel_zeros;
    if (!is_lowrank && pipeline->npu_type_b != rknpu2_configuration::NPU_TYPE_FP16) {
        per_channel_scales.resize(k_segments.size() * (size_t) N, 1.0f);
    }
    if (use_q4k_exact) {
        per_channel_zeros.resize(k_segments.size() * (size_t) N, 0.0f);
    }
    std::vector<LowRankSegmentFactors> lowrank_factors;
    if (is_lowrank) {
        lowrank_factors.resize(k_segments.size() * active_n_count);
    }

    for (size_t k_idx = 0; k_idx < k_segments.size(); ++k_idx) {
        const auto & k_seg = k_segments[k_idx];
        size_t n_idx = 0;
        for (const auto & n_seg : n_segments) {
            if (n_seg.size_n == 0) {
                continue;
            }
            if (use_q4k_exact) {
                extract_q4k_asym_segment(seg_npu, segment_scales, segment_zeros, tensor, tensor->data, K, N, k_seg, n_seg);
                for (int i = 0; i < n_seg.size_n; ++i) {
                    const size_t dst_idx = k_idx * (size_t) N + n_seg.offset_n + i;
                    per_channel_scales[dst_idx] = segment_scales[i];
                    per_channel_zeros[dst_idx] = segment_zeros[i];
                }
            } else {
                dequantize_tensor_segment(seg_fp32, tensor, ctx, tensor->data, K, N, K_op, k_seg, n_seg, pipeline->use_hadamard);
                if (is_lowrank) {
                    const float qmax = pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4 ? 7.0f : 127.0f;
                    const int K_seg = k_seg.size_k;
                    const int N_seg = n_seg.size_n;
                    auto & factors = lowrank_factors[k_idx * active_n_count + n_idx];
                    factors.v.resize(K_seg);
                    factors.u.resize(N_seg);
                    rknpu2_calibration::lowrank_decomposition(seg_fp32.data(), K_seg, N_seg, qmax, factors.u.data(), factors.v.data());
                    segment_scales.resize((size_t) N_seg * K_seg);
                    #pragma omp parallel for
                    for (int i = 0; i < N_seg; ++i) {
                        float * row_scales = segment_scales.data() + (size_t) i * K_seg;
                        for (int j = 0; j < K_seg; ++j) {
                            row_scales[j] = std::max(factors.v[j] * factors.u[i], 1e-7f);
                        }
                    }
                } else if (pipeline->npu_type_b != rknpu2_configuration::NPU_TYPE_FP16) {
                    const float quant_divisor = pipeline->npu_type_b == rknpu2_configuration::NPU_TYPE_INT4 ? 7.0f : 127.0f;
                    segment_scales.resize(n_seg.size_n);
                    #pragma omp parallel for
                    for (int i = 0; i < n_seg.size_n; ++i) {
                        const float * row = seg_fp32.data() + (size_t) i * k_seg.size_k;
                        float amax = 0.0f;
                        for (int j = 0; j < k_seg.size_k; ++j) {
                            amax = std::max(amax, std::abs(row[j]));
                        }
                        const float scale = amax == 0.0f ? 1.0f : amax / quant_divisor;
                        segment_scales[i] = scale;
                        per_channel_scales[k_idx * (size_t) N + n_seg.offset_n + i] = scale;
                    }
                } else {
                    segment_scales.assign(n_seg.size_n, 1.0f);
                }

                quantize_tensor_segment(seg_fp32, seg_npu, k_seg, n_seg, segment_scales, pipeline->npu_type_b);
            }
            current_write_ptr += pack_tensor_segment(seg_npu, current_write_ptr, k_seg, n_seg, pipeline);
            ++n_idx;
        }
    }
    GGML_ASSERT((size_t) (current_write_ptr - tensor_dma_ptr) == required_size);

    if (use_weight_cache) {
        std::vector<float> flattened_u;
        std::vector<float> flattened_v;
        flattened_u.reserve(cache_descriptor.lowrank_u_count);
        flattened_v.reserve(cache_descriptor.lowrank_v_count);
        for (const auto & factors : lowrank_factors) {
            flattened_u.insert(flattened_u.end(), factors.u.begin(), factors.u.end());
            flattened_v.insert(flattened_v.end(), factors.v.begin(), factors.v.end());
        }
        rknpu2_weight_cache::store(cache_descriptor, tensor_dma_ptr, per_channel_scales, signs, flattened_u, flattened_v);
    }

    RKNN_CHECK(rknn_mem_sync(sync_ctx, alloc.mem, RKNN_MEMORY_SYNC_TO_DEVICE), "sync B TO_DEVICE");
    std::lock_guard<std::mutex> lock(ctx->mutex);
    ctx->quantized_tensor_scales[tensor] = std::move(per_channel_scales);
    if (use_q4k_exact) {
        ctx->quantized_tensor_zeros[tensor] = std::move(per_channel_zeros);
    } else {
        ctx->quantized_tensor_zeros.erase(tensor);
    }
    if (is_lowrank) {
        ctx->lowrank_segment_factors[tensor] = std::move(lowrank_factors);
    } else {
        ctx->lowrank_segment_factors.erase(tensor);
    }
    if (!pipeline->use_hadamard) {
        ctx->hadamard_s_vectors.erase(tensor);
    }
    ++ctx->tensor_versions[tensor];
}

static void ggml_backend_rknpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    UNUSED(buffer);
    memcpy(data, (uint8_t*)tensor->data + offset, size);
}

static void ggml_backend_rknpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = (ggml_backend_rknpu_buffer_context *)buffer->context;
    std::vector<ggml_tensor *> packed_tensors;
    {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        memset(ctx->virtual_base, value, ctx->total_size);
        packed_tensors.reserve(ctx->tensor_versions.size());
        for (const auto & entry : ctx->tensor_versions) {
            packed_tensors.push_back(const_cast<ggml_tensor *>(entry.first));
        }
    }
    // The native layout and quantization metadata must reflect the host bytes.
    // Repacking also syncs the device and invalidates cached execution plans.
    for (auto * tensor : packed_tensors) {
        ggml_backend_rknpu_buffer_set_tensor(buffer, tensor, tensor->data, 0, ggml_nbytes(tensor));
    }
}


//
// Buffer Type
//

static const char * ggml_backend_rknpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return "RKNPU";
}

static ggml_backend_buffer_t ggml_backend_rknpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    UNUSED(buft);

    // Reserving virtual memory block
    void* virtual_base = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (virtual_base == MAP_FAILED) {
        return NULL;
    }

    // Initializing buffer context
    ggml_backend_rknpu_buffer_context * ctx = new ggml_backend_rknpu_buffer_context();
    ctx->virtual_base = virtual_base;
    ctx->total_size = size;
    ctx->name = "rknpu_virtual_buffer";

    static const ggml_backend_buffer_i rknpu_buffer_interface = {
        /* .free_buffer   = */ ggml_backend_rknpu_buffer_free_buffer,
        /* .get_base      = */ ggml_backend_rknpu_buffer_get_base,
        /* .init_tensor   = */ ggml_backend_rknpu_buffer_init_tensor,
        /* .memset_tensor = */ NULL,
        /* .set_tensor    = */ ggml_backend_rknpu_buffer_set_tensor,
        /* .get_tensor    = */ ggml_backend_rknpu_buffer_get_tensor,
        /* .set_tensor_2d = */ NULL,
        /* .get_tensor_2d = */ NULL,
        /* .cpy_tensor    = */ NULL,
        /* .clear         = */ ggml_backend_rknpu_buffer_clear,
        /* .reset         = */ NULL,
    };

    return ggml_backend_buffer_init(buft, rknpu_buffer_interface, ctx, size);
}

static size_t ggml_backend_rknpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return 64;
}

static size_t ggml_backend_rknpu_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    UNUSED(buft);
    size_t packed_size = get_tensor_packed_size(tensor);
    return packed_size > 0 ? std::max(packed_size, ggml_nbytes(tensor)) : ggml_nbytes(tensor);
}


static bool ggml_backend_rknpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    UNUSED(buft);
    return true;
}

//
// Device
//

static const char * ggml_backend_rknpu_device_get_name(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return "RKNPU";
}

static const char * ggml_backend_rknpu_device_get_description(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return "Rockchip NPU";
}

static void ggml_backend_rknpu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    UNUSED(dev);
    *free = 0;
    *total = 0;
}

static enum ggml_backend_dev_type ggml_backend_rknpu_device_get_type(ggml_backend_dev_t dev) {
    UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_rknpu_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name = ggml_backend_rknpu_device_get_name(dev);
    props->description = ggml_backend_rknpu_device_get_description(dev);
    props->type = ggml_backend_rknpu_device_get_type(dev);
    ggml_backend_rknpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id = NULL;

    props->caps.async = false;
    props->caps.host_buffer = false;
    props->caps.buffer_from_host_ptr = false;
    props->caps.events = false;
}

static bool ggml_backend_rknpu_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    UNUSED(dev);

    // Getting the current device configuration
    const auto& config = rknpu2_configuration::Rknpu2ConfigManager::get_instance().get_current_config();

    switch (op->op) {
        case GGML_OP_NONE:
            return true;

        case GGML_OP_MUL_MAT: {
            const struct ggml_tensor * src0 = op->src[0]; // Weights
            const struct ggml_tensor * src1 = op->src[1]; // Activations

            // Searching for available hardware pipeline for this tensor
            const auto* pipeline = config.resolve_op_support(src0);
            if (!pipeline) {
                return false;
            }

            // Rejecting zero-dimension ops
            if (src0->ne[0] == 0 || src0->ne[1] == 0 ||
                src1->ne[0] == 0 || src1->ne[1] == 0) {
                return false;
            }

            // Rejecting higher-dimension ops
            if (src0->ne[2] != 1 || src0->ne[3] != 1 ||
                src1->ne[2] != 1 || src1->ne[3] != 1 ||
                op->ne[2]   != 1 || op->ne[3]   != 1) {
                return false;
            }

            // Checking if activation type matches the supported operation
            if (src1->type != GGML_TYPE_F32) {
                return false;
            }

            if (src0->type == GGML_TYPE_Q4_K && ggml_nrows(src1) <= 4) {
                return false;
            }
            const bool staged_q4k = pipeline->use_q4k_staged_int8 && src0->type == GGML_TYPE_Q4_K;
            if ((!staged_q4k && get_tensor_packed_size(src0) == 0) || src1->ne[1] <= 0 || src1->ne[1] > (1 << 30) || op->type != GGML_TYPE_F32) {
                return false;
            }

            // Checking for K alignment
            if (src0->ne[0] % pipeline->k_align != 0) {
                return false;
            }

            // Checking for N alignment
            if (src0->ne[1] % pipeline->n_align != 0) {
                return false;
            }

            // Checking for exact dimensions
            if (src1->ne[0] != src0->ne[0]) {
                 return false;
            }

            // Checking contiguous memory
            if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
                return false;
            }

            return true;
        }
        default:
            return false;
    }
}

static ggml_backend_t ggml_backend_rknpu_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    UNUSED(dev);
    UNUSED(params);

    // Increase fd limit to avoid "Too many open files" from DMA-BUF allocations
    struct rlimit rlim;
    if (getrlimit(RLIMIT_NOFILE, &rlim) == 0) {
        const rlim_t target = (rlim_t)32768;
        if (rlim.rlim_cur < target) {
            const rlim_t new_cur = std::min(rlim.rlim_max, target);
            if (new_cur > rlim.rlim_cur) {
                rlim.rlim_cur = new_cur;
                setrlimit(RLIMIT_NOFILE, &rlim);
            }
        }
    }

    // Fetch device from environment variable, default to RK3588 if not set
    const char* env_device = std::getenv("RKNPU_DEVICE");
    std::string target_device = env_device ? env_device : "RK3588";
    if (!rknpu2_configuration::Rknpu2ConfigManager::get_instance().select_device(target_device)) return NULL;

    ggml_backend_rknpu_context * ctx = new ggml_backend_rknpu_context();

    static const struct ggml_backend_i rknpu_backend_interface = {
        /* .get_name           = */ ggml_backend_rknpu_name,
        /* .free               = */ ggml_backend_rknpu_free,
        /* .set_tensor_async   = */ NULL,
        /* .get_tensor_async   = */ NULL,
        /* .set_tensor_2d_async= */ NULL,
        /* .get_tensor_2d_async= */ NULL,
        /* .cpy_tensor_async   = */ NULL,
        /* .synchronize        = */ NULL,
        /* .graph_plan_create  = */ NULL,
        /* .graph_plan_free    = */ NULL,
        /* .graph_plan_update  = */ NULL,
        /* .graph_plan_compute = */ NULL,
        /* .graph_compute      = */ ggml_backend_rknpu_graph_compute,
        /* .event_record       = */ NULL,
        /* .event_wait         = */ NULL,
        /* .graph_optimize     = */ NULL,
    };

    return new ggml_backend{
        /* .guid    = */ {0},
        /* .iface   = */ rknpu_backend_interface,
        /* .device  = */ dev,
        /* .context = */ ctx,
    };
}


//
// Registry
//

static const char * ggml_backend_rknpu_reg_get_name(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return "RKNPU";
}

static size_t ggml_backend_rknpu_reg_get_device_count(ggml_backend_reg_t reg) {
    UNUSED(reg);
    return 1;
}

static ggml_backend_dev_t ggml_backend_rknpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    if (index != 0) {
        return NULL;
    }

    static const struct ggml_backend_buffer_type_i rknpu_buffer_type_interface = {
        /* .get_name         = */ ggml_backend_rknpu_buffer_type_get_name,
        /* .alloc_buffer     = */ ggml_backend_rknpu_buffer_type_alloc_buffer,
        /* .alloc_buffer_n   = */ NULL,
        /* .get_alignment    = */ ggml_backend_rknpu_buffer_type_get_alignment,
        /* .get_max_size     = */ NULL,
        /* .get_alloc_size   = */ ggml_backend_rknpu_buffer_type_get_alloc_size,
        /* .get_alloc_size_n = */ NULL,
        /* .is_host          = */ ggml_backend_rknpu_buffer_type_is_host,
    };

    static struct ggml_backend_buffer_type rknpu_buffer_type = {
        /* .iface   = */ rknpu_buffer_type_interface,
        /* .device  = */ NULL,
        /* .context = */ NULL,
    };

    static const struct ggml_backend_device_i rknpu_device_interface = {
        /* .get_name             = */ ggml_backend_rknpu_device_get_name,
        /* .get_description      = */ ggml_backend_rknpu_device_get_description,
        /* .get_memory           = */ ggml_backend_rknpu_device_get_memory,
        /* .get_type             = */ ggml_backend_rknpu_device_get_type,
        /* .get_props            = */ ggml_backend_rknpu_device_get_props,
        /* .init_backend         = */ ggml_backend_rknpu_device_init_backend,
        /* .get_buffer_type      = */ [](ggml_backend_dev_t dev) { UNUSED(dev); return &rknpu_buffer_type; },
        /* .get_host_buffer_type = */ NULL,
        /* .buffer_from_host_ptr = */ NULL,
        /* .supports_op          = */ ggml_backend_rknpu_device_supports_op,
        /* .supports_buft        = */ [](ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) { UNUSED(dev); return buft == &rknpu_buffer_type; },
        /* .offload_op           = */ NULL,
        /* .event_new            = */ NULL,
        /* .event_free           = */ NULL,
        /* .event_synchronize    = */ NULL,
    };

    static struct ggml_backend_device rknpu_device = {
        /* .iface   = */ rknpu_device_interface,
        /* .reg     = */ reg,
        /* .context = */ NULL,
    };

    if (rknpu_buffer_type.device == NULL) {
        rknpu_buffer_type.device = &rknpu_device;
    }

    return &rknpu_device;
}


//
// Public API
//

GGML_API ggml_backend_reg_t ggml_backend_rknpu2_reg(void) {
    static const struct ggml_backend_reg_i rknpu_reg_interface = {
        /* .get_name         = */ ggml_backend_rknpu_reg_get_name,
        /* .get_device_count = */ ggml_backend_rknpu_reg_get_device_count,
        /* .get_device       = */ ggml_backend_rknpu_reg_get_device,
        /* .get_proc_address = */ NULL,
    };

    static struct ggml_backend_reg rknpu_backend_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ rknpu_reg_interface,
        /* .context     = */ NULL,
    };

    return &rknpu_backend_reg;
}

#ifdef GGML_BACKEND_DL
GGML_BACKEND_DL_IMPL(ggml_backend_rknpu2_reg)
#endif
