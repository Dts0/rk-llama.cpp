#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rknpu2_weight_cache {

struct Hash128 {
    uint64_t lo = 0;
    uint64_t hi = 0;
};

struct SegmentK {
    int32_t offset = 0;
    int32_t size = 0;
};

struct SegmentN {
    int32_t offset = 0;
    int32_t size = 0;
    int32_t core = 0;
};

struct Descriptor {
    Hash128 source_hash;
    uint64_t source_size = 0;
    uint64_t packed_size = 0;
    uint64_t scale_count = 0;
    uint64_t sign_count = 0;

    std::string tensor_name;
    int32_t tensor_type = 0;
    std::array<int64_t, 4> ne = {};

    std::string device_name;
    std::string runtime_version;
    std::string pipeline_name;
    std::vector<int32_t> active_cores;
    std::vector<SegmentK> k_segments;
    std::vector<SegmentN> n_segments;

    int32_t max_k_limit = 0;
    int32_t npu_type_a = 0;
    int32_t npu_type_b = 0;
    int32_t npu_type_c = 0;
    int32_t matmul_type = 0;
    int32_t k_align = 0;
    int32_t n_align = 0;
    int32_t effective_k = 0;
    int32_t use_hadamard = 0;
};

bool enabled();
bool rebuild_requested();
Hash128 hash_source(const void * data, size_t size);
bool load(const Descriptor & descriptor, uint8_t * packed_data, std::vector<float> & scales, std::vector<float> & signs);
void store(const Descriptor & descriptor, const uint8_t * packed_data, const std::vector<float> & scales, const std::vector<float> & signs);

} // namespace rknpu2_weight_cache
