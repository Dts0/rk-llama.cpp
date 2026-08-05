#include "rknpu2-weight-cache.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr uint8_t CACHE_MAGIC[8] = { 'R', 'K', 'N', 'P', 'W', '0', '0', '1' };
constexpr uint32_t CACHE_VERSION = 1;
constexpr uint32_t CACHE_ENDIAN = 0x01020304;
constexpr uint64_t HASH_SEED_LO = 0x243f6a8885a308d3ULL;
constexpr uint64_t HASH_SEED_HI = 0x13198a2e03707344ULL;

struct CacheHeader {
    uint8_t magic[8];
    uint32_t version;
    uint32_t header_size;
    uint32_t endian;
    uint32_t tensor_type;
    int64_t ne[4];
    uint64_t source_size;
    uint64_t packed_size;
    uint64_t scale_count;
    uint64_t sign_count;
    uint64_t source_hash_lo;
    uint64_t source_hash_hi;
    uint64_t key_hash_lo;
    uint64_t key_hash_hi;
    uint64_t payload_hash_lo;
    uint64_t payload_hash_hi;
    uint64_t file_size;
    uint64_t runtime_hash;
    uint32_t core_count;
    uint32_t k_segment_count;
    uint32_t n_segment_count;
    int32_t max_k_limit;
    int32_t npu_type_a;
    int32_t npu_type_b;
    int32_t npu_type_c;
    int32_t matmul_type;
    int32_t k_align;
    int32_t n_align;
    int32_t effective_k;
    int32_t use_hadamard;
    uint32_t reserved[15];
};

struct CacheConfig {
    std::string directory;
    bool enabled = false;
    bool rebuild = false;
    bool trace = false;
};

static bool env_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && strcmp(value, "0") != 0;
}

static uint64_t rotate_left(uint64_t value, unsigned int shift) {
    return (value << shift) | (value >> (64 - shift));
}

static uint64_t avalanche(uint64_t value) {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

static rknpu2_weight_cache::Hash128 hash_bytes(const void * data, size_t size, rknpu2_weight_cache::Hash128 seed) {
    const uint8_t * ptr = static_cast<const uint8_t *>(data);
    uint64_t lo = seed.lo ^ avalanche((uint64_t) size + 0x9e3779b97f4a7c15ULL);
    uint64_t hi = seed.hi ^ avalanche((uint64_t) size + 0x6a09e667f3bcc909ULL);

    while (size >= sizeof(uint64_t)) {
        uint64_t lane;
        memcpy(&lane, ptr, sizeof(lane));
        lo ^= avalanche(lane + 0x3c6ef372fe94f82bULL);
        hi ^= avalanche(lane + 0xa54ff53a5f1d36f1ULL);
        lo = rotate_left(lo, 27) * 5 + 0x52dce729;
        hi = rotate_left(hi, 31) * 5 + 0x38495ab5;
        ptr += sizeof(uint64_t);
        size -= sizeof(uint64_t);
    }

    if (size > 0) {
        uint64_t tail = 0;
        memcpy(&tail, ptr, size);
        lo ^= avalanche(tail + 0x510e527fade682d1ULL);
        hi ^= avalanche(tail + 0x9b05688c2b3e6c1fULL);
    }

    return { avalanche(lo), avalanche(hi) };
}

static rknpu2_weight_cache::Hash128 hash_append(rknpu2_weight_cache::Hash128 hash, const void * data, size_t size) {
    return hash_bytes(data, size, { hash.lo ^ 0x1f83d9abfb41bd6bULL, hash.hi ^ 0x5be0cd19137e2179ULL });
}

template<typename T>
static rknpu2_weight_cache::Hash128 hash_value(rknpu2_weight_cache::Hash128 hash, const T & value) {
    return hash_append(hash, &value, sizeof(value));
}

static rknpu2_weight_cache::Hash128 hash_string(rknpu2_weight_cache::Hash128 hash, const std::string & value) {
    const uint64_t size = value.size();
    hash = hash_value(hash, size);
    return hash_append(hash, value.data(), value.size());
}

template<typename T>
static rknpu2_weight_cache::Hash128 hash_vector(rknpu2_weight_cache::Hash128 hash, const std::vector<T> & values) {
    const uint64_t size = values.size();
    hash = hash_value(hash, size);
    return hash_append(hash, values.data(), values.size() * sizeof(T));
}

static bool make_directory(const std::string & path) {
    if (path.empty()) {
        return false;
    }

    std::string current;
    if (path[0] == '/') {
        current = "/";
    }

    size_t start = path[0] == '/' ? 1 : 0;
    while (start <= path.size()) {
        const size_t slash = path.find('/', start);
        const std::string part = path.substr(start, slash == std::string::npos ? std::string::npos : slash - start);
        if (!part.empty()) {
            if (!current.empty() && current.back() != '/') {
                current += '/';
            }
            current += part;
            if (mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
                return false;
            }

            struct stat st;
            if (stat(current.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
                return false;
            }
        }
        if (slash == std::string::npos) {
            break;
        }
        start = slash + 1;
    }
    return true;
}

static const CacheConfig & cache_config() {
    static const CacheConfig config = []() {
        CacheConfig result;
        const char * directory = std::getenv("RKNPU_WEIGHT_CACHE_DIR");
        const bool requested = env_flag_enabled("RKNPU_WEIGHT_CACHE");
        result.rebuild = env_flag_enabled("RKNPU_WEIGHT_CACHE_REBUILD");
        result.trace = std::getenv("RKNPU_TRACE") != nullptr;
        if (!requested) {
            return result;
        }
        if (directory == nullptr || directory[0] == '\0') {
            fprintf(stderr, "RKNPU weight cache: RKNPU_WEIGHT_CACHE_DIR is not set\n");
            return result;
        }

        result.directory = directory;
        result.enabled = make_directory(result.directory);
        if (!result.enabled) {
            fprintf(stderr, "RKNPU weight cache: cannot create directory '%s': %s\n", result.directory.c_str(), strerror(errno));
        } else {
            fprintf(stderr, "RKNPU weight cache: enabled at '%s'%s\n", result.directory.c_str(), result.rebuild ? " (rebuild)" : "");
        }
        return result;
    }();
    return config;
}

static rknpu2_weight_cache::Hash128 descriptor_hash(const rknpu2_weight_cache::Descriptor & descriptor) {
    using namespace rknpu2_weight_cache;
    Hash128 hash = { HASH_SEED_LO, HASH_SEED_HI };
    hash = hash_value(hash, CACHE_VERSION);
    hash = hash_value(hash, CACHE_ENDIAN);
    hash = hash_value(hash, descriptor.source_hash);
    hash = hash_value(hash, descriptor.source_size);
    hash = hash_value(hash, descriptor.packed_size);
    hash = hash_value(hash, descriptor.scale_count);
    hash = hash_value(hash, descriptor.sign_count);
    hash = hash_string(hash, descriptor.tensor_name);
    hash = hash_value(hash, descriptor.tensor_type);
    hash = hash_append(hash, descriptor.ne.data(), descriptor.ne.size() * sizeof(descriptor.ne[0]));
    hash = hash_string(hash, descriptor.device_name);
    hash = hash_string(hash, descriptor.runtime_version);
    hash = hash_string(hash, descriptor.pipeline_name);
    hash = hash_vector(hash, descriptor.active_cores);
    hash = hash_vector(hash, descriptor.k_segments);
    hash = hash_vector(hash, descriptor.n_segments);
    hash = hash_value(hash, descriptor.max_k_limit);
    hash = hash_value(hash, descriptor.npu_type_a);
    hash = hash_value(hash, descriptor.npu_type_b);
    hash = hash_value(hash, descriptor.npu_type_c);
    hash = hash_value(hash, descriptor.matmul_type);
    hash = hash_value(hash, descriptor.k_align);
    hash = hash_value(hash, descriptor.n_align);
    hash = hash_value(hash, descriptor.effective_k);
    return hash_value(hash, descriptor.use_hadamard);
}

static uint64_t runtime_hash(const std::string & runtime_version) {
    const auto hash = hash_bytes(runtime_version.data(), runtime_version.size(), { HASH_SEED_LO, HASH_SEED_HI });
    return hash.lo ^ hash.hi;
}

static std::string safe_tensor_name(const std::string & name) {
    std::string result;
    result.reserve(std::min<size_t>(name.size(), 80));
    for (char ch : name) {
        if (result.size() == 80) {
            break;
        }
        const bool safe = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '.' || ch == '_' || ch == '-';
        result += safe ? ch : '_';
    }
    return result.empty() ? "tensor" : result;
}

static std::string cache_path(const rknpu2_weight_cache::Descriptor & descriptor, rknpu2_weight_cache::Hash128 key_hash) {
    char suffix[96];
    snprintf(suffix, sizeof(suffix), ".%016llx%016llx.%016llx%016llx.rknpu2w",
            (unsigned long long) descriptor.source_hash.lo, (unsigned long long) descriptor.source_hash.hi,
            (unsigned long long) key_hash.lo, (unsigned long long) key_hash.hi);
    return cache_config().directory + "/" + safe_tensor_name(descriptor.tensor_name) + suffix;
}

static bool add_size(uint64_t & total, uint64_t count, size_t element_size) {
    if (count > (std::numeric_limits<uint64_t>::max() - total) / element_size) {
        return false;
    }
    total += count * element_size;
    return true;
}

static bool expected_file_size(const rknpu2_weight_cache::Descriptor & descriptor, uint64_t & result) {
    result = sizeof(CacheHeader);
    return add_size(result, descriptor.active_cores.size(), sizeof(int32_t)) &&
           add_size(result, descriptor.k_segments.size(), sizeof(rknpu2_weight_cache::SegmentK)) &&
           add_size(result, descriptor.n_segments.size(), sizeof(rknpu2_weight_cache::SegmentN)) &&
           add_size(result, descriptor.scale_count, sizeof(float)) &&
           add_size(result, descriptor.sign_count, sizeof(float)) &&
           add_size(result, descriptor.packed_size, sizeof(uint8_t));
}

static rknpu2_weight_cache::Hash128 payload_hash(const rknpu2_weight_cache::Descriptor & descriptor, const std::vector<float> & scales,
                                                 const std::vector<float> & signs, const uint8_t * packed_data) {
    using namespace rknpu2_weight_cache;
    Hash128 hash = { HASH_SEED_LO, HASH_SEED_HI };
    hash = hash_vector(hash, descriptor.active_cores);
    hash = hash_vector(hash, descriptor.k_segments);
    hash = hash_vector(hash, descriptor.n_segments);
    hash = hash_vector(hash, scales);
    hash = hash_vector(hash, signs);
    return hash_append(hash, packed_data, descriptor.packed_size);
}

static CacheHeader make_header(const rknpu2_weight_cache::Descriptor & descriptor, rknpu2_weight_cache::Hash128 key_hash,
                               rknpu2_weight_cache::Hash128 data_hash, uint64_t file_size) {
    CacheHeader header = {};
    memcpy(header.magic, CACHE_MAGIC, sizeof(CACHE_MAGIC));
    header.version = CACHE_VERSION;
    header.header_size = sizeof(CacheHeader);
    header.endian = CACHE_ENDIAN;
    header.tensor_type = descriptor.tensor_type;
    memcpy(header.ne, descriptor.ne.data(), sizeof(header.ne));
    header.source_size = descriptor.source_size;
    header.packed_size = descriptor.packed_size;
    header.scale_count = descriptor.scale_count;
    header.sign_count = descriptor.sign_count;
    header.source_hash_lo = descriptor.source_hash.lo;
    header.source_hash_hi = descriptor.source_hash.hi;
    header.key_hash_lo = key_hash.lo;
    header.key_hash_hi = key_hash.hi;
    header.payload_hash_lo = data_hash.lo;
    header.payload_hash_hi = data_hash.hi;
    header.file_size = file_size;
    header.runtime_hash = runtime_hash(descriptor.runtime_version);
    header.core_count = descriptor.active_cores.size();
    header.k_segment_count = descriptor.k_segments.size();
    header.n_segment_count = descriptor.n_segments.size();
    header.max_k_limit = descriptor.max_k_limit;
    header.npu_type_a = descriptor.npu_type_a;
    header.npu_type_b = descriptor.npu_type_b;
    header.npu_type_c = descriptor.npu_type_c;
    header.matmul_type = descriptor.matmul_type;
    header.k_align = descriptor.k_align;
    header.n_align = descriptor.n_align;
    header.effective_k = descriptor.effective_k;
    header.use_hadamard = descriptor.use_hadamard;
    return header;
}

static bool header_matches(const CacheHeader & header, const rknpu2_weight_cache::Descriptor & descriptor,
                           rknpu2_weight_cache::Hash128 key_hash, uint64_t file_size) {
    return memcmp(header.magic, CACHE_MAGIC, sizeof(CACHE_MAGIC)) == 0 &&
           header.version == CACHE_VERSION && header.header_size == sizeof(CacheHeader) && header.endian == CACHE_ENDIAN &&
           header.tensor_type == (uint32_t) descriptor.tensor_type && memcmp(header.ne, descriptor.ne.data(), sizeof(header.ne)) == 0 &&
           header.source_size == descriptor.source_size && header.packed_size == descriptor.packed_size &&
           header.scale_count == descriptor.scale_count && header.sign_count == descriptor.sign_count &&
           header.source_hash_lo == descriptor.source_hash.lo && header.source_hash_hi == descriptor.source_hash.hi &&
           header.key_hash_lo == key_hash.lo && header.key_hash_hi == key_hash.hi && header.file_size == file_size &&
           header.runtime_hash == runtime_hash(descriptor.runtime_version) &&
           header.core_count == descriptor.active_cores.size() && header.k_segment_count == descriptor.k_segments.size() &&
           header.n_segment_count == descriptor.n_segments.size() && header.max_k_limit == descriptor.max_k_limit &&
           header.npu_type_a == descriptor.npu_type_a && header.npu_type_b == descriptor.npu_type_b &&
           header.npu_type_c == descriptor.npu_type_c && header.matmul_type == descriptor.matmul_type &&
           header.k_align == descriptor.k_align && header.n_align == descriptor.n_align &&
           header.effective_k == descriptor.effective_k && header.use_hadamard == descriptor.use_hadamard;
}

static bool read_full(int fd, void * data, size_t size) {
    uint8_t * ptr = static_cast<uint8_t *>(data);
    while (size > 0) {
        const ssize_t count = read(fd, ptr, size);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return false;
        }
        ptr += count;
        size -= count;
    }
    return true;
}

static bool write_full(int fd, const void * data, size_t size) {
    const uint8_t * ptr = static_cast<const uint8_t *>(data);
    while (size > 0) {
        const ssize_t count = write(fd, ptr, size);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            return false;
        }
        ptr += count;
        size -= count;
    }
    return true;
}

template<typename T>
static bool vectors_equal(const std::vector<T> & lhs, const std::vector<T> & rhs) {
    return lhs.size() == rhs.size() && (lhs.empty() || memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(T)) == 0);
}

static void trace_result(const char * result, const std::string & path) {
    if (cache_config().trace) {
        fprintf(stderr, "RKNPU weight cache: %s '%s'\n", result, path.c_str());
    }
}

} // namespace

namespace rknpu2_weight_cache {

bool enabled() {
    return cache_config().enabled;
}

bool rebuild_requested() {
    return cache_config().rebuild;
}

Hash128 hash_source(const void * data, size_t size) {
    return hash_bytes(data, size, { HASH_SEED_LO, HASH_SEED_HI });
}

bool load(const Descriptor & descriptor, uint8_t * packed_data, std::vector<float> & scales, std::vector<float> & signs) {
    if (!enabled() || rebuild_requested()) {
        return false;
    }

    const Hash128 key_hash = descriptor_hash(descriptor);
    const std::string path = cache_path(descriptor, key_hash);
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) {
        trace_result("miss", path);
        return false;
    }

    uint64_t file_size;
    struct stat st;
    CacheHeader header;
    bool valid = expected_file_size(descriptor, file_size) && fstat(fd, &st) == 0 && st.st_size >= 0 &&
                 (uint64_t) st.st_size == file_size && read_full(fd, &header, sizeof(header)) &&
                 header_matches(header, descriptor, key_hash, file_size);

    std::vector<int32_t> cores(descriptor.active_cores.size());
    std::vector<SegmentK> k_segments(descriptor.k_segments.size());
    std::vector<SegmentN> n_segments(descriptor.n_segments.size());
    scales.resize(valid ? descriptor.scale_count : 0);
    signs.resize(valid ? descriptor.sign_count : 0);

    if (valid) valid = read_full(fd, cores.data(), cores.size() * sizeof(cores[0]));
    if (valid) valid = read_full(fd, k_segments.data(), k_segments.size() * sizeof(k_segments[0]));
    if (valid) valid = read_full(fd, n_segments.data(), n_segments.size() * sizeof(n_segments[0]));
    if (valid) valid = vectors_equal(cores, descriptor.active_cores) && vectors_equal(k_segments, descriptor.k_segments) && vectors_equal(n_segments, descriptor.n_segments);
    if (valid) valid = read_full(fd, scales.data(), scales.size() * sizeof(scales[0]));
    if (valid) valid = read_full(fd, signs.data(), signs.size() * sizeof(signs[0]));
    if (valid) valid = read_full(fd, packed_data, descriptor.packed_size);

    if (close(fd) != 0) {
        valid = false;
    }

    if (valid) {
        const Hash128 actual = payload_hash(descriptor, scales, signs, packed_data);
        valid = actual.lo == header.payload_hash_lo && actual.hi == header.payload_hash_hi;
    }

    if (!valid) {
        scales.clear();
        signs.clear();
        trace_result("invalid", path);
        return false;
    }

    trace_result("hit", path);
    return true;
}

void store(const Descriptor & descriptor, const uint8_t * packed_data, const std::vector<float> & scales, const std::vector<float> & signs) {
    if (!enabled() || scales.size() != descriptor.scale_count || signs.size() != descriptor.sign_count) {
        return;
    }

    uint64_t file_size;
    if (!expected_file_size(descriptor, file_size)) {
        return;
    }

    const Hash128 key_hash = descriptor_hash(descriptor);
    const Hash128 data_hash = payload_hash(descriptor, scales, signs, packed_data);
    const CacheHeader header = make_header(descriptor, key_hash, data_hash, file_size);
    const std::string path = cache_path(descriptor, key_hash);
    std::string temporary = path + ".tmp.XXXXXX";
    std::vector<char> temporary_path(temporary.begin(), temporary.end());
    temporary_path.push_back('\0');

    const int fd = mkstemp(temporary_path.data());
    if (fd < 0) {
        trace_result("write failed", path);
        return;
    }

    fchmod(fd, 0644);
    bool valid = write_full(fd, &header, sizeof(header));
    if (valid) valid = write_full(fd, descriptor.active_cores.data(), descriptor.active_cores.size() * sizeof(descriptor.active_cores[0]));
    if (valid) valid = write_full(fd, descriptor.k_segments.data(), descriptor.k_segments.size() * sizeof(descriptor.k_segments[0]));
    if (valid) valid = write_full(fd, descriptor.n_segments.data(), descriptor.n_segments.size() * sizeof(descriptor.n_segments[0]));
    if (valid) valid = write_full(fd, scales.data(), scales.size() * sizeof(scales[0]));
    if (valid) valid = write_full(fd, signs.data(), signs.size() * sizeof(signs[0]));
    if (valid) valid = write_full(fd, packed_data, descriptor.packed_size);
    if (valid) valid = fdatasync(fd) == 0;
    if (close(fd) != 0) valid = false;
    if (valid) valid = rename(temporary_path.data(), path.c_str()) == 0;
    if (!valid) {
        unlink(temporary_path.data());
        trace_result("write failed", path);
        return;
    }

    trace_result("stored", path);
}

} // namespace rknpu2_weight_cache
