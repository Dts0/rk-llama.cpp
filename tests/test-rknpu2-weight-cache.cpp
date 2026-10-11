#include "rknpu2-weight-cache.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <limits>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

using namespace rknpu2_weight_cache;

#define CHECK(condition) do { if (!(condition)) { fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #condition); return false; } } while (0)

static std::vector<std::string> cache_files(const std::string & directory) {
    std::vector<std::string> result;
    DIR * dir = opendir(directory.c_str());
    if (dir == nullptr) {
        return result;
    }
    while (dirent * entry = readdir(dir)) {
        const std::string name = entry->d_name;
        if (name.size() >= 8 && name.substr(name.size() - 8) == ".rknpu2w") {
            result.push_back(directory + "/" + name);
        }
    }
    closedir(dir);
    std::sort(result.begin(), result.end());
    return result;
}

static Descriptor make_descriptor(const std::vector<uint8_t> & source, const std::vector<float> & signs) {
    Descriptor descriptor;
    descriptor.source_hash = hash_source(source.data(), source.size());
    descriptor.source_size = source.size();
    descriptor.packed_size = 48;
    descriptor.scale_count = 12;
    descriptor.sign_count = 8;
    descriptor.hadamard_sign_hash = hash_source(signs.data(), signs.size() * sizeof(float));
    descriptor.scale_scheme = SCALE_PER_CHANNEL_ABSMAX;
    descriptor.seed_scheme = SEED_TENSOR_NAME_STD_HASH_MT19937;
    descriptor.tensor_name = "blk.0/attn_q.weight";
    descriptor.tensor_type = 2;
    descriptor.ne = { 8, 6, 1, 1 };
    descriptor.device_name = "RK3576";
    descriptor.runtime_version = "test-runtime-1";
    descriptor.pipeline_name = "test-w8a16";
    descriptor.active_cores = { 0, 1, 2 };
    descriptor.k_segments = { { 0, 4 }, { 4, 4 } };
    descriptor.n_segments = { { 0, 2, 0 }, { 2, 4, 1 }, { 6, 0, 2 } };
    descriptor.max_k_limit = 4;
    descriptor.npu_type_a = 16;
    descriptor.npu_type_b = 8;
    descriptor.npu_type_c = 32;
    descriptor.matmul_type = 1;
    descriptor.k_align = 4;
    descriptor.n_align = 2;
    descriptor.effective_k = 4;
    descriptor.use_hadamard = 1;
    return descriptor;
}

static std::vector<uint8_t> read_file(const std::string & path) {
    std::ifstream stream(path, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

static bool write_file(const std::string & path, const std::vector<uint8_t> & bytes) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    return stream.good();
}

static bool run_enabled(const std::string & directory) {
    CHECK(enabled());
    CHECK(!rebuild_requested());
    const std::vector<uint8_t> source = { 1, 2, 3, 4, 5, 6 };
    const std::vector<float> signs = { -1, 1, -1, -1, 1, 1, -1, 1 };
    Descriptor descriptor = make_descriptor(source, signs);
    std::vector<uint8_t> packed(48);
    for (size_t i = 0; i < packed.size(); ++i) packed[i] = static_cast<uint8_t>(i * 3);
    const std::vector<float> scales = { 0.5f, 0.25f, 1, 2, 3, 4, 0.5f, 0.75f, 5, 6, 7, 8 };
    const std::vector<float> empty;
    std::vector<uint8_t> restored(packed.size(), 0);
    std::vector<float> loaded_scales = { 99 }, loaded_signs = { 98 }, loaded_u = { 97 }, loaded_v = { 96 };

    CHECK(!load(descriptor, restored.data(), loaded_scales, loaded_signs, loaded_u, loaded_v));
    CHECK(loaded_scales == std::vector<float>{ 99 });
    store(descriptor, packed.data(), scales, signs, empty, empty);
    auto paths = cache_files(directory);
    CHECK(paths.size() == 1);
    CHECK(load(descriptor, restored.data(), loaded_scales, loaded_signs, loaded_u, loaded_v));
    CHECK(restored == packed && loaded_scales == scales && loaded_signs == signs && loaded_u.empty() && loaded_v.empty());
    const std::string path = paths.front();
    const std::vector<uint8_t> original = read_file(path);
    CHECK(original.size() > packed.size());

    Descriptor wrong = descriptor;
    wrong.scale_count++;
    CHECK(!load(wrong, restored.data(), loaded_scales, loaded_signs, loaded_u, loaded_v));
    store(wrong, packed.data(), scales, signs, empty, empty);
    wrong = descriptor;
    wrong.scale_scheme = static_cast<ScaleScheme>(99);
    CHECK(!load(wrong, restored.data(), loaded_scales, loaded_signs, loaded_u, loaded_v));
    store(wrong, packed.data(), scales, signs, empty, empty);
    wrong = descriptor;
    wrong.seed_scheme = SEED_NONE;
    CHECK(!load(wrong, restored.data(), loaded_scales, loaded_signs, loaded_u, loaded_v));
    wrong = descriptor;
    wrong.hadamard_sign_hash.lo++;
    CHECK(!load(wrong, restored.data(), loaded_scales, loaded_signs, loaded_u, loaded_v));
    store(wrong, packed.data(), scales, signs, empty, empty);
    wrong = descriptor;
    wrong.runtime_version += "-changed";
    CHECK(!load(wrong, restored.data(), loaded_scales, loaded_signs, loaded_u, loaded_v));
    store(descriptor, packed.data(), empty, signs, empty, empty);
    CHECK(cache_files(directory).size() == 1);
    CHECK(read_file(path) == original);
    wrong = descriptor;
    wrong.packed_size = std::numeric_limits<uint64_t>::max();
    CHECK(!load(wrong, restored.data(), loaded_scales, loaded_signs, loaded_u, loaded_v));
    store(wrong, packed.data(), scales, signs, empty, empty);
    CHECK(cache_files(directory).size() == 1);

    std::vector<uint8_t> damaged = original;
    // scale_count starts after the fixed prefix, dimensions, source_size and packed_size.
    uint64_t corrupt_scale_count = descriptor.scale_count + 1;
    memcpy(damaged.data() + 72, &corrupt_scale_count, sizeof(corrupt_scale_count));
    CHECK(write_file(path, damaged));
    CHECK(!load(descriptor, restored.data(), loaded_scales, loaded_signs, loaded_u, loaded_v));
    damaged = original;
    damaged.back() ^= 0x40;
    CHECK(write_file(path, damaged));
    CHECK(!load(descriptor, restored.data(), loaded_scales, loaded_signs, loaded_u, loaded_v));
    CHECK(loaded_scales == scales && loaded_signs == signs);
    damaged = original;
    damaged.resize(damaged.size() - 1);
    CHECK(write_file(path, damaged));
    CHECK(!load(descriptor, restored.data(), loaded_scales, loaded_signs, loaded_u, loaded_v));
    damaged = original;
    damaged[7] = '1';
    uint32_t old_version = 1;
    memcpy(damaged.data() + 8, &old_version, sizeof(old_version));
    CHECK(write_file(path, damaged));
    CHECK(!load(descriptor, restored.data(), loaded_scales, loaded_signs, loaded_u, loaded_v));
    CHECK(read_file(path) == damaged);
    CHECK(write_file(path, original));
    CHECK(load(descriptor, restored.data(), loaded_scales, loaded_signs, loaded_u, loaded_v));

    Descriptor lowrank = descriptor;
    lowrank.scale_scheme = SCALE_LOWRANK;
    lowrank.use_lowrank = 1;
    lowrank.scale_count = 0;
    lowrank.lowrank_u_count = 12;
    lowrank.lowrank_v_count = 16;
    const std::vector<float> u = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const std::vector<float> v = { 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 };
    CHECK(!load(lowrank, restored.data(), loaded_scales, loaded_signs, loaded_u, loaded_v));
    store(lowrank, packed.data(), empty, signs, u, v);
    CHECK(cache_files(directory).size() == 2);
    CHECK(load(lowrank, restored.data(), loaded_scales, loaded_signs, loaded_u, loaded_v));
    CHECK(restored == packed && loaded_scales.empty() && loaded_signs == signs && loaded_u == u && loaded_v == v);
    wrong = lowrank;
    wrong.lowrank_v_count++;
    CHECK(!load(wrong, restored.data(), loaded_scales, loaded_signs, loaded_u, loaded_v));
    store(wrong, packed.data(), empty, signs, u, v);
    CHECK(cache_files(directory).size() == 2);

    Descriptor fp16 = descriptor;
    fp16.tensor_name = "fp16.weight";
    fp16.scale_scheme = SCALE_NONE;
    fp16.scale_count = 0;
    fp16.use_hadamard = 0;
    fp16.sign_count = 0;
    fp16.seed_scheme = SEED_NONE;
    fp16.hadamard_sign_hash = {};
    CHECK(!load(fp16, restored.data(), loaded_scales, loaded_signs, loaded_u, loaded_v));
    store(fp16, packed.data(), empty, empty, empty, empty);
    CHECK(load(fp16, restored.data(), loaded_scales, loaded_signs, loaded_u, loaded_v));
    CHECK(restored == packed && loaded_scales.empty() && loaded_signs.empty() && loaded_u.empty() && loaded_v.empty());
    return true;
}

static bool run_inactive(const std::string & directory, bool rebuild) {
    CHECK(enabled() == rebuild);
    CHECK(rebuild_requested() == rebuild);
    const std::vector<uint8_t> source = { 1, 2, 3, 4, 5, 6 };
    const std::vector<float> signs = { -1, 1, -1, -1, 1, 1, -1, 1 };
    const Descriptor descriptor = make_descriptor(source, signs);
    std::vector<uint8_t> packed(48, 42);
    std::vector<float> scales(12, 1), loaded_scales, loaded_signs, u, v;
    CHECK(!load(descriptor, packed.data(), loaded_scales, loaded_signs, u, v));
    const size_t count_before = cache_files(directory).size();
    store(descriptor, packed.data(), scales, signs, u, v);
    CHECK(cache_files(directory).size() == count_before);
    CHECK(!load(descriptor, packed.data(), loaded_scales, loaded_signs, u, v));
    return true;
}

int main() {
    char directory_template[] = "/tmp/rknpu2-weight-cache-test-XXXXXX";
    char * directory = mkdtemp(directory_template);
    if (directory == nullptr) return 1;
    bool valid = true;
    for (int mode = 0; mode < 3 && valid; ++mode) {
        const pid_t child = fork();
        if (child == 0) {
            setenv("RKNPU_WEIGHT_CACHE_DIR", directory, 1);
            setenv("RKNPU_WEIGHT_CACHE", mode == 1 ? "0" : "1", 1);
            setenv("RKNPU_WEIGHT_CACHE_REBUILD", mode == 2 ? "1" : "0", 1);
            unsetenv("RKNPU_TRACE");
            const bool passed = mode == 0 ? run_enabled(directory) : run_inactive(directory, mode == 2);
            _exit(passed ? 0 : 1);
        }
        if (child < 0) {
            valid = false;
        } else {
            int status = 0;
            valid = waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0;
        }
    }
    for (const auto & path : cache_files(directory)) unlink(path.c_str());
    rmdir(directory);
    if (valid) puts("rknpu2 weight cache tests passed");
    return valid ? 0 : 1;
}
