#include "rknpu2-quantization.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <csignal>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

static uint64_t checksum = 1469598103934665603ULL;
static size_t checks = 0;

static void require(bool condition, const char * message) {
    ++checks;
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

static void digest(uint8_t value) {
    checksum ^= value;
    checksum *= 1099511628211ULL;
}

static int reference(float input, float scale, bool array, int limit) {
    const float inverse = (scale == 0.0f || (array && std::abs(scale) < 1e-9f)) ? 0.0f : 1.0f / scale;
    const float scaled = input * inverse;
    if (std::isnan(scaled)) {
        return 0;
    }
    const double rounded = std::round((double)scaled);
    return rounded > limit ? limit : rounded < -limit ? -limit : (int)rounded;
}

static std::vector<float> make_input(size_t n) {
    static const float edge[] = {
        -0.5f, 0.5f, -1.5f, 1.5f, -2.5f, 2.5f, -6.5f, 6.5f,
        -127.5f, 127.5f, -128.0f, 128.0f, -256.0f, 256.0f, -1e20f, 1e20f,
        -std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
        -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(), -0.0f, 0.0f,
        std::nextafter(0.5f, 0.0f), std::nextafter(-0.5f, 0.0f),
        std::nextafter(0.5f, 1.0f), std::nextafter(-0.5f, -1.0f),
    };
    std::vector<float> input(n);
    for (size_t i = 0; i < n; ++i) {
        input[i] = i < std::size(edge) ? edge[i] : (float)((int)(i * 37 % 509) - 254) / 8.0f;
    }
    return input;
}

static void check_int8(const std::vector<float> & input, const std::vector<float> & scales) {
    const size_t n = input.size();
    const bool array = scales.size() != 1;
    std::vector<int8_t> output(n + 2, 0x55);
    rknpu2_quantization::quantize_fp32_to_int8(input.data(), output.data() + 1, n, scales.data(), scales.size());
    require(output.front() == 0x55 && output.back() == 0x55, "INT8 output bounds");
    for (size_t i = 0; i < n; ++i) {
        const float scale = scales[array ? i : 0];
        const int expected = reference(input[i], scale, array, 127);
        require(output[i + 1] == expected, "INT8 mathematical reference");
        if (!array) {
            int8_t scalar;
            rknpu2_quantization::quantize_fp32_to_int8(input.data() + i, &scalar, 1, &scale, 1);
            require(scalar == output[i + 1], "INT8 NEON and scalar agree");
        }
        digest((uint8_t)output[i + 1]);
    }
}

static void check_int4(const std::vector<float> & input, const std::vector<float> & scales) {
    const size_t n = input.size();
    const bool array = scales.size() != 1;
    std::vector<uint8_t> output(n / 2 + 2, 0x55);
    rknpu2_quantization::quantize_fp32_to_int4_packed(input.data(), output.data() + 1, n, scales.data(), scales.size());
    require(output.front() == 0x55 && output.back() == 0x55, "INT4 output bounds");
    for (size_t i = 0; i < n; i += 2) {
        const float s0 = scales[array ? i : 0];
        const float s1 = scales[array ? i + 1 : 0];
        const int q0 = reference(input[i], s0, array, 7);
        const int q1 = reference(input[i + 1], s1, array, 7);
        const uint8_t expected = (q0 & 15) | ((q1 & 15) << 4);
        require(output[i / 2 + 1] == expected, "INT4 adjacent low/high nibble reference");
        if (!array) {
            uint8_t scalar;
            rknpu2_quantization::quantize_fp32_to_int4_packed(input.data() + i, &scalar, 2, &s0, 1);
            require(scalar == output[i / 2 + 1], "INT4 NEON and scalar agree");
        }
        digest(output[i / 2 + 1]);
    }
}

static void check_explicit_rounding() {
    const float scale = 1.0f;
    const float input[] = {-0.5f, 0.5f, -1.5f, 1.5f, -2.5f, 2.5f, -6.5f, 6.5f,
                           7.5f, -7.5f, 127.5f, -127.5f, 128.0f, -128.0f, 256.0f, -256.0f,
                           0.5f, -0.5f};
    const int8_t expected8[] = {-1, 1, -2, 2, -3, 3, -7, 7, 8, -8, 127, -127, 127, -127, 127, -127, 1, -1};
    const uint8_t expected4[] = {0x1f, 0x2e, 0x3d, 0x79, 0x97, 0x97, 0x97, 0x97, 0xf1};
    int8_t output8[std::size(input)];
    uint8_t output4[std::size(input) / 2];
    rknpu2_quantization::quantize_fp32_to_int8(input, output8, std::size(input), &scale, 1);
    rknpu2_quantization::quantize_fp32_to_int4_packed(input, output4, std::size(input), &scale, 1);
    require(std::equal(std::begin(output8), std::end(output8), std::begin(expected8)), "literal INT8 ties-away and saturation");
    require(std::equal(std::begin(output4), std::end(output4), std::begin(expected4)), "literal INT4 ties-away, nibble order and tail");
}

static void check_invalid_arguments(int mode) {
    const pid_t child = fork();
    require(child >= 0, "fork for invalid argument test");
    if (child == 0) {
        const rlimit core_limit = {0, 0};
        setrlimit(RLIMIT_CORE, &core_limit);
        if (freopen("/dev/null", "w", stderr) == nullptr) {
            _exit(1);
        }
        const float input[4] = {};
        const float scales[4] = {1, 1, 1, 1};
        int8_t output8[4] = {};
        uint8_t output4[2] = {};
        switch (mode) {
            case 0:
                rknpu2_quantization::quantize_fp32_to_int8(input, output8, 4, scales, 2);
                break;
            case 1:
                rknpu2_quantization::quantize_fp32_to_int4_packed(input, output4, 4, scales, 2);
                break;
            case 2:
                rknpu2_quantization::quantize_fp32_to_int4_packed(input, output4, 3, scales, 1);
                break;
            case 3:
                rknpu2_quantization::quantize_fp32_to_int8(input, output8, 4, nullptr, 1);
                break;
        }
        _exit(1);
    }
    int status = 0;
    require(waitpid(child, &status, 0) == child, "wait for invalid argument test");
    require(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT, "invalid arguments abort");
}

static void check_constraints() {
    for (int mode = 0; mode < 4; ++mode) {
        check_invalid_arguments(mode);
    }
    rknpu2_quantization::quantize_fp32_to_int8(nullptr, nullptr, 0, nullptr, 0);
    rknpu2_quantization::quantize_fp32_to_int4_packed(nullptr, nullptr, 0, nullptr, 0);
}

static void check_conversion() {
    const float input[] = {0, -0.0f, 1, -1, 0.5f, 65504, 1e-5f, 1e-8f, 1.00048828125f, -3.5f};
    uint16_t fp16[std::size(input)];
    rknpu2_quantization::convert_fp32_to_fp16(input, fp16, std::size(input));
    for (size_t i = 0; i < std::size(input); ++i) {
        require(fp16[i] == GGML_FP32_TO_FP16(input[i]), "FP16 existing conversion and tail");
    }
    const int16_t i16[] = {-32768, -1, 0, 1, 32767};
    const int32_t i32[] = {-2147483647 - 1, -1, 0, 1, 2147483647};
    float output[5];
    rknpu2_quantization::dequantize_int16_to_fp32(i16, output, 5, 0.25f);
    for (size_t i = 0; i < 5; ++i) {
        require(output[i] == (float)i16[i] * 0.25f, "INT16 dequantization and tail");
    }
    rknpu2_quantization::dequantize_int32_to_fp32(i32, output, 5, 0.25f);
    for (size_t i = 0; i < 5; ++i) {
        require(output[i] == (float)i32[i] * 0.25f, "INT32 dequantization and tail");
    }
}

int main() {
    check_explicit_rounding();
    const size_t lengths[] = {1, 2, 14, 15, 16, 17, 18, 30, 31, 32, 33, 34, 63, 64, 66, 257, 258};
    const float scalar_scales[] = {1, 2, 0.5f, -1, 0, 1e-12f};
    const float array_scales[] = {1, 2, -1, 0.5f, 0, 1e-12f, 1e-9f, -1e-12f};
    for (size_t n : lengths) {
        const std::vector<float> input = make_input(n);
        for (float scale : scalar_scales) {
            const std::vector<float> scales = {scale};
            check_int8(input, scales);
            if (n % 2 == 0) {
                check_int4(input, scales);
            }
        }
        if (n > 1) {
            std::vector<float> scales(n);
            for (size_t i = 0; i < n; ++i) {
                scales[i] = array_scales[i % std::size(array_scales)];
            }
            check_int8(input, scales);
            if (n % 2 == 0) {
                check_int4(input, scales);
            }
        }
    }
    check_constraints();
    check_conversion();
    std::printf("PASS checks=%zu checksum=%016llx\n", checks, (unsigned long long)checksum);
}
