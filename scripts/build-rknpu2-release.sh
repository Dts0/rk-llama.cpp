#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build_rknpu2_release}"
RELEASE_DIR="${RELEASE_DIR:-${ROOT_DIR}/release}"
JOBS="${JOBS:-$(nproc)}"
RELEASE_DATE="${RELEASE_DATE:-$(date +%y%m%d)}"
SKIP_BUILD=0

usage() {
    cat <<'EOF'
Build and package the Linux aarch64 RK3576/RK3588 llama-server release.

Usage:
  scripts/build-rknpu2-release.sh [options]

Options:
  --build-dir DIR    CMake build directory (default: build_rknpu2_release)
  --release-dir DIR  Output directory (default: release)
  --jobs N           Parallel build jobs (default: number of online CPUs)
  --date YYMMDD      Override the release date used in the file name
  --skip-build       Package an existing build after validating its Git SHA
  -h, --help         Show this help

Environment equivalents:
  BUILD_DIR, RELEASE_DIR, JOBS, RELEASE_DATE, SOURCE_DATE_EPOCH
EOF
}

absolute_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *)  printf '%s/%s\n' "${ROOT_DIR}" "$1" ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)
            [[ $# -ge 2 ]] || { printf '%s\n' 'Missing value for --build-dir' >&2; exit 2; }
            BUILD_DIR="$2"
            shift 2
            ;;
        --release-dir)
            [[ $# -ge 2 ]] || { printf '%s\n' 'Missing value for --release-dir' >&2; exit 2; }
            RELEASE_DIR="$2"
            shift 2
            ;;
        --jobs)
            [[ $# -ge 2 ]] || { printf '%s\n' 'Missing value for --jobs' >&2; exit 2; }
            JOBS="$2"
            shift 2
            ;;
        --date)
            [[ $# -ge 2 ]] || { printf '%s\n' 'Missing value for --date' >&2; exit 2; }
            RELEASE_DATE="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ "${JOBS}" =~ ^[1-9][0-9]*$ ]] || { printf 'Invalid job count: %s\n' "${JOBS}" >&2; exit 2; }
[[ "${RELEASE_DATE}" =~ ^[0-9]{6}$ ]] || { printf 'Invalid release date: %s (expected YYMMDD)\n' "${RELEASE_DATE}" >&2; exit 2; }

BUILD_DIR="$(absolute_path "${BUILD_DIR}")"
RELEASE_DIR="$(absolute_path "${RELEASE_DIR}")"

for command in chmod cmake cmp file find git grep install ldd mktemp mv nproc readlink sha256sum tar uname unlink xargs; do
    command -v "${command}" >/dev/null 2>&1 || {
        printf 'Required command not found: %s\n' "${command}" >&2
        exit 1
    }
done

if [[ "$(uname -s)" != "Linux" || ! "$(uname -m)" =~ ^(aarch64|arm64)$ ]]; then
    printf '%s\n' 'This release script must run on Linux aarch64.' >&2
    exit 1
fi

if ! git -C "${ROOT_DIR}" diff --quiet --ignore-submodules -- ||
   ! git -C "${ROOT_DIR}" diff --cached --quiet --ignore-submodules --; then
    printf '%s\n' 'Refusing to package tracked source changes that are not committed.' >&2
    printf '%s\n' 'Commit or restore them first so the release Git SHA identifies its contents.' >&2
    exit 1
fi

FULL_SHA="$(git -C "${ROOT_DIR}" rev-parse HEAD)"
SHORT_SHA="$(git -C "${ROOT_DIR}" rev-parse --short=9 HEAD)"
VERSION_TAG="v${RELEASE_DATE}"
PACKAGE_NAME="rk-llama.cpp-${VERSION_TAG}-${SHORT_SHA}"
ARCHIVE_NAME="${PACKAGE_NAME}.tar.gz"
RKNNRT="${ROOT_DIR}/ggml/src/ggml-rknpu2/libs/librknnrt.so"

[[ -f "${RKNNRT}" ]] || { printf 'RKNN runtime not found: %s\n' "${RKNNRT}" >&2; exit 1; }

if [[ ${SKIP_BUILD} -eq 0 ]]; then
    printf 'Configuring release build: %s\n' "${BUILD_DIR}"
    cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DGGML_NATIVE=OFF \
        -DGGML_CPU_ARM_ARCH=armv8-a+crc+crypto \
        -DGGML_RKNPU2=ON \
        -DGGML_RKNPU2_DEBUG=OFF \
        -DLLAMA_BUILD_COMMON=ON \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_UI=OFF \
        -DLLAMA_USE_PREBUILT_UI=OFF \
        -DLLAMA_BUILD_APP=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_OPENSSL=ON

    printf 'Building llama-server with %s jobs\n' "${JOBS}"
    cmake --build "${BUILD_DIR}" --target llama-server --parallel "${JOBS}"
fi

BUILD_BIN="${BUILD_DIR}/bin"
SERVER="${BUILD_BIN}/llama-server"

[[ -x "${SERVER}" ]] || { printf 'llama-server not found: %s\n' "${SERVER}" >&2; exit 1; }
file "${SERVER}" | grep -q 'ARM aarch64' || {
    printf 'Unexpected llama-server architecture: %s\n' "$(file "${SERVER}")" >&2
    exit 1
}

VERSION_OUTPUT="$(LD_LIBRARY_PATH="${BUILD_BIN}:${ROOT_DIR}/ggml/src/ggml-rknpu2/libs${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "${SERVER}" --version 2>&1)"
if [[ "${VERSION_OUTPUT}" != *"${SHORT_SHA}"* ]]; then
    printf '%s\n' 'The build does not match the current Git commit.' >&2
    printf 'Expected SHA: %s\n' "${SHORT_SHA}" >&2
    printf 'Version output: %s\n' "${VERSION_OUTPUT}" >&2
    exit 1
fi

PROJECT_LIBS=(
    libggml-base.so.0
    libggml-cpu.so.0
    libggml-rknpu2.so.0
    libggml.so.0
    libllama-common.so.0
    libllama-server-impl.so
    libllama.so.0
    libmtmd.so.0
)

for library in "${PROJECT_LIBS[@]}"; do
    [[ -e "${BUILD_BIN}/${library}" ]] || {
        printf 'Required build library not found: %s\n' "${BUILD_BIN}/${library}" >&2
        exit 1
    }
done

mkdir -p "${RELEASE_DIR}"
STAGING_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/rk-llama-release.XXXXXX")"
TEMP_ARCHIVE="${RELEASE_DIR}/.${ARCHIVE_NAME}.tmp.$$"
TEMP_CHECKSUM="${RELEASE_DIR}/.${ARCHIVE_NAME}.sha256.tmp.$$"

cleanup() {
    if [[ -n "${STAGING_ROOT:-}" && -d "${STAGING_ROOT}" ]]; then
        find "${STAGING_ROOT}" -depth -delete
    fi
    [[ ! -e "${TEMP_ARCHIVE:-}" ]] || unlink "${TEMP_ARCHIVE}"
    [[ ! -e "${TEMP_CHECKSUM:-}" ]] || unlink "${TEMP_CHECKSUM}"
}
trap cleanup EXIT

PACKAGE_DIR="${STAGING_ROOT}/${PACKAGE_NAME}"
mkdir -p "${PACKAGE_DIR}/bin" "${PACKAGE_DIR}/lib"

install -m 0755 "${SERVER}" "${PACKAGE_DIR}/bin/llama-server"
for library in "${PROJECT_LIBS[@]}"; do
    install -m 0755 "$(readlink -f "${BUILD_BIN}/${library}")" "${PACKAGE_DIR}/lib/${library}"
done
install -m 0755 "${RKNNRT}" "${PACKAGE_DIR}/lib/librknnrt.so"
install -m 0644 "${ROOT_DIR}/LICENSE" "${PACKAGE_DIR}/LICENSE"

cat > "${PACKAGE_DIR}/start.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SERVER="${ROOT_DIR}/bin/llama-server"

usage() {
    cat <<'HELP'
Usage:
  ./start.sh [model.gguf] [additional llama-server arguments]

Environment overrides:
  MODEL, RKNPU_DEVICE, HOST, PORT, THREADS, THREADS_BATCH,
  PARALLEL, CTX_SIZE, CACHE_RAM, BATCH_SIZE, UBATCH_SIZE

Examples:
  ./start.sh
  ./start.sh /path/to/model.gguf
  RKNPU_DEVICE=RK3588 PORT=8081 ./start.sh /path/to/model.gguf
HELP
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

detect_rknpu_device() {
    local compatible=""
    if [[ -r /proc/device-tree/compatible ]]; then
        compatible="$(tr '\0' '\n' < /proc/device-tree/compatible)"
    fi

    case "${compatible,,}" in
        *rk3588*) printf '%s\n' RK3588 ;;
        *rk3576*) printf '%s\n' RK3576 ;;
        *)        printf '%s\n' RK3576 ;;
    esac
}

export LD_LIBRARY_PATH="${ROOT_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export RKNPU_DEVICE="${RKNPU_DEVICE:-$(detect_rknpu_device)}"

case "${RKNPU_DEVICE}" in
    RK3576|RK3588) ;;
    *)
        printf 'Unsupported RKNPU_DEVICE: %s (expected RK3576 or RK3588)\n' "${RKNPU_DEVICE}" >&2
        exit 2
        ;;
esac

if [[ -n "${MODEL:-}" ]]; then
    model="${MODEL}"
elif [[ $# -gt 0 && "${1}" != -* ]]; then
    model="$1"
    shift
else
    model="/root/mnt/module/gguf/Qwen3.5-2B-Q8_0.gguf"
fi

if [[ ! -f "${model}" ]]; then
    printf 'Model not found: %s\n' "${model}" >&2
    printf '%s\n' 'Pass a GGUF path as the first argument or set MODEL.' >&2
    exit 2
fi

host="${HOST:-0.0.0.0}"
port="${PORT:-8080}"
threads="${THREADS:-4}"
threads_batch="${THREADS_BATCH:-8}"
parallel="${PARALLEL:-2}"
ctx_size="${CTX_SIZE:-32000}"
cache_ram="${CACHE_RAM:-2048}"
batch_size="${BATCH_SIZE:-512}"
ubatch_size="${UBATCH_SIZE:-512}"

printf 'Starting llama-server: device=%s model=%s listen=%s:%s\n' \
    "${RKNPU_DEVICE}" "${model}" "${host}" "${port}"

exec "${SERVER}" \
    -m "${model}" \
    --host "${host}" \
    --port "${port}" \
    --n-gpu-layers 99 \
    -tb "${threads_batch}" \
    -t "${threads}" \
    --parallel "${parallel}" \
    --slots \
    --ctx-size "${ctx_size}" \
    -ctk q8_0 \
    -ctv q8_0 \
    --kv-unified \
    --cache-ram "${cache_ram}" \
    -b "${batch_size}" \
    -ub "${ubatch_size}" \
    --no-warmup \
    -fa auto \
    --no-mmproj \
    --reasoning auto \
    --spec-type ngram-cache \
    --spec-draft-n-max 16 \
    --spec-draft-n-min 1 \
    "$@"
EOF
chmod 0755 "${PACKAGE_DIR}/start.sh"

cat > "${PACKAGE_DIR}/README.md" <<EOF
# RK llama.cpp runtime

Release: ${VERSION_TAG}
Git commit: ${FULL_SHA}
Architecture: Linux aarch64
Supported RKNPU targets: RK3576 and RK3588
CPU baseline: armv8-a+crc+crypto

## Start

The default model is \`/root/mnt/module/gguf/Qwen3.5-2B-Q8_0.gguf\`:

\`\`\`sh
./start.sh
\`\`\`

Specify another model as the first argument or with \`MODEL\`:

\`\`\`sh
./start.sh /path/to/model.gguf
MODEL=/path/to/model.gguf ./start.sh
\`\`\`

The script detects RK3576/RK3588 from the device tree. Override it when needed:

\`\`\`sh
RKNPU_DEVICE=RK3588 PORT=8081 ./start.sh /path/to/model.gguf
\`\`\`

Additional arguments are appended to the \`llama-server\` command:

\`\`\`sh
./start.sh /path/to/model.gguf --api-key secret
\`\`\`

The server listens on \`0.0.0.0:8080\` by default. This build does not include
the embedded Web UI; use the HTTP/OpenAI-compatible API endpoints.

The host system must provide glibc, libstdc++, libgcc, libgomp, and OpenSSL 3.
Run \`./start.sh --help\` for the available environment overrides.
EOF

(
    cd "${PACKAGE_DIR}"
    {
        find ./bin ./lib -type f -print
        printf '%s\n' ./LICENSE ./README.md ./start.sh
    } | LC_ALL=C sort | xargs sha256sum
) > "${PACKAGE_DIR}/MANIFEST.sha256"

(
    cd "${PACKAGE_DIR}"
    sha256sum --check MANIFEST.sha256
)

PACKAGE_LD_LIBRARY_PATH="${PACKAGE_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
PACKAGED_VERSION="$(LD_LIBRARY_PATH="${PACKAGE_LD_LIBRARY_PATH}" "${PACKAGE_DIR}/bin/llama-server" --version 2>&1)"
[[ "${PACKAGED_VERSION}" == *"${SHORT_SHA}"* ]] || {
    printf '%s\n' 'Packaged llama-server version verification failed.' >&2
    exit 1
}

DEPENDENCIES="$(LD_LIBRARY_PATH="${PACKAGE_LD_LIBRARY_PATH}" ldd "${PACKAGE_DIR}/bin/llama-server")"
if grep -q 'not found' <<< "${DEPENDENCIES}"; then
    printf '%s\n' 'Packaged llama-server has unresolved dependencies:' >&2
    printf '%s\n' "${DEPENDENCIES}" >&2
    exit 1
fi

cmp -s "$(readlink -f "${BUILD_BIN}/libggml-rknpu2.so.0")" "${PACKAGE_DIR}/lib/libggml-rknpu2.so.0" || {
    printf '%s\n' 'Packaged RKNPU backend does not match the current build.' >&2
    exit 1
}
"${PACKAGE_DIR}/start.sh" --help >/dev/null

SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git -C "${ROOT_DIR}" show -s --format=%ct HEAD)}"
[[ "${SOURCE_DATE_EPOCH}" =~ ^[0-9]+$ ]] || {
    printf 'Invalid SOURCE_DATE_EPOCH: %s\n' "${SOURCE_DATE_EPOCH}" >&2
    exit 2
}

tar \
    --sort=name \
    --format=gnu \
    --owner=0 \
    --group=0 \
    --numeric-owner \
    --mtime="@${SOURCE_DATE_EPOCH}" \
    -czf "${TEMP_ARCHIVE}" \
    -C "${STAGING_ROOT}" \
    "${PACKAGE_NAME}"

tar -tzf "${TEMP_ARCHIVE}" >/dev/null
mv -f "${TEMP_ARCHIVE}" "${RELEASE_DIR}/${ARCHIVE_NAME}"
TEMP_ARCHIVE=""

(
    cd "${RELEASE_DIR}"
    sha256sum "${ARCHIVE_NAME}"
) > "${TEMP_CHECKSUM}"
mv -f "${TEMP_CHECKSUM}" "${RELEASE_DIR}/${ARCHIVE_NAME}.sha256"
TEMP_CHECKSUM=""

printf '\nRelease created:\n'
printf '  %s\n' "${RELEASE_DIR}/${ARCHIVE_NAME}"
printf '  %s\n' "${RELEASE_DIR}/${ARCHIVE_NAME}.sha256"
printf '  %s\n' "${PACKAGED_VERSION}"
