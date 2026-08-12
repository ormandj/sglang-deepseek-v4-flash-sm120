# DeepSeek-V4-Flash-0731 on SGLang for RTX PRO 6000 Blackwell (SM120).
#
# v0.2.0 is a clean reimage. It starts from the current official CUDA 13 /
# Torch 2.13 nightly, advances both source trees to the audited main commits,
# and applies only the required model-support and correctness carries recorded
# in stack.lock.json. Performance experiments from the v0.1.0 line are absent.
ARG DSV4_0731_RELEASE_VERSION=0.2.0
ARG DSV4_0731_RELEASE_CANDIDATE=0
ARG DSV4_0731_CACHE_SCHEMA=v10
ARG DSV4_0731_SGLANG_BASE=lmsysorg/sglang:nightly-dev-cu13-20260812-c7c03ec5@sha256:d7538b2bae8aff4b00b826442f7abd69d45ded936bc16fd0a493a2466df52050
ARG DSV4_0731_SGLANG_BASE_HEAD=c7c03ec53b1e664c2d415db4f02e43f86661f31d
ARG DSV4_0731_SGLANG_MAIN_HEAD=dc5f6c488317645d96dc630b1f410e4dfb6f9667
ARG DSV4_0731_SGLANG_MAIN_TREE=d117eca486f68ecbdf7d6168a47cfab7e2c8110b
ARG DSV4_0731_SGLANG_EFFECTIVE_TREE=8c336f4426844b2028938f7c542c0a403d37b804
ARG DSV4_0731_SGLANG_INTEGRATION_HEAD=f5356c7daec4565c2a0a06f9499b775545937db3
ARG DSV4_0731_SGLANG_PR29927_HEAD=e5ea881c5dd487acef17d58ba1a9d2b7ecfeee91
ARG DSV4_0731_SGLANG_PR33614_HEAD=fca0998feda2bfc2a735286d34d354b979850d72
ARG DSV4_0731_SGLANG_PR32686_HEAD=15c0902eefc59cb8aae919d16d4a0cf60f1a9a2b
ARG DSV4_0731_SGLANG_PR33568_HEAD=cab45a29997f8898e076e9253741a5119a401db0
ARG DSV4_0731_SGLANG_PR33805_HEAD=26a2a3981798b8deb97d053d163fa3c48668e03f
ARG DSV4_0731_FLASHINFER_MAIN_HEAD=065971254bca6ad0509d775e5806de53b64ac7b9
ARG DSV4_0731_FLASHINFER_MAIN_TREE=5d0f69e41f414efba115dcd13c03cbd46e40d640
ARG DSV4_0731_FLASHINFER_EFFECTIVE_TREE=09b10c6dc66ca0c96c62a13dfa5ea3b63f1018e4
ARG DSV4_0731_FLASHINFER_INTEGRATION_HEAD=44f0789e872d6f3abea652dc3fb435b6e37a0001
ARG DSV4_0731_FLASHINFER_PR3930_HEAD=e855cc25993d11d4707678d68fbde108d0578bef
ARG DSV4_0731_FLASHINFER_CUDART_RESOLVER_SOURCE_HEAD=28be41230a979ffdfd423706d1a4c7e82c6988eb
ARG DSV4_0731_FLASHINFER_VERSION=0.6.18.dev20260811
ARG DSV4_0731_FLASHINFER_CUBIN_URL=https://github.com/flashinfer-ai/flashinfer/releases/download/nightly-v0.6.18-20260811/flashinfer_cubin-0.6.18.dev20260811-py3-none-any.whl
ARG DSV4_0731_FLASHINFER_CUBIN_SHA256=300bc87236f646be7e46d3a382e5d74961d5685598ea4015954f76a3baf5b873
ARG IMAGE_SOURCE
ARG IMAGE_SOURCE_REVISION

FROM ${DSV4_0731_SGLANG_BASE} AS runtime
ARG DSV4_0731_RELEASE_VERSION
ARG DSV4_0731_RELEASE_CANDIDATE
ARG DSV4_0731_CACHE_SCHEMA
ARG DSV4_0731_SGLANG_BASE_HEAD
ARG DSV4_0731_SGLANG_MAIN_HEAD
ARG DSV4_0731_SGLANG_MAIN_TREE
ARG DSV4_0731_SGLANG_EFFECTIVE_TREE
ARG DSV4_0731_SGLANG_INTEGRATION_HEAD
ARG DSV4_0731_SGLANG_PR29927_HEAD
ARG DSV4_0731_SGLANG_PR33614_HEAD
ARG DSV4_0731_SGLANG_PR32686_HEAD
ARG DSV4_0731_SGLANG_PR33568_HEAD
ARG DSV4_0731_SGLANG_PR33805_HEAD
ARG DSV4_0731_FLASHINFER_MAIN_HEAD
ARG DSV4_0731_FLASHINFER_MAIN_TREE
ARG DSV4_0731_FLASHINFER_EFFECTIVE_TREE
ARG DSV4_0731_FLASHINFER_INTEGRATION_HEAD
ARG DSV4_0731_FLASHINFER_PR3930_HEAD
ARG DSV4_0731_FLASHINFER_CUDART_RESOLVER_SOURCE_HEAD
ARG DSV4_0731_FLASHINFER_VERSION
ARG DSV4_0731_FLASHINFER_CUBIN_URL
ARG DSV4_0731_FLASHINFER_CUBIN_SHA256
ARG IMAGE_SOURCE
ARG IMAGE_SOURCE_REVISION

COPY patches/sglang/0001-sglang-dsv4-0731-v0.2.0-rc.0.patch /tmp/sglang-release.patch
RUN set -e; cd /sgl-workspace/sglang; \
    git config --local --unset-all http.https://github.com/.extraheader || true; \
    git remote set-url origin https://github.com/sgl-project/sglang.git; \
    git fetch --depth=1 origin "${DSV4_0731_SGLANG_MAIN_HEAD}"; \
    git checkout --detach FETCH_HEAD; \
    git reset --hard "${DSV4_0731_SGLANG_MAIN_HEAD}"; \
    test "$(git rev-parse HEAD)" = "${DSV4_0731_SGLANG_MAIN_HEAD}"; \
    test "$(git rev-parse HEAD^{tree})" = "${DSV4_0731_SGLANG_MAIN_TREE}"; \
    git apply --index --binary /tmp/sglang-release.patch; \
    test "$(git write-tree)" = "${DSV4_0731_SGLANG_EFFECTIVE_TREE}"; \
    uv run --no-project python -m compileall -q \
      python/sglang/kernels/ops/attention/flash_mla_sm120.py \
      python/sglang/srt/distributed/bootstrap.py \
      python/sglang/srt/distributed/parallel_state.py \
      python/sglang/srt/entrypoints/openai \
      python/sglang/srt/layers/attention/deepseek_v4_backend.py \
      python/sglang/srt/layers/attention/dsv4 \
      python/sglang/srt/layers/deep_gemm_wrapper \
      python/sglang/srt/layers/moe/moe_runner/deep_gemm.py \
      python/sglang/srt/layers/moe/moe_runner/deep_gemm_sm120.py \
      python/sglang/srt/models/deepseek_v4.py \
      python/sglang/srt/speculative/dspark_components \
      python/sglang/srt/speculative/spec_utils.py; \
    rm /tmp/sglang-release.patch

COPY patches/flashinfer/0001-flashinfer-dsv4-0731-v0.2.0-rc.0.patch /tmp/flashinfer-release.patch
RUN set -e; \
    git init /tmp/flashinfer-src; \
    git -C /tmp/flashinfer-src remote add origin https://github.com/flashinfer-ai/flashinfer.git; \
    git -C /tmp/flashinfer-src fetch --depth=1 origin "${DSV4_0731_FLASHINFER_MAIN_HEAD}"; \
    git -C /tmp/flashinfer-src checkout --detach FETCH_HEAD; \
    test "$(git -C /tmp/flashinfer-src rev-parse HEAD)" = "${DSV4_0731_FLASHINFER_MAIN_HEAD}"; \
    test "$(git -C /tmp/flashinfer-src rev-parse HEAD^{tree})" = "${DSV4_0731_FLASHINFER_MAIN_TREE}"; \
    git -C /tmp/flashinfer-src apply --index --binary /tmp/flashinfer-release.patch; \
    test "$(git -C /tmp/flashinfer-src write-tree)" = "${DSV4_0731_FLASHINFER_EFFECTIVE_TREE}"; \
    git -C /tmp/flashinfer-src submodule update --init --depth=1 \
      3rdparty/cccl 3rdparty/cutlass 3rdparty/spdlog; \
    cd /tmp/flashinfer-src; \
    PYTHONPATH=/tmp/flashinfer-src uv run --no-project pytest -q tests/comm/test_cuda_ipc.py; \
    cd /; \
    SOURCE_DATE_EPOCH="$(git -C /tmp/flashinfer-src show -s --format=%ct HEAD)" \
      BUILD_NVEP=0 \
      FLASHINFER_DEV_RELEASE_SUFFIX="${DSV4_0731_FLASHINFER_VERSION##*.dev}" \
      uv build --wheel --out-dir /tmp/flashinfer-wheel /tmp/flashinfer-src; \
    uv pip uninstall --system --break-system-packages flashinfer-python flashinfer-jit-cache || true; \
    uv pip install --system --break-system-packages --no-cache --no-deps --reinstall \
      "${DSV4_0731_FLASHINFER_CUBIN_URL}#sha256=${DSV4_0731_FLASHINFER_CUBIN_SHA256}"; \
    uv pip install --system --break-system-packages --no-cache --no-deps \
      /tmp/flashinfer-wheel/flashinfer_python-*.whl; \
    uv run --no-project python -c "import flashinfer, importlib.metadata as m; expected='${DSV4_0731_FLASHINFER_VERSION}'; assert flashinfer.__version__ == expected, flashinfer.__version__; assert flashinfer.__git_version__ == '${DSV4_0731_FLASHINFER_MAIN_HEAD}', flashinfer.__git_version__; assert m.version('flashinfer-cubin') == expected, m.version('flashinfer-cubin'); assert all(d.metadata['Name'] != 'flashinfer-jit-cache' for d in m.distributions()); print('flashinfer', expected)"; \
    rm -rf /tmp/flashinfer-src /tmp/flashinfer-wheel /tmp/flashinfer-release.patch

RUN set -e; cd /sgl-workspace/sglang; \
    uv run --no-project python -c "import deep_gemm, importlib.metadata as m; assert m.version('sgl-deep-gemm') == '0.1.5.post2', m.version('sgl-deep-gemm'); missing=[n for n in ('m_grouped_fp8_fp4_gemm_nt_contiguous', 'fp8_paged_mqa_logits', 'fp8_fp4_paged_mqa_logits', 'tf32_hc_prenorm_gemm') if not hasattr(deep_gemm, n)]; assert not missing, missing; print('sgl-deep-gemm', m.version('sgl-deep-gemm'))"; \
    uv run --no-project python -c "import sglang; from sglang.srt.entrypoints.openai import encoding_dsv4; print('sglang', sglang.__version__, 'dsv4 encoding OK')"; \
    uv run --no-project python test/registered/unit/layers/deep_gemm_wrapper/test_compile_utils.py; \
    uv run --no-project python test/registered/unit/entrypoints/openai/test_serving_chat.py; \
    uv run --no-project python test/registered/unit/spec/test_decode_bookkeeping_ownership.py; \
    uv run --no-project python test/registered/unit/speculative/test_spec_prepare_swa_eviction.py

ENV SGLANG_BUILD_COMMIT=${DSV4_0731_SGLANG_MAIN_HEAD} \
    SGLANG_BUILD_TREE=${DSV4_0731_SGLANG_EFFECTIVE_TREE} \
    FLASHINFER_VERSION=${DSV4_0731_FLASHINFER_VERSION} \
    FLASHINFER_CUDA_ARCH_LIST=12.0f
LABEL org.opencontainers.image.title="sglang-deepseek-v4-flash-sm120" \
      org.opencontainers.image.description="SGLang for DeepSeek-V4-Flash-0731 on RTX PRO 6000 Blackwell (SM120)" \
      org.opencontainers.image.source=${IMAGE_SOURCE} \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version=${DSV4_0731_RELEASE_VERSION} \
      org.opencontainers.image.revision=${IMAGE_SOURCE_REVISION} \
      ai.release.candidate=rc.${DSV4_0731_RELEASE_CANDIDATE} \
      ai.release.cache-schema=${DSV4_0731_CACHE_SCHEMA} \
      ai.sglang.base.head=${DSV4_0731_SGLANG_BASE_HEAD} \
      ai.sglang.main.head=${DSV4_0731_SGLANG_MAIN_HEAD} \
      ai.sglang.main.tree=${DSV4_0731_SGLANG_MAIN_TREE} \
      ai.sglang.effective.tree=${DSV4_0731_SGLANG_EFFECTIVE_TREE} \
      ai.sglang.integration.head=${DSV4_0731_SGLANG_INTEGRATION_HEAD} \
      ai.sglang.pr29927.head=${DSV4_0731_SGLANG_PR29927_HEAD} \
      ai.sglang.pr33614.head=${DSV4_0731_SGLANG_PR33614_HEAD} \
      ai.sglang.pr32686.head=${DSV4_0731_SGLANG_PR32686_HEAD} \
      ai.sglang.pr33568.head=${DSV4_0731_SGLANG_PR33568_HEAD} \
      ai.sglang.pr33805.head=${DSV4_0731_SGLANG_PR33805_HEAD} \
      ai.flashinfer.version=${DSV4_0731_FLASHINFER_VERSION} \
      ai.flashinfer.main.head=${DSV4_0731_FLASHINFER_MAIN_HEAD} \
      ai.flashinfer.main.tree=${DSV4_0731_FLASHINFER_MAIN_TREE} \
      ai.flashinfer.effective.tree=${DSV4_0731_FLASHINFER_EFFECTIVE_TREE} \
      ai.flashinfer.integration.head=${DSV4_0731_FLASHINFER_INTEGRATION_HEAD} \
      ai.flashinfer.pr3930.head=${DSV4_0731_FLASHINFER_PR3930_HEAD} \
      ai.flashinfer.local.cudart-resolver.source=${DSV4_0731_FLASHINFER_CUDART_RESOLVER_SOURCE_HEAD} \
      ai.flashinfer.cubin.sha256=${DSV4_0731_FLASHINFER_CUBIN_SHA256} \
      ai.flashinfer.sm120.module=persistent-runtime-jit
