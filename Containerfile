# DeepSeek-V4-Flash-0731 on SGLang for RTX PRO 6000 Blackwell (SM120).
#
# v0.7.0 refreshes the source heads and removes carries now provided by
# upstream while preserving the reviewed SM120-specific effective behavior.
ARG DSV4_0731_RELEASE_VERSION=0.7.0
ARG DSV4_0731_RELEASE_CANDIDATE=1
ARG DSV4_0731_CACHE_SCHEMA=v26
ARG DSV4_0731_SGLANG_BASE=lmsysorg/sglang:nightly-dev-cu13-20260818-c0b6474b@sha256:51e576f02368480c055c7aadb67590d82b172e2392123ce4cf4cc8251b2d8caf
ARG DSV4_0731_SGLANG_BASE_HEAD=c0b6474b43363c2f4bc60fe3d7817d393fb51d32
ARG DSV4_0731_SGLANG_MAIN_HEAD=7f8f030000b628ea2cb033e7457a13dd0ac80f99
ARG DSV4_0731_SGLANG_MAIN_TREE=f392a43569c7021f78a6046bcefd84d14f7a18d7
ARG DSV4_0731_SGLANG_EFFECTIVE_TREE=6696e98a1f037f16774eeea793c05d3eb1316d6d
ARG DSV4_0731_SGLANG_INTEGRATION_HEAD=678c5226279c416a42e8fa77dd8b9f0ccca09b71
ARG DSV4_0731_SGLANG_PR29927_HEAD=b19fd53b923614f4cbafc21ae99a853737ede9bd
ARG DSV4_0731_SGLANG_PR35116_HEAD=65eef2612efba5759aaebbabfae37763cf56a277
ARG DSV4_0731_SGLANG_PR35118_HEAD=23d2ac6f51f2696da42663b2160bc89754680ca9
ARG DSV4_0731_SGLANG_PR33614_HEAD=652e5ed0a5dfbd888c0c6657c20823e4af578140
ARG DSV4_0731_SGLANG_PR32686_HEAD=15c0902eefc59cb8aae919d16d4a0cf60f1a9a2b
ARG DSV4_0731_SGLANG_PR33568_HEAD=cab45a29997f8898e076e9253741a5119a401db0
ARG DSV4_0731_SGLANG_PR34018_HEAD=c3ffe8cfd3cf6cf9c30fc470cf7b76754954f3f0
ARG DSV4_0731_SGLANG_PR34528_HEAD=f28d875d121a1ec0ab879ef54873220c2ed23c6a
ARG DSV4_0731_SGLANG_PR35217_HEAD=8460babe10361637deee4c59bd7a1372eda51fc6
ARG DSV4_0731_FLASHINFER_MAIN_HEAD=5366177a074e27df7db527f5b744c77dfd748484
ARG DSV4_0731_FLASHINFER_MAIN_TREE=4395863927a93b3b84501143b05bf4d36921c01d
ARG DSV4_0731_FLASHINFER_EFFECTIVE_TREE=917a439a4cd74f5f0fa4f7dbb13543c606ffe346
ARG DSV4_0731_FLASHINFER_INTEGRATION_HEAD=fb61a80b2d5da03b119ae4e7a6424ede137ba2c1
ARG DSV4_0731_FLASHINFER_PR3930_HEAD=e855cc25993d11d4707678d68fbde108d0578bef
ARG DSV4_0731_FLASHINFER_CUDART_RESOLVER_SOURCE_HEAD=441a07a8b34b631345c942dab865ab0602cd1066
ARG DSV4_0731_FLASHINFER_VERSION=0.6.18.dev20260819
ARG DSV4_0731_FLASHINFER_CUBIN_URL=https://github.com/flashinfer-ai/flashinfer/releases/download/nightly-v0.6.18-20260819/flashinfer_cubin-0.6.18.dev20260819-py3-none-any.whl
ARG DSV4_0731_FLASHINFER_CUBIN_SHA256=277c3f2ef478dd8da5f315f21c3ce56c4437dbb47b170ceb4b47185f9c46560b
ARG DSV4_0731_DEEPGEMM_MAIN_HEAD=80b2c44b9ae95b90c1e0a1626a05b6c4f7f09f1f
ARG DSV4_0731_DEEPGEMM_MAIN_TREE=c27ad7254f740eb7635f2eea56da948e09883328
ARG DSV4_0731_DEEPGEMM_EFFECTIVE_TREE=b7e23a6fb5ac6571046cc12e85352d43af63f27d
ARG DSV4_0731_DEEPGEMM_INTEGRATION_HEAD=9e3b08724b54fbcfe340a5fecc3bbeb7ed173da9
ARG DSV4_0731_DEEPGEMM_VERSION=0.0.0+sm120jit5
ARG DSV4_0731_DEEPGEMM_PR76_HEAD=4900cbd750b4fb10bf756bd1be1f4357b66eac74
ARG DSV4_0731_DEEPGEMM_PR77_HEAD=c5dd8bfde3fb1ec8c8537e218e201dd0199189f1
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
ARG DSV4_0731_SGLANG_PR35116_HEAD
ARG DSV4_0731_SGLANG_PR35118_HEAD
ARG DSV4_0731_SGLANG_PR33614_HEAD
ARG DSV4_0731_SGLANG_PR32686_HEAD
ARG DSV4_0731_SGLANG_PR33568_HEAD
ARG DSV4_0731_SGLANG_PR34018_HEAD
ARG DSV4_0731_SGLANG_PR34528_HEAD
ARG DSV4_0731_SGLANG_PR35217_HEAD
ARG DSV4_0731_FLASHINFER_MAIN_HEAD
ARG DSV4_0731_FLASHINFER_MAIN_TREE
ARG DSV4_0731_FLASHINFER_EFFECTIVE_TREE
ARG DSV4_0731_FLASHINFER_INTEGRATION_HEAD
ARG DSV4_0731_FLASHINFER_PR3930_HEAD
ARG DSV4_0731_FLASHINFER_CUDART_RESOLVER_SOURCE_HEAD
ARG DSV4_0731_FLASHINFER_VERSION
ARG DSV4_0731_FLASHINFER_CUBIN_URL
ARG DSV4_0731_FLASHINFER_CUBIN_SHA256
ARG DSV4_0731_DEEPGEMM_MAIN_HEAD
ARG DSV4_0731_DEEPGEMM_MAIN_TREE
ARG DSV4_0731_DEEPGEMM_EFFECTIVE_TREE
ARG DSV4_0731_DEEPGEMM_INTEGRATION_HEAD
ARG DSV4_0731_DEEPGEMM_VERSION
ARG DSV4_0731_DEEPGEMM_PR76_HEAD
ARG DSV4_0731_DEEPGEMM_PR77_HEAD
ARG IMAGE_SOURCE
ARG IMAGE_SOURCE_REVISION

COPY patches/sglang/0001-sglang-dsv4-0731-v0.7.0-rc.1.patch /tmp/sglang-release.patch
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
    uv run --no-project --python /usr/bin/python python -m compileall -q \
      python/sglang/kernels/ops/attention/flash_mla_sm120.py \
      python/sglang/srt/distributed/bootstrap.py \
      python/sglang/srt/distributed/device_communicators/pcie_ipc_ar.py \
      python/sglang/srt/distributed/parallel_state.py \
      python/sglang/srt/environ.py \
      python/sglang/srt/entrypoints/warmup.py \
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

COPY patches/flashinfer/0001-flashinfer-dsv4-0731-v0.7.0-rc.1.patch /tmp/flashinfer-release.patch
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
    PYTHONPATH=/tmp/flashinfer-src uv run --no-project --python /usr/bin/python pytest -q \
      tests/comm/test_cuda_ipc.py \
      tests/comm/test_pcie_ipc_policy.py \
      tests/comm/test_pcie_ipc_tuning.py; \
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
    uv run --no-project --python /usr/bin/python python -c "import flashinfer, importlib.metadata as m; expected='${DSV4_0731_FLASHINFER_VERSION}'; assert flashinfer.__version__ == expected, flashinfer.__version__; assert flashinfer.__git_version__ == '${DSV4_0731_FLASHINFER_MAIN_HEAD}', flashinfer.__git_version__; assert m.version('flashinfer-cubin') == expected, m.version('flashinfer-cubin'); assert all(d.metadata['Name'] != 'flashinfer-jit-cache' for d in m.distributions()); print('flashinfer', expected)"; \
    rm -rf /tmp/flashinfer-src /tmp/flashinfer-wheel /tmp/flashinfer-release.patch

COPY patches/deepgemm/0001-deepgemm-sm120-v0.7.0-rc.1.patch /tmp/deepgemm-release.patch
RUN set -e; \
    git init /tmp/deepgemm-src; \
    git -C /tmp/deepgemm-src remote add origin https://github.com/sgl-project/DeepGEMM.git; \
    git -C /tmp/deepgemm-src fetch --depth=1 origin "${DSV4_0731_DEEPGEMM_MAIN_HEAD}"; \
    git -C /tmp/deepgemm-src checkout --detach FETCH_HEAD; \
    test "$(git -C /tmp/deepgemm-src rev-parse HEAD)" = "${DSV4_0731_DEEPGEMM_MAIN_HEAD}"; \
    test "$(git -C /tmp/deepgemm-src rev-parse HEAD^{tree})" = "${DSV4_0731_DEEPGEMM_MAIN_TREE}"; \
    git -C /tmp/deepgemm-src apply --index --binary /tmp/deepgemm-release.patch; \
    test "$(git -C /tmp/deepgemm-src write-tree)" = "${DSV4_0731_DEEPGEMM_EFFECTIVE_TREE}"; \
    git -C /tmp/deepgemm-src submodule update --init --depth=1 \
      third-party/cutlass third-party/fmt; \
    cd /tmp/deepgemm-src; \
    bash build_sgl_deep_gemm.sh; \
    uv pip uninstall --system --break-system-packages sgl-deep-gemm || true; \
    uv pip install --system --break-system-packages --no-cache --no-deps \
      /tmp/deepgemm-src/dist/sgl_deep_gemm-*.whl; \
    cd /; \
    uv run --no-project --python /usr/bin/python python -c "import deep_gemm, importlib.metadata as m; expected='${DSV4_0731_DEEPGEMM_VERSION}'; assert m.version('sgl-deep-gemm') == expected, m.version('sgl-deep-gemm'); print('sgl-deep-gemm', expected)"; \
    rm -rf /tmp/deepgemm-src /tmp/deepgemm-release.patch

RUN set -e; cd /sgl-workspace/sglang; \
    uv run --no-project --python /usr/bin/python python -c "import deep_gemm, importlib.metadata as m; assert m.version('sgl-deep-gemm') == '${DSV4_0731_DEEPGEMM_VERSION}', m.version('sgl-deep-gemm'); missing=[n for n in ('m_grouped_fp8_fp4_gemm_nt_contiguous', 'fp8_paged_mqa_logits', 'fp8_fp4_paged_mqa_logits', 'tf32_hc_prenorm_gemm') if not hasattr(deep_gemm, n)]; assert not missing, missing; print('sgl-deep-gemm', m.version('sgl-deep-gemm'))"; \
    uv run --no-project --python /usr/bin/python python -c "import sglang; from sglang.srt.entrypoints.openai import encoding_dsv4; print('sglang', sglang.__version__, 'dsv4 encoding OK')"; \
    uv run --no-project --python /usr/bin/python python test/registered/unit/layers/deep_gemm_wrapper/test_compile_utils.py; \
    uv run --no-project --python /usr/bin/python python test/registered/attention/test_dsv4_indexer_row_slice.py; \
    uv run --no-project --python /usr/bin/python python test/registered/unit/layers/test_dsv4_nonpaged_indexer.py; \
    uv run --no-project --python /usr/bin/python python test/registered/unit/entrypoints/openai/test_serving_chat.py; \
    uv run --no-project --python /usr/bin/python python test/registered/unit/entrypoints/test_warmup.py; \
    uv run --no-project --python /usr/bin/python python test/registered/unit/distributed/test_pcie_ipc_ar.py; \
    uv run --no-project --python /usr/bin/python python test/registered/unit/spec/test_decode_bookkeeping_ownership.py; \
    uv run --no-project --python /usr/bin/python python test/registered/spec/dspark/test_dspark_dp_tier.py

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
      ai.sglang.pr35116.head=${DSV4_0731_SGLANG_PR35116_HEAD} \
      ai.sglang.pr35118.head=${DSV4_0731_SGLANG_PR35118_HEAD} \
      ai.sglang.pr33614.head=${DSV4_0731_SGLANG_PR33614_HEAD} \
      ai.sglang.pr32686.head=${DSV4_0731_SGLANG_PR32686_HEAD} \
      ai.sglang.pr33568.head=${DSV4_0731_SGLANG_PR33568_HEAD} \
      ai.sglang.pr34018.head=${DSV4_0731_SGLANG_PR34018_HEAD} \
      ai.sglang.pr34528.head=${DSV4_0731_SGLANG_PR34528_HEAD} \
      ai.sglang.pr35217.head=${DSV4_0731_SGLANG_PR35217_HEAD} \
      ai.flashinfer.version=${DSV4_0731_FLASHINFER_VERSION} \
      ai.flashinfer.main.head=${DSV4_0731_FLASHINFER_MAIN_HEAD} \
      ai.flashinfer.main.tree=${DSV4_0731_FLASHINFER_MAIN_TREE} \
      ai.flashinfer.effective.tree=${DSV4_0731_FLASHINFER_EFFECTIVE_TREE} \
      ai.flashinfer.integration.head=${DSV4_0731_FLASHINFER_INTEGRATION_HEAD} \
      ai.flashinfer.pr3930.head=${DSV4_0731_FLASHINFER_PR3930_HEAD} \
      ai.flashinfer.local.cudart-resolver.source=${DSV4_0731_FLASHINFER_CUDART_RESOLVER_SOURCE_HEAD} \
      ai.flashinfer.cubin.sha256=${DSV4_0731_FLASHINFER_CUBIN_SHA256} \
      ai.flashinfer.sm120.module=persistent-runtime-jit \
      ai.deepgemm.version=${DSV4_0731_DEEPGEMM_VERSION} \
      ai.deepgemm.main.head=${DSV4_0731_DEEPGEMM_MAIN_HEAD} \
      ai.deepgemm.main.tree=${DSV4_0731_DEEPGEMM_MAIN_TREE} \
      ai.deepgemm.effective.tree=${DSV4_0731_DEEPGEMM_EFFECTIVE_TREE} \
      ai.deepgemm.integration.head=${DSV4_0731_DEEPGEMM_INTEGRATION_HEAD} \
      ai.deepgemm.pr76.head=${DSV4_0731_DEEPGEMM_PR76_HEAD} \
      ai.deepgemm.pr77.head=${DSV4_0731_DEEPGEMM_PR77_HEAD}
