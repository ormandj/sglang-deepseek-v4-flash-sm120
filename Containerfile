# DeepSeek-V4-Flash-0731 on SGLang for RTX PRO 6000 Blackwell (SM120).
#
# v0.3.3 retains the reviewed v0.3.2 SM120 stack and corrects DeepGEMM's
# compiled-dimension mapping for SM120 operand-swapped GEMMs.
# The DSpark EP1 allocation fix remains supplied directly by SGLang main.
ARG DSV4_0731_RELEASE_VERSION=0.3.3
ARG DSV4_0731_RELEASE_CANDIDATE=0
ARG DSV4_0731_CACHE_SCHEMA=v15
ARG DSV4_0731_SGLANG_BASE=lmsysorg/sglang:nightly-dev-cu13-20260813-273d978b@sha256:5e012cc3cfe06fd7718bab6f7b8183fad56df28a6b934058edb4d59afc42d440
ARG DSV4_0731_SGLANG_BASE_HEAD=273d978bedc89bc8cb1a5c4d57d9ea04aea2cc9c
ARG DSV4_0731_SGLANG_MAIN_HEAD=9d34c2809f58f3d84ef5dd343733e3f5e86395d5
ARG DSV4_0731_SGLANG_MAIN_TREE=c03a1221e78f29172f6a86a80be1497f201eafb5
ARG DSV4_0731_SGLANG_EFFECTIVE_TREE=2833e60bfdb9c17820095e2e1aa478bb2eb041ef
ARG DSV4_0731_SGLANG_INTEGRATION_HEAD=684d1561e94b8868ec282a764814fee5338fc407
ARG DSV4_0731_SGLANG_PR29927_HEAD=e5ea881c5dd487acef17d58ba1a9d2b7ecfeee91
ARG DSV4_0731_SGLANG_PR33614_HEAD=fca0998feda2bfc2a735286d34d354b979850d72
ARG DSV4_0731_SGLANG_PR32686_HEAD=15c0902eefc59cb8aae919d16d4a0cf60f1a9a2b
ARG DSV4_0731_SGLANG_PR33568_HEAD=cab45a29997f8898e076e9253741a5119a401db0
ARG DSV4_0731_SGLANG_PR33805_HEAD=26a2a3981798b8deb97d053d163fa3c48668e03f
ARG DSV4_0731_SGLANG_PR34018_HEAD=c3ffe8cfd3cf6cf9c30fc470cf7b76754954f3f0
ARG DSV4_0731_SGLANG_PR34528_HEAD=f28d875d121a1ec0ab879ef54873220c2ed23c6a
ARG DSV4_0731_FLASHINFER_MAIN_HEAD=ed6c709849fe1c02d4545b4e743a436405f6ca5b
ARG DSV4_0731_FLASHINFER_MAIN_TREE=767ee24272d9a686a0e720a6007ada2b38570fb7
ARG DSV4_0731_FLASHINFER_EFFECTIVE_TREE=b68d15ef203d0e0e11b3734ade4ccf6dca1b6b4d
ARG DSV4_0731_FLASHINFER_INTEGRATION_HEAD=26804401d5c7700ead975d20bb29784adff66d66
ARG DSV4_0731_FLASHINFER_PR3930_HEAD=e855cc25993d11d4707678d68fbde108d0578bef
ARG DSV4_0731_FLASHINFER_PR4393_HEAD=dca29052ac92789df4df95455170209a93b1ee73
ARG DSV4_0731_FLASHINFER_CUDART_RESOLVER_SOURCE_HEAD=28be41230a979ffdfd423706d1a4c7e82c6988eb
ARG DSV4_0731_FLASHINFER_OUTER_AUTOTUNE_FIX_HEAD=26804401d5c7700ead975d20bb29784adff66d66
ARG DSV4_0731_FLASHINFER_VERSION=0.6.18.dev20260813
ARG DSV4_0731_FLASHINFER_CUBIN_URL=https://github.com/flashinfer-ai/flashinfer/releases/download/nightly-v0.6.18-20260813/flashinfer_cubin-0.6.18.dev20260813-py3-none-any.whl
ARG DSV4_0731_FLASHINFER_CUBIN_SHA256=078d3a14267dcc24c75534e9d53beb6c639ab6ddde325c2de241fc96fcf509bb
ARG DSV4_0731_DEEPGEMM_MAIN_HEAD=7509acb3e261b5acba06087e91c70c409a43419c
ARG DSV4_0731_DEEPGEMM_MAIN_TREE=c4f9e5af9ab88563caa2a100deb0fba7b1217504
ARG DSV4_0731_DEEPGEMM_EFFECTIVE_TREE=b166d085065d39155a8f745126d6db88597d268c
ARG DSV4_0731_DEEPGEMM_INTEGRATION_HEAD=fad8fc7f27a681183a6648c85250cbca6241be27
ARG DSV4_0731_DEEPGEMM_VERSION=0.1.5.post2+sm120jit2
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
ARG DSV4_0731_SGLANG_PR34018_HEAD
ARG DSV4_0731_SGLANG_PR34528_HEAD
ARG DSV4_0731_FLASHINFER_MAIN_HEAD
ARG DSV4_0731_FLASHINFER_MAIN_TREE
ARG DSV4_0731_FLASHINFER_EFFECTIVE_TREE
ARG DSV4_0731_FLASHINFER_INTEGRATION_HEAD
ARG DSV4_0731_FLASHINFER_PR3930_HEAD
ARG DSV4_0731_FLASHINFER_PR4393_HEAD
ARG DSV4_0731_FLASHINFER_CUDART_RESOLVER_SOURCE_HEAD
ARG DSV4_0731_FLASHINFER_OUTER_AUTOTUNE_FIX_HEAD
ARG DSV4_0731_FLASHINFER_VERSION
ARG DSV4_0731_FLASHINFER_CUBIN_URL
ARG DSV4_0731_FLASHINFER_CUBIN_SHA256
ARG DSV4_0731_DEEPGEMM_MAIN_HEAD
ARG DSV4_0731_DEEPGEMM_MAIN_TREE
ARG DSV4_0731_DEEPGEMM_EFFECTIVE_TREE
ARG DSV4_0731_DEEPGEMM_INTEGRATION_HEAD
ARG DSV4_0731_DEEPGEMM_VERSION
ARG IMAGE_SOURCE
ARG IMAGE_SOURCE_REVISION

COPY patches/sglang/0001-sglang-dsv4-0731-v0.3.2-rc.0.patch /tmp/sglang-release.patch
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

COPY patches/flashinfer/0001-flashinfer-dsv4-0731-v0.3.2-rc.0.patch /tmp/flashinfer-release.patch
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

COPY patches/deepgemm/0001-sm120-preserve-runtime-einsum-token-dim.patch /tmp/deepgemm-release.patch
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
    uv run --no-project --python /usr/bin/python python test/registered/unit/entrypoints/openai/test_serving_chat.py; \
    uv run --no-project --python /usr/bin/python python test/registered/unit/distributed/test_pcie_ipc_ar.py; \
    uv run --no-project --python /usr/bin/python python test/registered/unit/spec/test_decode_bookkeeping_ownership.py; \
    uv run --no-project --python /usr/bin/python python test/registered/spec/dspark/test_dspark_dp_tier.py; \
    uv run --no-project --python /usr/bin/python python test/registered/unit/speculative/test_spec_prepare_swa_eviction.py

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
      ai.sglang.pr34018.head=${DSV4_0731_SGLANG_PR34018_HEAD} \
      ai.sglang.pr34528.head=${DSV4_0731_SGLANG_PR34528_HEAD} \
      ai.flashinfer.version=${DSV4_0731_FLASHINFER_VERSION} \
      ai.flashinfer.main.head=${DSV4_0731_FLASHINFER_MAIN_HEAD} \
      ai.flashinfer.main.tree=${DSV4_0731_FLASHINFER_MAIN_TREE} \
      ai.flashinfer.effective.tree=${DSV4_0731_FLASHINFER_EFFECTIVE_TREE} \
      ai.flashinfer.integration.head=${DSV4_0731_FLASHINFER_INTEGRATION_HEAD} \
      ai.flashinfer.pr3930.head=${DSV4_0731_FLASHINFER_PR3930_HEAD} \
      ai.flashinfer.pr4393.head=${DSV4_0731_FLASHINFER_PR4393_HEAD} \
      ai.flashinfer.local.cudart-resolver.source=${DSV4_0731_FLASHINFER_CUDART_RESOLVER_SOURCE_HEAD} \
      ai.flashinfer.local.outer-autotune-fix.head=${DSV4_0731_FLASHINFER_OUTER_AUTOTUNE_FIX_HEAD} \
      ai.flashinfer.cubin.sha256=${DSV4_0731_FLASHINFER_CUBIN_SHA256} \
      ai.flashinfer.sm120.module=persistent-runtime-jit \
      ai.deepgemm.version=${DSV4_0731_DEEPGEMM_VERSION} \
      ai.deepgemm.main.head=${DSV4_0731_DEEPGEMM_MAIN_HEAD} \
      ai.deepgemm.main.tree=${DSV4_0731_DEEPGEMM_MAIN_TREE} \
      ai.deepgemm.effective.tree=${DSV4_0731_DEEPGEMM_EFFECTIVE_TREE} \
      ai.deepgemm.integration.head=${DSV4_0731_DEEPGEMM_INTEGRATION_HEAD}
