# DeepSeek-V4-Flash-0731 on SGLang for 2x RTX PRO 6000 Blackwell (SM120, TP=2).
#
# The image starts from the official CUDA 13 SGLang nightly by immutable digest
# and inherits nothing else. SGLang is main d2c405f1 plus a single source patch
# carrying the pinned heads of PRs #29927, #33614, #33140, #32194, #30700,
# #32330 (rebased, with an mnnvl preflight multicast-granularity fix), #32686
# and #32815, five local fixes, and a local DSV4/DSV32 tool-schema
# encoding fix. FlashInfer is
# current main 67f76379 built from source with a single patch carrying PRs #4308
# and #4309 (#3903 is already in main); the matching cubin wheel is installed by
# URL and pinned by SHA256, and the stale JIT-cache wheel is removed so the
# patched SM120 modules compile once into the persistent runtime cache.
#
# Every pin below is duplicated in stack.lock.json, which scripts/verify-patches.sh
# checks against this file and against the patch bytes.
ARG DSV4_0731_SGLANG_BASE=lmsysorg/sglang:nightly-dev-cu13-20260803-12eadf86@sha256:687efca081e85f4e3126456ff389b1af515fc08a604de4c61f947f531963aba7
ARG DSV4_0731_SGLANG_BASE_HEAD=12eadf86f12aec2e6f81a6e38b61b964a4c6b529
ARG DSV4_0731_SGLANG_MAIN_HEAD=4ad990ba7d75bb9f948f5f6bd8d79a66b5d3fd63
ARG DSV4_0731_SGLANG_MAIN_TREE=671cb86de1275cfc17a4f48c5cea5bf86d3c82d6
ARG DSV4_0731_SGLANG_EFFECTIVE_TREE=d62dbaf3bb1dd402b0f08185248a15a78c7377b2
ARG DSV4_0731_SGLANG_PR29927_HEAD=21a5bc8e4e05909a7c946d3467561a8e024a108a
ARG DSV4_0731_SGLANG_PR32183_HEAD=22ef431215b1d8529eaebd8e8c6de9510390afaf
ARG DSV4_0731_SGLANG_PR33614_HEAD=56eae704773c168c687603dcb24b40130d1a9594
ARG DSV4_0731_SGLANG_PR32194_HEAD=f8fac3913dbfbbc9048692d79de3587d226fa421
ARG DSV4_0731_SGLANG_PR30700_HEAD=aead319d06c7a2ab2a21e575000fcdaea7c17675
ARG DSV4_0731_SGLANG_PR32330_HEAD=f330c748ade470e806347f24dffae4eec38eb878
ARG DSV4_0731_SGLANG_PR32686_HEAD=15c0902eefc59cb8aae919d16d4a0cf60f1a9a2b
ARG DSV4_0731_SGLANG_PR32815_HEAD=1dbf09f6e0157279f5350357d0309397f0257025
ARG DSV4_0731_SGLANG_SM120_DEEPGEMM_HEAD=d4dc7502cd4469c37e935d4fbce946b8a2331212
ARG DSV4_0731_SGLANG_PREFILL_WORKSPACE_HEAD=73d125d0432c80354380b77d729d2d686cda38c8
ARG DSV4_0731_SGLANG_SM120_ALLREDUCE_GATE_HEAD=861b99ca17a021361963c6a3f43a65999252e41e
ARG DSV4_0731_SGLANG_ALLREDUCE_TOKEN_CAP_HEAD=f99f4a0bbc34ca250071a51d1159e76c1a5d2d88
ARG DSV4_0731_SGLANG_DSPARK_SWA_EVICTION_HEAD=bcc988b92823c5a1be8d6cee0a3527770f07c6cf
ARG DSV4_0731_FLASHINFER_MAIN_HEAD=67f76379a145f19793896394974e29e610cda912
ARG DSV4_0731_FLASHINFER_MAIN_TREE=eb3d93ed91e895c40b5c8c8ebdcb6bfae88637a8
ARG DSV4_0731_FLASHINFER_EFFECTIVE_TREE=d8567dec0ff48f11c0ab1d17a05e944d50e48f7a
ARG DSV4_0731_FLASHINFER_PR4308_HEAD=4ec7f230447320c9f32585d16596cdda8133029f
ARG DSV4_0731_FLASHINFER_PR4309_HEAD=bf136350f1dbbec177e2892b094df6e99145f9ea
ARG DSV4_0731_FLASHINFER_VERSION=0.6.17.dev20260804
ARG DSV4_0731_FLASHINFER_CUBIN_URL=https://github.com/flashinfer-ai/flashinfer/releases/download/nightly-v0.6.17-20260804/flashinfer_cubin-0.6.17.dev20260804-py3-none-any.whl
ARG DSV4_0731_FLASHINFER_CUBIN_SHA256=d64f7bb13c4f1528230054908240fd0f20fcfacd80a8e8a55df24d0f1e9a2c3c

FROM ${DSV4_0731_SGLANG_BASE} AS runtime
ARG DSV4_0731_SGLANG_BASE_HEAD
ARG DSV4_0731_SGLANG_MAIN_HEAD
ARG DSV4_0731_SGLANG_MAIN_TREE
ARG DSV4_0731_SGLANG_EFFECTIVE_TREE
ARG DSV4_0731_SGLANG_PR29927_HEAD
ARG DSV4_0731_SGLANG_PR32183_HEAD
ARG DSV4_0731_SGLANG_PR33614_HEAD
ARG DSV4_0731_SGLANG_PR32194_HEAD
ARG DSV4_0731_SGLANG_PR30700_HEAD
ARG DSV4_0731_SGLANG_PR32330_HEAD
ARG DSV4_0731_SGLANG_PR32686_HEAD
ARG DSV4_0731_SGLANG_PR32815_HEAD
ARG DSV4_0731_SGLANG_SM120_DEEPGEMM_HEAD
ARG DSV4_0731_SGLANG_PREFILL_WORKSPACE_HEAD
ARG DSV4_0731_SGLANG_SM120_ALLREDUCE_GATE_HEAD
ARG DSV4_0731_SGLANG_ALLREDUCE_TOKEN_CAP_HEAD
ARG DSV4_0731_SGLANG_DSPARK_SWA_EVICTION_HEAD
ARG DSV4_0731_FLASHINFER_MAIN_HEAD
ARG DSV4_0731_FLASHINFER_MAIN_TREE
ARG DSV4_0731_FLASHINFER_EFFECTIVE_TREE
ARG DSV4_0731_FLASHINFER_PR4308_HEAD
ARG DSV4_0731_FLASHINFER_PR4309_HEAD
ARG DSV4_0731_FLASHINFER_VERSION
ARG DSV4_0731_FLASHINFER_CUBIN_URL
ARG DSV4_0731_FLASHINFER_CUBIN_SHA256

COPY patches/sglang/0001-sglang-dsv4-0731-perf-r8.patch /tmp/sglang-perf-r8.patch
# The tree check pins the single source patch: the pinned PR heads, the four
# local fixes, and the tool-schema encoding fix, applied as one diff.
RUN set -e; cd /sgl-workspace/sglang; \
    git fetch --depth=1 origin "${DSV4_0731_SGLANG_MAIN_HEAD}"; \
    git checkout --detach FETCH_HEAD; \
    git reset --hard "${DSV4_0731_SGLANG_MAIN_HEAD}"; \
    test "$(git rev-parse HEAD)" = "${DSV4_0731_SGLANG_MAIN_HEAD}"; \
    test "$(git rev-parse HEAD^{tree})" = "${DSV4_0731_SGLANG_MAIN_TREE}"; \
    git apply --index --binary /tmp/sglang-perf-r8.patch; \
    test "$(git write-tree)" = "${DSV4_0731_SGLANG_EFFECTIVE_TREE}"; \
    uv run --no-project python -m compileall -q \
      python/sglang/srt/arg_groups/overrides.py \
      python/sglang/srt/distributed/bootstrap.py \
      python/sglang/srt/distributed/parallel_state.py \
      python/sglang/srt/entrypoints/openai \
      python/sglang/srt/layers/attention/deepseek_v4_backend.py \
      python/sglang/srt/layers/communicator.py \
      python/sglang/srt/layers/deep_gemm_wrapper \
      python/sglang/srt/layers/flashinfer_comm_fusion.py \
      python/sglang/srt/model_executor/runner/base_runner.py \
      python/sglang/srt/model_loader/utils.py \
      python/sglang/srt/server_args.py \
      test/registered/unit/entrypoints/openai/test_serving_chat.py \
      test/registered/unit/layers/deep_gemm_wrapper \
      test/registered/unit/layers/test_flashinfer_comm_fusion.py \
      test/registered/unit/layers/test_layer_communicator_fusion_gate.py \
      test/registered/unit/model_loader/test_deepgemm_sm120.py \
      test/registered/unit/test_model_overrides.py; \
    rm /tmp/sglang-perf-r8.patch

COPY patches/flashinfer/0001-flashinfer-dsv4-0731-perf-refresh.patch /tmp/flashinfer-perf-refresh.patch
# SOURCE_DATE_EPOCH comes from the checked-out commit so the wheel build is
# reproducible. The cubin wheel is fetched by URL with a pinned SHA256 fragment,
# which uv enforces. Removing flashinfer-jit-cache leaves the patched SM120
# modules to compile once into the persistent runtime cache.
RUN set -e; \
    git init /tmp/flashinfer-src; \
    git -C /tmp/flashinfer-src remote add origin https://github.com/flashinfer-ai/flashinfer.git; \
    git -C /tmp/flashinfer-src fetch --depth=1 origin "${DSV4_0731_FLASHINFER_MAIN_HEAD}"; \
    git -C /tmp/flashinfer-src checkout --detach FETCH_HEAD; \
    test "$(git -C /tmp/flashinfer-src rev-parse HEAD)" = "${DSV4_0731_FLASHINFER_MAIN_HEAD}"; \
    test "$(git -C /tmp/flashinfer-src rev-parse HEAD^{tree})" = "${DSV4_0731_FLASHINFER_MAIN_TREE}"; \
    git -C /tmp/flashinfer-src apply --index --binary /tmp/flashinfer-perf-refresh.patch; \
    test "$(git -C /tmp/flashinfer-src write-tree)" = "${DSV4_0731_FLASHINFER_EFFECTIVE_TREE}"; \
    git -C /tmp/flashinfer-src submodule update --init --depth=1 \
      3rdparty/cccl 3rdparty/cutlass 3rdparty/spdlog; \
    SOURCE_DATE_EPOCH="$(git -C /tmp/flashinfer-src show -s --format=%ct HEAD)" \
      BUILD_NVEP=0 FLASHINFER_DEV_RELEASE_SUFFIX=20260804 \
      uv build --wheel --out-dir /tmp/flashinfer-wheel /tmp/flashinfer-src; \
    uv pip uninstall --system --break-system-packages flashinfer-python flashinfer-jit-cache || true; \
    uv pip install --system --break-system-packages --no-cache --no-deps --reinstall \
      "${DSV4_0731_FLASHINFER_CUBIN_URL}#sha256=${DSV4_0731_FLASHINFER_CUBIN_SHA256}"; \
    uv pip install --system --break-system-packages --no-cache --no-deps \
      /tmp/flashinfer-wheel/flashinfer_python-*.whl; \
    uv run --no-project python -c "import flashinfer, importlib.metadata as m; expected='${DSV4_0731_FLASHINFER_VERSION}'; assert flashinfer.__version__ == expected, flashinfer.__version__; assert flashinfer.__git_version__ == '${DSV4_0731_FLASHINFER_MAIN_HEAD}', flashinfer.__git_version__; assert m.version('flashinfer-cubin') == expected, m.version('flashinfer-cubin'); assert all(d.metadata['Name'] != 'flashinfer-jit-cache' for d in m.distributions()); print('flashinfer', expected)"; \
    rm -rf /tmp/flashinfer-src /tmp/flashinfer-wheel /tmp/flashinfer-perf-refresh.patch

# CPU-only checks of the parts this image actually changes: the DeepGEMM entry
# points the SM120 paths require, the DSV4 encoder, and the SM120 DeepGEMM
# capability, model-override, communicator-fusion, and all-reduce workspace
# behavior.
RUN set -e; cd /sgl-workspace/sglang; \
    uv run --no-project python -c "import deep_gemm, importlib.metadata as m; assert m.version('sgl-deep-gemm') == '0.1.5', m.version('sgl-deep-gemm'); missing=[n for n in ('m_grouped_fp8_fp4_gemm_nt_contiguous', 'fp8_fp4_paged_mqa_logits') if not hasattr(deep_gemm, n)]; assert not missing, missing; print('sgl-deep-gemm', m.version('sgl-deep-gemm'))"; \
    uv run --no-project python -c "import sglang; from sglang.srt.entrypoints.openai import encoding_dsv4; print('sglang', sglang.__version__, 'dsv4 encoding OK')"; \
    uv run --no-project python test/registered/unit/layers/deep_gemm_wrapper/test_configurer.py; \
    uv run --no-project python test/registered/unit/model_loader/test_deepgemm_sm120.py; \
    uv run --no-project python test/registered/unit/layers/deep_gemm_wrapper/test_compile_utils.py; \
    uv run --no-project python test/registered/unit/entrypoints/openai/test_serving_chat.py; \
    uv run --no-project python test/registered/unit/layers/test_layer_communicator_fusion_gate.py; \
    uv run --no-project python test/registered/unit/layers/test_flashinfer_comm_fusion.py; \
    uv run --no-project python test/registered/unit/test_model_overrides.py \
      TestGoldenModelOverrides.test_deepseek_v4_sm120_moe_pass \
      TestGoldenModelOverrides.test_sm120_fp8_wo_a_gemm_default_gates_on_deepgemm_capability \
      TestGoldenModelOverrides.test_flashinfer_allreduce_fusion_passes

ENV SGLANG_BUILD_COMMIT=${DSV4_0731_SGLANG_MAIN_HEAD} \
    SGLANG_BUILD_TREE=${DSV4_0731_SGLANG_EFFECTIVE_TREE} \
    FLASHINFER_VERSION=${DSV4_0731_FLASHINFER_VERSION} \
    FLASHINFER_CUDA_ARCH_LIST=12.0f
LABEL org.opencontainers.image.title="sglang-deepseek-v4-flash-sm120" \
      org.opencontainers.image.description="SGLang for DeepSeek-V4-Flash-0731 on RTX PRO 6000 Blackwell (SM120, TP=2)" \
      org.opencontainers.image.source="https://github.com/ormandj/sglang-deepseek-v4-flash-sm120" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.revision=${DSV4_0731_SGLANG_MAIN_HEAD} \
      ai.sglang.base.head=${DSV4_0731_SGLANG_BASE_HEAD} \
      ai.sglang.main.head=${DSV4_0731_SGLANG_MAIN_HEAD} \
      ai.sglang.main.tree=${DSV4_0731_SGLANG_MAIN_TREE} \
      ai.sglang.effective.tree=${DSV4_0731_SGLANG_EFFECTIVE_TREE} \
      ai.sglang.pr29927.head=${DSV4_0731_SGLANG_PR29927_HEAD} \
      ai.sglang.pr32183.head=${DSV4_0731_SGLANG_PR32183_HEAD} \
      ai.sglang.pr33614.head=${DSV4_0731_SGLANG_PR33614_HEAD} \
      ai.sglang.pr32194.head=${DSV4_0731_SGLANG_PR32194_HEAD} \
      ai.sglang.pr30700.head=${DSV4_0731_SGLANG_PR30700_HEAD} \
      ai.sglang.pr32330.head=${DSV4_0731_SGLANG_PR32330_HEAD} \
      ai.sglang.pr32686.head=${DSV4_0731_SGLANG_PR32686_HEAD} \
      ai.sglang.pr32815.head=${DSV4_0731_SGLANG_PR32815_HEAD} \
      ai.sglang.local.sm120-deepgemm.head=${DSV4_0731_SGLANG_SM120_DEEPGEMM_HEAD} \
      ai.sglang.local.prefill-workspace.head=${DSV4_0731_SGLANG_PREFILL_WORKSPACE_HEAD} \
      ai.sglang.local.sm120-allreduce-gate.head=${DSV4_0731_SGLANG_SM120_ALLREDUCE_GATE_HEAD} \
      ai.sglang.local.allreduce-token-cap.head=${DSV4_0731_SGLANG_ALLREDUCE_TOKEN_CAP_HEAD} \
      ai.sglang.local.dspark-swa-eviction.head=${DSV4_0731_SGLANG_DSPARK_SWA_EVICTION_HEAD} \
      ai.flashinfer.version=${DSV4_0731_FLASHINFER_VERSION} \
      ai.flashinfer.main.head=${DSV4_0731_FLASHINFER_MAIN_HEAD} \
      ai.flashinfer.main.tree=${DSV4_0731_FLASHINFER_MAIN_TREE} \
      ai.flashinfer.effective.tree=${DSV4_0731_FLASHINFER_EFFECTIVE_TREE} \
      ai.flashinfer.pr4308.head=${DSV4_0731_FLASHINFER_PR4308_HEAD} \
      ai.flashinfer.pr4309.head=${DSV4_0731_FLASHINFER_PR4309_HEAD} \
      ai.flashinfer.cubin.sha256=${DSV4_0731_FLASHINFER_CUBIN_SHA256} \
      ai.flashinfer.sm120.module=persistent-runtime-jit
