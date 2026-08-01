# DeepSeek-V4-Flash-0731 on SGLang for 2x RTX PRO 6000 Blackwell (SM120, TP=2).
#
# The image starts from the official CUDA 13 SGLang nightly by immutable digest
# and inherits nothing else. SGLang is main 9dcaf6bf plus exactly the current
# heads of PRs #29927, #32815, #30700, #32330, and #32686, followed by two local
# follow-ups. FlashInfer is current main 668a1ba1 built from source with PR #3930
# and two local patches; the matching cubin wheel is installed by URL and pinned
# by SHA256, and the stale JIT-cache wheel is removed so the patched SM120
# modules compile once into the persistent runtime cache.
#
# Every pin below is duplicated in stack.lock.json, which scripts/verify-patches.sh
# checks against this file and against the patch bytes.
ARG DSV4_0731_SGLANG_BASE=lmsysorg/sglang:nightly-dev-cu13-20260731-68d44294@sha256:484e90a21e97e418f55254e5f9b36e5c5864151916a09f197764767a67bba9b2
ARG DSV4_0731_SGLANG_BASE_HEAD=68d442945f9c5e10eced40a1a3da254efe7e96be
ARG DSV4_0731_SGLANG_MAIN_HEAD=9dcaf6bfdff89b4b29611725ef44161be4e429dd
ARG DSV4_0731_SGLANG_MAIN_TREE=d16546bd3173bcb31c5257f6b8f4b703e5455c46
ARG DSV4_0731_SGLANG_EFFECTIVE_TREE=5aa769769a7bf7fc9553a85fe6be2c2a99fad31c
ARG DSV4_0731_SGLANG_WORKSPACE_TREE=98411f8b7105969e68aadd6a2c86974bbf7e79c5
ARG DSV4_0731_SGLANG_ENCODING_HEAD=5912c5d3c2fbb52d9667f93db16e4e29c881bcbd
ARG DSV4_0731_SGLANG_PR29927_HEAD=bfc395a8d85411a75caac354099b2f127bcc38b3
ARG DSV4_0731_SGLANG_PR32815_HEAD=1dbf09f6e0157279f5350357d0309397f0257025
ARG DSV4_0731_SGLANG_PR30700_HEAD=2960b75149cd9888f4fe39c59a4be7337331a5d6
ARG DSV4_0731_SGLANG_PR32330_HEAD=34c9d59633fa97e11200b3b82c2dbfe97df40f5a
ARG DSV4_0731_SGLANG_PR32686_HEAD=15c0902eefc59cb8aae919d16d4a0cf60f1a9a2b
ARG DSV4_0731_FLASHINFER_BASE_HEAD=668a1ba1ca86432c79f6adad37ecfce8d06ec083
ARG DSV4_0731_FLASHINFER_BASE_TREE=99ad66494d6cfe67a4736d25a100ad17650be0b0
ARG DSV4_0731_FLASHINFER_MAIN_HEAD=668a1ba1ca86432c79f6adad37ecfce8d06ec083
ARG DSV4_0731_FLASHINFER_EFFECTIVE_TREE=164ff8cf34d74856829650798f6400484e1d7155
ARG DSV4_0731_FLASHINFER_PR3903_HEAD=2deed6c15e8c8410763d6e8adedc8ca69597e957
ARG DSV4_0731_FLASHINFER_PR3930_HEAD=e855cc25993d11d4707678d68fbde108d0578bef
ARG DSV4_0731_FLASHINFER_TOPK192_HEAD=4d42fdbbe8e7ee9fcb58ed3daeebb6c2b31503ab
ARG DSV4_0731_FLASHINFER_PROFILER_HEAD=c8fb671d7effea6e9d5b0bd766cd1e160b6923b5
ARG DSV4_0731_FLASHINFER_VERSION=0.6.17.dev20260731
ARG DSV4_0731_FLASHINFER_CUBIN_URL=https://github.com/flashinfer-ai/flashinfer/releases/download/nightly-v0.6.17-20260731/flashinfer_cubin-0.6.17.dev20260731-py3-none-any.whl
ARG DSV4_0731_FLASHINFER_CUBIN_SHA256=5813d73a6f6620cfabcc4a85bfede4f20f4576d55dbbe0fe2ef57803ed9d1cd2

FROM ${DSV4_0731_SGLANG_BASE} AS runtime
ARG DSV4_0731_SGLANG_BASE_HEAD
ARG DSV4_0731_SGLANG_MAIN_HEAD
ARG DSV4_0731_SGLANG_MAIN_TREE
ARG DSV4_0731_SGLANG_EFFECTIVE_TREE
ARG DSV4_0731_SGLANG_WORKSPACE_TREE
ARG DSV4_0731_SGLANG_ENCODING_HEAD
ARG DSV4_0731_SGLANG_PR29927_HEAD
ARG DSV4_0731_SGLANG_PR32815_HEAD
ARG DSV4_0731_SGLANG_PR30700_HEAD
ARG DSV4_0731_SGLANG_PR32330_HEAD
ARG DSV4_0731_SGLANG_PR32686_HEAD
ARG DSV4_0731_FLASHINFER_BASE_HEAD
ARG DSV4_0731_FLASHINFER_BASE_TREE
ARG DSV4_0731_FLASHINFER_MAIN_HEAD
ARG DSV4_0731_FLASHINFER_EFFECTIVE_TREE
ARG DSV4_0731_FLASHINFER_PR3903_HEAD
ARG DSV4_0731_FLASHINFER_PR3930_HEAD
ARG DSV4_0731_FLASHINFER_TOPK192_HEAD
ARG DSV4_0731_FLASHINFER_PROFILER_HEAD
ARG DSV4_0731_FLASHINFER_VERSION
ARG DSV4_0731_FLASHINFER_CUBIN_URL
ARG DSV4_0731_FLASHINFER_CUBIN_SHA256

COPY patches/sglang/0001-sglang-pr29927-pr32815-pr30700-pr32330.patch /tmp/sglang-pr-stack.patch
COPY patches/sglang/0002-sglang-pr32686-deepgemm-warmup.patch /tmp/sglang-pr32686.patch
COPY patches/sglang/0003-sglang-flashinfer-allreduce-prefill-workspace.patch /tmp/sglang-workspace.patch
COPY patches/sglang/0004-sglang-dsv4-0731-reasoning-effort.patch /tmp/sglang-0731-encoding.patch
# The intermediate tree check pins the pure upstream PR stack; the second check
# pins the two local follow-ups on top of it.
RUN set -e; cd /sgl-workspace/sglang; \
    git fetch --depth=1 origin "${DSV4_0731_SGLANG_MAIN_HEAD}"; \
    git checkout --detach FETCH_HEAD; \
    git reset --hard "${DSV4_0731_SGLANG_MAIN_HEAD}"; \
    test "$(git rev-parse HEAD)" = "${DSV4_0731_SGLANG_MAIN_HEAD}"; \
    test "$(git rev-parse HEAD^{tree})" = "${DSV4_0731_SGLANG_MAIN_TREE}"; \
    git apply --index --binary /tmp/sglang-pr-stack.patch; \
    git apply --index --binary /tmp/sglang-pr32686.patch; \
    test "$(git write-tree)" = "${DSV4_0731_SGLANG_EFFECTIVE_TREE}"; \
    git apply --index --binary /tmp/sglang-workspace.patch; \
    git apply --index --binary /tmp/sglang-0731-encoding.patch; \
    test "$(git write-tree)" = "${DSV4_0731_SGLANG_WORKSPACE_TREE}"; \
    uv run --no-project python -m compileall -q \
      python/sglang/kernels/ops/attention/flash_mla_sm120.py \
      python/sglang/kernels/ops/layernorm/mhc.py \
      python/sglang/kernels/ops/moe/ep_moe_kernels.py \
      python/sglang/srt/arg_groups/overrides.py \
      python/sglang/srt/distributed/bootstrap.py \
      python/sglang/srt/distributed/parallel_state.py \
      python/sglang/srt/layers/attention/deepseek_v4_backend.py \
      python/sglang/srt/layers/attention/dsv4 \
      python/sglang/srt/layers/communicator.py \
      python/sglang/srt/layers/deep_gemm_wrapper \
      python/sglang/srt/layers/flashinfer_comm_fusion.py \
      python/sglang/srt/layers/moe/moe_runner/deep_gemm.py \
      python/sglang/srt/model_executor/runner/base_runner.py \
      python/sglang/srt/model_loader/utils.py \
      python/sglang/srt/models/deepseek_v4.py \
      python/sglang/srt/entrypoints/openai/encoding_dsv4.py \
      python/sglang/srt/entrypoints/openai/serving_chat.py \
      python/sglang/srt/server_args.py \
      test/registered/unit/entrypoints/openai/test_serving_chat.py \
      test/registered/unit/layers/test_flashinfer_comm_fusion.py; \
    rm /tmp/sglang-pr-stack.patch /tmp/sglang-pr32686.patch \
      /tmp/sglang-workspace.patch /tmp/sglang-0731-encoding.patch

COPY patches/flashinfer/0001-flashinfer-pr3930-exact-cuda-runtime-library-match.patch /tmp/flashinfer-pr3930.patch
COPY patches/flashinfer/0002-flashinfer-sm120-dsv4-topk192.patch /tmp/flashinfer-topk192.patch
COPY patches/flashinfer/0003-flashinfer-mxfp8-mxfp4-profiler-quant-params.patch /tmp/flashinfer-mxfp4-profiler.patch
# SOURCE_DATE_EPOCH comes from the checked-out commit so the wheel build is
# reproducible. The cubin wheel is fetched by URL with a pinned SHA256 fragment,
# which uv enforces. Removing flashinfer-jit-cache leaves the patched SM120
# modules to compile once into the persistent runtime cache.
RUN set -e; \
    git init /tmp/flashinfer-src; \
    git -C /tmp/flashinfer-src remote add origin https://github.com/flashinfer-ai/flashinfer.git; \
    git -C /tmp/flashinfer-src fetch --depth=1 origin "${DSV4_0731_FLASHINFER_BASE_HEAD}"; \
    git -C /tmp/flashinfer-src checkout --detach FETCH_HEAD; \
    test "$(git -C /tmp/flashinfer-src rev-parse HEAD)" = "${DSV4_0731_FLASHINFER_BASE_HEAD}"; \
    test "$(git -C /tmp/flashinfer-src rev-parse HEAD^{tree})" = "${DSV4_0731_FLASHINFER_BASE_TREE}"; \
    git -C /tmp/flashinfer-src apply --index --binary /tmp/flashinfer-pr3930.patch; \
    git -C /tmp/flashinfer-src apply --index --binary /tmp/flashinfer-topk192.patch; \
    git -C /tmp/flashinfer-src apply --index --binary /tmp/flashinfer-mxfp4-profiler.patch; \
    test "$(git -C /tmp/flashinfer-src write-tree)" = "${DSV4_0731_FLASHINFER_EFFECTIVE_TREE}"; \
    git -C /tmp/flashinfer-src submodule update --init --depth=1 \
      3rdparty/cccl 3rdparty/cutlass 3rdparty/spdlog; \
    SOURCE_DATE_EPOCH="$(git -C /tmp/flashinfer-src show -s --format=%ct HEAD)" \
      BUILD_NVEP=0 FLASHINFER_DEV_RELEASE_SUFFIX=20260731 \
      uv build --wheel --out-dir /tmp/flashinfer-wheel /tmp/flashinfer-src; \
    uv pip uninstall --system --break-system-packages flashinfer-python flashinfer-jit-cache || true; \
    uv pip install --system --break-system-packages --no-cache --no-deps --reinstall \
      "${DSV4_0731_FLASHINFER_CUBIN_URL}#sha256=${DSV4_0731_FLASHINFER_CUBIN_SHA256}"; \
    uv pip install --system --break-system-packages --no-cache --no-deps \
      /tmp/flashinfer-wheel/flashinfer_python-*.whl; \
    uv run --no-project python -c "import flashinfer, importlib.metadata as m; expected='${DSV4_0731_FLASHINFER_VERSION}'; assert flashinfer.__version__ == expected, flashinfer.__version__; assert flashinfer.__git_version__ == '${DSV4_0731_FLASHINFER_BASE_HEAD}', flashinfer.__git_version__; assert m.version('flashinfer-cubin') == expected; assert all(d.metadata['Name'] != 'flashinfer-jit-cache' for d in m.distributions()); print('flashinfer', expected)"; \
    rm -rf /tmp/flashinfer-src /tmp/flashinfer-wheel \
      /tmp/flashinfer-pr3930.patch /tmp/flashinfer-topk192.patch \
      /tmp/flashinfer-mxfp4-profiler.patch

# CPU-only checks of the parts this image actually changes: the DeepGEMM entry
# points the SM120 paths require, the 0731 reasoning-effort encoder, and the
# model-override, communicator-fusion, and all-reduce workspace behavior.
RUN set -e; cd /sgl-workspace/sglang; \
    uv run --no-project python -c "import deep_gemm, importlib.metadata as m; assert m.version('sgl-deep-gemm') == '0.1.5'; missing=[n for n in ('m_grouped_fp8_fp4_gemm_nt_contiguous', 'fp8_fp4_paged_mqa_logits') if not hasattr(deep_gemm, n)]; assert not missing, missing; print('sgl-deep-gemm', m.version('sgl-deep-gemm'))"; \
    uv run --no-project python -c "import sglang; from sglang.srt.entrypoints.openai import encoding_dsv4; print('sglang', sglang.__version__, 'dsv4 encoding OK')"; \
    uv run --no-project python test/registered/unit/layers/deep_gemm_wrapper/test_compile_utils.py; \
    uv run --no-project python test/registered/unit/layers/test_layer_communicator_fusion_gate.py; \
    uv run --no-project python test/registered/unit/test_model_overrides.py \
      TestGoldenModelOverrides.test_deepseek_v4_sm120_moe_pass \
      TestGoldenModelOverrides.test_sm120_fp8_wo_a_gemm_default_gates_on_deepgemm_capability \
      TestGoldenModelOverrides.test_flashinfer_allreduce_fusion_passes; \
    uv run --no-project python test/registered/unit/entrypoints/openai/test_serving_chat.py \
      ServingChatTestCase.test_dpsk_v32_encoding_path \
      ServingChatTestCase.test_dsv4_0731_reasoning_effort_detection \
      ServingChatTestCase.test_dsv4_task_and_reminder_encode_end_to_end \
      ServingChatTestCase.test_dsv4_reasoning_effort_checkpoint_compatibility; \
    uv run --no-project python test/registered/unit/layers/test_flashinfer_comm_fusion.py \
      TestFlashInferCommFusion.test_preinitialized_workspace_covers_largest_prefill_forward \
      TestFlashInferCommFusion.test_trtllm_workspace_preflight_does_not_require_multicast \
      TestFlashInferCommFusion.test_auto_backend_resolves_by_arch \
      TestFlashInferCommFusion.test_explicit_backend_validation

ENV SGLANG_BUILD_COMMIT=${DSV4_0731_SGLANG_MAIN_HEAD} \
    SGLANG_BUILD_TREE=${DSV4_0731_SGLANG_WORKSPACE_TREE} \
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
      ai.sglang.upstream.tree=${DSV4_0731_SGLANG_EFFECTIVE_TREE} \
      ai.sglang.effective.tree=${DSV4_0731_SGLANG_WORKSPACE_TREE} \
      ai.sglang.patch.flashinfer-workspace=prefill-capacity \
      ai.sglang.patch.dsv4-0731-encoding.head=${DSV4_0731_SGLANG_ENCODING_HEAD} \
      ai.sglang.pr29927.head=${DSV4_0731_SGLANG_PR29927_HEAD} \
      ai.sglang.pr32815.head=${DSV4_0731_SGLANG_PR32815_HEAD} \
      ai.sglang.pr30700.head=${DSV4_0731_SGLANG_PR30700_HEAD} \
      ai.sglang.pr32330.head=${DSV4_0731_SGLANG_PR32330_HEAD} \
      ai.sglang.pr32686.head=${DSV4_0731_SGLANG_PR32686_HEAD} \
      ai.flashinfer.base.head=${DSV4_0731_FLASHINFER_BASE_HEAD} \
      ai.flashinfer.current-main.head=${DSV4_0731_FLASHINFER_MAIN_HEAD} \
      ai.flashinfer.effective.tree=${DSV4_0731_FLASHINFER_EFFECTIVE_TREE} \
      ai.flashinfer.pr3903.head=${DSV4_0731_FLASHINFER_PR3903_HEAD} \
      ai.flashinfer.pr3930.head=${DSV4_0731_FLASHINFER_PR3930_HEAD} \
      ai.flashinfer.patch.mxfp8-mxfp4-profiler.head=${DSV4_0731_FLASHINFER_PROFILER_HEAD} \
      ai.flashinfer.cubin.sha256=${DSV4_0731_FLASHINFER_CUBIN_SHA256} \
      ai.flashinfer.sm120.topk192.head=${DSV4_0731_FLASHINFER_TOPK192_HEAD} \
      ai.flashinfer.sm120.module=persistent-runtime-jit
