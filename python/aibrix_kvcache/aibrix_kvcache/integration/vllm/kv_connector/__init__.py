# Copyright 2024 The Aibrix Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import logging

logger = logging.getLogger(__name__)

VLLM_V1_WORKER_GPU_MODEL_RUNNER_MODULE = "vllm.v1.worker.gpu_model_runner"

# Track if patches are applied
_patches_applied = False


def _apply_gpu_model_runner_patches(module):
    """Apply patches to an already-imported gpu_model_runner module."""
    GPUModelRunner = module.GPUModelRunner

    # ------------------------------------------------------------------
    # Source-patch detection: if GPUModelRunner already has patched signature,
    # the source-code patch is in place and we should not double-patch.
    # ------------------------------------------------------------------
    if getattr(GPUModelRunner, "_aibrix_patched", False):
        logger.info("[AIBrix] vLLM source patch detected, skipping patch")
        return

    # Check if _update_states already accepts load_results (source patch)
    import inspect

    sig = inspect.signature(GPUModelRunner._update_states)
    if "load_results" in sig.parameters:
        logger.info(
            "[AIBrix] vLLM source patch detected, _update_states has "
            "load_results, skipping patch"
        )
        return

    logger.info("[AIBrix] Applying patches to vLLM GPUModelRunner...")

    # Import has_kv_transfer_group at patch time to avoid import errors
    from vllm.distributed.kv_transfer import has_kv_transfer_group

    # ---- Patch 1: GPUModelRunner.execute_model -----------------------
    # This patch intercepts execute_model to:
    # 1. Call kv_connector_load_before_update() before _update_states
    #    to load KV from external cache into GPU buffers.
    #
    # NOTE: With get_num_new_matched_tokens() properly implemented,
    # the scheduler already adjusts num_computed_tokens and
    # num_scheduled_tokens. The monkey-patch no longer needs to
    # preprocess scheduler_output — it only needs to trigger the
    # actual KV data transfer.
    _orig_execute_model = GPUModelRunner.execute_model

    def _patched_execute_model(self, scheduler_output, *args, **kwargs):
        """Wrapped execute_model that calls KV connector before state updates"""
        # Get load_results from KV connector before _update_states
        if has_kv_transfer_group() and hasattr(
            self, "kv_connector_load_before_update"
        ):
            self.kv_connector_load_before_update(scheduler_output)
        elif has_kv_transfer_group():
            # Fallback: call directly using mixin method
            from vllm.distributed.kv_transfer import get_kv_transfer_group

            kv_connector = get_kv_transfer_group()
            if scheduler_output.kv_connector_metadata is not None:
                kv_connector.bind_connector_metadata(
                    scheduler_output.kv_connector_metadata
                )
                kv_connector.start_load_kv_before_update()

        # Call original execute_model - it will call _update_states
        return _orig_execute_model(self, scheduler_output, *args, **kwargs)

    GPUModelRunner.execute_model = _patched_execute_model

    # Mark class so we never double-patch
    GPUModelRunner._aibrix_patched = True
    logger.info("[AIBrix] GPUModelRunner patched successfully")


def aibrix_patch_vllm():
    """Apply AIBrix patches to vLLM"""
    global _patches_applied
    if _patches_applied:
        logger.info("[AIBrix] Already patched — skipping")
        return

    # Patch GPUModelRunner
    try:
        module = importlib.import_module(VLLM_V1_WORKER_GPU_MODEL_RUNNER_MODULE)
        _apply_gpu_model_runner_patches(module)
    except ImportError as e:
        logger.warning("[AIBrix] Failed to patch gpu_model_runner: %s", e)

    _patches_applied = True


aibrix_patch_vllm()
