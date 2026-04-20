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
import os
import sys

logger = logging.getLogger(__name__)


def _loud(msg: str) -> None:
    """Unconditional stderr write + logger. For debugging fork race."""
    pid = os.getpid()
    sys.stderr.write(f"[AIBRIX-DEBUG pid={pid}] {msg}\n")
    sys.stderr.flush()
    logger.warning("[AIBRIX-DEBUG pid=%d] %s", pid, msg)

VLLM_V1_WORKER_GPU_MODEL_RUNNER_MODULE = "vllm.v1.worker.gpu_model_runner"

# Track if patches are applied
_patches_applied = False


def _apply_gpu_model_runner_patches(module):
    """Apply patches to an already-imported gpu_model_runner module."""
    GPUModelRunner = module.GPUModelRunner

    _loud(f"_apply_gpu_model_runner_patches: GPUModelRunner id={id(GPUModelRunner)} module={GPUModelRunner.__module__} execute_model.__name__={GPUModelRunner.execute_model.__name__} execute_model id={id(GPUModelRunner.execute_model)}")

    if GPUModelRunner.execute_model.__name__ == "_patched_execute_model":
        _loud("Already monkey-patched, skipping")
        return

    # Source-patch detection: _update_states accepts load_results
    import inspect
    sig = inspect.signature(GPUModelRunner._update_states)
    if "load_results" in sig.parameters:
        _loud("vLLM source patch detected, skipping monkey-patch")
        return

    _loud("Applying patches to vLLM GPUModelRunner...")

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
        has_group = has_kv_transfer_group()
        has_mixin_method = hasattr(self, "kv_connector_load_before_update")
        has_meta = getattr(scheduler_output, "kv_connector_metadata", None) is not None
        _loud(f"_patched_execute_model: has_group={has_group} has_mixin_method={has_mixin_method} has_meta={has_meta}")
        if has_group and has_mixin_method:
            _loud("branch: mixin")
            self.kv_connector_load_before_update(scheduler_output)
        elif has_group:
            _loud("branch: fallback")
            from vllm.distributed.kv_transfer import get_kv_transfer_group
            kv_connector = get_kv_transfer_group()
            if scheduler_output.kv_connector_metadata is not None:
                kv_connector.bind_connector_metadata(
                    scheduler_output.kv_connector_metadata
                )
                _loud("calling start_load_kv_before_update")
                kv_connector.start_load_kv_before_update()
            else:
                _loud("skipping: no metadata")

        # Forward pass
        result = _orig_execute_model(self, scheduler_output, *args, **kwargs)

        # SAVE phase — explicit wait_for_save() call (Attempt A)
        # vLLM's native _get_kv_connector_output context manager may clear
        # _connector_metadata between the forward and this point, so we
        # re-bind metadata before calling wait_for_save.
        if has_group:
            from vllm.distributed.kv_transfer import get_kv_transfer_group
            kv_connector = get_kv_transfer_group()
            if scheduler_output.kv_connector_metadata is not None:
                try:
                    kv_connector.bind_connector_metadata(
                        scheduler_output.kv_connector_metadata
                    )
                    _loud("calling wait_for_save")
                    kv_connector.wait_for_save()
                except Exception as e:
                    _loud(f"wait_for_save failed: {e!r}")

        return result

    GPUModelRunner.execute_model = _patched_execute_model

    # Mark class so we never double-patch
    GPUModelRunner._aibrix_patched = True
    _loud(f"GPUModelRunner patched. id={id(GPUModelRunner)} New execute_model id={id(GPUModelRunner.execute_model)}")


def aibrix_patch_vllm():
    """Apply AIBrix patches to vLLM."""
    _loud("aibrix_patch_vllm() invoked")
    # Patch GPUModelRunner
    try:
        module = importlib.import_module(VLLM_V1_WORKER_GPU_MODEL_RUNNER_MODULE)
        _apply_gpu_model_runner_patches(module)
    except ImportError as e:
        _loud(f"Failed to patch gpu_model_runner: {e}")

    _patches_applied = True


_loud("aibrix_kvcache.integration.vllm.kv_connector __init__.py LOADING")
aibrix_patch_vllm()
_loud("aibrix_kvcache.integration.vllm.kv_connector __init__.py DONE")
