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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = logging.getLogger(__name__)


def _loud(msg: str) -> None:
    """Unconditional stderr write + logger. For debugging fork race."""
    pid = os.getpid()
    sys.stderr.write(f"[AIBRIX-DEBUG pid={pid}] {msg}\n")
    sys.stderr.flush()
    logger.warning("[AIBRIX-DEBUG pid=%d] %s", pid, msg)


VLLM_V1_WORKER_GPU_MODEL_RUNNER_MODULE = "vllm.v1.worker.gpu_model_runner"

_patches_applied = False


def _apply_gpu_model_runner_patches(module):
    """Apply patches to an already-imported gpu_model_runner module."""
    GPUModelRunner = module.GPUModelRunner

    _loud(
        f"_apply_gpu_model_runner_patches: execute_model.__name__="
        f"{GPUModelRunner.execute_model.__name__}"
    )

    if GPUModelRunner.execute_model.__name__ == "_patched_execute_model":
        _loud("Already monkey-patched, skipping")
        return

    import inspect
    sig = inspect.signature(GPUModelRunner._update_states)
    if "load_results" in sig.parameters:
        _loud("vLLM source patch detected, skipping monkey-patch")
        return

    _loud("Applying patches to vLLM GPUModelRunner...")
    from vllm.distributed.kv_transfer import has_kv_transfer_group

    # ---- Patch 1: execute_model wraps load + _preprocess_* adjust ----
    _orig_execute_model = GPUModelRunner.execute_model

    def _patched_execute_model(self, scheduler_output, *args, **kwargs):
        """Patched execute_model (Attempt C):
        - Call start_load_kv_before_update() to load from L1 into GPU
        - Apply _preprocess_* with delta detection (avoid double-adjust
          of tokens already counted by scheduler via get_num_new_matched_tokens)
        """
        self._aibrix_load_results = {}

        if has_kv_transfer_group():
            from vllm.distributed.kv_transfer import get_kv_transfer_group
            kv_connector = get_kv_transfer_group()
            if scheduler_output.kv_connector_metadata is not None:
                kv_connector.bind_connector_metadata(
                    scheduler_output.kv_connector_metadata
                )
                self._aibrix_load_results = (
                    kv_connector.start_load_kv_before_update()
                )

        return _orig_execute_model(self, scheduler_output, *args, **kwargs)

    GPUModelRunner.execute_model = _patched_execute_model

    # ---- Patch 2: _update_states with delta-aware preprocessing ----
    _orig_update_states = GPUModelRunner._update_states

    def _patched_update_states(self, scheduler_output):
        load_results = getattr(self, "_aibrix_load_results", {})

        # scheduler_external_tokens is populated via
        # AIBrixOffloadingConnector.connector_scheduler._request_external_tokens.
        # But we only have access to that on the scheduler process, not worker.
        # For the delta detection, we piggyback on the connector metadata:
        # the AIBrixOffloadingConnectorRequestMetadata has a load_len field
        # which represents num_external_tokens promised by scheduler.
        meta = getattr(scheduler_output, "kv_connector_metadata", None)
        scheduler_adjusted_by_req: dict[str, int] = {}
        if meta is not None and hasattr(meta, "requests"):
            for req_id, req_meta in meta.requests.items():
                scheduler_adjusted_by_req[req_id] = getattr(
                    req_meta, "load_len", 0
                )

        _preprocess_new_reqs(
            scheduler_output, load_results, scheduler_adjusted_by_req
        )
        _preprocess_cached_reqs(
            scheduler_output, load_results, scheduler_adjusted_by_req
        )

        _orig_update_states(self, scheduler_output)

    GPUModelRunner._update_states = _patched_update_states

    GPUModelRunner._aibrix_patched = True
    _loud("GPUModelRunner patched successfully")


def _preprocess_new_reqs(
    scheduler_output: "SchedulerOutput",
    load_results: dict,
    scheduler_adjusted_by_req: dict,
) -> None:
    """Adjust new requests — but only by the DELTA between what the worker
    actually loaded and what the scheduler already counted externally.
    """
    if not load_results:
        return

    for new_req_data in scheduler_output.scheduled_new_reqs:
        req_id = new_req_data.req_id
        num_worker_loaded = load_results.get(req_id, 0)
        if num_worker_loaded <= 0:
            continue

        scheduler_adjusted = scheduler_adjusted_by_req.get(req_id, 0)
        # The scheduler already incremented num_computed_tokens by
        # scheduler_adjusted. The worker loaded num_worker_loaded in total
        # (may be equal or more, never less in the happy path).
        # We only adjust by the DELTA to avoid double-counting.
        delta = num_worker_loaded - scheduler_adjusted

        if delta <= 0:
            continue

        num_scheduled = scheduler_output.num_scheduled_tokens[req_id]
        # leave at least 1 token for compute
        if delta >= num_scheduled:
            delta = num_scheduled - 1

        if delta <= 0:
            continue

        new_req_data.num_computed_tokens += delta
        scheduler_output.num_scheduled_tokens[req_id] -= delta
        scheduler_output.total_num_scheduled_tokens -= delta


def _preprocess_cached_reqs(
    scheduler_output: "SchedulerOutput",
    load_results: dict,
    scheduler_adjusted_by_req: dict,
) -> None:
    """Adjust cached/running requests with delta-only logic."""
    if not load_results:
        return

    req_data = scheduler_output.scheduled_cached_reqs

    for i, req_id in enumerate(req_data.req_ids):
        num_worker_loaded = load_results.get(req_id, 0)
        if num_worker_loaded <= 0:
            continue

        scheduler_adjusted = scheduler_adjusted_by_req.get(req_id, 0)
        delta = num_worker_loaded - scheduler_adjusted

        if delta <= 0:
            continue

        num_scheduled = scheduler_output.num_scheduled_tokens[req_id]
        if delta >= num_scheduled:
            delta = num_scheduled - 1

        if delta <= 0:
            continue

        req_data.num_computed_tokens[i] += delta
        if req_data.new_token_ids and req_data.new_token_ids[i]:
            req_data.new_token_ids[i] = req_data.new_token_ids[i][delta:]
        scheduler_output.num_scheduled_tokens[req_id] -= delta
        scheduler_output.total_num_scheduled_tokens -= delta


def aibrix_patch_vllm():
    """Apply AIBrix patches to vLLM."""
    _loud("aibrix_patch_vllm() invoked")
    try:
        module = importlib.import_module(VLLM_V1_WORKER_GPU_MODEL_RUNNER_MODULE)
        _apply_gpu_model_runner_patches(module)
    except ImportError as e:
        _loud(f"Failed to patch gpu_model_runner: {e}")

    global _patches_applied
    _patches_applied = True


_loud("aibrix_kvcache.integration.vllm.kv_connector __init__.py LOADING")
aibrix_patch_vllm()
_loud("aibrix_kvcache.integration.vllm.kv_connector __init__.py DONE")
