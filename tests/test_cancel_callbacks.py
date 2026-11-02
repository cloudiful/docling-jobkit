# SPDX-License-Identifier: MIT

"""Winner-only completion callbacks for queued cancellation.

RQ and Ray arbitrate through Redis, so those tests require a real server
(``DOCLING_TEST_REDIS_URL``, default DB15) and fail loudly instead of skipping.
"""

import asyncio
import json
import os
import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import redis as sync_redis

from docling.datamodel.service.callbacks import (
    CallbackSpec,
    ProgressCallbackRequest,
    ProgressKind,
)
from docling.datamodel.service.options import ConvertDocumentsOptions
from docling.datamodel.service.targets import InBodyTarget
from docling.datamodel.service.tasks import TaskType

from docling_jobkit.datamodel.task import Task
from docling_jobkit.datamodel.task_meta import TaskStatus
from docling_jobkit.orchestrators.base_orchestrator import (
    TaskCancellationConflictError,
)
from docling_jobkit.orchestrators.completion_callbacks import (
    emit_task_completed_callback,
)

_REDIS_URL = os.getenv("DOCLING_TEST_REDIS_URL", "redis://127.0.0.1:6379/15")


def _unique() -> str:
    return uuid.uuid4().hex[:12]


def _callback_spec() -> CallbackSpec:
    return CallbackSpec(
        url="https://example.invalid/hook",
        headers={"Authorization": "Bearer token"},
    )


class _RecordingInvoker:
    calls: list[dict[str, Any]] = []

    def invoke_callbacks_async(self, **kwargs: Any) -> None:
        _RecordingInvoker.calls.append(kwargs)


class _ExplodingInvoker:
    def invoke_callbacks_async(self, **kwargs: Any) -> None:
        raise RuntimeError("callback sender is down")


class _FailingNotifier:
    async def add_task(self, task_id: str) -> None:
        pass

    async def remove_task(self, task_id: str) -> None:
        pass

    async def notify_task_subscribers(self, task_id: str) -> None:
        raise RuntimeError("notifier is down")

    async def notify_queue_positions(self) -> None:
        raise AssertionError(
            "queue notification must not follow a failed subscriber notification"
        )


@pytest.fixture
def recorded_callbacks(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    _RecordingInvoker.calls = []
    monkeypatch.setattr(
        "docling_jobkit.orchestrators.completion_callbacks.CallbackInvoker",
        _RecordingInvoker,
    )
    return _RecordingInvoker.calls


@pytest.fixture
def sync_conn():
    conn = sync_redis.Redis.from_url(_REDIS_URL)
    try:
        conn.ping()
    except Exception as exc:  # pragma: no cover - environment failure
        pytest.fail(
            f"Redis at {_REDIS_URL} is required for cancellation-callback "
            f"coverage: {exc}"
        )
    yield conn
    conn.close()


class _RedisBaseline:
    """Captures the pre-test exact keyspace so cleanup removes only new keys."""

    def __init__(self, conn) -> None:
        self._conn = conn
        self._keys = set(conn.keys())
        self._tenants = set(conn.smembers("tenants:known"))
        self._queues = set(conn.smembers("rq:queues"))
        self._had_tenants = bool(conn.exists("tenants:known"))
        self._had_queues = bool(conn.exists("rq:queues"))

    def purge(
        self,
        token: str,
        task_ids: list[str],
        *,
        owned_tenants: tuple[str, ...] = (),
        owned_queues: tuple[str, ...] = (),
    ) -> None:
        """Delete only keys proven to belong to this fixture run."""
        token_bytes = token.encode("utf-8")
        new_keys = set(self._conn.keys()) - self._keys
        owned_keys = {key for key in new_keys if token_bytes in key}
        for task_id in task_ids:
            for suffix in ("", ":dispatch", ":execution"):
                key = f"task:{task_id}{suffix}".encode()
                if key not in self._keys:
                    owned_keys.add(key)
            for key in (
                f"docling:tasks:{task_id}:metadata".encode(),
                f"rq:job:{task_id}".encode(),
            ):
                if key not in self._keys:
                    owned_keys.add(key)
        if owned_keys:
            self._conn.delete(*owned_keys)

        current_tenants = set(self._conn.smembers("tenants:known"))
        added_tenants = current_tenants - self._tenants
        owned_tenants_bytes = {
            tenant.encode("utf-8")
            for tenant in owned_tenants
            if tenant.encode("utf-8") in added_tenants
        }
        if owned_tenants_bytes:
            self._conn.srem("tenants:known", *owned_tenants_bytes)
        if not self._had_tenants and self._conn.scard("tenants:known") == 0:
            self._conn.delete("tenants:known")

        owned_queues_bytes = {
            queue.encode("utf-8")
            for queue in owned_queues
            if queue.encode("utf-8") not in self._queues
        }
        if owned_queues_bytes:
            self._conn.srem("rq:queues", *owned_queues_bytes)
        if not self._had_queues and self._conn.scard("rq:queues") == 0:
            self._conn.delete("rq:queues")


async def _wait_until(predicate, timeout: float = 10.0, message: str = "timeout"):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"{message} (waited {timeout}s)")


def _assert_cancelled_callback(calls: list[dict[str, Any]], task_id: str) -> None:
    assert len(calls) == 1, f"expected exactly one callback, got {len(calls)}"
    call = calls[0]
    assert call["task_id"] == task_id
    progress = call["progress"]
    assert progress.kind == ProgressKind.TASK_COMPLETED
    assert progress.task_status == "cancelled"
    assert progress.failure is None
    request = ProgressCallbackRequest(task_id=task_id, progress=progress)
    body = json.loads(request.model_dump_json())
    assert body == {
        "task_id": task_id,
        "progress": {
            "kind": "task_completed",
            "task_status": "cancelled",
            "failure": None,
        },
    }


def test_task_status_is_the_canonical_docling_enum() -> None:
    from docling.datamodel.service.tasks import TaskStatus as CanonicalTaskStatus

    assert TaskStatus is CanonicalTaskStatus
    assert {status.value for status in TaskStatus} == {
        "pending",
        "started",
        "success",
        "failure",
        "cancelled",
    }


def test_emit_task_completed_callback_accepts_cancelled(
    recorded_callbacks: list[dict[str, Any]],
) -> None:
    task = Task(task_id="t-cancelled", callbacks=[_callback_spec()])

    emit_task_completed_callback(task, "cancelled")

    _assert_cancelled_callback(recorded_callbacks, "t-cancelled")


def _local_orchestrator(num_workers: int = 1):
    from docling_jobkit.orchestrators.local.orchestrator import (
        LocalOrchestrator,
        LocalOrchestratorConfig,
    )

    converter_manager = MagicMock()
    converter_manager.config.allow_external_plugins = False
    converter_manager.config.max_file_size = None
    converter_manager.convert_documents.return_value = []
    config = LocalOrchestratorConfig(num_workers=num_workers, shared_models=True)
    return LocalOrchestrator(config=config, converter_manager=converter_manager)


async def _enqueue_local(orch, callbacks: list[CallbackSpec]) -> Task:
    return await orch.enqueue(
        sources=[],
        convert_options=ConvertDocumentsOptions(),
        targets=[InBodyTarget()],
        callbacks=callbacks,
    )


class TestLocalCancelCallbacks:
    @pytest.mark.asyncio
    async def test_winner_emits_one_cancelled_callback(
        self, recorded_callbacks: list[dict[str, Any]]
    ):
        orch = _local_orchestrator()
        task = await _enqueue_local(orch, [_callback_spec()])

        cancelled = await orch.cancel_queued_task(task.task_id)

        assert cancelled.task_status is TaskStatus.CANCELLED
        _assert_cancelled_callback(recorded_callbacks, task.task_id)

    @pytest.mark.asyncio
    async def test_repeated_cancel_emits_once(
        self, recorded_callbacks: list[dict[str, Any]]
    ):
        orch = _local_orchestrator()
        task = await _enqueue_local(orch, [_callback_spec()])

        await orch.cancel_queued_task(task.task_id)
        await orch.cancel_queued_task(task.task_id)

        _assert_cancelled_callback(recorded_callbacks, task.task_id)

    @pytest.mark.asyncio
    async def test_claim_winner_emits_nothing(
        self, recorded_callbacks: list[dict[str, Any]]
    ):
        orch = _local_orchestrator()
        task = await _enqueue_local(orch, [_callback_spec()])
        assert orch.claim_task_started(task.task_id) is True

        with pytest.raises(TaskCancellationConflictError):
            await orch.cancel_queued_task(task.task_id)

        assert recorded_callbacks == []

    @pytest.mark.asyncio
    async def test_no_callback_without_configured_callbacks(
        self, recorded_callbacks: list[dict[str, Any]]
    ):
        orch = _local_orchestrator()
        task = await _enqueue_local(orch, [])

        await orch.cancel_queued_task(task.task_id)

        assert recorded_callbacks == []

    @pytest.mark.asyncio
    async def test_worker_skip_does_not_emit_again(
        self, recorded_callbacks: list[dict[str, Any]]
    ):
        orch = _local_orchestrator()
        task = await _enqueue_local(orch, [_callback_spec()])
        await orch.cancel_queued_task(task.task_id)
        assert len(recorded_callbacks) == 1

        queue_task = asyncio.create_task(orch.process_queue())
        try:
            # The cancelled task keeps its (now meaningless) queue entry; the
            # worker must dequeue and skip it without a second callback.
            await _wait_until(
                orch.task_queue.empty, message="worker never drained the queue"
            )
        finally:
            queue_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queue_task

        assert orch.tasks[task.task_id].task_status is TaskStatus.CANCELLED
        _assert_cancelled_callback(recorded_callbacks, task.task_id)

    @pytest.mark.asyncio
    async def test_sender_failure_keeps_cancel_terminal(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(
            "docling_jobkit.orchestrators.completion_callbacks.CallbackInvoker",
            _ExplodingInvoker,
        )
        orch = _local_orchestrator()
        task = await _enqueue_local(orch, [_callback_spec()])

        cancelled = await orch.cancel_queued_task(task.task_id)

        assert cancelled.task_status is TaskStatus.CANCELLED
        assert cancelled.is_completed()

    @pytest.mark.asyncio
    async def test_notifier_failure_keeps_callback_and_durable_cancel(
        self, recorded_callbacks: list[dict[str, Any]]
    ):
        orch = _local_orchestrator()
        task = await _enqueue_local(orch, [_callback_spec()])
        orch.notifier = _FailingNotifier()

        with pytest.raises(RuntimeError, match="notifier is down"):
            await orch.cancel_queued_task(task.task_id)

        assert orch.tasks[task.task_id].task_status is TaskStatus.CANCELLED
        _assert_cancelled_callback(recorded_callbacks, task.task_id)

        orch.notifier = None
        await orch.cancel_queued_task(task.task_id)
        _assert_cancelled_callback(recorded_callbacks, task.task_id)


@pytest.fixture
def rq_token() -> str:
    return _unique()


@pytest.fixture
async def rq_orchestrator(rq_token: str, sync_conn):
    from docling_jobkit.orchestrators.rq.orchestrator import (
        RQOrchestrator,
        RQOrchestratorConfig,
    )

    config = RQOrchestratorConfig(
        redis_url=_REDIS_URL,
        queue_name=f"cancelcb-{rq_token}",
        results_prefix=f"docling:cancelcb:{rq_token}:results",
        heartbeat_key_prefix=f"docling:cancelcb:{rq_token}:alive",
        sub_channel=f"docling:cancelcb:{rq_token}:updates",
        results_ttl=600,
    )
    orch = RQOrchestrator(config=config)
    baseline = _RedisBaseline(sync_conn)
    created: list[str] = []
    yield orch, created
    await orch.close()
    baseline.purge(
        rq_token,
        created,
        owned_queues=(f"rq:queue:cancelcb-{rq_token}",),
    )


async def _enqueue_rq(orch, created: list[str], callbacks: list[CallbackSpec]) -> Task:
    task = await orch.enqueue(
        sources=[],
        convert_options=ConvertDocumentsOptions(),
        targets=[InBodyTarget()],
        callbacks=callbacks,
    )
    created.append(task.task_id)
    return task


class TestRqCancelCallbacks:
    @pytest.mark.asyncio
    async def test_winner_emits_one_cancelled_callback(
        self, rq_orchestrator, recorded_callbacks: list[dict[str, Any]]
    ):
        orch, created = rq_orchestrator
        task = await _enqueue_rq(orch, created, [_callback_spec()])

        cancelled = await orch.cancel_queued_task(task.task_id)

        assert cancelled.task_status is TaskStatus.CANCELLED
        _assert_cancelled_callback(recorded_callbacks, task.task_id)

    @pytest.mark.asyncio
    async def test_repeated_cancel_emits_once(
        self, rq_orchestrator, recorded_callbacks: list[dict[str, Any]]
    ):
        orch, created = rq_orchestrator
        task = await _enqueue_rq(orch, created, [_callback_spec()])

        await orch.cancel_queued_task(task.task_id)
        await orch.cancel_queued_task(task.task_id)

        _assert_cancelled_callback(recorded_callbacks, task.task_id)

    @pytest.mark.asyncio
    async def test_claim_winner_emits_nothing(
        self, rq_orchestrator, sync_conn, recorded_callbacks: list[dict[str, Any]]
    ):
        from docling_jobkit.orchestrators.rq.worker import claim_task_started

        orch, created = rq_orchestrator
        task = await _enqueue_rq(orch, created, [_callback_spec()])
        assert (
            claim_task_started(sync_conn, task.task_id, metadata_ttl=600)
            == TaskStatus.STARTED
        )

        with pytest.raises(TaskCancellationConflictError):
            await orch.cancel_queued_task(task.task_id)

        assert recorded_callbacks == []

    @pytest.mark.asyncio
    async def test_worker_skip_does_not_emit_again(
        self, rq_orchestrator, sync_conn, tmp_path, recorded_callbacks
    ):
        from docling_jobkit.orchestrators.rq.worker import _run_docling_task

        orch, created = rq_orchestrator
        task = await _enqueue_rq(orch, created, [_callback_spec()])
        await orch.cancel_queued_task(task.task_id)
        assert len(recorded_callbacks) == 1

        converter = MagicMock()
        converter.config.max_file_size = None
        job = MagicMock()
        job.connection = sync_conn

        with (
            patch(
                "docling_jobkit.orchestrators.rq.worker.get_current_job",
                return_value=job,
            ),
            patch.object(sync_conn, "publish") as publish,
        ):
            result = _run_docling_task(task, converter, orch.config, tmp_path)

        assert result is None
        converter.convert_documents.assert_not_called()
        publish.assert_not_called()
        _assert_cancelled_callback(recorded_callbacks, task.task_id)

    @pytest.mark.asyncio
    async def test_sender_failure_keeps_cancel_terminal(
        self, rq_orchestrator, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(
            "docling_jobkit.orchestrators.completion_callbacks.CallbackInvoker",
            _ExplodingInvoker,
        )
        orch, created = rq_orchestrator
        task = await _enqueue_rq(orch, created, [_callback_spec()])

        cancelled = await orch.cancel_queued_task(task.task_id)

        assert cancelled.task_status is TaskStatus.CANCELLED
        durable = await orch._get_task_from_redis(task.task_id)
        assert durable is not None
        assert durable.task_status is TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_notifier_failure_keeps_callback_and_durable_cancel(
        self,
        rq_orchestrator,
        recorded_callbacks: list[dict[str, Any]],
        monkeypatch: pytest.MonkeyPatch,
    ):
        orch, created = rq_orchestrator
        task = await _enqueue_rq(orch, created, [_callback_spec()])

        gate_released: list[bool] = []
        real_emit = emit_task_completed_callback

        def emit_and_record(task_obj: Task, status: str) -> None:
            gate_released.append(not orch._redis_gate._semaphore.locked())
            real_emit(task_obj, status)

        monkeypatch.setattr(
            "docling_jobkit.orchestrators.rq.orchestrator.emit_task_completed_callback",
            emit_and_record,
        )
        orch.notifier = _FailingNotifier()

        with pytest.raises(RuntimeError, match="notifier is down"):
            await orch.cancel_queued_task(task.task_id)

        durable = await orch._get_task_from_redis(task.task_id)
        assert durable is not None
        assert durable.task_status is TaskStatus.CANCELLED
        _assert_cancelled_callback(recorded_callbacks, task.task_id)
        assert gate_released == [True]

        orch.notifier = None
        await orch.cancel_queued_task(task.task_id)
        _assert_cancelled_callback(recorded_callbacks, task.task_id)
        assert gate_released == [True]


@pytest.fixture
def ray_token() -> str:
    return _unique()


@pytest.fixture
async def ray_orchestrator(ray_token: str, sync_conn):
    pytest.importorskip("ray", reason="Ray extra is required")
    from docling_jobkit.orchestrators._redis_gate import RedisCallerGate
    from docling_jobkit.orchestrators.ray.config import RayOrchestratorConfig
    from docling_jobkit.orchestrators.ray.orchestrator import RayOrchestrator
    from docling_jobkit.orchestrators.ray.redis_helper import RedisStateManager

    state_manager = RedisStateManager(
        redis_url=_REDIS_URL,
        results_prefix=f"docling:cancelcb:{ray_token}:results",
        sub_channel=f"docling:cancelcb:{ray_token}:updates",
    )
    config = RayOrchestratorConfig(redis_url=_REDIS_URL)
    with patch.object(RayOrchestrator, "__init__", lambda self, **kw: None):
        orch = object.__new__(RayOrchestrator)
    orch.config = config
    orch.allowed_target_kinds = None
    orch.tasks = {}
    orch.notifier = None
    orch.cm = MagicMock()
    orch.cm.config = MagicMock()
    orch.redis_manager = state_manager
    orch._redis_gate = RedisCallerGate(config.redis_gate_concurrency or 1)
    orch.dispatcher = None
    baseline = _RedisBaseline(sync_conn)
    created: list[str] = []
    yield orch, created
    await state_manager.disconnect()
    baseline.purge(
        ray_token,
        created,
        owned_tenants=(f"cancelcb-{ray_token}",),
    )


async def _enqueue_ray(
    orch, created: list[str], tenant_id: str, callbacks: list[CallbackSpec]
) -> Task:
    task = Task(
        task_id=str(uuid.uuid4()),
        task_type=TaskType.CONVERT,
        target=InBodyTarget(),
        convert_options=ConvertDocumentsOptions(),
        sources=[],
        metadata={"tenant_id": tenant_id},
        callbacks=callbacks,
    )
    await orch.redis_manager.set_task_metadata(
        task_id=task.task_id,
        tenant_id=tenant_id,
        task_type=TaskType.CONVERT,
        task_size=0,
        status=TaskStatus.PENDING,
    )
    await orch.redis_manager.enqueue_task(tenant_id, task)
    created.append(task.task_id)
    return task


class TestRayCancelCallbacks:
    @pytest.mark.asyncio
    async def test_queued_winner_emits_one_cancelled_callback(
        self, ray_orchestrator, ray_token: str, recorded_callbacks: list[dict[str, Any]]
    ):
        orch, created = ray_orchestrator
        tenant_id = f"cancelcb-{ray_token}"
        task = await _enqueue_ray(orch, created, tenant_id, [_callback_spec()])

        cancelled = await orch.cancel_queued_task(task.task_id)

        assert cancelled.task_status is TaskStatus.CANCELLED
        # The returned task must not leak the credential-bearing callback specs.
        assert cancelled.callbacks == []
        _assert_cancelled_callback(recorded_callbacks, task.task_id)

    @pytest.mark.asyncio
    async def test_dispatched_winner_emits_callback_after_dispatch_state_deleted(
        self, ray_orchestrator, ray_token: str, recorded_callbacks: list[dict[str, Any]]
    ):
        orch, created = ray_orchestrator
        tenant_id = f"cancelcb-{ray_token}"
        task = await _enqueue_ray(orch, created, tenant_id, [_callback_spec()])
        assert await orch.redis_manager.dispatch_task_atomic(tenant_id, task.task_id, 0)

        cancelled = await orch.cancel_queued_task(task.task_id)

        assert cancelled.task_status is TaskStatus.CANCELLED
        _assert_cancelled_callback(recorded_callbacks, task.task_id)
        assert await orch.redis_manager.get_task_dispatch_hash(task.task_id) == {}

    @pytest.mark.asyncio
    async def test_repeated_cancel_emits_once(
        self, ray_orchestrator, ray_token: str, recorded_callbacks: list[dict[str, Any]]
    ):
        orch, created = ray_orchestrator
        tenant_id = f"cancelcb-{ray_token}"
        task = await _enqueue_ray(orch, created, tenant_id, [_callback_spec()])

        await orch.cancel_queued_task(task.task_id)
        await orch.cancel_queued_task(task.task_id)

        _assert_cancelled_callback(recorded_callbacks, task.task_id)

    @pytest.mark.asyncio
    async def test_claim_winner_emits_nothing(
        self, ray_orchestrator, ray_token: str, recorded_callbacks: list[dict[str, Any]]
    ):
        orch, created = ray_orchestrator
        tenant_id = f"cancelcb-{ray_token}"
        task = await _enqueue_ray(orch, created, tenant_id, [_callback_spec()])
        assert await orch.redis_manager.mark_task_started(task.task_id, tenant_id)

        with pytest.raises(TaskCancellationConflictError):
            await orch.cancel_queued_task(task.task_id)

        assert recorded_callbacks == []

    @pytest.mark.asyncio
    async def test_sender_failure_keeps_cancel_terminal(
        self,
        ray_orchestrator,
        ray_token: str,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(
            "docling_jobkit.orchestrators.completion_callbacks.CallbackInvoker",
            _ExplodingInvoker,
        )
        orch, created = ray_orchestrator
        tenant_id = f"cancelcb-{ray_token}"
        task = await _enqueue_ray(orch, created, tenant_id, [_callback_spec()])

        cancelled = await orch.cancel_queued_task(task.task_id)

        assert cancelled.task_status is TaskStatus.CANCELLED
        metadata = await orch.redis_manager.get_task_metadata_model(task.task_id)
        assert metadata is not None
        assert metadata.status is TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_notifier_failure_keeps_callback_and_durable_cancel(
        self,
        ray_orchestrator,
        ray_token: str,
        recorded_callbacks: list[dict[str, Any]],
        monkeypatch: pytest.MonkeyPatch,
    ):
        orch, created = ray_orchestrator
        tenant_id = f"cancelcb-{ray_token}"
        task = await _enqueue_ray(orch, created, tenant_id, [_callback_spec()])

        gate_released: list[bool] = []
        real_emit = emit_task_completed_callback

        def emit_and_record(task_obj: Task, status: str) -> None:
            gate_released.append(not orch._redis_gate._semaphore.locked())
            real_emit(task_obj, status)

        monkeypatch.setattr(
            "docling_jobkit.orchestrators.ray.orchestrator.emit_task_completed_callback",
            emit_and_record,
        )
        orch.notifier = _FailingNotifier()

        with pytest.raises(RuntimeError, match="notifier is down"):
            await orch.cancel_queued_task(task.task_id)

        metadata = await orch.redis_manager.get_task_metadata_model(task.task_id)
        assert metadata is not None
        assert metadata.status is TaskStatus.CANCELLED
        _assert_cancelled_callback(recorded_callbacks, task.task_id)
        assert gate_released == [True]

        orch.notifier = None
        await orch.cancel_queued_task(task.task_id)
        _assert_cancelled_callback(recorded_callbacks, task.task_id)
        assert gate_released == [True]
