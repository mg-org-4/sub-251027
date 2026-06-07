import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

try:
    from aiohttp import web  # noqa: F401

    AIOHTTP_AVAILABLE = True
except ModuleNotFoundError:
    AIOHTTP_AVAILABLE = False


class TestScheduleDeliveryContractModel(unittest.TestCase):
    def test_legacy_delivery_dict_migrates_to_canonical_contract(self):
        from services.scheduler.models import Schedule, TriggerType

        schedule = Schedule.from_dict(
            {
                "schedule_id": "sched_legacy_delivery",
                "name": "Legacy delivery",
                "template_id": "tmpl_1",
                "trigger_type": TriggerType.INTERVAL.value,
                "interval_sec": 300,
                "delivery": {
                    "platform": "telegram",
                    "channel_id": "-1001234567890",
                    "message_thread_id": 456,
                },
            }
        )

        self.assertEqual(
            schedule.delivery,
            {
                "enabled": True,
                "platform": "telegram",
                "target_id": "-1001234567890",
                "thread_id": "456",
                "mode": "reply",
                "failure_alert": True,
            },
        )

    def test_explicit_no_delivery_contract_is_preserved(self):
        from services.scheduler.models import Schedule, TriggerType

        schedule = Schedule(
            schedule_id="sched_no_delivery",
            name="No delivery",
            template_id="tmpl_1",
            trigger_type=TriggerType.INTERVAL,
            interval_sec=300,
            delivery={"mode": "none"},
        )

        self.assertEqual(schedule.delivery, {"enabled": False, "mode": "none"})


@unittest.skipIf(not AIOHTTP_AVAILABLE, "aiohttp not available")
class TestScheduleDeliveryContractApi(unittest.IsolatedAsyncioTestCase):
    async def test_create_normalizes_telegram_topic_delivery_before_persistence(self):
        from api.schedules import ScheduleHandlers

        store = MagicMock()
        store.add.return_value = True

        with patch("api.schedules.get_schedule_store", return_value=store):
            handlers = ScheduleHandlers(template_checker=lambda template_id: True)

        request = AsyncMock()
        request.json = AsyncMock(
            return_value={
                "name": "Topic delivery",
                "template_id": "tmpl_1",
                "trigger_type": "interval",
                "interval_sec": 300,
                "delivery": {
                    "platform": "telegram",
                    "channel_id": "-1001234567890",
                    "message_thread_id": 456,
                },
            }
        )

        response = await handlers.create_schedule(request)

        self.assertEqual(response.status, 201)
        created_schedule = store.add.call_args.args[0]
        self.assertEqual(
            created_schedule.delivery,
            {
                "enabled": True,
                "platform": "telegram",
                "target_id": "-1001234567890",
                "thread_id": "456",
                "mode": "reply",
                "failure_alert": True,
            },
        )
        body = json.loads(response.body)
        self.assertEqual(body["schedule"]["delivery"], created_schedule.delivery)

    async def test_update_delivery_patch_semantics_preserve_clear_and_no_delivery(self):
        from api.schedules import ScheduleHandlers
        from services.scheduler.models import Schedule, TriggerType

        existing = Schedule(
            schedule_id="sched_patch",
            name="Patch delivery",
            template_id="tmpl_1",
            trigger_type=TriggerType.INTERVAL,
            interval_sec=300,
            delivery={
                "platform": "telegram",
                "target_id": "-1001234567890",
                "thread_id": "456",
            },
        )
        store = MagicMock()
        store.get.return_value = existing
        store.update.return_value = True

        with patch("api.schedules.get_schedule_store", return_value=store):
            handlers = ScheduleHandlers(template_checker=lambda template_id: True)

        preserve_request = AsyncMock()
        preserve_request.match_info = {"schedule_id": "sched_patch"}
        preserve_request.json = AsyncMock(return_value={"name": "Renamed"})

        response = await handlers.update_schedule(preserve_request)
        self.assertEqual(response.status, 200)
        self.assertEqual(existing.delivery["thread_id"], "456")

        clear_request = AsyncMock()
        clear_request.match_info = {"schedule_id": "sched_patch"}
        clear_request.json = AsyncMock(return_value={"delivery": None})

        response = await handlers.update_schedule(clear_request)
        self.assertEqual(response.status, 200)
        self.assertIsNone(existing.delivery)

        no_delivery_request = AsyncMock()
        no_delivery_request.match_info = {"schedule_id": "sched_patch"}
        no_delivery_request.json = AsyncMock(
            return_value={"delivery": {"enabled": False}}
        )

        response = await handlers.update_schedule(no_delivery_request)
        self.assertEqual(response.status, 200)
        self.assertEqual(existing.delivery, {"enabled": False, "mode": "none"})

    async def test_malformed_delivery_target_returns_bounded_error_code(self):
        from api.schedules import ScheduleHandlers

        store = MagicMock()

        with patch("api.schedules.get_schedule_store", return_value=store):
            handlers = ScheduleHandlers(template_checker=lambda template_id: True)

        request = AsyncMock()
        request.json = AsyncMock(
            return_value={
                "name": "Bad delivery",
                "template_id": "tmpl_1",
                "trigger_type": "interval",
                "interval_sec": 300,
                "delivery": {
                    "platform": "telegram",
                    "target_id": "-1001234567890",
                    "thread_id": "not-a-topic",
                },
            }
        )

        response = await handlers.create_schedule(request)

        self.assertEqual(response.status, 400)
        body = json.loads(response.body)
        self.assertEqual(body["code"], "delivery_malformed")
        store.add.assert_not_called()

    async def test_conflicting_delivery_target_aliases_return_bounded_error_code(self):
        from api.schedules import ScheduleHandlers

        store = MagicMock()

        with patch("api.schedules.get_schedule_store", return_value=store):
            handlers = ScheduleHandlers(template_checker=lambda template_id: True)

        request = AsyncMock()
        request.json = AsyncMock(
            return_value={
                "name": "Ambiguous delivery",
                "template_id": "tmpl_1",
                "trigger_type": "interval",
                "interval_sec": 300,
                "delivery": {
                    "platform": "feishu",
                    "target_id": "oc_abc123",
                    "channel_id": "oc_other",
                    "workspace_id": "tenant-alpha",
                },
            }
        )

        response = await handlers.create_schedule(request)

        self.assertEqual(response.status, 400)
        body = json.loads(response.body)
        self.assertEqual(body["code"], "delivery_ambiguous")
        store.add.assert_not_called()


class TestScheduleDeliveryContractRunner(unittest.TestCase):
    def test_execute_schedule_submits_normalized_delivery(self):
        from services.scheduler.models import Schedule, TriggerType
        from services.scheduler.runner import SchedulerRunner

        submitted = {}

        async def submit_fn(**kwargs):
            submitted.update(kwargs)
            return {"prompt_id": "prompt_1"}

        schedule = Schedule(
            schedule_id="sched_runner_delivery",
            name="Runner delivery",
            template_id="tmpl_1",
            trigger_type=TriggerType.INTERVAL,
            interval_sec=300,
            delivery={
                "platform": "slack",
                "channel_id": "C123456789",
                "thread_ts": "1234567890.123456",
                "workspace_id": "T123456789",
            },
        )
        runner = SchedulerRunner(submit_fn=submit_fn, tick_interval=10)
        runner._store = MagicMock()
        runner._store.update.return_value = True

        with patch("services.scheduler.runner.get_run_history") as mock_history:
            history = MagicMock()
            history.is_processed.return_value = False
            mock_history.return_value = history

            runner._execute_schedule(schedule, 1000.0)

        self.assertEqual(
            submitted["delivery"],
            {
                "enabled": True,
                "platform": "slack",
                "target_id": "C123456789",
                "thread_id": "1234567890.123456",
                "workspace_id": "T123456789",
                "mode": "reply",
                "failure_alert": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
