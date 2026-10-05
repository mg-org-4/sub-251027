import copy
import unittest
from types import SimpleNamespace

from services.parameter_lab_queue_receipt import (
    PARAMETER_LAB_RECEIPT_KEY,
    PARAMETER_LAB_RECEIPT_VERSION,
    consume_parameter_lab_queue_receipt,
    register_parameter_lab_queue_receipt_handler,
)


def _payload(marker):
    return {
        "prompt": {"1": {"class_type": "Test", "inputs": {}}},
        "extra_data": {
            "extra_pnginfo": {
                "workflow": {
                    "nodes": [],
                    "extra": {
                        "preserved": {"safe": True},
                        PARAMETER_LAB_RECEIPT_KEY: marker,
                    },
                }
            }
        },
    }


class TestParameterLabQueueReceipt(unittest.TestCase):
    def test_valid_marker_promotes_native_uuid_and_is_removed_from_metadata(self):
        prompt_id = "11111111-1111-4111-8111-111111111111"
        source = _payload(
            {"version": PARAMETER_LAB_RECEIPT_VERSION, "prompt_id": prompt_id}
        )
        result = consume_parameter_lab_queue_receipt(source)

        self.assertEqual(result["prompt_id"], prompt_id)
        extra = result["extra_data"]["extra_pnginfo"]["workflow"]["extra"]
        self.assertNotIn(PARAMETER_LAB_RECEIPT_KEY, extra)
        self.assertEqual(extra["preserved"], {"safe": True})
        self.assertNotIn(PARAMETER_LAB_RECEIPT_KEY, repr(result))

    def test_invalid_or_conflicting_markers_are_stripped_without_identifier_authority(
        self,
    ):
        invalid_markers = (
            None,
            "bad",
            {},
            {"version": 999, "prompt_id": "11111111-1111-4111-8111-111111111111"},
            {"version": PARAMETER_LAB_RECEIPT_VERSION, "prompt_id": "not-a-uuid"},
            {
                "version": PARAMETER_LAB_RECEIPT_VERSION,
                "prompt_id": "11111111-1111-4111-8111-111111111111",
                "extra": True,
            },
        )
        for marker in invalid_markers:
            with self.subTest(marker=marker):
                result = consume_parameter_lab_queue_receipt(_payload(marker))
                self.assertNotIn("prompt_id", result)
                self.assertNotIn(
                    PARAMETER_LAB_RECEIPT_KEY,
                    result["extra_data"]["extra_pnginfo"]["workflow"]["extra"],
                )

        source = _payload(
            {
                "version": PARAMETER_LAB_RECEIPT_VERSION,
                "prompt_id": "11111111-1111-4111-8111-111111111111",
            }
        )
        source["prompt_id"] = "22222222-2222-4222-8222-222222222222"
        result = consume_parameter_lab_queue_receipt(source)
        # CRITICAL: the transient marker is the exact ID the frontend will own after
        # promptQueued. Preserving a different earlier handler value would cross-assign.
        self.assertEqual(result["prompt_id"], "11111111-1111-4111-8111-111111111111")
        self.assertNotIn(
            PARAMETER_LAB_RECEIPT_KEY,
            result["extra_data"]["extra_pnginfo"]["workflow"]["extra"],
        )

    def test_copy_on_write_preserves_input_and_unrelated_shapes(self):
        source = _payload(
            {
                "version": PARAMETER_LAB_RECEIPT_VERSION,
                "prompt_id": "33333333-3333-4333-8333-333333333333",
            }
        )
        original = copy.deepcopy(source)
        result = consume_parameter_lab_queue_receipt(source)

        self.assertEqual(source, original)
        self.assertIsNot(result, source)
        self.assertEqual(result["extra_data"]["extra_pnginfo"]["workflow"]["nodes"], [])
        untouched = {"prompt": {}}
        self.assertIs(consume_parameter_lab_queue_receipt(untouched), untouched)

    def test_registration_is_idempotent_and_uses_official_host_handler(self):
        handlers = []
        server = SimpleNamespace(
            on_prompt_handlers=handlers,
            add_on_prompt_handler=handlers.append,
        )

        self.assertTrue(register_parameter_lab_queue_receipt_handler(server))
        self.assertFalse(register_parameter_lab_queue_receipt_handler(server))
        self.assertEqual(len(handlers), 1)
        promoted = handlers[0](
            _payload(
                {
                    "version": PARAMETER_LAB_RECEIPT_VERSION,
                    "prompt_id": "44444444-4444-4444-8444-444444444444",
                }
            )
        )
        self.assertEqual(promoted["prompt_id"], "44444444-4444-4444-8444-444444444444")


if __name__ == "__main__":
    unittest.main()
