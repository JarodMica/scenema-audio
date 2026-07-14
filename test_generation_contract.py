import unittest
from pathlib import Path

from scenema_audio import ScenemaAudioTTSEngine


class GenerationContractTests(unittest.TestCase):
    def test_generate_mode_preserves_reference_and_vc_choice(self):
        reference = Path("voice.wav")
        mode, resolved_reference, skip_vc = ScenemaAudioTTSEngine._resolve_generation_contract(
            "generate",
            reference,
            False,
        )
        self.assertEqual(mode, "generate")
        self.assertEqual(resolved_reference, reference)
        self.assertFalse(skip_vc)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "generate.*voice_design"):
            ScenemaAudioTTSEngine._resolve_generation_contract("invalid", None, False)

    def test_voice_design_uses_dynamic_generate_without_reference(self):
        mode, reference, skip_vc = ScenemaAudioTTSEngine._resolve_generation_contract(
            "voice_design",
            Path("voice.wav"),
            False,
        )
        self.assertEqual(mode, "generate")
        self.assertIsNone(reference)
        self.assertTrue(skip_vc)


if __name__ == "__main__":
    unittest.main()
