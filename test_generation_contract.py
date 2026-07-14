import inspect
import unittest
from pathlib import Path

from scenema_audio import ScenemaAudioTTSEngine


class GenerationContractTests(unittest.TestCase):
    def test_default_seed_requests_random_generation(self):
        signature = inspect.signature(ScenemaAudioTTSEngine.tts_inference)
        self.assertEqual(signature.parameters["seed"].default, -1)

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

    def test_prompt_controls_are_transparent_and_unmodified(self):
        prompt = ScenemaAudioTTSEngine._build_prompt(
            "Get away from the door right now!",
            voice_description="A young woman yelling at the top of her lungs",
            gender="female",
            scene="Women yelling in a station",
            language="en",
            shot="scene",
            action="She points at the door while yelling",
        )
        self.assertEqual(
            prompt,
            '<speak voice="A young woman yelling at the top of her lungs" gender="female" '
            'language="en" shot="scene" scene="Women yelling in a station">'
            '<action>She points at the door while yelling</action>'
            'Get away from the door right now!</speak>',
        )

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
