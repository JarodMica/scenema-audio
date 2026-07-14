import unittest
from pathlib import Path

from scenema_audio import ScenemaAudioTTSEngine


class GenerationContractTests(unittest.TestCase):
    def test_delivery_only_yelling_controls_become_one_quoted_speech_instruction(self):
        prompt = ScenemaAudioTTSEngine._build_prompt(
            "Get away from the door right now!",
            voice_description="Yelling at top of lungs",
            gender="female",
            scene="Yelling at top of lungs",
            language="en",
            shot="closeup",
            action="Yelling at top of lungs",
        )
        self.assertEqual(
            prompt,
            '<speak voice="A forceful female voice" gender="female" language="en" '
            'shot="closeup"><action>She delivers the quoted words at maximum vocal '
            'intensity</action>Get away from the door right now!</speak>',
        )
        self.assertNotIn("Yelling", prompt)

    def test_delivery_action_preserves_physical_action_without_fragment(self):
        prompt = ScenemaAudioTTSEngine._build_prompt(
            "Get out!",
            voice_description="A tense woman",
            gender="female",
            scene="",
            language="en",
            shot="closeup",
            action="She points at the door while yelling at top of lungs",
        )
        self.assertIn(
            "<action>She points at the door. She delivers the quoted words at maximum "
            "vocal intensity</action>",
            prompt,
        )
        self.assertNotIn("yelling", prompt.lower())

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

    def test_identity_is_preserved_when_yelling_is_part_of_voice_description(self):
        prompt = ScenemaAudioTTSEngine._build_prompt(
            "Run!",
            voice_description="A young woman yelling at the top of her lungs",
            gender="female",
            scene="A dark hallway",
            language="en",
            shot="closeup",
            action="",
        )
        self.assertIn('voice="A young woman using maximum vocal intensity"', prompt)
        self.assertIn('scene="A dark hallway"', prompt)
        self.assertIn(
            "<action>She delivers the quoted words at maximum vocal intensity</action>",
            prompt,
        )
        self.assertNotIn("yelling", prompt.lower())

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "generate.*voice_design"):
            ScenemaAudioTTSEngine._resolve_generation_contract("invalid", None, False)

    def test_intense_delivery_synonyms_are_normalized_case_insensitively(self):
        for cue in (
            "SHOUTING as loudly as possible",
            "Screaming at the top of his lungs",
            "yells at top of lungs",
        ):
            with self.subTest(cue=cue):
                prompt = ScenemaAudioTTSEngine._build_prompt(
                    "Run!",
                    voice_description=cue,
                    gender="male",
                    scene="",
                    language="en",
                    shot="closeup",
                    action="",
                )
                self.assertIn('voice="A forceful male voice"', prompt)
                self.assertIn(
                    "<action>He delivers the quoted words at maximum vocal intensity</action>",
                    prompt,
                )
                self.assertNotRegex(prompt.lower(), r"yell|shout|scream")

    def test_non_delivery_controls_are_unchanged(self):
        prompt = ScenemaAudioTTSEngine._build_prompt(
            "Welcome.",
            voice_description="A warm narrator",
            gender="male",
            scene="A quiet library",
            language="en",
            shot="closeup",
            action="He smiles",
        )
        self.assertEqual(
            prompt,
            '<speak voice="A warm narrator" gender="male" language="en" shot="closeup" '
            'scene="A quiet library"><action>He smiles</action>Welcome.</speak>',
        )

    def test_scene_with_environmental_shouting_is_not_reinterpreted_as_delivery(self):
        prompt = ScenemaAudioTTSEngine._build_prompt(
            "The train arrived.",
            voice_description="A calm narrator",
            gender="female",
            scene="A crowd shouting in the station",
            language="en",
            shot="wide",
            action="",
        )
        self.assertIn('scene="A crowd shouting in the station"', prompt)
        self.assertNotIn("quoted words", prompt)

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
