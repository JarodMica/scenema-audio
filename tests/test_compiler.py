# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

"""Tests for XML prompt compilation."""

from audio_core.compiler import (
    compile_prompt,
    extract_speech_text,
    compile_chunk_prompt,
    extract_sentence_actions,
)


class TestCompilePrompt:
    def test_full_prompt_with_scene_and_actions(self):
        xml = (
            '<speak voice="Deep male voice, gravitas" scene="A dimly lit room" gender="male">'
            "<action>He takes a slow breath</action>"
            "Many years later, as he faced the firing squad."
            "<action>He pauses</action>"
            "Colonel Aureliano Buendia was to remember."
            "</speak>"
        )
        result = compile_prompt(xml)

        assert "Close-up in A dimly lit room." in result.prompt
        assert "He takes a slow breath." in result.prompt
        assert '"Many years later, as he faced the firing squad."' in result.prompt
        assert '"Colonel Aureliano Buendia was to remember."' in result.prompt
        assert "Deep male voice, gravitas." in result.prompt
        assert result.voice == "Deep male voice, gravitas"
        assert result.scene == "A dimly lit room"
        assert result.language == "en"
        assert result.gender == "male"

    def test_quoted_speech_delivery_instruction_is_attached_in_closeup_mode(self):
        xml = (
            '<speak voice="Forceful female voice" gender="female">'
            '<action>She delivers the quoted words at maximum vocal intensity</action>'
            'Run!'
            '</speak>'
        )
        result = compile_prompt(xml)
        assert (
            'She delivers the quoted words at maximum vocal intensity: "Run!"'
            in result.prompt
        )
        assert "intensity. \"Run!\"" not in result.prompt

    def test_minimal_prompt_no_scene_no_actions(self):
        xml = '<speak voice="Narrator" gender="male">Hello world.</speak>'
        result = compile_prompt(xml)

        assert "Close-up in a person speaking to camera." in result.prompt
        assert '"Hello world."' in result.prompt
        assert "Narrator." in result.prompt

    def test_multiple_text_blocks(self):
        xml = (
            '<speak voice="V" gender="female">'
            "First sentence."
            "<action>Pause</action>"
            "Second sentence."
            "<action>Laugh</action>"
            "Third sentence."
            "</speak>"
        )
        result = compile_prompt(xml)

        assert '"First sentence."' in result.prompt
        assert '"Second sentence."' in result.prompt
        assert '"Third sentence."' in result.prompt
        # Closeup mode: no connector
        assert "speaks:" not in result.prompt
        assert "continues:" not in result.prompt

    def test_action_only_before_text(self):
        xml = (
            '<speak voice="V" gender="female">'
            "<action>She clears her throat</action>"
            "Good morning everyone."
            "</speak>"
        )
        result = compile_prompt(xml)

        assert "She clears her throat." in result.prompt
        assert '"Good morning everyone."' in result.prompt

    def test_trailing_punctuation_added(self):
        xml = '<speak voice="V" gender="male">No punctuation here</speak>'
        result = compile_prompt(xml)

        assert '"No punctuation here."' in result.prompt

    def test_existing_punctuation_preserved(self):
        xml = '<speak voice="V" gender="male">Already has punctuation!</speak>'
        result = compile_prompt(xml)

        assert '"Already has punctuation!"' in result.prompt

    def test_question_mark_preserved(self):
        xml = '<speak voice="V" gender="male">Is this a question?</speak>'
        result = compile_prompt(xml)

        assert '"Is this a question?"' in result.prompt

    def test_speech_text_extraction(self):
        xml = (
            '<speak voice="V" gender="male">'
            "<action>He sighs</action>"
            "First part."
            "<action>He pauses</action>"
            "Second part."
            "</speak>"
        )
        text = extract_speech_text(xml)
        assert text == "First part. Second part."

    def test_speech_text_excludes_actions_and_sounds(self):
        xml = (
            '<speak voice="V" gender="male">'
            "<action>Action one</action>"
            "<sound>A door slams</sound>"
            "Speech only."
            "<action>Action two</action>"
            "</speak>"
        )
        text = extract_speech_text(xml)
        assert "Action" not in text
        assert "door" not in text
        assert text == "Speech only."

    def test_language_attribute(self):
        xml = '<speak voice="V" language="es" gender="male">Hola mundo.</speak>'
        result = compile_prompt(xml)
        assert result.language == "es"

    def test_gender_attribute(self):
        xml = '<speak voice="V" gender="female">Hello.</speak>'
        result = compile_prompt(xml)
        assert result.gender == "female"

    def test_gender_defaults_to_male(self):
        xml = '<speak voice="V">Hello.</speak>'
        result = compile_prompt(xml)
        assert result.gender == "male"

    def test_shot_closeup_default(self):
        xml = '<speak voice="V" gender="male" scene="A room">Hello.</speak>'
        result = compile_prompt(xml)
        assert "Close-up in A room." in result.prompt
        assert result.shot == "closeup"

    def test_shot_wide(self):
        xml = '<speak voice="V" gender="male" scene="A busy street" shot="wide">Hello.</speak>'
        result = compile_prompt(xml)
        assert "Wide shot of A busy street." in result.prompt
        assert "Close-up" not in result.prompt
        assert result.shot == "wide"

    def test_shot_scene(self):
        xml = '<speak voice="V" gender="male" scene="Thunder and rain" shot="scene">Hello.</speak>'
        result = compile_prompt(xml)
        assert "Thunder and rain." in result.prompt
        assert "Close-up" not in result.prompt
        assert "Wide shot" not in result.prompt
        assert result.shot == "scene"

    def test_sound_block_in_closeup(self):
        xml = (
            '<speak voice="V" gender="male">'
            "<sound>A door creaks open</sound>"
            "Hello there."
            "</speak>"
        )
        result = compile_prompt(xml)
        assert "A door creaks open." in result.prompt
        assert '"Hello there."' in result.prompt

    def test_scene_mode_action_connector(self):
        xml = (
            '<speak voice="Tense whisper" gender="male" scene="Dark room" shot="scene">'
            "<action>He whispers urgently</action>"
            "Get out now."
            "</speak>"
        )
        result = compile_prompt(xml)
        assert 'He whispers urgently: "Get out now."' in result.prompt

    def test_scene_mode_sfx_reinforcement(self):
        xml = (
            '<speak voice="V" gender="male" scene="Heavy rain, thunder" shot="scene">'
            "<sound>Lightning cracks</sound>"
            "Run!"
            "</speak>"
        )
        result = compile_prompt(xml)
        # Scene repeated at end as SFX reinforcement
        assert result.prompt.endswith("Heavy rain, thunder.")

    def test_scene_mode_full_radio_drama(self):
        xml = (
            '<speak voice="Tense male whisper" gender="male" '
            'scene="Dark room, heavy rain" shot="scene">'
            "<sound>A phone rings twice then stops</sound>"
            "<action>He picks up the receiver and speaks in a low whisper</action>"
            "Its done. The package is at the location."
            "<sound>Thunder rumbles in the distance</sound>"
            "<action>He continues urgently</action>"
            "You have thirty minutes."
            "</speak>"
        )
        result = compile_prompt(xml)
        assert "Dark room, heavy rain." in result.prompt
        assert "A phone rings twice then stops." in result.prompt
        assert (
            'He picks up the receiver and speaks in a low whisper: "Its done.'
            in result.prompt
        )
        assert "Thunder rumbles in the distance." in result.prompt
        assert 'He continues urgently: "You have thirty minutes."' in result.prompt
        assert "Tense male whisper." in result.prompt
        # SFX reinforcement at end
        assert result.prompt.endswith("Dark room, heavy rain.")

    def test_closeup_no_sfx_reinforcement(self):
        xml = '<speak voice="V" gender="male" scene="A room">Hello.</speak>'
        result = compile_prompt(xml)
        # Closeup mode: no scene repeated at end
        assert result.prompt.count("A room") == 1


class TestCompileChunkPrompt:
    def test_basic_chunk(self):
        prompt = compile_chunk_prompt(
            speech_text="Hello world.",
            voice="Deep voice",
            scene="A dark room",
        )
        assert "Close-up in A dark room." in prompt
        assert '"Hello world."' in prompt
        assert "Deep voice." in prompt

    def test_chunk_with_actions(self):
        prompt = compile_chunk_prompt(
            speech_text="The story begins.",
            voice="Narrator",
            actions_before=["He takes a breath"],
            actions_after=["He pauses dramatically"],
        )
        assert "He takes a breath." in prompt
        assert '"The story begins."' in prompt
        assert "He pauses dramatically." in prompt

    def test_chunk_no_scene(self):
        prompt = compile_chunk_prompt(
            speech_text="Just text.",
            voice="Voice",
        )
        assert "Close-up in a person speaking to camera." in prompt


class TestExtractSentenceActions:
    def test_single_action_before_text(self):
        xml = '<speak voice="V" gender="male"><action>He whispers.</action>Hello world.</speak>'
        mapping = extract_sentence_actions(xml)
        assert 0 in mapping
        assert mapping[0] == ["He whispers."]

    def test_multiple_actions_before_text(self):
        xml = (
            '<speak voice="V" gender="male">'
            "<action>First direction.</action>"
            "<action>Second direction.</action>"
            "Hello world."
            "</speak>"
        )
        mapping = extract_sentence_actions(xml)
        assert 0 in mapping
        assert mapping[0] == ["First direction.", "Second direction."]

    def test_action_between_text_blocks(self):
        xml = (
            '<speak voice="V" gender="female">'
            "First sentence."
            "<action>She pauses.</action>"
            "Second sentence."
            "</speak>"
        )
        mapping = extract_sentence_actions(xml)
        # Action before second text block -> sentence index 1
        assert 1 in mapping
        assert mapping[1] == ["She pauses."]

    def test_no_actions(self):
        xml = '<speak voice="V" gender="male">Hello world.</speak>'
        mapping = extract_sentence_actions(xml)
        assert mapping == {}

    def test_action_before_multi_sentence_text(self):
        xml = (
            '<speak voice="V" gender="male">'
            "<action>He speaks slowly.</action>"
            "First sentence. Second sentence. Third sentence."
            "</speak>"
        )
        mapping = extract_sentence_actions(xml)
        # Action maps to first sentence (index 0) only
        assert 0 in mapping
        assert mapping[0] == ["He speaks slowly."]
        assert 1 not in mapping
        assert 2 not in mapping
