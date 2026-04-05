"""MiniMax-native realtime handler using Silero VAD + STT + LLM + TTS pipeline."""

import io
import json
import ssl
import wave
import uuid
import asyncio
import logging
import threading
import certifi
from typing import Any, AsyncGenerator, Dict, Optional, Tuple

import numpy as np
import gradio as gr
import httpx
from faster_whisper import WhisperModel
from elevenlabs.client import AsyncElevenLabs
from openai import AsyncOpenAI
from fastrtc import AdditionalOutputs, ReplyOnPause, SileroVadOptions
from numpy.typing import NDArray
from scipy.signal import resample as scipy_resample
from scipy.signal import resample_poly

from reachy_mini_conversation_app.config import config, ELEVENLABS_VOICE_MAP
from reachy_mini_conversation_app.prompts import get_session_instructions, get_session_voice
from reachy_mini_conversation_app.tools.core_tools import (
    ToolDependencies,
    get_tool_specs,
)
from reachy_mini_conversation_app.tools.background_tool_manager import (
    ToolCallRoutine,
    ToolNotification,
    BackgroundToolManager,
)


logger = logging.getLogger(__name__)

MINIMAX_BASE_URL = "https://api.minimax.io/v1"
OUTPUT_SAMPLE_RATE = 24000

ELEVENLABS_MODEL_ID = "eleven_turbo_v2_5"
DEFAULT_ELEVENLABS_VOICE = "Rachel"

# Reachy Mini speaker: float32 stereo at 16 kHz
ROBOT_SPEAKER_SAMPLE_RATE = 16000
ROBOT_SPEAKER_CHANNELS = 2

# Whisper model (lazy-loaded on first use, protected by lock to prevent concurrent loads)
_whisper_model: Optional["WhisperModel"] = None
_whisper_lock = threading.Lock()


def _get_whisper() -> "WhisperModel":
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            logger.info("Loading Whisper 'base' model (first run may take a moment)...")
            _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
            logger.info("Whisper model loaded.")
        return _whisper_model


def _realtime_specs_to_chat_tools(realtime_specs: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Convert OpenAI Realtime tool specs to chat completions tool format."""
    tools = []
    for spec in realtime_specs:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec.get("description", ""),
                    "parameters": spec.get("parameters", {}),
                },
            }
        )
    return tools


def _audio_to_wav_bytes(audio_array: NDArray[np.int16], sample_rate: int) -> bytes:
    """Convert numpy int16 audio array to WAV bytes."""
    if audio_array.ndim > 1:
        audio_array = audio_array.flatten()
    audio_array = audio_array.astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_array.tobytes())
    return buf.getvalue()


class MinimaxHandler(ReplyOnPause):
    """MiniMax-native handler: Silero VAD → STT → LLM (MiniMax-M2.7) → TTS.

    Replaces OpenaiRealtimeHandler with MiniMax's own APIs:
    - VAD:  Silero (via fastrtc ReplyOnPause)
    - STT:  MiniMax /v1/audio/transcriptions
    - LLM:  MiniMax chat completions (MiniMax-M2.7)
    - TTS:  MiniMax T2A WebSocket (wss://api.minimax.io/ws/v1/t2a_v2)
    """

    def __init__(
        self,
        deps: ToolDependencies,
        gradio_mode: bool = False,
        instance_path: Optional[str] = None,
    ) -> None:
        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path

        # MiniMax client (lazy-initialized in _ensure_ready)
        self._api_key: Optional[str] = None
        self._openai_client: Optional[AsyncOpenAI] = None   # chat completions
        self._eleven_client: Optional[AsyncElevenLabs] = None
        self._ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        self._initialized = False

        # Conversation state
        self.chat_history: list[Dict[str, Any]] = []
        self._session_voice: str = DEFAULT_ELEVENLABS_VOICE
        self._session_instructions: str = ""

        # Prevent concurrent speech turns from corrupting chat history
        self._reply_lock = asyncio.Lock()

        # Robot speaker state
        self._speaker_started = False

        # Tool manager for background tool execution
        self.tool_manager = BackgroundToolManager()
        self._tool_output_queue: asyncio.Queue[AdditionalOutputs] = asyncio.Queue()

        # Initialize ReplyOnPause with Silero VAD
        super().__init__(
            fn=self._reply_fn,
            needs_args=True,
            output_sample_rate=OUTPUT_SAMPLE_RATE,
            input_sample_rate=16000,
            algo_options=None,
            model_options=None,
        )

    def copy(self) -> "MinimaxHandler":
        return MinimaxHandler(self.deps, self.gradio_mode, self.instance_path)

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #

    def _ensure_ready(self, api_key: Optional[str] = None) -> None:
        """Lazy-initialize the MiniMax client and session parameters."""
        if self._initialized:
            return

        # api_key from Gradio textbox may be a list (chatbot component) if indexing is off
        key = (api_key if isinstance(api_key, str) else None) or config.OPENAI_API_KEY or ""
        if not key.strip():
            logger.warning("MiniMax API key not found; requests will likely fail.")
            key = "MISSING"
        self._api_key = key.strip()

        base_url = config.OPENAI_BASE_URL or MINIMAX_BASE_URL
        self._openai_client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=base_url,
            http_client=httpx.AsyncClient(verify=self._ssl_ctx),
        )
        eleven_key = config.ELEVENLABS_API_KEY or ""
        if not eleven_key.strip():
            logger.warning("ElevenLabs API key not found; TTS requests will fail.")
        self._eleven_client = AsyncElevenLabs(api_key=eleven_key.strip())

        self._session_instructions = get_session_instructions()
        self._session_voice = get_session_voice(default=DEFAULT_ELEVENLABS_VOICE)

        self.tool_manager.start_up(tool_callbacks=[self._handle_tool_result])
        self._initialized = True
        logger.info(
            "MinimaxHandler initialized: model=%s tts=elevenlabs voice=%s",
            config.MODEL_NAME,
            self._session_voice,
        )

    # ------------------------------------------------------------------ #
    # Main reply pipeline (called by ReplyOnPause on each speech turn)
    # ------------------------------------------------------------------ #

    async def _reply_fn(
        self,
        audio_tuple: Tuple[int, NDArray],
        *args: Any,
    ) -> AsyncGenerator:
        """Async generator: VAD pause detected → STT → LLM → TTS → yield audio."""
        # Skip this turn if another turn is already being processed
        # (prevents concurrent turns from corrupting chat history with dangling tool calls)
        if self._reply_lock.locked():
            logger.debug("Another turn in progress; skipping this turn.")
            return

        async with self._reply_lock:
            # Extract Gradio additional inputs
            api_key_textbox: Optional[str] = args[1] if len(args) > 1 else None
            self._ensure_ready(api_key_textbox)

            sample_rate, audio_array = audio_tuple

            # Ensure mono int16 at 16 kHz for STT
            if audio_array.ndim > 1:
                audio_array = audio_array.flatten()
            if sample_rate != 16000:
                audio_array = scipy_resample(
                    audio_array, int(len(audio_array) * 16000 / sample_rate)
                ).astype(np.int16)
            else:
                audio_array = audio_array.astype(np.int16)

            if audio_array.size == 0:
                return

            # 1. Transcribe speech → text
            transcript = await self._transcribe(audio_array, 16000)
            transcript = transcript.strip()
            if not transcript:
                logger.debug("Empty transcript; skipping turn.")
                return

            logger.info("User said: %s", transcript)
            yield AdditionalOutputs({"role": "user", "content": transcript})

            # Notify movement manager
            if self.deps.movement_manager is not None:
                self.deps.movement_manager.set_listening(False)

            # 2. LLM: chat completion (may include tool calls)
            response_text, tool_calls = await self._chat_completion(transcript)

            # 3. Handle tool calls (sequential, one round)
            if tool_calls:
                tool_messages = await self._execute_tool_calls(tool_calls)
                # Flush UI notifications queued by tool handler
                while not self._tool_output_queue.empty():
                    yield self._tool_output_queue.get_nowait()

                if tool_messages:
                    # Follow-up completion with tool results
                    response_text, _ = await self._chat_completion_with_results(tool_messages)

            if response_text:
                logger.info("Assistant: %s", response_text)
                yield AdditionalOutputs({"role": "assistant", "content": response_text})

                # 4. TTS: stream audio back
                async for audio_chunk in self._tts_stream(response_text):
                    yield audio_chunk

    # ------------------------------------------------------------------ #
    # STT
    # ------------------------------------------------------------------ #

    async def _transcribe(self, audio_array: NDArray[np.int16], sample_rate: int) -> str:
        """Transcribe audio using local Whisper (faster-whisper)."""
        wav_bytes = _audio_to_wav_bytes(audio_array, sample_rate)
        try:
            loop = asyncio.get_event_loop()

            def _run_whisper() -> str:
                model = _get_whisper()
                segments, _ = model.transcribe(io.BytesIO(wav_bytes), beam_size=5)
                return " ".join(s.text for s in segments).strip()

            return await loop.run_in_executor(None, _run_whisper)
        except Exception as e:
            logger.error("STT error: %s", e)
            return ""

    # ------------------------------------------------------------------ #
    # LLM
    # ------------------------------------------------------------------ #

    def _build_messages(self) -> list[Dict[str, Any]]:
        return [
            {"role": "system", "content": self._session_instructions},
            *self.chat_history,
        ]

    async def _chat_completion(self, user_text: str) -> Tuple[str, list[Any]]:
        """Single-turn chat completion with tool support."""
        self.chat_history.append({"role": "user", "content": user_text})
        chat_tools = _realtime_specs_to_chat_tools(get_tool_specs())
        try:
            logger.info("Calling LLM (%s)...", config.MODEL_NAME)
            resp = await self._openai_client.chat.completions.create(  # type: ignore[union-attr]
                model=config.MODEL_NAME,
                messages=self._build_messages(),
                tools=chat_tools or None,  # type: ignore[arg-type]
                tool_choice="auto" if chat_tools else None,
            )
            logger.info("LLM responded.")
        except Exception as e:
            logger.error("LLM error: %s", e)
            return "Sorry, I encountered an error.", []

        msg = resp.choices[0].message
        self.chat_history.append(msg.model_dump())
        # Strip <think>...</think> reasoning blocks (MiniMax-M2.7 is a reasoning model)
        import re
        content = re.sub(r"<think>.*?</think>", "", msg.content or "", flags=re.DOTALL).strip()
        return content, msg.tool_calls or []

    async def _chat_completion_with_results(
        self, tool_messages: list[Dict[str, Any]]
    ) -> Tuple[str, list[Any]]:
        """Follow-up chat completion after tool results are added to history."""
        for tm in tool_messages:
            self.chat_history.append(tm)
        try:
            resp = await self._openai_client.chat.completions.create(  # type: ignore[union-attr]
                model=config.MODEL_NAME,
                messages=self._build_messages(),
            )
        except Exception as e:
            logger.error("LLM follow-up error: %s", e)
            return "Sorry, I encountered an error.", []

        msg = resp.choices[0].message
        self.chat_history.append(msg.model_dump())
        return msg.content or "", []

    # ------------------------------------------------------------------ #
    # Tool execution
    # ------------------------------------------------------------------ #

    async def _execute_tool_calls(self, tool_calls: list[Any]) -> list[Dict[str, Any]]:
        """Execute tool calls and return tool result messages for history."""
        tool_messages: list[Dict[str, Any]] = []
        for tc in tool_calls:
            call_id = tc.id
            tool_name = tc.function.name
            args_str = tc.function.arguments
            logger.info("Tool call: %s(%s) id=%s", tool_name, args_str, call_id)

            try:
                args = json.loads(args_str or "{}")
            except json.JSONDecodeError:
                args = {}

            # Show in Gradio UI
            await self._tool_output_queue.put(
                AdditionalOutputs(
                    {
                        "role": "assistant",
                        "content": f"🛠️ Calling tool: {tool_name}({args_str})",
                        "metadata": {"title": f"🛠️ {tool_name}", "status": "pending"},
                    }
                )
            )

            # Execute via BackgroundToolManager and wait for the task to finish
            bg = await self.tool_manager.start_tool(
                call_id=call_id,
                tool_call_routine=ToolCallRoutine(
                    tool_name=tool_name,
                    args_json_str=args_str or "{}",
                    deps=self.deps,
                ),
                is_idle_tool_call=False,
            )

            # Wait for the underlying asyncio task to complete
            result: Dict[str, Any] = {}
            try:
                if bg._task is not None:
                    await asyncio.wait_for(asyncio.shield(bg._task), timeout=15.0)
                result = bg.result or ({"error": bg.error} if bg.error else {})
            except asyncio.TimeoutError:
                result = {"error": f"Tool '{tool_name}' timed out"}
            except Exception as e:
                result = {"error": str(e)}

            await self._tool_output_queue.put(
                AdditionalOutputs(
                    {
                        "role": "assistant",
                        "content": json.dumps(result),
                        "metadata": {"title": f"🛠️ {tool_name} done", "status": "done"},
                    }
                )
            )

            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(result),
                }
            )
        return tool_messages

    async def _handle_tool_result(self, bg_tool: ToolNotification) -> None:
        """Callback from BackgroundToolManager when a tool finishes."""
        # Results are already handled inline in _execute_tool_calls
        pass

    # ------------------------------------------------------------------ #
    # TTS
    # ------------------------------------------------------------------ #

    def _ensure_speaker_started(self) -> None:
        """Start the Reachy Mini speaker pipeline (once)."""
        if self._speaker_started:
            return
        try:
            audio = self.deps.reachy_mini.media_manager.audio
            if audio is not None:
                audio.start_playing()
                self._speaker_started = True
                logger.info("Reachy Mini speaker pipeline started.")
        except Exception as e:
            logger.warning("Could not start Reachy Mini speaker: %s", e)

    def _push_to_robot_speaker(self, pcm_int16: NDArray[np.int16]) -> None:
        """Push a PCM int16 mono chunk (24 kHz) to Reachy Mini's speaker.

        Converts: int16 mono 24 kHz → float32 stereo 16 kHz.
        """
        try:
            audio = self.deps.reachy_mini.media_manager.audio
            if audio is None:
                return
            # int16 → float32 in [-1, 1]
            mono_f32 = pcm_int16.astype(np.float32) / 32768.0
            # Resample 24000 → 16000 (ratio 2:3)
            resampled = resample_poly(mono_f32, up=2, down=3).astype(np.float32)
            # Mono → stereo shape (N, 2)
            stereo = np.stack([resampled, resampled], axis=1)
            audio.push_audio_sample(stereo)
        except Exception as e:
            logger.debug("Robot speaker push error: %s", e)

    async def _tts_stream(self, text: str) -> AsyncGenerator[Tuple[int, NDArray[np.int16]], None]:
        """Stream TTS audio using ElevenLabs API and play on Reachy Mini speaker."""
        if not text.strip():
            return

        try:
            # Notify head wobbler that audio is coming
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.reset()

            self._ensure_speaker_started()

            voice_id = ELEVENLABS_VOICE_MAP.get(self._session_voice, self._session_voice)
            async for pcm_chunk in self._eleven_client.text_to_speech.stream(  # type: ignore[union-attr]
                voice_id=voice_id,
                text=text,
                model_id=ELEVENLABS_MODEL_ID,
                output_format="pcm_24000",
            ):
                if not pcm_chunk:
                    continue
                # Ensure we have raw bytes
                if not isinstance(pcm_chunk, (bytes, bytearray)):
                    try:
                        pcm_chunk = bytes(pcm_chunk)
                    except Exception:
                        continue
                # int16 requires even-length buffers
                if len(pcm_chunk) % 2 != 0:
                    pcm_chunk = pcm_chunk[:-1]
                if len(pcm_chunk) < 2:
                    continue
                mono_int16 = np.frombuffer(pcm_chunk, dtype=np.int16)
                # Push to Reachy Mini speaker
                self._push_to_robot_speaker(mono_int16)
                if self.deps.head_wobbler is not None:
                    self.deps.head_wobbler.feed(bytes(pcm_chunk).hex())
                # Also yield to browser/Gradio
                yield (OUTPUT_SAMPLE_RATE, mono_int16.reshape(1, -1))

        except Exception as e:
            logger.error("TTS error: %s", e)

    # ------------------------------------------------------------------ #
    # Personality / voice management
    # ------------------------------------------------------------------ #

    async def apply_personality(self, profile: Optional[str]) -> str:
        """Update session personality (profile) at runtime."""
        from reachy_mini_conversation_app.config import set_custom_profile

        try:
            set_custom_profile(profile)
            self._session_instructions = get_session_instructions()
            self._session_voice = get_session_voice(default=DEFAULT_ELEVENLABS_VOICE)
            # Reset chat history so new instructions apply from next turn
            self.chat_history.clear()
            logger.info("Applied personality: %s voice: %s", profile, self._session_voice)
            return f"Applied personality '{profile or 'default'}'. Will take effect on next turn."
        except Exception as e:
            logger.error("Error applying personality '%s': %s", profile, e)
            return f"Failed to apply personality: {e}"

    async def get_available_voices(self) -> list[str]:
        """Return ElevenLabs voice names."""
        return list(ELEVENLABS_VOICE_MAP.keys())
