import asyncio
import os
import json
import numpy as np
from typing import Optional, List, Dict, Any, Union
from autogen.agentchat import GroupChat, Agent, UserProxyAgent
from fastrtc import get_tts_model, get_stt_model, get_twilio_turn_credentials, KokoroTTSOptions
from aiortc.mediastreams import AudioStreamTrack as FastRTCAudioStreamTrack
from fastrtc.reply_on_pause import ReplyOnPause, AlgoOptions, AppState
from fastrtc.stream import Stream as FastRTCStream
import time
import random
from threading import Lock

class AudioGroupChat(GroupChat):
    """Real-time audio group chat implementation enabling voice and text communication between humans and AI agents."""

    def __init__(self, agents=None, messages=None, max_round=10, speaker_selection_method="round_robin", allow_repeat_speaker=False):
        """Initialize AudioGroupChat with audio processing capabilities."""
        super().__init__(
            agents=agents or [],
            messages=messages or [],
            max_round=max_round,
            speaker_selection_method=speaker_selection_method,
            allow_repeat_speaker=allow_repeat_speaker,
        )

        # Enable agent voice by default
        self.agent_voice_enabled = True

        # Initialize audio processing components
        print("Initializing TTS models...")
        # Map of agent names to voice models and options
        self.tts_models = {}
        self.voice_options = {}  # Store voice options separately
        # Create distinct voices using Kokoro's options
        self.available_voices = [
            ("energetic", KokoroTTSOptions(speed=1.5, lang="en-us")),     # Fast, energetic voice
            ("calm", KokoroTTSOptions(speed=0.75, lang="en-us")),         # Slower, calmer voice
            ("british", KokoroTTSOptions(speed=1.0, lang="en-gb")),       # British accent
            ("authoritative", KokoroTTSOptions(speed=0.9, lang="en-us")), # Slightly slower, authoritative
            ("default", KokoroTTSOptions(speed=1.0, lang="en-us")),       # Normal voice
        ]
        self.next_voice_index = 0
        # Default model for unassigned agents
        self.default_tts_model = get_tts_model("kokoro")
        print("Default TTS model initialized")

        # Assign voices to initial agents
        if agents:
            for agent in agents:
                if isinstance(agent, UserProxyAgent):
                    print(f"Skipping voice assignment for human user {agent.name}")
                    continue
                voice_name, voice_options = self.available_voices[self.next_voice_index % len(self.available_voices)]
                print(f"Assigning voice '{voice_name}' to agent {agent.name} (speed={voice_options.speed}, lang={voice_options.lang})")
                self.tts_models[agent.name] = get_tts_model("kokoro")
                self.voice_options[agent.name] = voice_options  # Store voice options
                self.next_voice_index += 1

        self.tts_sample_rate = 24000  # Standard sample rate for TTS output

        print("Initializing STT model...")
        self.stt_model = get_stt_model()
        print("STT model initialized")

        # Configure WebRTC settings
        self.rtc_config = get_twilio_turn_credentials() if os.environ.get("TWILIO_ACCOUNT_SID") else None

        # Create FastRTC stream for audio handling with ReplyOnPause
        algo_options = AlgoOptions(
            audio_chunk_duration=1.0,  # Longer chunks for better transcription
            started_talking_threshold=0.2,  # More sensitive to speech start
            speech_threshold=0.1,  # More sensitive to ongoing speech
        )

        # Create a handler that maintains state per participant
        self.participant_states = {}

        # Create main handler for audio processing
        async def audio_callback(frame, additional_inputs=None):
            try:
                # Get user ID from additional inputs
                user_id = None
                if additional_inputs and len(additional_inputs) > 0:
                    if isinstance(additional_inputs[0], str):
                        user_id = additional_inputs[0]
                    elif hasattr(additional_inputs[0], 'value'):
                        user_id = additional_inputs[0].value

                if not user_id:
                    print("No user ID provided in audio callback")
                    return None

                print(f"Processing audio from user: {user_id}")

                # Handle different audio input formats
                if isinstance(frame, tuple) and len(frame) == 2:
                    sample_rate, audio_array = frame
                elif isinstance(frame, dict):
                    # Handle Gradio's audio format
                    sample_rate = frame.get('sr', 48000)  # Default to 48kHz
                    audio_array = frame.get('value', None)
                    if audio_array is None:
                        print("No audio array in data")
                        return None
                else:
                    print(f"Unexpected audio format: {type(frame)}")
                    return None

                print(f"Processing audio: sr={sample_rate}, shape={getattr(audio_array, 'shape', None)}")

                # Ensure audio data is in correct format
                if isinstance(audio_array, bytes):
                    audio_array = np.frombuffer(audio_array, dtype=np.float32)
                elif isinstance(audio_array, list):
                    audio_array = np.array(audio_array, dtype=np.float32)

                # Convert to mono if stereo
                if len(audio_array.shape) > 1:
                    audio_array = np.mean(audio_array, axis=1)

                # Ensure 1D array
                audio_array = audio_array.reshape(-1)

                # Normalize audio to [-1, 1] range
                if audio_array.dtype != np.float32:
                    audio_array = audio_array.astype(np.float32)
                if np.max(np.abs(audio_array)) > 1.0:
                    audio_array = audio_array / 32768.0

                print("Before STT model call") # ADDED LOGGING
                print(f"Audio data shape before STT: {audio_array.shape}, dtype: {audio_array.dtype}") # ADDED LOGGING
                text = self.stt_model.stt((sample_rate, audio_array))
                print("After STT model call") # ADDED LOGGING

                if not text:
                    print("No speech detected")
                    return None

                print(f"Transcribed text: {text}")

                # Create chat message
                message = {
                    "role": "user",
                    "content": text,
                    "name": user_id
                }

                # Add to chat history
                await self.text_queue.put({
                    "type": "chat",
                    "text": text,
                    "sender": user_id,
                    "channel": "voice"
                })

                # Add message to messages list for UI updates
                self.messages.append(message)

                # Process the message through the chat handler
                await self._handle_chat_message(user_id, message)

                # Return messages for Gradio Chatbot
                return self.messages

            except Exception as e:
                print(f"Error in audio callback: {str(e)}")
                import traceback
                traceback.print_exc()
                return None

        # Create main handler using FastRTC's ReplyOnPause
        handler = ReplyOnPause(
            fn=audio_callback,
            algo_options=algo_options,
            can_interrupt=True,
            expected_layout="mono",
            output_sample_rate=24000,
            output_frame_size=480,  # Standard frame size for 24kHz
            input_sample_rate=48000,
        )

        # Initialize handler with required attributes
        handler._loop = asyncio.get_event_loop()
        handler.queue = asyncio.Queue()
        handler.args_set = asyncio.Event()  # Required by StreamHandlerBase
        handler.channel_set = asyncio.Event()  # Required by StreamHandlerBase
        handler._channel = None  # Will be set by AudioCallback
        handler._phone_mode = False  # Required by StreamHandlerBase
        self._lock = Lock()

        # Create a clear_queue function that clears the handler's queue
        def clear_queue():
            while not handler.queue.empty():
                handler.queue.get_nowait()
        handler._clear_queue = clear_queue

        # Create FastRTC stream for audio handling
        self.stream = FastRTCStream(
            modality="audio",
            mode="send-receive",
            handler=handler,  # This sets event_handler internally
            rtc_configuration=self.rtc_config,
            concurrency_limit=10,  # Allow up to 10 concurrent connections
            time_limit=None,  # No time limit on sessions
            additional_inputs=[],  # Initialize empty, will be set by UI
            additional_outputs=[],  # Initialize empty, will be set by UI
            additional_outputs_handler=lambda prev, curr: curr,  # Simple handler
            ui_args={
                "title": "Huddle Audio Group Chat",
                "subtitle": "Click the microphone button to start speaking",
                "show_audio_input": True,
                "show_audio_output": True,
                "show_text": True
            }
        )

        # Verify handler is properly set
        if not self.stream.event_handler:
            raise RuntimeError("Failed to initialize stream event_handler")

        # Initialize participant tracking
        self.human_participants = {}
        self.active_calls = {}

        # Channel configuration
        self.voice_enabled = True  # Enable voice by default
        self.text_enabled = True   # Enable text by default
        self.agent_voice_enabled = True  # Enable agent voice by default

        # Track last message count to detect new messages
        self._last_message_count = 0

        # Required for autogen compatibility
        self.client_cache = []
        self.previous_cache = []

        # Message queues for each channel: Initialize queues
        self.voice_queue = asyncio.Queue()
        self.text_queue = asyncio.Queue()

    async def initialize(self):
        """Initialize async tasks and processors for real-time operation."""
        try:
            print("Initializing audio group chat...")

            # Start background tasks
            self._process_voice_queue_task = asyncio.create_task(self._process_voice_queue())
            self._process_text_queue_task = asyncio.create_task(self._process_text_queue())
            self._monitor_agent_messages_task = asyncio.create_task(self._monitor_agent_messages())

            # Add error handlers and names for better tracking
            for task, name in [
                (self._process_voice_queue_task, "voice_queue"),
                (self._process_text_queue_task, "text_queue"),
                (self._monitor_agent_messages_task, "agent_monitor")
            ]:
                task.set_name(name)
                task.add_done_callback(self._task_error_handler)

            # Wait for tasks to start
            await asyncio.sleep(0.1)

            # Verify tasks are running
            for task in [self._process_voice_queue_task, self._process_text_queue_task, self._monitor_agent_messages_task]:
                if task.done():
                    exc = task.exception()
                    if exc:
                        print(f"Task {task.get_name()} failed to start: {exc}")
                        raise exc
                else:
                    print(f"Task {task.get_name()} started successfully")

            print("Audio group chat initialized successfully")

        except Exception as e:
            print(f"Error initializing audio group chat: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _task_error_handler(self, task):
        """Handle errors in real-time processing tasks."""
        try:
            # Skip if task was cancelled normally
            if task.cancelled():
                return

            # Get the exception if any
            exc = task.exception()
            if exc:
                task_name = task.get_name()
                print(f"Background task '{task_name}' error: {exc}")
                import traceback
                traceback.print_exc()

                # Restart the task if it's one of our monitored tasks
                if task in [self._process_voice_queue_task, self._process_text_queue_task, self._monitor_agent_messages_task]:
                    print(f"Restarting failed task '{task_name}'...")

                    # Create new task with same name
                    new_task = asyncio.create_task(task.get_coro())
                    new_task.set_name(task_name)
                    new_task.add_done_callback(self._task_error_handler)

                    # Update task reference
                    if task == self._process_voice_queue_task:
                        self._process_voice_queue_task = new_task
                    elif task == self._process_text_queue_task:
                        self._process_text_queue_task = new_task
                    elif task == self._monitor_agent_messages_task:
                        self._monitor_agent_messages_task = new_task

                    print(f"Task '{task_name}' restarted")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Error in task error handler: {e}")
            import traceback
            traceback.print_exc()

    def add_agent(self, agent: Agent):
        """Add an agent to the chat and assign it a unique voice."""
        # Skip voice assignment for human users
        if isinstance(agent, UserProxyAgent):
            print(f"Skipping voice assignment for human user {agent.name}")
            # Add agent to group chat
            if not hasattr(self, 'agents'):
                self.agents = []
            self.agents.append(agent)
            return

        # Add agent to group chat
        if not hasattr(self, 'agents'):
            self.agents = []
        self.agents.append(agent)

        # Assign a unique voice to the agent if not already assigned
        if agent.name not in self.tts_models:
            voice_name, voice_options = self.available_voices[self.next_voice_index % len(self.available_voices)]
            print(f"Assigning voice '{voice_name}' to agent {agent.name} (speed={voice_options.speed}, lang={voice_options.lang})")
            self.tts_models[agent.name] = get_tts_model("kokoro")
            self.voice_options[agent.name] = voice_options  # Store voice options
            self.next_voice_index += 1
        return agent

    def add_human_participant(self, user_id: str) -> str:
        """Add a human participant to the group chat."""
        print(f"\nAdding human participant: {user_id}")

        try:
            # Create session ID
            session_id = str(random.getrandbits(64))

            # Create user proxy agent
            user_agent = UserProxyAgent(
                name=user_id,
                human_input_mode="NEVER",
                code_execution_config={"use_docker": False}
            )

            # Add participant info
            self.human_participants[user_id] = {
                "session_id": session_id,
                "agent": user_agent,
                "active": False,
                "stream": None
            }

            # Add agent to chat
            if user_agent not in self.agents:
                self.agents.append(user_agent)

            print(f"Participant {user_id} added successfully")
            print("Current participants:", list(self.human_participants.keys()))

            return session_id

        except Exception as e:
            print(f"Error adding participant {user_id}: {e}")
            import traceback
            traceback.print_exc()
            raise

    async def start_audio_session(self, user_id: str):
        """Start an audio session for a participant and initialize the group chat."""
        try:
            # Verify participant exists
            participant = self.human_participants.get(user_id)
            if not participant:
                print(f"Error: Participant {user_id} not found")
                print("Available participants:", list(self.human_participants.keys()))
                return

            print(f"\nStarting audio session for {user_id}...")

            # Mark participant as active
            participant["active"] = True

            # Use the main AudioGroupChat stream for this participant
            if "stream" not in participant:
                participant["stream"] = self.stream
                print(f"Using main stream for participant {user_id}")

            print(f"Audio session started for {user_id}")

        except Exception as e:
            print(f"Error in start_audio_session: {e}")
            import traceback
            traceback.print_exc()
            raise

    async def _handle_audio_input(self, audio_data: tuple[int, np.ndarray], user_id: str = None) -> dict:
        """Process incoming audio from a user in real-time."""
        try:
            print("Received audio input:", type(audio_data))
            if audio_data is None:
                print("No audio data received")
                return None

            # Handle different audio input formats
            if isinstance(audio_data, tuple) and len(audio_data) == 2:
                sample_rate, audio_array = audio_data
            elif isinstance(audio_data, dict):
                # Handle Gradio's audio format
                sample_rate = audio_data.get('sr', 48000)  # Default to 48kHz
                audio_array = audio_data.get('value', None)
                if audio_array is None:
                    print("No audio array in data")
                    return None
            else:
                print(f"Unexpected audio format: {type(audio_data)}")
                return None

            print(f"Processing audio: sr={sample_rate}, shape={getattr(audio_array, 'shape', None)}")

            # Ensure audio data is in correct format
            if isinstance(audio_array, bytes):
                audio_array = np.frombuffer(audio_array, dtype=np.float32)
            elif isinstance(audio_array, list):
                audio_array = np.array(audio_array, dtype=np.float32)

            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)

            # Ensure 1D array (sequence_length,)
            audio_array = audio_array.reshape(-1)

            # Normalize audio to [-1, 1] range
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            if np.max(np.abs(audio_array)) > 1.0:
                audio_array = audio_array / 32768.0

            print("Audio preprocessing complete")

            # Get participant's state
            participant = self.human_participants.get(user_id)
            if not participant:
                print(f"No participant found for user_id: {user_id}")
                return None

            # Use STT model to convert accumulated speech to text
            text = self.stt_model.stt((sample_rate, audio_array))

            if text and text.strip():
                # Create message for processing
                message = {
                    "type": "chat",
                    "text": text,
                    "sender": user_id,
                    "channel": "voice"
                }

                print(f"Transcribed text: {text}")

                # Process the message through the group chat manager
                await self._handle_chat_message(user_id, message)

                return {"text": text}
            else:
                print("No text transcribed from audio")

            return None

        except Exception as e:
            print(f"Error in handle_audio_input: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def handle_audio_output(self):
        """Process and return the next audio output for real-time playback."""
        try:
            # Check the voice queue
            try:
                # Get next message from voice queue without blocking
                message = self.voice_queue.get_nowait()
                print(f"Processing voice queue message: {message}")

                if isinstance(message, dict):
                    if message.get("type") == "chat":
                        text = message.get("text")
                        sender = message.get("sender")

                        if text and sender and sender not in self.human_participants:
                            print(f"Converting to speech from voice queue: {text} from {sender}")
                            try:
                                # Extract text from ChatResult if needed
                                if hasattr(text, 'content'):
                                    text = text.content
                                elif isinstance(text, dict):
                                    text = text.get("content", "")
                                elif isinstance(text, str):
                                    text = text

                                # Skip empty messages
                                if not text or not text.strip():
                                    print("Empty text content, skipping")
                                    return None

                                # Convert to speech with timeout
                                tts_result = self.tts_models.get(sender, self.default_tts_model).tts(text, options=self.voice_options.get(sender, None))
                                if asyncio.iscoroutine(tts_result):
                                    sample_rate, audio_data = await asyncio.wait_for(tts_result, timeout=5.0)
                                else:
                                    sample_rate, audio_data = tts_result

                                if audio_data is not None:
                                    # Ensure correct format
                                    if audio_data.dtype != np.float32:
                                        audio_data = audio_data.astype(np.float32)
                                    # Normalize audio to [-1, 1] range
                                    max_val = np.max(np.abs(audio_data))
                                    if max_val > 0:  # Avoid division by zero
                                        audio_data = audio_data / max_val

                                    print(f"Broadcasting TTS audio for {sender}: {text}")
                                    print(f"Audio stats - min: {np.min(audio_data):.3f}, max: {np.max(audio_data):.3f}, mean: {np.mean(audio_data):.3f}")

                                    # Broadcast to all participants
                                    await self._broadcast_audio_to_participants((sample_rate, audio_data))
                                    # Return for Gradio UI
                                    return audio_data
                                else:
                                    print(f"TTS failed for message from {sender}: {text}")

                            except asyncio.TimeoutError:
                                print(f"TTS timed out for message: {text}")
                            except Exception as e:
                                print(f"Error converting message to speech: {e}")
                                import traceback
                                traceback.print_exc()
                    elif message.get("type") == "audio":
                        # Direct audio data
                        audio_data = message.get("audio_data")
                        if audio_data is not None:
                            return audio_data
            except asyncio.QueueEmpty:
                pass

            return None

        except Exception as e:
            print(f"Error in audio output handler: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def text_to_speech(self, text: str | dict, user_id: str = None) -> tuple[int, np.ndarray] | None:
        """Convert text to speech with real-time processing and delivery."""
        try:
            # Check if TTS is enabled and model is initialized
            if not hasattr(self, 'tts_models') or not self.tts_models:
                print("TTS models not initialized")
                return None

            # Extract text from chat result if needed
            if isinstance(text, dict):
                text = text.get("content", "")
            elif not isinstance(text, str):
                text = str(text)

            # Skip empty messages
            if not text or text.strip() == "":
                return None

            # Extract sender from chat result if needed
            if isinstance(text, dict):
                sender = text.get("sender", user_id)
            else:
                sender = user_id or ""

            # Skip TTS for human participants
            if sender in self.human_participants:
                print(f"Skipping TTS for human participant {sender}")
                return None

            # Convert to speech using tts() method with timeout
            try:
                # Get the appropriate TTS model and voice options for the sender
                tts_model = self.tts_models.get(sender, self.default_tts_model)
                voice_options = self.voice_options.get(sender, None)
                print(f"Using voice model for {sender} with options: {voice_options}")

                # Call tts() with the voice options
                tts_result = tts_model.tts(text, options=voice_options)
                if not hasattr(tts_result, '__await__'):
                    # If not awaitable, assume it's already the result
                    sample_rate, audio_data = tts_result
                else:
                    # If awaitable, wait for result with timeout
                    sample_rate, audio_data = await asyncio.wait_for(
                        tts_result,
                        timeout=5.0  # 5 second timeout
                    )

                print(f"TTS generated audio: sr={sample_rate}, shape={audio_data.shape}")

            except asyncio.TimeoutError:
                print("TTS timed out")
                return None

            if audio_data is None:
                print("TTS failed to generate audio")
                return None

            # Ensure correct format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # AUTOMATIC GAIN CONTROL
            peak = np.max(np.abs(audio_data))
            if peak > 0:
                target_peak = 0.9  # Scale to 90% of max volume
                gain = target_peak / peak
                audio_data *= gain
            else:
                print("Silent audio detected")
                return None

            # Clip to [-1.0, 1.0] and convert to int16
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_data = (audio_data * 32767).astype(np.int16)  # Convert to 16-bit integers


            # Add to voice queue for broadcasting
            await self.voice_queue.put({
                "type": "audio",
                "sample_rate": sample_rate,
                "audio_data": audio_data,
                "sender": sender
            })

            print(f"Added TTS audio to voice queue: {text}")
            return sample_rate, audio_data

        except Exception as e:
            print(f"Error in text_to_speech: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def _send_audio_to_participant(self, user_id: str, audio_frame: tuple[int, np.ndarray], max_retries: int = 3) -> bool:
        """Send audio data to a specific participant."""
        participant = self.human_participants.get(user_id)
        if not participant or not participant.get("active"):
            print(f"Participant {user_id} not found or not active")
            return False

        stream = participant.get("stream")
        if not stream:
            print(f"No stream found for participant {user_id}")
            return False

        sample_rate, audio_data = audio_frame

        # Ensure audio data is in correct format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / 32768.0

        for attempt in range(max_retries):
            try:
                print(f"Sending audio to {user_id} (attempt {attempt + 1})")
                # Send audio through FastRTC stream
                if stream and stream.event_handler:
                    print(f"Sending audio to {user_id}")
                    # Create coroutine to send audio through queue
                    async def send_audio():
                        await stream.event_handler.queue.put((sample_rate, audio_data))
                    await send_audio()
                    print(f"Successfully sent audio to {user_id}")
                    return True

            except Exception as e:
                print(f"Error sending audio to {user_id} (attempt {attempt + 1}): {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(0.5)  # Brief delay before retry

        return False

    async def broadcast_audio(self, audio_frame: tuple[int, np.ndarray], sender_id: str = None):
        """Broadcast audio to all participants except the sender."""
        print("\n=== Broadcasting Audio ===")
        sample_rate, audio_data = audio_frame
        print(f"Audio frame: sr={sample_rate}Hz, shape={audio_data.shape}, dtype={audio_data.dtype}")

        # Get list of active participants excluding sender
        participants = [
            user_id for user_id, participant in self.human_participants.items()
            if user_id != sender_id and participant.get("active") and participant.get("stream") is not None
        ]

        if not participants:
            print("No active participants with streams to broadcast to")
            return False

        print(f"Broadcasting to participants: {participants}")

        # Ensure audio data is in correct format
        if audio_data.dtype == np.int16:
            pass  # Already in correct format (no scaling needed)
        else:
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / 32768.0  # Only for non-integer data


        # Add to voice queue for local playback (REMOVE THIS LINE for debugging if needed, it might cause repeated processing)
        # await self.voice_queue.put({
        #     "type": "audio",
        #     "sample_rate": sample_rate,
        #     "audio_data": audio_data,
        #     "sender": sender_id
        # })

        # Send to all participants in parallel
        tasks = []
        for user_id in participants:
            try:
                participant = self.human_participants[user_id]
                stream = participant.get("stream")
                if stream and stream.event_handler:
                    print(f"Preparing to send audio to {user_id}")
                    tasks.append(self._send_audio_to_participant(user_id, (sample_rate, audio_data)))
                else:
                    print(f"Stream for {user_id} is closed or invalid")
            except Exception as e:
                print(f"Error preparing to send to {user_id}: {e}")
                import traceback
                traceback.print_exc()

        if not tasks:
            print("No tasks created for broadcasting")
            return False

        # Wait for all sends to complete
        print(f"Sending audio to {len(tasks)} participants...")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for any failures
        success = False
        for user_id, result in zip(participants, results):
            if isinstance(result, Exception):
                print(f"Failed to send to {user_id}: {result}")
            elif not result:
                print(f"Failed to send to {user_id}")
            else:
                success = True
                print(f"Successfully sent audio to {user_id}")

        if success:
            print("Successfully broadcasted audio to at least one participant")
        else:
            print("Failed to broadcast audio to any participants")

        return success

    async def _handle_chat_message(self, user_id: str, msg: Dict[str, Any]) -> None:
        """Handle incoming chat messages."""
        try:
            if msg.get("type") == "chat" and msg.get("text"):
                # Create task to handle message asynchronously
                asyncio.create_task(self._process_chat_message(user_id, msg))
        except Exception as e:
            print(f"Error handling chat message: {e}")
            import traceback
            traceback.print_exc()

    async def _process_chat_message(self, user_id: str, message: Dict[str, Any]) -> None:
        """Process chat messages and handle responses in real-time."""
        try:
            print("\n=== Processing Chat Message ===")
            print(f"From User: {user_id}")
            print(f"Message: {message}")

            # Get message content
            text = None
            if isinstance(message, dict):
                text = message.get("content", message.get("text", ""))
            elif hasattr(message, 'content'):  # Handle ChatResult objects
                text = message.content

            if not text or not text.strip():
                print("Empty message, skipping")
                return

            # Create chat message
            chat_message = {
                "role": "user" if user_id in self.human_participants else "assistant",
                "content": text,
                "name": user_id
            }

            # Add to voice queue if it's an agent message and voice is enabled
            if user_id not in self.human_participants and self.agent_voice_enabled:
                await self.voice_queue.put({
                    "type": "chat",
                    "text": text,
                    "sender": user_id,
                    "channel": "both"
                })
                print(f"Added agent message to voice queue: {text[:100]}..." if len(text) > 100 else f"Added agent message to voice queue: {text}")
                chat_message["_voice_queued"] = True  # Use dict entry instead of attribute

            # Add message to messages list for UI updates
            #print(f"Adding message to chat history: {chat_message}")
            #self.messages.append(chat_message)

            # Add message to both queues to ensure it's displayed and spoken
            message_for_queues = {
                "type": "chat",
                "text": text,
                "sender": user_id,
                "channel": "both"
            }

            # Always add to text queue for display
            await self.text_queue.put(message_for_queues)
            print(f"Added message to text queue: {text[:100]}..." if len(text) > 100 else f"Added message to text queue: {text}")

            # Get the first available agent as recipient
            recipient = next((agent for agent in self.agents if agent.name != user_id), None)
            if recipient:
                # Get sender agent object
                sender = self.agent(user_id)
                if sender:
                    # Initiate chat with the message
                    response = await self.manager.a_initiate_chat(
                        message=text,
                        sender=sender,
                        recipient=recipient
                    )

                    # Process response if any
                    if response:
                        # Extract latest message from ChatResult
                        if hasattr(response, 'chat_history') and response.chat_history:
                            # Get the last message from chat_history
                            latest_msg = response.chat_history[-1]
                            response_text = latest_msg.get('content', '') if isinstance(latest_msg, dict) else str(latest_msg)
                        elif hasattr(response, 'summary'):
                            response_text = response.summary
                        elif hasattr(response, 'content'):
                            response_text = response.content
                        elif isinstance(response, dict):
                            response_text = response.get('content', '')
                        else:
                            response_text = str(response)

                        response_msg = {
                            "role": "assistant",
                            "content": response_text,
                            "name": recipient.name
                        }
                        self.messages.append(response_msg)

                        # Create base message
                        base_message = {
                            "type": "chat",
                            "text": response_text,
                            "sender": recipient.name
                        }

                        # Add to text queue
                        text_message = base_message.copy()
                        text_message["channel"] = "text"
                        await self.text_queue.put(text_message)
                        print(f"Added agent response to text queue: {response_text[:100]}..." if len(response_text) > 100 else f"Added agent response to text queue: {response_text}")

                        # Add to voice queue if agent voice is enabled
                        if self.agent_voice_enabled:
                            voice_message = base_message.copy()
                            voice_message["channel"] = "voice"
                            await self.voice_queue.put(voice_message)
                            print(f"Added agent response to voice queue: {response_text[:100]}..." if len(response_text) > 100 else f"Added agent response to voice queue: {voice_text}") # Corrected log var name
                            # Mark the original message and response as queued
                            message["_voice_queued"] = True
                            response_msg["_voice_queued"] = True

        except Exception as e:
            print(f"Error processing chat message: {e}")
            import traceback
            traceback.print_exc()

    async def _monitor_agent_messages(self):
        """Monitor messages between agents and add them to the voice queue."""
        processed_messages = set()  # Track processed messages by content hash
        print("\n=== Starting Agent Message Monitor ===\n")

        while True:
            await asyncio.sleep(0.1)  # Prevent tight loop
            try:
                with self._lock:
                    current_count = len(self.messages)
                    message_count_changed = current_count > self._last_message_count
                    
                if message_count_changed:
                    with self._lock:  # Atomic update of last count
                        self._last_message_count = current_count

                if message_count_changed: # CHECK FLAG OUTSIDE LOCK
                    print(f"\n=== Monitoring Agent Messages ===")
                    print(f"Current message count: {len(self.messages)}, Last processed: {self._last_message_count}")

                    # Get only the latest message
                    last_message = self.messages[-1]
                    print(f"Processing latest message: {last_message}")

                    # Extract text and sender from message
                    text = None
                    sender = None

                    if isinstance(last_message, tuple):
                        text = last_message[1]  # Agent's response is in second position
                        sender = last_message[0].name if hasattr(last_message[0], 'name') else str(last_message[0])
                    elif isinstance(last_message, dict): # ASSUMING DICTIONARY - MIGHT BE WRONG FOR AGENT-AGENT MESSAGES
                        text = last_message.get('content', '')
                        sender = last_message.get('name', '') # **THIS IS POTENTIALLY WRONG FOR AGENT-AGENT MESSAGES**
                    else: # ADDED ELSE BLOCK TO HANDLE UNEXPECTED TYPES
                        print(f"Unexpected message type in monitor: {type(last_message)}") # ADDED LOGGING
                        continue # SKIP PROCESSING IF UNEXPECTED TYPE

                    if not sender: # ADDED CHECK FOR EMPTY SENDER
                        print("Warning: Sender is empty, skipping message") # ADDED LOGGING
                        continue # SKIP PROCESSING IF NO SENDER


                    message_hash = hash(f"{sender}:{text}")
                    print(f"Message hash: {message_hash}")

                    # Check processing conditions
                    is_agent = sender not in self.human_participants
                    voice_enabled = self.agent_voice_enabled
                    not_processed = message_hash not in processed_messages
                    not_queued = not (isinstance(last_message, dict) and last_message.get("_voice_queued", False))

                    print(f"Message Analysis:")
                    print(f"- Sender: {sender}")
                    print(f"- Is agent: {is_agent} - Result: {is_agent}") # ADDED LOGGING
                    print(f"- Voice enabled: {voice_enabled} - Result: {voice_enabled}") # ADDED LOGGING
                    print(f"- Not processed: {not_processed} - Result: {not_processed}") # ADDED LOGGING
                    print(f"- Not queued: {not_queued} - Result: {not_queued}") # ADDED LOGGING
                    print(f"- Text preview: {text[:100]}..." if len(text) > 100 else f"- Text: {text}")

                    # Process agent messages that haven't been processed
                    if is_agent and voice_enabled and not_processed and not_queued:
                        print(f"\n✓ Processing new agent message from {sender}")

                        # Create message for voice queue
                        message_for_queues = {
                            "type": "chat",
                            "text": text,
                            "sender": sender,
                            "channel": "voice"
                        }

                        # Add to voice queue
                        print(f"Adding to voice queue: {text[:100]}..." if len(text) > 100 else f"Adding to voice queue: {text}")
                        await self.voice_queue.put(message_for_queues)

                        # Mark original message as queued and track processed message
                        if isinstance(last_message, dict):
                            last_message["_voice_queued"] = True
                            print("Marked message as queued")
                        processed_messages.add(message_hash)
                        print(f"Added to processed messages (total: {len(processed_messages)})")

                        # Add small delay for conversation flow
                        print("Adding delay for conversation flow...")
                        await asyncio.sleep(0.2)
                    else:
                        print(f"✗ Message skipped - Conditions not met for sender: {sender}")
                else:
                    print("No new messages detected in monitor loop") # ADDED LOGGING

                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)

            except Exception as e:
                print(f"Error in agent message monitor: {str(e)}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1)  # Longer delay on error
            print("Agent Message Monitor Loop iteration finished") # ADDED LOGGING
            
    async def _process_voice_queue(self):
        """Process messages in the voice queue for real-time audio output."""
        while True:
            try:
                # Get next message from voice queue
                message = await self.voice_queue.get()
                print(f"\n=== Processing Voice Queue Message ===")
                print(f"Message type: {message.get('type')}")
                print(f"Sender: {message.get('sender')}")
                print(f"Channel: {message.get('channel')}")
                print(f"Text content: {message.get('text', '')[:100]}..." if message.get('text') and len(message.get('text')) > 100 else f"Text content: {message.get('text', '')}")

                if isinstance(message, dict):
                    message_type = message.get("type")
                    sender = message.get("sender")
                    channel = message.get("channel")

                    print(f"Message details - Type: {message_type}, Sender: {sender}, Channel: {channel}")

                    if message_type == "chat":
                        text = message.get("text")

                        # Only process messages from agents if voice is enabled
                        if text and sender and sender not in self.human_participants and self.agent_voice_enabled:
                            print(f"Converting to speech from voice queue: {text} from {sender}")
                            try:
                                # Extract text from ChatResult if needed
                                original_text = text
                                if hasattr(text, 'summary'):
                                    text = text.summary
                                elif hasattr(text, 'content'):
                                    text = text.content
                                elif isinstance(text, dict):
                                    text = text.get("content", "")
                                elif isinstance(text, str):
                                    text = text
                                else:
                                    text = str(text)

                                if text != original_text:
                                    print(f"Extracted text content: {text[:100]}..." if len(text) > 100 else f"Extracted text content: {text}")

                                # Skip empty messages
                                if not text or not isinstance(text, str) or not text.strip():
                                    print(f"Skipping empty or invalid message from {sender}")
                                    continue

                                # Convert text to speech
                                print(f"Converting to speech - Sender: {sender}, Text: {text[:100]}..." if len(text) > 100 else f"Converting to speech - Sender: {sender}, Text: {text}")
                                audio_result = await self.text_to_speech(text, sender)

                                if audio_result:
                                    # Broadcast audio to all participants
                                    print(f"Broadcasting audio from {sender}")
                                    success = await self.broadcast_audio(audio_result, sender)
                                    if success:
                                        print(f"Successfully broadcast TTS audio for: {text}")
                                    else:
                                        print(f"Failed to broadcast TTS audio for: {text}")
                                else:
                                    print(f"Failed to convert text to speech: {text}")
                            except Exception as e:
                                print(f"Error processing chat message in voice queue: {e}") # More specific error log
                                import traceback
                                traceback.print_exc()
                                continue

                    elif message_type == "audio":
                        # Handle raw audio data
                        audio_frame = message.get("audio_data")
                        if audio_frame:
                            try:
                                success = await self.broadcast_audio(audio_frame, sender)
                                if success:
                                    print("Successfully broadcast direct audio data")
                                else:
                                    print("Failed to broadcast direct audio data")
                            except Exception as e:
                                print(f"Error broadcasting audio in voice queue: {e}") # More specific error log
                                import traceback
                                traceback.print_exc()
                                continue
                        else:
                            # Try alternate audio format
                            sample_rate = message.get("sample_rate")
                            audio_data = message.get("audio_data")
                            if sample_rate and audio_data is not None:
                                try:
                                    print(f"Broadcasting formatted audio from {sender}")
                                    success = await self.broadcast_audio((sample_rate, audio_data), sender)
                                    if success:
                                        print(f"Successfully broadcast formatted audio from {sender}")
                                    else:
                                        print(f"Failed to broadcast formatted audio from {sender}")
                                except Exception as e:
                                    print(f"Error broadcasting formatted audio from {sender} in voice queue: {e}") # More specific error log
                                    import traceback
                                    traceback.print_exc()
                                    continue
                            else:
                                print(f"Invalid audio message format from {sender}")
                    else:
                        print(f"Unknown message type: {message_type} from {sender}")
                else:
                    print(f"Invalid message format: {message}")

                # Mark task as done
                self.voice_queue.task_done() # ENSURE THIS IS CALLED

            except asyncio.CancelledError:
                # Allow clean shutdown
                print("Voice queue processor cancelled")
                raise
            except Exception as e:
                print(f"Error processing voice queue: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Small delay to prevent tight loop
            await asyncio.sleep(0.1)

    async def _process_text_queue(self):
        """Process messages in the text queue."""
        while True:
            try:
                # Get next message from text queue
                message = await self.text_queue.get()
                print(f"Processing text queue message: {message}")

                if isinstance(message, dict):
                    text = message.get("text", "")
                    sender = message.get("sender", "")

                    if text and sender:
                        # Process the message through the chat manager
                        await self._process_chat_message(sender, {
                            "role": "user",
                            "content": text,
                            "name": sender
                        })

                # Mark task as done
                self.text_queue.task_done() # ENSURE THIS IS CALLED

            except Exception as e:
                print(f"Error in text queue processor: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1)

    async def _broadcast_audio_to_participants(self, audio_frame: tuple[int, np.ndarray]):
        """Broadcast audio to all active participants."""
        # This is now just a wrapper around broadcast_audio for backward compatibility
        return await self.broadcast_audio(audio_frame)

    async def send_message(self, message: Dict[str, Any]) -> Optional[str]:
        """Send a message and handle responses."""
        try:
            print(f"Processing message from {message.get('sender', 'unknown')}")

            # Add message to chat history
            self.messages.append({
                "role": "user",
                "content": message["content"],
                "name": message["sender"]
            })

            # Get response from group chat manager
            if hasattr(self, "manager"):
                print(f"Getting response from group chat manager for {message['sender']}")
                response = await self.manager.a_run_chat(
                    content=message["content"],
                    sender=message["sender"],
                )

                if response:
                    # Get the responding agent's name
                    agent_name = next((agent.name for agent in self.agents if agent != message["sender"]), "Assistant")
                    print(f"Got response from {agent_name}")

                    # Add response to chat history
                    self.messages.append({
                        "role": "assistant",
                        "content": response,
                        "name": agent_name
                    })
                    print(f"Added response to chat history from {agent_name}")

                    # Add to text queue for UI updates
                    print(f"Adding response to text queue from {agent_name}")
                    await self.text_queue.put({
                        "type": "chat",
                        "text": response,
                        "sender": agent_name,
                        "channel": "both"
                    })

                    # Convert to speech if agent voice is enabled
                    if self.agent_voice_enabled:
                        print(f"Converting response to speech for {agent_name}")
                        # Prefix with agent name for clarity
                        speech_text = f"{agent_name} says: {response}"

                        # Add to voice queue for speech synthesis
                        print(f"Adding response to voice queue from {agent_name}")
                        await self.voice_queue.put({
                            "type": "chat",
                            "text": speech_text,
                            "sender": agent_name,
                            "channel": "voice"
                        })

                        # Mark message as queued
                        self.messages[-1]["_voice_queued"] = True  # Use dict entry instead of attribute
                        print(f"Marked message as queued for {agent_name}")
                    else:
                        print(f"Agent voice is disabled, skipping speech synthesis for {agent_name}")

                    return response
                else:
                    print(f"No response received from group chat manager for {message['sender']}")
            else:
                print("No group chat manager available")

            return None

        except Exception as e:
            print(f"Error in send_message: {e}")
            import traceback
            traceback.print_exc()
            return None

    def agent(self, name_or_id: str) -> Optional[Agent]:
        """Find an agent by name or ID."""
        if not name_or_id:
            return None

        for agent in self.agents:
            if hasattr(agent, 'name') and agent.name == name_or_id:
                return agent
        return None

    def _log_error(self, message: str) -> None:
        """Log error messages."""
        print(f"Error: {message}")

    def set_channel_config(self, voice_enabled: bool = None, text_enabled: bool = None, agent_voice_enabled: bool = None):
        """Configure real-time communication channels."""
        if voice_enabled is not None:
            self.voice_enabled = voice_enabled
        if text_enabled is not None:
            self.text_enabled = text_enabled
        if agent_voice_enabled is not None:
            self.agent_voice_enabled = agent_voice_enabled

    def append(self, message: dict[str, Any], speaker: Agent) -> None:
        """Override append to intercept messages before they are added to the chat."""
        print("\n=== GroupChat Message Appended ===")
        print(f"From: {speaker.name if speaker else 'Unknown'}")

        # Extract content from message
        content = None
        if message and isinstance(message, dict):
            content = message.get("content", "")
            # Handle ChatResult objects
            if hasattr(content, 'summary'):
                content = content.summary
            elif hasattr(content, 'chat_history') and content.chat_history:
                # Get the last message from chat_history
                latest = content.chat_history[-1]
                content = latest.get('content', '') if isinstance(latest, dict) else str(latest)
            elif hasattr(content, 'content'):
                content = content.content
            elif not isinstance(content, str):
                content = str(content)

        print(f"Extracted content: {content[:100]}..." if content and len(content) > 100 else f"Extracted content: {content}")

        # Add message to our queues for processing
        if content and isinstance(content, str) and content.strip():
            # Create base message
            message_for_queues = {
                "type": "chat",
                "text": content,
                "sender": speaker.name if speaker else "Unknown"
            }
            with self._lock:
                # Update last message count atomically with message addition
                self._last_message_count = len(self.messages)
                # Add to text queue for display
                text_message = message_for_queues.copy()
                text_message["channel"] = "text"
                print("Adding message to text queue")
                asyncio.create_task(self.text_queue.put(text_message))

                # Add to voice queue if this is an agent message and voice is enabled
                if (speaker and
                    speaker.name not in self.human_participants and
                    self.agent_voice_enabled):

                    voice_message = message_for_queues.copy()
                    voice_message["channel"] = "voice"
                    print("Adding message to voice queue")
                    asyncio.create_task(self.voice_queue.put(voice_message))
                    # Mark the original message as queued
                    if isinstance(message, dict):
                        message["_voice_queued"] = True  # Mark the original message
                    print(f"Marked message as queued for {speaker.name}")
        # Call parent class's append
        super().append(message, speaker)

        # Update _last_message_count
        self._last_message_count = len(self.messages)
        print(f"Updated _last_message_count in append to: {self._last_message_count}") # ADDED LOGGING