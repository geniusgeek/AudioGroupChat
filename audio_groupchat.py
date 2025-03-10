import asyncio
import os
import json
import numpy as np
from typing import Optional, List, Dict, Any, Union
from autogen.agentchat import GroupChat, Agent, UserProxyAgent
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from fastrtc import get_tts_model, get_stt_model, get_twilio_turn_credentials
from aiortc.mediastreams import AudioStreamTrack as FastRTCAudioStreamTrack
from fastrtc.tracks import StreamHandlerBase, AudioCallback
from fastrtc.stream import Stream as FastRTCStream
import time
import random

class AudioStreamHandler(StreamHandlerBase):
    """Handles audio stream processing for the group chat."""
    
    def __init__(self, expected_layout: str = "mono", output_sample_rate: int = 24000):
        super().__init__(
            expected_layout=expected_layout,
            output_sample_rate=output_sample_rate,
            output_frame_size=960,  # Standard frame size for 20ms
            input_sample_rate=48000  # Standard WebRTC sample rate
        )
        self.stt_model = get_stt_model()
        self.audio_buffer = []
        self.last_processed_time = time.time()
        self.processing_interval = 1.0  # Process every 1 second
        self._current_frame = None

    def receive(self, frame: tuple[int, np.ndarray]) -> None:
        """Process incoming audio frame.
        
        Args:
            frame: Tuple of (sample_rate, audio_data)
        """
        try:
            sample_rate, audio_data = frame
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Ensure int16 format
            if audio_data.dtype != np.int16:
                if audio_data.dtype == np.float32:
                    audio_data = (audio_data * 32767).astype(np.int16)
                else:
                    audio_data = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
            
            # Store processed frame
            self._current_frame = (sample_rate, audio_data)
            
            # Add to buffer for STT processing
            self.audio_buffer.append(audio_data)
            
            # Process buffer if enough time has passed
            current_time = time.time()
            if current_time - self.last_processed_time >= self.processing_interval:
                self._process_buffer()
                
        except Exception as e:
            print(f"Error processing audio frame: {e}")
            self._current_frame = (48000, np.zeros(960, dtype=np.int16))
    
    def emit(self) -> tuple[int, np.ndarray] | None:
        """Emit processed audio frame."""
        return self._current_frame
    
    def copy(self) -> 'AudioStreamHandler':
        """Create a copy of this handler."""
        new_handler = AudioStreamHandler(
            expected_layout=self.expected_layout,
            output_sample_rate=self.output_sample_rate
        )
        new_handler.stt_model = self.stt_model
        return new_handler
    
    def _process_buffer(self) -> None:
        """Process accumulated audio buffer."""
        if not self.audio_buffer:
            return
            
        try:
            # Concatenate buffer
            audio_data = np.concatenate(self.audio_buffer)
            
            # Convert speech to text
            text = self.stt_model.transcribe(audio_data)
            
            # Send transcribed text through data channel if available
            if text and self._channel:
                self._channel.send(json.dumps({"type": "transcription", "text": text}))
            
            # Clear buffer and update time
            self.audio_buffer = []
            self.last_processed_time = time.time()
            
        except Exception as e:
            print(f"Error processing audio buffer: {e}")

class AudioGroupChat(GroupChat):
    def __init__(self, agents=None, messages=None, max_round=10, speaker_selection_method="round_robin", allow_repeat_speaker=False):
        # Initialize GroupChat with proper parameters
        super().__init__(
            agents=agents or [], 
            messages=messages or [],
            max_round=max_round,
            speaker_selection_method=speaker_selection_method,
            allow_repeat_speaker=allow_repeat_speaker,
        )
        
        # Initialize audio processing components
        print("Initializing TTS model...")
        self.tts_model = get_tts_model("kokoro")
        print("TTS model initialized")
        
        print("Initializing STT model...")
        self.stt_model = get_stt_model()
        print("STT model initialized")
        
        # Configure WebRTC settings
        self.rtc_config = get_twilio_turn_credentials() if os.environ.get("TWILIO_ACCOUNT_SID") else None
        
        # Create FastRTC stream for audio handling
        self.stream = FastRTCStream(
            modality="audio",
            mode="send-receive",
            handler=AudioStreamHandler(),
            rtc_configuration=self.rtc_config,
            concurrency_limit=10,  # Allow up to 10 concurrent connections
            time_limit=None  # No time limit on sessions
        )
        
        # Initialize participant tracking
        self.human_participants = {}
        self.active_calls = {}
        
        # Channel configuration
        self.voice_enabled = True  # Enable voice by default
        self.text_enabled = True   # Enable text by default
        self.agent_voice_enabled = True  # Enable agent voice by default
        
        # Message queues for each channel
        self.voice_queue = asyncio.Queue()
        self.text_queue = asyncio.Queue()
        
        # Track last message count to detect new messages
        self._last_message_count = 0
        
        # Required for autogen compatibility
        self.client_cache = []
        self.previous_cache = []
        
        # Initialize queues
        self.voice_queue = asyncio.Queue()
        self.text_queue = asyncio.Queue()
        
    async def initialize(self):
        """Initialize async tasks and processors."""
        try:
            print("Initializing audio group chat...")
            
            # Store tasks to prevent garbage collection
            self._tasks = [
                asyncio.create_task(self._process_voice_queue()),
                asyncio.create_task(self._process_text_queue()),
                asyncio.create_task(self._monitor_agent_messages())
            ]
            
            # Set up error handlers
            for task in self._tasks:
                task.add_done_callback(self._task_error_handler)
                
            print("Audio group chat initialized successfully")
            
        except Exception as e:
            print(f"Error initializing audio group chat: {e}")
            import traceback
            traceback.print_exc()
            raise
            
    def _task_error_handler(self, task):
        """Handle errors in background tasks."""
        try:
            # Get the exception if any
            exc = task.exception()
            if exc:
                print(f"Background task error: {exc}")
                import traceback
                traceback.print_exc()
                
                # Restart the task if it's one of our monitored tasks
                if task in self._tasks:
                    print("Restarting failed task...")
                    new_task = asyncio.create_task(task.get_coro())
                    new_task.add_done_callback(self._task_error_handler)
                    self._tasks[self._tasks.index(task)] = new_task
        except asyncio.CancelledError:
            pass

    def add_agent(self, agent: Agent):
        if not hasattr(self, 'agents'):
            self.agents = []
        self.agents.append(agent)
        return agent
        
    def add_human_participant(self, user_id: str) -> str:
        """Add a human participant to the group chat.
        
        Args:
            user_id: ID for the new participant
            
        Returns:
            str: Session ID for the participant
        """
        print(f"\nAdding human participant: {user_id}")
        
        try:
            # Create session ID
            session_id = str(random.getrandbits(64))
            
            # Create audio handler for this participant
            audio_handler = AudioStreamHandler()
            
            # Create user proxy agent
            user_agent = UserProxyAgent(
                name=user_id,
                human_input_mode="NEVER",
                code_execution_config={"use_docker": False}
            )
            
            # Add participant info
            self.human_participants[user_id] = {
                "session_id": session_id,
                "handler": audio_handler,
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
        """Start an audio session for a participant and initialize the group chat.
        
        Args:
            user_id: ID of the participant to start session for
        """
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
            
            # Initialize FastRTC stream for this participant if not already done
            if "stream" not in participant:
                # Create audio handler for this participant
                audio_handler = participant["handler"]
                
                # Set up handler's audio callback
                async def on_audio(frame):
                    await self._handle_audio_input(frame, user_id)
                audio_handler.audio_callback = on_audio
                
                # Create unique stream for this participant
                participant["stream"] = FastRTCStream(
                    modality="audio",
                    mode="send-receive",
                    handler=audio_handler,
                    rtc_configuration=self.rtc_config,
                    concurrency_limit=1,  # Only one connection per participant
                    time_limit=None
                )
                
            print(f"Audio session started for {user_id}")
            
        except Exception as e:
            print(f"Error in start_audio_session: {e}")
            import traceback
            traceback.print_exc()
            raise

    async def _handle_audio_input(self, audio_data: tuple[int, np.ndarray], user_id: str = None) -> dict:
        """Process incoming audio from a user.
        
        Args:
            user_id: ID of the user sending audio
            audio_data: Tuple of (sample_rate, audio_array)
            
        Returns:
            dict: Response containing text and/or audio
        """
        try:
            # Extract audio data
            sample_rate, audio_array = audio_data
            
            # Ensure audio is in correct format
            if isinstance(audio_array, bytes):
                audio_array = np.frombuffer(audio_array, dtype=np.float32)
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Ensure 1D array (sequence_length,)
            audio_array = audio_array.reshape(-1)
            
            # Normalize audio to [-1, 1] range
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            if np.max(np.abs(audio_array)) > 1.0:
                audio_array = audio_array / 32768.0  # Convert from int16 range to float32 [-1, 1]
            
            # Use STT model to convert speech to text
            text = self.stt_model.stt((sample_rate, audio_array))
            
            if not text:
                return None
                
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
            
        except Exception as e:
            print(f"Error in audio input handler: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    async def text_to_speech(self, text: str | dict, user_id: str = None) -> np.ndarray | None:
        """Convert text to speech and send it to participants.
        
        Args:
            text: Text to convert to speech or chat result dict
            user_id: Optional target user ID. If None, broadcast to all
            
        Returns:
            np.ndarray: Audio data if successful, None otherwise
        """
        try:
            # Extract text from chat result if needed
            if isinstance(text, dict):
                text = text.get("content", "")
            elif not isinstance(text, str):
                text = str(text)
            
            # Skip empty messages
            if not text or text.strip() == "":
                return None
                
            print(f"Converting to speech: {text}")
                
            # Convert to speech using tts() method with timeout
            try:
                # Call tts() and await the result
                tts_result = self.tts_model.tts(text)
                if not hasattr(tts_result, '__await__'):
                    # If not awaitable, assume it's already the result
                    sample_rate, audio_data = tts_result
                else:
                    # If awaitable, wait for result with timeout
                    sample_rate, audio_data = await asyncio.wait_for(
                        tts_result,
                        timeout=5.0  # 5 second timeout
                    )
            except asyncio.TimeoutError:
                print("TTS timed out")
                return None
                
            if audio_data is None:
                print("TTS failed to generate audio")
                return None
                
            # Ensure correct format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / 32768.0
                
            print("Broadcasting audio to participants")
            # Broadcast to all participants or send to specific user
            if user_id:
                await self._send_audio_to_participant(user_id, (sample_rate, audio_data))
            else:
                await self._broadcast_audio_to_participants((sample_rate, audio_data))
                
            return audio_data
            
        except Exception as e:
            print(f"Error in text_to_speech: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def _send_audio_to_participant(self, user_id: str, audio_frame: tuple[int, np.ndarray], max_retries: int = 3) -> bool:
        """Send audio data to a specific participant.
        
        Args:
            user_id: Target participant ID
            audio_frame: Tuple of (sample_rate, audio_data)
            max_retries: Maximum number of retry attempts
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
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
                await stream.send_audio((sample_rate, audio_data))
                print(f"Successfully sent audio to {user_id}")
                return True
                    
            except Exception as e:
                print(f"Error sending audio to {user_id} (attempt {attempt + 1}): {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(0.5)  # Brief delay before retry
                
        return False

    async def broadcast_audio(self, audio_frame: tuple[int, np.ndarray], sender_id: str = None):
        """Broadcast audio to all participants except the sender.
        
        Args:
            audio_frame: Tuple of (sample_rate, audio_data)
            sender_id: Optional ID of the sender to exclude from broadcast
        """
        print(f"Broadcasting audio to all participants (except {sender_id})")
        
        # Get list of active participants excluding sender
        participants = [
            user_id for user_id, participant in self.human_participants.items()
            if user_id != sender_id and participant.get("active")
        ]
        
        if not participants:
            print("No active participants to broadcast to")
            return
            
        print(f"Broadcasting to participants: {participants}")
        
        # Send to all participants in parallel
        tasks = [
            self._send_audio_to_participant(user_id, audio_frame)
            for user_id in participants
        ]
        
        # Wait for all sends to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check for any failures
        for user_id, result in zip(participants, results):
            if isinstance(result, Exception):
                print(f"Failed to send to {user_id}: {result}")
            elif not result:
                print(f"Failed to send to {user_id}")

    async def _handle_chat_message(self, user_id: str, msg: Dict[str, Any]):
        """Handle incoming chat messages.
        
        Args:
            user_id: ID of the participant sending the message
            msg: Message dictionary
        """
        try:
            if msg.get("type") == "chat" and msg.get("text"):
                # Create task to handle message asynchronously
                asyncio.create_task(self._process_chat_message(user_id, msg))
        except Exception as e:
            print(f"Error handling chat message: {e}")
            import traceback
            traceback.print_exc()
    
    async def _process_chat_message(self, user_id: str, message: Dict[str, Any]):
        """Process chat messages and handle responses.
        
        Args:
            user_id: ID of the participant sending the message
            message: Message dictionary
        """
        try:
            # Get message content
            text = message.get("content", message.get("text", ""))
            if not text or not text.strip():
                return
                
            # Create chat message
            chat_message = {
                "role": "user" if user_id in self.human_participants else "assistant",
                "content": text,
                "name": user_id
            }
            
            # Add to messages and voice queue
            self.messages.append(chat_message)
            await self.voice_queue.put({
                "type": "chat",
                "text": text,
                "sender": user_id,
                "channel": "voice"
            })
            
            # Get response from group chat manager
            if user_id in self.human_participants:
                # Create message for the chat
                message = {
                    "role": "user",
                    "content": text,
                    "name": user_id
                }
                
                # Get the first agent as recipient
                recipient = next((agent for agent in self.agents if agent.name != user_id), None)
                if not recipient:
                    print(f"No available agents to receive message from {user_id}")
                    return
                
                # Initiate chat with the message
                await self.manager.a_initiate_chat(
                    message=text,
                    sender=self.agent(user_id),
                    recipient=recipient
                )
                    
                # Convert response to speech if agent voice is enabled
                if self.agent_voice_enabled:
                    await self.text_to_speech(text,user_id)
                        
            # Determine channel for message propagation
            channel = message.get("channel", "both")
            
            # Add original message to appropriate queue for UI updates
            if channel in ["text", "both"] and self.text_enabled:
                await self.text_queue.put({
                    "type": "chat",
                    "text": text,
                    "sender": user_id,
                    "channel": "text"
                })
                
            if channel in ["audio", "both"] and self.voice_enabled:
                await self.voice_queue.put({
                    "type": "chat",
                    "text": text,
                    "sender": user_id,
                    "channel": "voice"
                })
                
        except Exception as e:
            print(f"Error processing chat message: {e}")
            import traceback
            traceback.print_exc()
            
    async def _process_voice_queue(self):
        """Process messages in the voice queue."""
        while True:
            try:
                # Get next message from queue
                message = await self.voice_queue.get()
                
                # Skip if voice channel is disabled
                if not self.voice_enabled:
                    continue
                    
                print(f"Processing voice message: {message}")
                
                if isinstance(message, tuple):
                    # Direct audio data (sample_rate, audio_data)
                    sample_rate, audio_data = message
                    await self._broadcast_audio_to_participants((sample_rate, audio_data))
                    
                elif isinstance(message, dict):
                    # Text message that needs TTS
                    text = message.get("text", "")
                    sender = message.get("sender", "Assistant")
                    
                    if text and text.strip():
                        try:
                            print(f"Converting to speech: {text}")
                            # Convert to speech with timeout
                            sample_rate, audio_data = await asyncio.wait_for(
                                self.tts_model.tts(text),
                                timeout=5.0
                            )
                            
                            if audio_data is not None:
                                # Ensure correct format
                                if audio_data.dtype != np.float32:
                                    audio_data = audio_data.astype(np.float32)
                                if np.max(np.abs(audio_data)) > 1.0:
                                    audio_data = audio_data / 32768.0
                                    
                                print(f"Broadcasting TTS audio for {sender}")
                                await self._broadcast_audio_to_participants((sample_rate, audio_data))
                            else:
                                print(f"TTS failed for message: {text}")
                                
                        except asyncio.TimeoutError:
                            print(f"TTS timed out for message: {text}")
                        except Exception as e:
                            print(f"Error processing voice message: {e}")
                            import traceback
                            traceback.print_exc()
                
                try:
                    if isinstance(message, tuple) and len(message) == 2:
                        # Direct audio data
                        sample_rate, audio_array = message
                        
                        # Ensure audio is in correct format
                        if audio_array.dtype != np.float32:
                            audio_array = audio_array.astype(np.float32)
                        if np.max(np.abs(audio_array)) > 1.0:
                            audio_array = audio_array / 32768.0
                        
                        # Broadcast to all participants
                        await self._broadcast_audio_to_participants((sample_rate, audio_array))
                        
                    elif isinstance(message, dict):
                        # Get message content
                        text = message.get("text", "")
                        sender = message.get("sender", "unknown")
                        
                        # Skip empty messages
                        if not text or not text.strip():
                            continue
                            
                        print(f"Processing voice message from {sender}: {text}")
                        
                        # Convert to speech if needed
                        if self.agent_voice_enabled and message.get("type") == "chat":
                            try:
                                # If it's an agent message, add the agent name
                                if sender not in self.human_participants:
                                    text = f"{sender} says: {text}"
                                
                                # Convert to speech
                                sample_rate, audio_data = await asyncio.wait_for(
                                    self.tts_model.tts(text),
                                    timeout=5.0  # 5 second timeout for TTS
                                )
                                
                                if audio_data is not None:
                                    # Ensure correct format
                                    if audio_data.dtype != np.float32:
                                        audio_data = audio_data.astype(np.float32)
                                    if np.max(np.abs(audio_data)) > 1.0:
                                        audio_data = audio_data / 32768.0
                                    
                                    # Broadcast to all participants
                                    await self._broadcast_audio_to_participants((sample_rate, audio_data))
                                    print(f"Broadcasted TTS audio for message from {sender}")
                                else:
                                    print(f"TTS failed to generate audio for message from {sender}")
                                    
                            except asyncio.TimeoutError:
                                print(f"TTS timed out for message from {sender}")
                            except Exception as e:
                                print(f"Error converting message to speech: {e}")
                                import traceback
                                traceback.print_exc()
                    
                except Exception as e:
                    print(f"Error processing voice message: {e}")
                    import traceback
                    traceback.print_exc()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in voice queue processor: {e}")
                await asyncio.sleep(1)
                
    async def _broadcast_audio_to_participants(self, audio_frame: tuple[int, np.ndarray]):
        """Broadcast audio to all active participants.
        
        Args:
            audio_frame: Tuple of (sample_rate, audio_data)
        """
        print("Broadcasting audio to all participants")
        
        # Get list of active participants
        participants = [
            user_id for user_id, participant in self.human_participants.items()
            if participant.get("active")
        ]
        
        if not participants:
            print("No active participants to broadcast to")
            return
            
        print(f"Broadcasting to participants: {participants}")
        
        # Ensure audio data is in correct format
        sample_rate, audio_data = audio_frame
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / 32768.0
            
        # Send to all participants in parallel
        tasks = []
        for user_id in participants:
            try:
                participant = self.human_participants[user_id]
                stream = participant.get("stream")
                if stream:
                    print(f"Sending audio to {user_id}")
                    tasks.append(stream.send_audio((sample_rate, audio_data)))
                else:
                    print(f"No stream found for {user_id}")
            except Exception as e:
                print(f"Error preparing to send to {user_id}: {e}")
                import traceback
                traceback.print_exc()
                
        if tasks:
            # Wait for all sends to complete
            try:
                await asyncio.gather(*tasks)
                print("Successfully broadcasted audio to all participants")
            except Exception as e:
                print(f"Error during broadcast: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("No tasks created for broadcasting")
                
    async def _process_text_queue(self):
        """Process messages in the text queue."""
        while True:
            try:
                # Get next message from queue
                message = await self.text_queue.get()
                
                # Skip if text channel is disabled
                if not self.text_enabled:
                    continue
                    
                print(f"Processing text message: {message}")
                    
                # Process message
                try:
                    # Get message content
                    if isinstance(message, tuple):
                        sender, text = message
                    else:
                        text = message.get("text", "")
                        sender = message.get("sender", "unknown")
                        
                    print(f"Message from {sender}: {text}")
                    
                    if not text or not text.strip():
                        continue
                        
                    # Add message to messages list for UI updates
                    chat_message = {
                        "role": "user" if sender in self.human_participants else "assistant",
                        "content": text,
                        "name": sender
                    }
                    self.messages.append(chat_message)
                    print(f"Added message to chat history: {chat_message}")
                    
                    # If it's an agent message, convert to speech
                    if sender not in self.human_participants and self.agent_voice_enabled:
                        try:
                            print(f"Converting agent message to speech: {text}")
                            # Call tts() and handle result
                            tts_result = self.tts_model.tts(text)
                            
                            # Handle both awaitable and non-awaitable results
                            if not hasattr(tts_result, '__await__'):
                                sample_rate, audio_data = tts_result
                            else:
                                sample_rate, audio_data = await asyncio.wait_for(
                                    tts_result,
                                    timeout=5.0
                                )
                                
                            if audio_data is not None:
                                # Ensure correct format
                                if audio_data.dtype != np.float32:
                                    audio_data = audio_data.astype(np.float32)
                                if np.max(np.abs(audio_data)) > 1.0:
                                    audio_data = audio_data / 32768.0
                                    
                                print(f"Broadcasting TTS audio for {sender}")
                                await self._broadcast_audio_to_participants((sample_rate, audio_data))
                            else:
                                print(f"TTS failed for message: {text}")
                                
                        except asyncio.TimeoutError:
                            print(f"TTS timed out for message: {text}")
                        except Exception as e:
                            print(f"Error converting message to speech: {e}")
                            import traceback
                            traceback.print_exc()
                except Exception as e:
                    print(f"Error processing text message: {e}")
                    import traceback
                    traceback.print_exc()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in text queue processor: {e}")
                await asyncio.sleep(1)
                
    async def _monitor_agent_messages(self):
        """Monitor messages between agents and convert to speech."""
        while True:
            try:
                # Check for new messages
                if len(self.messages) > self._last_message_count:
                    # Get new messages
                    new_messages = self.messages[self._last_message_count:]
                    self._last_message_count = len(self.messages)
                    
                    # Process each new message
                    for msg in new_messages:
                        # Process all agent messages (including agent-to-agent)
                        if msg.get("role") == "assistant":
                            sender = msg.get("name", "Agent")
                            content = msg.get("content", "")
                            
                            if content and content.strip():
                                print(f"Processing agent message from {sender}: {content}")
                                
                                # Add to text queue for UI updates
                                await self.text_queue.put({
                                    "type": "chat",
                                    "text": content,
                                    "sender": sender,
                                    "channel": "text"
                                })
                                
                                # Convert to speech if agent voice is enabled
                                if self.agent_voice_enabled:
                                    try:
                                        # Convert to speech
                                        print(f"Converting message to speech: {content}")
                                        sample_rate, audio_data = await asyncio.wait_for(
                                            self.tts_model.tts(content),
                                            timeout=5.0  # 5 second timeout for TTS
                                        )
                                        
                                        if audio_data is not None:
                                            # Ensure correct format
                                            if audio_data.dtype != np.float32:
                                                audio_data = audio_data.astype(np.float32)
                                            if np.max(np.abs(audio_data)) > 1.0:
                                                audio_data = audio_data / 32768.0
                                                
                                            # Broadcast audio to all participants
                                            await self._broadcast_audio_to_participants((sample_rate, audio_data))
                                            print(f"Broadcasted TTS audio for message from {sender}")
                                        else:
                                            print(f"TTS failed to generate audio for message from {sender}")
                                            
                                    except asyncio.TimeoutError:
                                        print(f"TTS timed out for message from {sender}")
                                    except Exception as e:
                                        print(f"Error converting message to speech: {e}")
                                        import traceback
                                        traceback.print_exc()
                                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
                    
            except Exception as e:
                print(f"Error monitoring agent messages: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1)  # Longer delay on error
                
    async def send_message(self, message: Dict[str, Any]) -> Optional[str]:
        """Send a message and handle responses.
        
        Args:
            message: Message dictionary containing content, sender, recipients, and channel
            
        Returns:
            Optional response string from the group chat manager
        """
        try:
            # Add message to chat history
            self.messages.append({
                "role": "user",
                "content": message["content"],
                "name": message["sender"]
            })
            
            # Get response from group chat manager
            if hasattr(self, "manager"):
                response = await self.manager.a_run_chat(
                    content=message["content"],
                    sender=message["sender"],
                )
                
                if response:
                    # Get the responding agent's name
                    agent_name = next((agent.name for agent in self.agents if agent != message["sender"]), "Assistant")
                    
                    # Add response to chat history
                    self.messages.append({
                        "role": "assistant",
                        "content": response,
                        "name": agent_name
                    })
                    
                    # Add to text queue for UI updates
                    await self.text_queue.put({
                        "type": "chat",
                        "text": response,
                        "sender": agent_name,
                        "channel": "both"
                    })
                    
                    # Convert to speech if agent voice is enabled
                    if self.agent_voice_enabled:
                        # Prefix with agent name for clarity
                        speech_text = f"{agent_name} says: {response}"
                        
                        # Add to voice queue for speech synthesis
                        await self.voice_queue.put({
                            "type": "chat",
                            "text": speech_text,
                            "sender": agent_name,
                            "channel": "voice"
                        })
                    
                    # Convert to speech if needed
                    if message.get("channel") in ["audio", "both"] and self.agent_voice_enabled:
                        await self.text_to_speech(response)
                    
                    return response
            
            return None
                
        except Exception as e:
            print(f"Error sending message: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def agent(self, name_or_id: str) -> Optional[Agent]:
        """Find an agent by name or ID.
        
        Args:
            name_or_id: Name or ID of the agent to find
            
        Returns:
            Agent if found, None otherwise
        """
        if not name_or_id:
            return None
        
        for agent in self.agents:
            if hasattr(agent, 'name') and agent.name == name_or_id:
                return agent
        return None

    def _log_error(self, message: str) -> None:
        """Log error messages.
        
        Args:
            message: Error message to log
        """
        print(f"Error: {message}")

    def set_channel_config(self, voice_enabled: bool = None, text_enabled: bool = None, agent_voice_enabled: bool = None):
        """Configure channel settings.
        
        Args:
            voice_enabled: Enable/disable voice channel
            text_enabled: Enable/disable text channel
            agent_voice_enabled: Enable/disable agent voice responses
        """
        if voice_enabled is not None:
            self.voice_enabled = voice_enabled
        if text_enabled is not None:
            self.text_enabled = text_enabled
        if agent_voice_enabled is not None:
            self.agent_voice_enabled = agent_voice_enabled