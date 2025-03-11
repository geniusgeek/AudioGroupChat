import asyncio
import os
import json
import numpy as np
from typing import Optional, List, Dict, Any, Union
from fastrtc.tracks import AsyncStreamHandler
from aiortc.contrib.media import AudioFrame
from autogen.agentchat import GroupChat, Agent, UserProxyAgent
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from fastrtc import get_tts_model, get_stt_model, get_twilio_turn_credentials
from aiortc.mediastreams import AudioStreamTrack as FastRTCAudioStreamTrack
from fastrtc.reply_on_pause import ReplyOnPause, AlgoOptions, AppState
from fastrtc.stream import Stream as FastRTCStream
import time
import random

class AudioGroupChat(GroupChat):
    """Real-time audio group chat implementation enabling voice and text communication between humans and AI agents.
    
    This class extends GroupChat to provide real-time audio processing capabilities:
    1. Real-time audio streaming via WebRTC
    2. Speech-to-Text (STT) for human voice input
    3. Text-to-Speech (TTS) for agent responses
    4. Parallel message processing for voice and text
    
    Key Features:
    - Real-time audio processing and playback
    - Multi-participant support
    - AI agent integration
    - WebRTC-based streaming
    - Configurable voice/text channels
    
    Audio Pipeline:
    1. Input: WebRTC → ReplyOnPause → STT → Text
    2. Processing: Text → Agent → Response
    3. Output: Response → TTS → WebRTC → Audio
    
    Example:
        chat = AudioGroupChat(agents=[agent1, agent2])
        user_id = chat.add_human_participant('user1')
        chat.start_audio_session(user_id)
        await chat.send_message({'content': 'Hello!', 'sender': 'user1'})
    """
    
    def __init__(self, agents=None, messages=None, max_round=10, speaker_selection_method="round_robin", allow_repeat_speaker=False):
        """Initialize AudioGroupChat with audio processing capabilities.
        
        Args:
            agents (List[Agent], optional): List of AI agents. Defaults to None.
            messages (List[Dict], optional): Initial messages. Defaults to None.
            max_round (int, optional): Maximum conversation rounds. Defaults to 10.
            speaker_selection_method (str, optional): Method to select next speaker. Defaults to "round_robin".
            allow_repeat_speaker (bool, optional): Allow same speaker consecutively. Defaults to False.
            
        Components Initialized:
        1. TTS Model: For converting text to speech
        2. STT Model: For converting speech to text
        3. WebRTC: For real-time audio streaming
        4. Message Queues: For parallel processing
        5. Participant Tracking: For managing users
        """
        # Initialize GroupChat with proper parameters
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
        print("Initializing TTS model...")
        self.tts_model = get_tts_model("kokoro")
        print("TTS model initialized")
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
                
                # Process complete utterance
                text = self.stt_model.stt((sample_rate, audio_array))
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
        """Initialize async tasks and processors for real-time operation.
        
        This method sets up the core processing components:
        1. Task Creation: Start background processors
        2. Queue Setup: Initialize message queues
        3. Error Handling: Configure error callbacks
        4. State Management: Initialize tracking
        
        Components:
        - Voice Queue Processor: Handle audio messages
        - Text Queue Processor: Handle text messages
        - Agent Monitor: Track agent interactions
        - Error Handler: Manage task failures
        
        Processing Flow:
        1. Create async tasks
        2. Set error handlers
        3. Start processors
        4. Initialize state
        5. Begin monitoring
        """
        try:
            print("Initializing audio group chat...")
            
            # Start background tasks
            self._process_voice_queue_task = asyncio.create_task(self._process_voice_queue())
            self._process_text_queue_task = asyncio.create_task(self._process_text_queue())
            self._monitor_agent_messages_task = asyncio.create_task(self._monitor_agent_messages())
            
            # Add error handlers
            for task in [self._process_voice_queue_task, self._process_text_queue_task, self._monitor_agent_messages_task]:
                task.add_done_callback(self._task_error_handler)
                
            print("Audio group chat initialized successfully")
            
        except Exception as e:
            print(f"Error initializing audio group chat: {e}")
            import traceback
            traceback.print_exc()
            raise
            
    def _task_error_handler(self, task):
        """Handle errors in real-time processing tasks.
        
        This method implements error handling:
        1. Error Detection: Task monitoring
        2. Error Recovery: State restoration
        3. Task Management: Cleanup/restart
        4. Logging: Error tracking
        
        Error Types:
        - Task failures
        - Queue errors
        - Connection issues
        - Resource problems
        
        Args:
            task (asyncio.Task): Failed task with:
                - Exception info
                - Stack trace
                - State data
        
        Recovery Flow:
        1. Detect error
        2. Log details
        3. Clean resources
        4. Restore state
        5. Restart if needed
        """
        try:
            # Get the exception if any
            exc = task.exception()
            if exc:
                print(f"Background task error: {exc}")
                import traceback
                traceback.print_exc()
                
                # Restart the task if it's one of our monitored tasks
                if task in [self._process_voice_queue_task, self._process_text_queue_task, self._monitor_agent_messages_task]:
                    print("Restarting failed task...")
                    new_task = asyncio.create_task(task.get_coro())
                    new_task.add_done_callback(self._task_error_handler)
                    if task == self._process_voice_queue_task:
                        self._process_voice_queue_task = new_task
                    elif task == self._process_text_queue_task:
                        self._process_text_queue_task = new_task
                    elif task == self._monitor_agent_messages_task:
                        self._monitor_agent_messages_task = new_task
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
        """Process incoming audio from a user in real-time.
        
        This method implements the audio input pipeline:
        1. Audio Validation: Check format and content
        2. Speech Detection: Convert to text
        3. Message Creation: Format for chat
        4. Queue Management: Add to processing queues
        
        Real-time Features:
        - Immediate processing
        - Format validation
        - Error handling
        - Queue integration
        
        Args:
            user_id (str): ID of the user sending audio
            audio_data (tuple[int, np.ndarray]): Audio containing:
                - Sample rate (int)
                - Audio data (numpy.ndarray)
        
        Returns:
            dict: Response with:
                - text: Transcribed text
                - Additional metadata
        
        Processing Flow:
        1. Validate audio format
        2. Convert to text
        3. Create message
        4. Add to queues
        5. Return response
        """
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
            print(f"Error in audio input handler: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def handle_audio_output(self):
        """Process and return the next audio output for real-time playback.
        
        This method implements a non-blocking audio output processor that:
        1. Checks the voice queue for pending messages
        2. Processes text-to-speech conversion
        3. Formats audio data for playback
        4. Broadcasts audio to all participants
        
        The audio processing pipeline ensures:
        - Immediate processing of new messages
        - Proper audio data formatting
        - Real-time broadcasting to participants
        - Error handling and recovery
        
        Audio Format Specifications:
        - Sample Rate: 24kHz (standard for TTS output)
        - Data Type: 32-bit float
        - Amplitude Range: [-1.0, 1.0]
        - Channel Layout: Mono
        
        Returns:
            numpy.ndarray: Processed audio data ready for playback, or None if no audio
        
        Audio Processing Steps:
        1. Get message from voice queue
        2. Extract text content
        3. Convert text to speech
        4. Format audio data
        5. Broadcast to participants
        6. Return for UI playback
        """
        """Get the next audio output for Gradio UI.
        
        Returns:
            numpy.ndarray: Audio data array for Gradio audio output, or None if no audio
        """
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
                                tts_result = self.tts_model.tts(text)
                                if asyncio.iscoroutine(tts_result):
                                    sample_rate, audio_data = await asyncio.wait_for(tts_result, timeout=5.0)
                                else:
                                    sample_rate, audio_data = tts_result
                                
                                if audio_data is not None:
                                    # Ensure correct format
                                    if audio_data.dtype != np.float32:
                                        audio_data = audio_data.astype(np.float32)
                                    if np.max(np.abs(audio_data)) > 1.0:
                                        audio_data = audio_data / 32768.0
                                    
                                    print(f"Broadcasting TTS audio for {sender}: {text}")
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
        """Convert text to speech with real-time processing and delivery.
        
        This method handles the text-to-speech conversion pipeline:
        1. Text content extraction
        2. TTS conversion with timeout protection
        3. Audio data formatting
        4. Broadcasting to participants
        
        The TTS process is optimized for real-time chat:
        - Asynchronous processing with timeouts
        - Automatic format conversion
        - Direct broadcasting capability
        - Error handling and recovery
        
        Args:
            text (str | dict): Text to convert or message dict with 'content'
            user_id (str, optional): Target participant for direct sending
        
        Returns:
            tuple[int, np.ndarray] | None: Tuple of (sample_rate, audio_data) if successful
        
        Audio Specifications:
        - Sample Rate: 24kHz
        - Format: 32-bit float
        - Range: [-1.0, 1.0]
        - Layout: Mono
        
        Processing Steps:
        1. Extract text content
        2. Skip empty messages
        3. Convert to speech (with timeout)
        4. Format audio data
        5. Add to voice queue
        6. Return audio data
        """
        """Convert text to speech and send it to participants.
        
        Args:
            text: Text to convert to speech or chat result dict
            user_id: Optional target user ID. If None, broadcast to all
            
        Returns:
            tuple: A tuple of (sample_rate, audio_data) if successful, None otherwise
        """
        try:
            # Check if TTS is enabled and model is initialized
            if not hasattr(self, 'tts_model') or self.tts_model is None:
                print("TTS model not initialized")
                return None

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
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / 32768.0
                
            # Add to voice queue for broadcasting
            await self.voice_queue.put({
                "type": "audio",
                "sample_rate": sample_rate,
                "audio_data": audio_data,
                "text": text
            })
                
            print(f"Added TTS audio to voice queue: {text}")
            return sample_rate, audio_data
            
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
        """Broadcast audio to all participants except the sender.
        
        Args:
            audio_frame: Tuple of (sample_rate, audio_data)
            sender_id: Optional ID of the sender to exclude from broadcast
            
        Returns:
            bool: True if at least one participant received the audio successfully
        """
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
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / 32768.0
            
        # Add to voice queue for local playback
        await self.voice_queue.put({
            "type": "audio",
            "sample_rate": sample_rate,
            "audio_data": audio_data
        })
            
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
    
    async def _process_chat_message(self, user_id: str, message: Dict[str, Any]) -> None:
        """Process chat messages and handle responses in real-time.
        
        This method implements the core message processing pipeline:
        1. Content Extraction: Get text from various formats
        2. Message Formatting: Create proper chat message
        3. History Updates: Add to chat history
        4. Voice Processing: Queue agent messages for TTS
        5. Response Generation: Get and process agent responses
        
        Real-time Optimizations:
        - Immediate message processing
        - Async response generation
        - Direct voice queue integration
        - Smart message routing
        
        Args:
            user_id (str): Participant ID sending the message
            message (Dict[str, Any]): Message containing:
                - content/text: Message text
                - type: Usually 'chat'
                - sender: Message sender ID
                - Additional metadata
        
        Flow:
        1. Extract text content
        2. Create chat message
        3. Update UI history
        4. Process agent messages
        5. Generate responses
        6. Queue for voice output
        
        Voice Processing:
        - Human messages are processed directly
        - Agent responses are queued for TTS
        - Messages are broadcast to all participants
        """
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
            
            # Add message to messages list for UI updates
            print(f"Adding message to chat history: {chat_message}")
            self.messages.append(chat_message)
            
            # Add to voice queue for audio processing if from agent
            if user_id not in self.human_participants:
                await self.voice_queue.put({
                    "type": "chat",
                    "text": text,  # Send the actual text content
                    "sender": user_id,
                    "channel": "voice"
                })
                print(f"Added agent message to voice queue: {text}")
            
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
                            # Get the last message from chat history
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
                        
                        # Add agent response to voice queue for TTS
                        if self.agent_voice_enabled and response_text and isinstance(response_text, str) and response_text.strip():
                            message = {
                                "type": "chat",
                                "text": response_text,  # Send only the latest message content
                                "sender": recipient.name,
                                "channel": "voice"
                            }
                            await self.voice_queue.put(message)
                            print(f"Added latest agent response to voice queue: {response_text[:100]}..." if len(response_text) > 100 else f"Added latest agent response to voice queue: {response_text}")
                
        except Exception as e:
            print(f"Error processing chat message: {e}")
            import traceback
            traceback.print_exc()
                
    async def _process_voice_queue(self):
        """Process messages in the voice queue for real-time audio output.
        
        This method implements the voice processing pipeline:
        1. Queue Monitoring: Check for new messages
        2. Content Processing: Extract and validate text
        3. TTS Conversion: Generate audio output
        4. Broadcasting: Deliver to participants
        
        Real-time Features:
        - Continuous processing
        - Immediate conversion
        - Direct broadcasting
        - Error recovery
        
        Message Types:
        1. Chat Messages:
           - Contains text for TTS
           - Includes sender info
           - Specifies channel
        2. Audio Messages:
           - Contains raw audio
           - Ready for broadcast
        
        Processing Flow:
        1. Get queue message
        2. Identify message type
        3. Process accordingly
        4. Generate audio
        5. Broadcast output
        6. Handle errors
        """
        while True:
            try:
                # Get next message from voice queue
                message = await self.voice_queue.get()
                print(f"Processing voice queue message: {message}")
                
                if isinstance(message, dict):
                    message_type = message.get("type")
                    sender = message.get("sender")

                    if message_type == "chat":
                        text = message.get("text")
                        
                        # Only process messages from agents if voice is enabled
                        if text and sender and sender not in self.human_participants and self.agent_voice_enabled:
                            print(f"Converting to speech from voice queue: {text} from {sender}")
                            try:
                                # Extract text from ChatResult if needed
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
                                    
                                # Skip empty messages
                                if not text or not isinstance(text, str) or not text.strip():
                                    print(f"Skipping empty or invalid message: {text}")
                                    continue
                                    
                                # Convert text to speech
                                audio_result = await self.text_to_speech(text)
                                if audio_result:
                                    # Broadcast audio to all participants
                                    success = await self.broadcast_audio(audio_result, sender)
                                    if success:
                                        print(f"Successfully broadcast TTS audio for: {text}")
                                    else:
                                        print(f"Failed to broadcast TTS audio for: {text}")
                                else:
                                    print(f"Failed to convert text to speech: {text}")
                            except Exception as e:
                                print(f"Error processing chat message: {e}")
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
                                print(f"Error broadcasting audio: {e}")
                                import traceback
                                traceback.print_exc()
                                continue
                        else:
                            # Try alternate audio format
                            sample_rate = message.get("sample_rate")
                            audio_data = message.get("audio_data")
                            if sample_rate and audio_data is not None:
                                try:
                                    success = await self.broadcast_audio((sample_rate, audio_data), sender)
                                    if success:
                                        print("Successfully broadcast audio data")
                                    else:
                                        print("Failed to broadcast audio data")
                                except Exception as e:
                                    print(f"Error broadcasting audio: {e}")
                                    import traceback
                                    traceback.print_exc()
                                    continue
                            else:
                                print("Invalid audio message format")
                    
            except asyncio.CancelledError:
                # Allow clean shutdown
                print("Voice queue processor cancelled")
                raise
            except Exception as e:
                print(f"Error processing voice queue: {e}")
                import traceback
                traceback.print_exc()
                continue
                
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
                self.text_queue.task_done()
                
            except Exception as e:
                print(f"Error in text queue processor: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1)

    async def _monitor_agent_messages(self):
        """Monitor messages between agents and add them to the text queue."""
        while True:
            try:
                # Check for new messages
                if self.messages and len(self.messages) > self._last_message_count:
                    last_message = self.messages[-1]
                    self._last_message_count = len(self.messages)
                    
                    # Extract text from message
                    text = None
                    sender = None
                    
                    if isinstance(last_message, tuple):
                        text = last_message[1]  # Agent's response is in second position
                        sender = last_message[0].name if hasattr(last_message[0], 'name') else str(last_message[0])
                    elif isinstance(last_message, dict):
                        text = last_message.get('content', '')
                        sender = last_message.get('name', '')
                    
                    if text and text.strip() and sender:
                        # Add to voice queue for TTS
                        await self.voice_queue.put({
                            "type": "chat",
                            "text": text,
                            "sender": sender,
                            "channel": "text"
                        })
                
                await asyncio.sleep(0.1)  # Small delay to prevent tight loop
                
            except Exception as e:
                print(f"Error in agent message monitor: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1)
                
    async def _broadcast_audio_to_participants(self, audio_frame: tuple[int, np.ndarray]):
        """Broadcast audio to all active participants.
        
        Args:
            audio_frame: Tuple of (sample_rate, audio_data)
        """
        # This is now just a wrapper around broadcast_audio for backward compatibility
        return await self.broadcast_audio(audio_frame)
                
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
        """Configure real-time communication channels.
        
        This method manages channel settings:
        1. Voice Channel: Audio streaming
        2. Text Channel: Message passing
        3. Agent Voice: TTS responses
        4. Channel States: Active tracking
        
        Configuration Options:
        - Independent channel control
        - Selective agent responses
        - Real-time updates
        - State persistence
        
        Args:
            voice_enabled (bool, optional): Enable voice streaming
            text_enabled (bool, optional): Enable text messages
            agent_voice_enabled (bool, optional): Enable agent TTS
        
        Config Flow:
        1. Update settings
        2. Apply changes
        3. Update state
        4. Notify handlers
        """
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
                # Get the last message from chat history
                latest = content.chat_history[-1]
                content = latest.get('content', '') if isinstance(latest, dict) else str(latest)
            elif hasattr(content, 'content'):
                content = content.content
            elif not isinstance(content, str):
                content = str(content)
        
        print(f"Extracted content: {content[:100]}..." if content and len(content) > 100 else f"Extracted content: {content}")
        
        # Add message to our queue for processing
        if content and isinstance(content, str) and content.strip():
            print("Adding message to text queue")
            asyncio.create_task(self.text_queue.put({
                "type": "chat",
                "text": content,
                "sender": message.get("name", speaker.name if speaker else "Agent")
            }))
        
        # Call parent class method to maintain normal functionality
        super().append(message, speaker)