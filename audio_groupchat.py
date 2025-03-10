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
from fastrtc.pause_detection import get_silero_model
import time
import random

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
        """Process incoming audio from a user.
        
        Args:
            user_id: ID of the user sending audio
            audio_data: Tuple of (sample_rate, audio_array)
            
        Returns:
            dict: Response containing text and/or audio
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
            
            # Ensure audio is in correct format
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
                audio_array = audio_array / 32768.0  # Convert from int16 range to float32 [-1, 1]
            
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
        """Stream audio output for Gradio UI.
        
        This method continuously monitors messages and converts them to speech.
        
        Yields:
            tuple: A tuple of (sample_rate, audio_data) for Gradio audio output
        """
        try:
            # Get the latest message from the chat
            if self.messages and len(self.messages) > self._last_message_count:
                last_message = self.messages[-1]
                self._last_message_count = len(self.messages)
                
                # Extract text from message
                text = None
                if isinstance(last_message, tuple):
                    text = last_message[1]  # Agent's response is in second position
                elif isinstance(last_message, dict):
                    text = last_message.get('content', '')
                    
                if text and text.strip():
                    print(f"Converting to speech: {text}")
                    # Convert to speech and yield audio
                    audio = await self.text_to_speech(text)
                    if audio is not None:
                        print("Audio generated successfully")
                        return (48000, audio)  # Use WebRTC standard sample rate
                    else:
                        print("Failed to generate audio")
                        
        except Exception as e:
            print(f"Error in handle_audio_output: {e}")
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
            print("\n=== Processing Chat Message ===")
            print(f"From User: {user_id}")
            print(f"Message: {message}")
            
            # Verify this is a valid participant
            if user_id not in self.human_participants:
                print(f"Warning: Message from unknown user {user_id}")
                return
                
            # Get message content
            text = message.get("content", message.get("text", ""))
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
            
            # Add to voice queue for audio processing
            await self.voice_queue.put({
                "type": "chat",
                "text": text,
                "sender": user_id,
                "channel": "voice"
            })
            
            # Get the first available agent as recipient
            recipient = next((agent for agent in self.agents if agent.name != user_id), None)
            if not recipient:
                print(f"No available agents to receive message from {user_id}")
                return
            
            # Get sender agent object
            sender = self.agent(user_id)
            if not sender:
                print(f"Error: Could not find agent for user {user_id}")
                return
            
            # Initiate chat with the message
            response = await self.manager.a_initiate_chat(
                message=text,
                sender=sender,
                recipient=recipient
            )
                
            # Process response if any
            if response:
                response_msg = {
                    "role": "assistant",
                    "content": response,
                    "name": recipient.name
                }
                self.messages.append(response_msg)
                    
                # Convert response to speech if agent voice is enabled
                if self.agent_voice_enabled:
                    await self.text_to_speech(response, user_id)
                    
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
                
                if isinstance(message, tuple) and len(message) == 2:
                    # Direct audio data (sample_rate, audio_data)
                    sample_rate, audio_data = message
                    
                    # Ensure audio is in correct format
                    if audio_data.dtype != np.float32:
                        audio_data = audio_data.astype(np.float32)
                    if np.max(np.abs(audio_data)) > 1.0:
                        audio_data = audio_data / 32768.0
                    
                    # Broadcast to all participants
                    await self._broadcast_audio_to_participants((sample_rate, audio_data))
                    
                elif isinstance(message, dict):
                    # Text message that needs TTS
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
                                print(f"Successfully broadcasted TTS audio for {sender}")
                            else:
                                print(f"TTS failed for message: {text}")
                                
                        except asyncio.TimeoutError:
                            print(f"TTS timed out for message: {text}")
                        except Exception as e:
                            print(f"Error converting message to speech: {e}")
                            import traceback
                            traceback.print_exc()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in voice queue processor: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in voice queue processor: {e}")
                await asyncio.sleep(1)
                
    async def _process_text_queue(self):
        """Process messages in the text queue."""
        while True:
            try:
                # Get next message from queue
                message = await self.text_queue.get()
                
                # Skip if text channel is disabled
                if not self.text_enabled:
                    continue
                    
                print(f"\n=== Processing Text Message ===")
                print(f"Message: {message}")
                    
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
                    
                    # Only convert agent messages to speech here, not in _monitor_agent_messages
                    if sender not in self.human_participants and self.agent_voice_enabled:
                        try:
                            print(f"Converting agent message to speech: {text}")
                            # Call tts() - it returns (sample_rate, audio_data) directly, no await needed
                            sample_rate, audio_data = self.tts_model.tts(text)
                                
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
        """Monitor messages between agents and add them to the text queue."""
        while True:
            try:
                # Check for new messages
                if len(self.messages) > self._last_message_count:
                    print("\n=== Checking New Messages ===")
                    print(f"Total messages: {len(self.messages)}")
                    print(f"Last processed: {self._last_message_count}")
                    
                    # Get new messages
                    new_messages = self.messages[self._last_message_count:]
                    self._last_message_count = len(self.messages)
                    
                    print(f"Found {len(new_messages)} new messages")
                    
                    # Process each new message
                    for msg in new_messages:
                        print("\n=== Processing Message ===")
                        print(f"Message: {msg}")
                        
                        # Only add agent messages to text queue (TTS happens in _process_text_queue)
                        if msg.get("role") == "assistant":
                            sender = msg.get("name", "Agent")
                            content = msg.get("content", "")
                            
                            if content and content.strip():
                                print(f"Adding agent message to text queue from {sender}: {content}")
                            # Add to text queue for processing
                            await self.text_queue.put({
                                "type": "chat",
                                "text": content,
                                "sender": sender,
                                "channel": "text"
                            })
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error monitoring agent messages: {e}")
                await asyncio.sleep(1)

    async def _broadcast_audio_to_participants(self, audio_frame: tuple[int, np.ndarray]):
        """Broadcast audio to all active participants.
        
        Args:
            audio_frame: Tuple of (sample_rate, audio_data)
        """
        print("\n=== Broadcasting Audio ===")
        
        # Get list of active participants
        participants = [
            user_id for user_id, participant in self.human_participants.items()
            if participant.get("active") and participant.get("stream") is not None
        ]
        
        if not participants:
            print("No active participants with streams to broadcast to")
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
                if stream and stream.event_handler:
                    print(f"Sending audio to {user_id}")
                    # Create coroutine to send audio through queue
                    async def send_audio():
                        await stream.event_handler.queue.put((sample_rate, audio_data))
                    tasks.append(asyncio.create_task(send_audio()))
                else:
                    print(f"Stream for {user_id} is closed or invalid")
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
            
    def append(self, message: dict[str, Any], speaker: Agent) -> None:
        """Override append to intercept messages before they are added to the chat."""
        print("\n=== GroupChat Message Appended ===")
        print(f"From: {speaker.name if speaker else 'Unknown'}")
        print(f"Message: {message}")
        
        # Add message to our queue for processing
        if message and isinstance(message, dict):
            content = message.get("content", "")
            if content and isinstance(content, str):
                print("Adding message to text queue")
                asyncio.create_task(self.text_queue.put({
                    "type": "chat",
                    "text": content,
                    "sender": message.get("name", speaker.name if speaker else "Agent")
                }))
            
        # Call parent class method to maintain normal functionality
        super().append(message, speaker)