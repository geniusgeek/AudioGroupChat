import asyncio
import os
import json
import numpy as np
from typing import Optional, List, Dict, Any, Union
from autogen.agentchat import GroupChat, Agent
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from fastrtc import get_tts_model, get_stt_model, get_twilio_turn_credentials
from aiortc.mediastreams import AudioStreamTrack as FastRTCAudioStreamTrack
from fastrtc.tracks import StreamHandlerBase, AudioCallback
from fastrtc.stream import Stream as FastRTCStream

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
        self.tts_model = get_tts_model("kokoro")
        self.stt_model = get_stt_model()
        
        # Configure WebRTC settings
        self.rtc_config = get_twilio_turn_credentials() if os.environ.get("TWILIO_ACCOUNT_SID") else None
        
        # Create FastRTC stream for audio handling
        self.stream = FastRTCStream(
            modality="audio",
            mode="send-receive",
            handler=AudioStreamHandler(),
            rtc_configuration=self.rtc_config,
            concurrency_limit=5,  # Allow up to 5 concurrent connections
            time_limit=None  # No time limit on sessions
        )
        
        # Initialize participant tracking
        self.human_participants = {}
        self.active_calls = {}

    def add_agent(self, agent: Agent):
        if not hasattr(self, 'agents'):
            self.agents = []
        self.agents.append(agent)
        return agent
        
    def add_human_participant(self, user_id: str) -> str:
        """Add a human participant to the group chat.
        
        Args:
            user_id: ID of the participant to add
            
        Returns:
            A unique session ID for the participant
        """
        try:
            print(f"\nAdding human participant: {user_id}")
            
            # Validate user_id
            if not user_id:
                raise ValueError("user_id cannot be empty")
            if user_id in self.human_participants:
                print(f"Participant {user_id} already exists")
                return str(hash(user_id))
            
            # Create a new audio handler for this participant
            audio_handler = AudioStreamHandler()
            
            # Store participant info
            self.human_participants[user_id] = {
                "handler": audio_handler,
                "active": False,
                "session_id": str(hash(user_id))
            }
            
            print(f"Participant {user_id} added successfully")
            print("Current participants:", list(self.human_participants.keys()))
            
            return self.human_participants[user_id]["session_id"]
            
        except Exception as e:
            print(f"Error adding human participant: {e}")
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
            
            # Create and set WebRTC offer
            try:
                offer = await participant["pc"].createOffer()
                await participant["pc"].setLocalDescription(offer)
                print("WebRTC offer created and set successfully")
            except Exception as e:
                print(f"Error creating WebRTC offer: {e}")
                return
            
            # Setup ICE candidate handling
            participant["pc"].onicecandidate = lambda candidate: self._handle_ice_candidate(user_id, candidate)
            print("ICE candidate handling configured")
            
            # Start the group chat if not already started
            if self.manager and not self.messages:
                print("\nInitializing group chat...")
                
                # Find required agents
                agent1 = next((agent for agent in self.agents if agent.name == "Agent1"), None)
                human_agent = next((agent for agent in self.agents if agent.name == user_id), None)
                
                print("Available agents:", [agent.name for agent in self.agents])
                print(f"Found Agent1: {agent1 is not None}")
                print(f"Found human agent: {human_agent is not None}")
                
                if agent1 and human_agent:
                    print("Starting chat with welcome message...")
                    try:
                        await self.manager.a_initiate_chat(
                            agent1,
                            message={
                                "content": "Hello! I'm Agent1. Welcome to our audio group chat. How can we assist you today?",
                                "sender": agent1,
                                "recipients": [human_agent],
                                "channel": "both"
                            }
                        )
                        print("Welcome message sent successfully")
                    except Exception as e:
                        print(f"Error sending welcome message: {e}")
                else:
                    print("Error: Missing required agents for chat initialization")
            else:
                print("Chat already initialized or manager not set")
                
        except Exception as e:
            print(f"Error in start_audio_session: {e}")
            import traceback
            traceback.print_exc()

    async def text_to_speech(self, text: str) -> np.ndarray:
        try:
            # Ensure text is not empty or just whitespace
            if not text or text.strip() == "":
                print("Empty text provided to TTS, returning silence")
                return np.zeros(48000, dtype=np.int16)  # 1 second of silence at 48kHz
                
            # Preprocess text to ensure it's suitable for TTS
            text = text.strip()
            
            # Call TTS model with error handling
            try:
                result = self.tts_model.tts(text)
            except Exception as e:
                print(f"TTS model error: {e}")
                return np.zeros(48000, dtype=np.int16)
            
            # Process TTS result
            if isinstance(result, tuple) and len(result) >= 2:
                _, audio_data = result
                if isinstance(audio_data, np.ndarray) and len(audio_data) > 0:
                    # Ensure correct format for WebRTC
                    if audio_data.dtype != np.int16:
                        audio_data = np.clip(audio_data * 32768, -32768, 32767).astype(np.int16)
                    return audio_data
            
            # If we got here, something went wrong with the TTS output
            print(f"TTS model returned invalid output format: {type(result)}")
            return np.zeros(24000, dtype=np.int16)
        except Exception as e:
            print(f"TTS conversion failed: {e}")
            # Return a short silence instead of empty data
            return np.zeros(24000, dtype=np.int16)

    async def _handle_audio_input(self, audio_data: tuple[int, np.ndarray]):
        """Handle incoming audio from a user.
        
        Args:
            audio_data: A tuple containing (sample_rate, audio_array)
        """
        try:
            # Validate audio data format
            if not isinstance(audio_data, tuple) or len(audio_data) != 2:
                print(f"Invalid audio data format: {type(audio_data)}")
                return
            
            sample_rate, audio_array = audio_data
            
            # Process audio through our handler
            audio_handler = AudioStreamHandler()
            processed_audio = audio_handler.process_audio(audio_array)
            
            if processed_audio is None or len(processed_audio) == 0:
                print("Warning: No audio data after processing")
                return
            
            # Convert speech to text
            try:
                # Convert to int16 if needed
                if processed_audio.dtype != np.int16:
                    processed_audio = (processed_audio * 32768).astype(np.int16)
                text = self.stt_model.stt(processed_audio)
            except Exception as e:
                print(f"STT processing failed: {e}")
                return
            
            if text and text.strip():
                print(f"Transcribed text: {text}")
                # Return both text and processed audio
                return {
                    "text": text,
                    "audio": processed_audio
                }
            else:
                print("No text transcribed from audio")
                return None
        except Exception as e:
            print(f"Error in audio input handler: {e}")
            import traceback
            traceback.print_exc()

    async def _setup_negotiation(self, user_id: str):
        participant = self.human_participants[user_id]
        if self.negotiator:
            await self.negotiator.create_offer(
                participant["audio_session"],
                lambda candidate: self._handle_ice_candidate(user_id, candidate)
            )

    def _handle_ice_candidate(self, user_id: str, candidate: dict):
        if self.negotiator:
            self.negotiator.add_ice_candidate(
                self.human_participants[user_id]["audio_session"],
                candidate
            )

    async def broadcast_audio(self, stream, sender_id: str):
        for user_id, participant in self.human_participants.items():
            if user_id != sender_id:
                max_retries = 2
                for attempt in range(max_retries):
                    try:
                        # Create a new audio handler for each participant
                        audio_handler = AudioStreamHandler()
                        
                        # Process different types of input streams
                        if isinstance(stream, FastRTCStream):
                            # Copy stream configuration
                            audio_handler.sample_rate = getattr(stream, 'sample_rate', 48000)
                            audio_handler.frame_size = getattr(stream, 'frame_size', 960)
                            
                            # Process audio frames
                            try:
                                frame = await stream.event_handler.next_frame()
                                if frame is not None:
                                    audio_handler.receive(frame)
                                else:
                                    print("No audio frame available")
                                    continue
                            except Exception as e:
                                print(f"Error processing frame: {e}")
                                continue
                                
                        elif isinstance(stream, np.ndarray):
                            # Convert numpy array to proper format
                            if stream.dtype != np.int16:
                                stream = np.clip(stream * 32768, -32768, 32767).astype(np.int16)
                            if len(stream.shape) > 1:
                                stream = stream.mean(axis=1).astype(np.int16)
                            audio_handler.receive(stream)
                            
                        elif isinstance(stream, bytes):
                            try:
                                # Convert bytes to numpy array
                                audio_array = np.frombuffer(stream, dtype=np.int16)
                                if len(audio_array.shape) > 1:
                                    audio_array = audio_array.mean(axis=1).astype(np.int16)
                                audio_handler.receive(audio_array)
                            except Exception as e:
                                print(f"Failed to process audio bytes: {e}")
                                continue
                        else:
                            print(f"Unsupported stream type: {type(stream)}")
                            continue

                        # Create new stream with proper configuration and track
                        audio_stream = FastRTCStream(
                            handler=audio_handler,
                            mode="send-receive",
                            modality="audio"
                        )
                        # Initialize audio track
                        audio_track = AudioCallback(
                            track=FastRTCAudioStreamTrack(),
                            event_handler=audio_handler,
                            channel=None
                        )
                        audio_stream.track = audio_track
                        
                        # Update participant's audio track using proper FastRTC pattern
                        # Use existing audio track instead of creating new
                        participant["pc_track"].track = audio_stream.track
                        print(f"Successfully delivered audio to {user_id}")
                        break
                    except Exception as e:
                        print(f"Stream delivery failed to {user_id} (attempt {attempt + 1}): {e}")
                        await asyncio.sleep(1)

    async def send_message(self, message: Dict[str, Any]) -> Optional[str]:
        try:
            # Create properly formatted GroupChat message
            chat_message = {
                "content": message["content"],
                "sender": self.agent(message["sender"]),
                "recipient": tuple(self.agent(r) for r in message["recipients"]),
                "channel": message.get("channel", "both")
            }

            # Get response from group chat manager
            if self.manager:
                response = await self.manager.a_run(chat_message)
                if response:
                    # Process based on channel
                    if chat_message["channel"] in ["text", "both"]:
                        # Add to chat history
                        self.messages.append({
                            "content": response,
                            "role": "assistant",
                            "name": "Agent"
                        })

                    if chat_message["channel"] in ["audio", "both"]:
                        # Convert to speech and broadcast
                        try:
                            audio_data = await self.text_to_speech(response)
                            if audio_data is not None:
                                # Broadcast to all participants
                                await self.broadcast_audio(audio_data, "agent")
                        except Exception as e:
                            print(f"Error in TTS processing: {e}")
                            import traceback
                            traceback.print_exc()

                    return response

            return None
        except Exception as e:
            print(f"Error sending message: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def _process_audio_message(self, message: Dict[str, Any]) -> None:
        try:
            if not message.get("content"):
                return
                
            # Convert text to speech
            audio_data = await self.text_to_speech(message["content"])
            
            # Create audio stream
            audio_handler = AudioStreamHandler()
            audio_stream = FastRTCStream(
                handler=audio_handler,
                mode="send-receive",
                modality="audio"
            )
            
            # Process audio through stream handler
            audio_handler.receive(audio_data)
            
            # Broadcast audio to all participants except sender
            await self.broadcast_audio(audio_stream, message.get("sender", ""))
        except Exception as e:
            print(f"Error in audio message processing: {e}")
            import traceback
            traceback.print_exc()

    def _handle_chat_message(self, user_id: str, msg):
        """Handle incoming chat messages from WebRTC data channel."""
        try:
            message = json.loads(msg)
            if message.get("type") == "chat" and message.get("text"):
                # Create task to handle message asynchronously
                asyncio.create_task(self._process_chat_message(user_id, message))
        except Exception as e:
            print(f"Error handling chat message: {e}")
            import traceback
            traceback.print_exc()
    
    async def _process_chat_message(self, user_id: str, message: Dict[str, Any]):
        """Process chat messages and handle responses."""
        try:
            # Send message and get response
            response = await self.send_message({
                "content": message["text"],
                "sender": user_id,
                "recipients": [a.name for a in self.agents],
                "channel": message.get("channel", "both")
            })
            
            if response:
                # Send response back through data channel
                participant = self.human_participants.get(user_id)
                if participant and participant["data_channel"].readyState == "open":
                    participant["data_channel"].send(json.dumps({
                        "type": "chat",
                        "text": response,
                        "sender": "Agent"
                    }))
        except Exception as e:
            print(f"Error processing chat message: {e}")
            import traceback
            traceback.print_exc()

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