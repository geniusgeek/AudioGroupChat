import gradio as gr
import asyncio
import numpy as np
from starlette.requests import ClientDisconnect
from fastrtc import (
    get_stt_model, get_tts_model, ReplyOnPause, AdditionalOutputs,
    get_hf_turn_credentials, AlgoOptions, SileroVadOptions, wait_for_item
)
from fastrtc.stream import Stream

from audio_groupchat import AudioGroupChat

class GradioUI:
    def __init__(self, audio_chat=None):
        self.current_user_id = "user_123"
        
        # Use provided AudioGroupChat instance or create new one
        self.audio_chat = audio_chat or AudioGroupChat()
        
        # Initialize message queues if not present
        if not hasattr(self.audio_chat, 'voice_queue'):
            self.audio_chat.voice_queue = asyncio.Queue()
        if not hasattr(self.audio_chat, 'text_queue'):
            self.audio_chat.text_queue = asyncio.Queue()
        
        # Create Gradio components
        self.messages = gr.Chatbot(
            label="Conversation History",
            type="messages",
            height=400,
            show_label=True,
            value=[]
        )

        # Initialize FastRTC stream with ReplyOnPause handler
        self.stream = Stream(
            modality="audio",
            mode="send-receive",
            handler=ReplyOnPause(
                self.audio_input_handler,
                input_sample_rate=48000,
            ),
            additional_inputs=[gr.Textbox(value=self.current_user_id, visible=False)],
            additional_outputs=[self.messages],
            additional_outputs_handler=lambda prev, current: current,
            rtc_configuration=self.audio_chat.rtc_config,
            time_limit=90,
            ui_args={
                "title": "Huddle Audio Group Chat",
                "icon_button_color": "#5c5c5c",
                "pulse_color": "#a7c6fc",
                "icon_radius": 0,
            }
        )
        
        # Add user to audio chat
        self.audio_chat.add_human_participant(self.current_user_id)
        print("   - Human participants:", list(self.audio_chat.human_participants.keys()))
    
    def _log_error(self, message: str, error: Exception = None):
        """Log error messages with optional exception details."""
        print(f"[ERROR] {message}")
        if error:
            print(f"Exception details: {str(error)}")
            import traceback
            traceback.print_exc()

    async def audio_input_handler(self, audio_data: tuple[int, np.ndarray], user_id: str, chatbot: list[dict] | None = None) -> tuple[tuple[int, np.ndarray] | None, list[dict]]:
        """Handle incoming audio from the user with optimized streaming."""
        try:
            chatbot = chatbot or []
            
            # Extract sample rate and audio data
            if isinstance(audio_data, tuple) and len(audio_data) == 2:
                sample_rate, audio_array = audio_data
            else:
                print(f"Warning: Unexpected audio data format: {type(audio_data)}")
                return None, chatbot
            
            # Ensure audio data is in the correct format
            if not isinstance(audio_array, np.ndarray):
                print(f"Warning: Audio data is not a numpy array: {type(audio_array)}")
                return None, chatbot
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
                audio_array = audio_array.mean(axis=1)
            
            # Ensure 1D array for STT model
            audio_array = audio_array.reshape(-1)
            
            # Normalize audio to [-1, 1] range for STT
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            if np.max(np.abs(audio_array)) > 1.0:
                audio_array = audio_array / 32768.0
            
            try:
                # Process audio through AudioGroupChat
                response = await self.audio_chat._handle_audio_input((sample_rate, audio_array), user_id)
                if not response:
                    return None, chatbot
                
                # Update chat history with user's message
                if response.get("text"):
                    text = response["text"]
                    print(f"Transcribed text: {text}")
                    chatbot.append({"role": "user", "content": text})
                    
                    # Add to voice queue for processing
                    await self.audio_chat.voice_queue.put({
                        "type": "chat",
                        "text": text,
                        "sender": user_id,
                        "channel": "voice"
                    })
                
                # Add agent responses to chat history
                agent_responses = response.get("agent_responses", [])
                for agent_response in agent_responses:
                    if isinstance(agent_response, dict):
                        content = agent_response.get("content")
                        sender = agent_response.get("name", "Assistant")
                        if content:
                            chat_message = {
                                "role": "assistant",
                                "content": content,
                                "name": sender
                            }
                            chatbot.append(chat_message)
                            
                            # Queue for TTS
                            await self.audio_chat.voice_queue.put({
                                "type": "chat",
                                "text": content,
                                "sender": sender,
                                "channel": "voice"
                            })
                
                return None, chatbot
                
            except ClientDisconnect:
                self._log_error("Client disconnected during audio processing")
                return None, chatbot
                    
        except Exception as e:
            self._log_error("Audio input processing error", e)
            return None, chatbot

    async def audio_output_handler(self):
        """Handle audio output stream."""
        # Initialize chatbot history
        chatbot = []
        
        # Return empty audio initially
        yield (48000, np.zeros((1, 48000), dtype=np.float32)), chatbot
        
        while True:
            try:
                # Get audio stream from AudioGroupChat
                participant = self.audio_chat.human_participants.get(self.current_user_id)
                if participant and participant.get("active"):
                    # Process audio frames from the queue
                    try:
                        audio_data = await asyncio.wait_for(self.audio_chat.voice_queue.get(), timeout=0.1)
                        if isinstance(audio_data, dict):
                            # Handle chat message
                            if audio_data.get("type") == "chat" and audio_data.get("text"):
                                sender = audio_data.get("sender", "Assistant")
                                text = audio_data["text"]
                                
                                # Add message to chat history with sender info
                                chat_message = {
                                    "role": "assistant" if sender not in self.audio_chat.human_participants else "user",
                                    "content": text,
                                    "name": sender
                                }
                                chatbot.append(chat_message)
                                
                                # Convert text to speech if it's from an agent
                                if sender not in self.audio_chat.human_participants:
                                    try:
                                        # Call TTS and handle result
                                        tts_result = self.audio_chat.tts_model.tts(text)
                                        
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
                                                
                                            # Broadcast audio
                                            await self.audio_chat._broadcast_audio_to_participants((sample_rate, audio_data))
                                        else:
                                            print(f"TTS failed for message: {text}")
                                    except asyncio.TimeoutError:
                                        print(f"TTS timed out for message: {text}")
                                    except Exception as e:
                                        self._log_error("Text-to-speech error", e)
                                
                                # Return updated chat history
                                yield (48000, np.zeros((1, 48000), dtype=np.float32)), chatbot
                                
                        elif isinstance(audio_data, tuple) and len(audio_data) == 2:
                            # Handle audio data
                            sample_rate, audio_array = audio_data
                            if len(audio_array.shape) == 1:
                                audio_array = audio_array.reshape(1, -1)
                            yield (sample_rate, audio_array), chatbot
                        else:
                            yield (48000, np.zeros((1, 48000), dtype=np.float32)), chatbot
                            
                    except asyncio.TimeoutError:
                        # No audio available, yield silence
                        yield (48000, np.zeros((1, 48000), dtype=np.float32)), chatbot
                    except ClientDisconnect:
                        # Client disconnected, clean up and exit
                        self._log_error("Client disconnected")
                        return
                    except Exception as e:
                        self._log_error("Error processing audio output", e)
                        yield (48000, np.zeros((1, 48000), dtype=np.float32)), chatbot
                    
                    await asyncio.sleep(0.02)  # Small delay to prevent busy-waiting
                else:
                    # No audio stream available, yield silence
                    yield (48000, np.zeros((1, 48000), dtype=np.float32)), chatbot
                    await asyncio.sleep(0.1)  # Longer delay when no stream
            except ClientDisconnect:
                # Client disconnected, clean up and exit
                self._log_error("Client disconnected")
                return
            except Exception as e:
                self._log_error("Audio output error", e)
                # Yield silence and wait before retrying
                yield (48000, np.zeros((1, 48000), dtype=np.float32)), chatbot
                await asyncio.sleep(1)

    async def init_audio_session(self):
        """Initialize the audio session and start group chat."""
        print("\nInitializing audio session...")
        await self.audio_chat.start_audio_session(self.current_user_id)
        print("Audio session started successfully")

    def create_interface(self):
        """Create the Gradio interface."""
        with gr.Blocks(title="Huddle Audio Chat") as demo:
            with gr.Row():
                with gr.Column(scale=1):
                    # User ID field (read-only)
                    user_id = gr.Textbox(
                        label="User ID",
                        value=self.current_user_id,
                        interactive=False
                    )
                    
                    # Audio input with microphone
                    audio_input = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        streaming=True,
                        label="Speak here",
                        format="wav"
                    )
                    
                with gr.Column(scale=2):
                    # Chat history
                    self.messages.render()
                    
                    # Audio output
                    audio_output = gr.Audio(
                        label="AI Response",
                        autoplay=True,
                        show_label=True,
                        type="numpy",
                        format="wav",
                        elem_id="audio-output"
                    )
            
            # Initialize audio session when interface loads
            demo.load(fn=self.init_audio_session)
            
            # Handle audio streams
            audio_input.stream(
                fn=self.audio_input_handler,
                inputs=[audio_input, user_id],
                outputs=[audio_output, self.messages],
                show_progress=False
            )
            
            # Handle audio output
            audio_output.stream(
                fn=self.audio_output_handler,
                inputs=None,
                outputs=[audio_output, self.messages],
                show_progress=False,
                queue=True
            )
            
            # Enable queuing for smoother audio handling
            demo.queue()
            
            return demo

if __name__ == "__main__":
    ui = GradioUI()
    ui.create_interface().launch()