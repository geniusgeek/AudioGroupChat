import gradio as gr
import asyncio
import numpy as np
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

    async def audio_input_handler(self, audio_data: tuple[int, np.ndarray], user_id: str, chatbot: list[dict] | None = None) -> tuple[int, np.ndarray] | None:
        """Handle incoming audio from the user with optimized streaming."""
        try:
            chatbot = chatbot or []
            
            # Extract sample rate and audio data
            if isinstance(audio_data, tuple) and len(audio_data) == 2:
                sample_rate, audio_array = audio_data
            else:
                print(f"Warning: Unexpected audio data format: {type(audio_data)}")
                return
            
            # Ensure audio data is in the correct format
            if not isinstance(audio_array, np.ndarray):
                print(f"Warning: Audio data is not a numpy array: {type(audio_array)}")
                return
            
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
            
            # Process audio through AudioGroupChat
            response = await self.audio_chat._handle_audio_input((sample_rate, audio_array), user_id)
            if not response:
                return None, chatbot
            
            # Update chat history with user's message
            if response.get("text"):
                text = response["text"]
                print(f"Transcribed text: {text}")
                chatbot.append({
                    "role": "user",
                    "content": text,
                    "name": user_id
                })
                
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
                    agent_name = agent_response.get("name", "Assistant")
                    content = agent_response.get("content")
                    if content:
                        chatbot.append({
                            "role": "assistant",
                            "content": content,
                            "name": agent_name
                        })
            
            return None, chatbot
                    
        except Exception as e:
            self._log_error("Audio input processing error", e)
            return None, chatbot

    async def audio_output_handler(self):
        """Handle audio output stream."""
        # Return empty audio initially
        yield (48000, np.zeros((1, 48000), dtype=np.float32))
        
        while True:
            try:
                # Get audio stream from AudioGroupChat
                participant = self.audio_chat.human_participants.get(self.current_user_id)
                if participant and participant.get("active") and "stream" in participant:
                    # Process audio frames from the queue
                    try:
                        audio_data = await asyncio.wait_for(self.audio_chat.voice_queue.get(), timeout=0.1)
                        if isinstance(audio_data, dict):
                            # Handle chat message
                            if audio_data.get("type") == "chat" and audio_data.get("text"):
                                yield AdditionalOutputs([{
                                    "role": "assistant",
                                    "content": audio_data["text"],
                                    "name": audio_data.get("sender", "Assistant")
                                }])
                        elif isinstance(audio_data, tuple) and len(audio_data) == 2:
                            # Handle audio data
                            sample_rate, audio_array = audio_data
                            if len(audio_array.shape) == 1:
                                audio_array = audio_array.reshape(1, -1)
                            yield (sample_rate, audio_array)
                        else:
                            yield (48000, np.zeros((1, 48000), dtype=np.float32))
                    except asyncio.TimeoutError:
                        # No audio available, yield silence
                        yield (48000, np.zeros((1, 48000), dtype=np.float32))
                    except Exception as e:
                        self._log_error("Error processing audio output", e)
                        yield (48000, np.zeros((1, 48000), dtype=np.float32))
                    
                    await asyncio.sleep(0.02)  # Small delay to prevent busy-waiting
                else:
                    # No audio stream available, yield silence
                    yield (48000, np.zeros((1, 48000), dtype=np.float32))
                    await asyncio.sleep(0.1)  # Longer delay when no stream
            except Exception as e:
                self._log_error("Audio output error", e)
                # Yield silence and wait before retrying
                yield (48000, np.zeros((1, 48000), dtype=np.float32))
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
            
            return demo

if __name__ == "__main__":
    ui = GradioUI()
    ui.create_interface().launch()