import asyncio
import numpy as np
import gradio as gr
from audio_groupchat import AudioGroupChat

class GradioUI:
    def __init__(self, audio_chat=None):
        self.current_user_id = "user_123"
        
        # Use provided AudioGroupChat instance or create new one
        self.audio_chat = audio_chat or AudioGroupChat()
        
        # Get stream from AudioGroupChat
        self.stream = self.audio_chat.stream
        
        # Create chat history component
        self.messages = gr.Chatbot(
            label="Conversation History",
            type="messages",
            height=400,
            show_label=True,
            value=[]
        )
        
        # Add user to audio chat
        session_id = self.audio_chat.add_human_participant(self.current_user_id)
        if session_id:
            participant = self.audio_chat.human_participants.get(self.current_user_id)
            if participant:
                participant["stream"] = self.stream
                participant["active"] = True
                print(f"Stream attached to participant {self.current_user_id}")
            
        print("   - Human participants:", list(self.audio_chat.human_participants.keys()))
        
    def _update_messages(self, prev_messages, new_messages):
        """Update chat history with new messages"""
        try:
            # Handle ChatResult objects
            if hasattr(new_messages, 'chat_history'):
                messages_to_process = new_messages.chat_history
            else:
                messages_to_process = new_messages

            # Convert to list if single message
            if isinstance(messages_to_process, dict):
                messages_to_process = [messages_to_process]
            
            # Process messages
            result = []
            for msg in messages_to_process:
                if isinstance(msg, dict):
                    # Skip empty messages
                    if not msg.get('content', '').strip():
                        continue
                        
                    # Ensure proper role assignment
                    if msg.get('name') in self.audio_chat.human_participants:
                        msg['role'] = 'user'
                    else:
                        msg['role'] = 'assistant'
                        
                    result.append([msg['content'], None] if msg['role'] == 'user' else [None, msg['content']])
                
            return result if result else prev_messages
                
        except Exception as e:
            print(f"Error updating messages: {e}")
            return prev_messages
    
    def _log_error(self, message: str, error: Exception = None):
        """Log error messages with optional exception details."""
        print(f"[ERROR] {message}")
        if error:
            print(f"Exception details: {str(error)}")
            import traceback
            traceback.print_exc()

    async def init_audio_session(self):
        """Initialize the audio session and start group chat."""
        try:
            print("\nInitializing audio session...")
            # Initialize audio chat components
            await self.audio_chat.initialize()
            
            # Start audio session for current user
            await self.audio_chat.start_audio_session(self.current_user_id)
            
            # Ensure stream is ready
            if not self.stream or not self.stream.event_handler:
                raise RuntimeError("Stream or handler not properly initialized")
                
            print("Audio session started successfully")
            
        except Exception as e:
            self._log_error("Failed to initialize audio session", e)
            raise

    def _init_session_sync(self):
        """Initialize audio session synchronously."""
        try:
            print("Initializing audio session...")
            asyncio.run(self.init_audio_session())
            print("Audio session initialized successfully")
        except Exception as e:
            print(f"Error initializing audio session: {e}")
            raise

    def create_interface(self):
        """Create the Gradio interface."""
        with gr.Blocks(title="Huddle Audio Chat") as demo:
            with gr.Row():
                with gr.Column():
                    user_id = gr.Textbox(label="User ID", value=self.current_user_id)
                    # Audio input with microphone
                    audio_input = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        streaming=True,
                        label="Speak here",
                        interactive=True,
                        format="wav",
                    )
                with gr.Column():
                    # Create Gradio components
                    self.messages = gr.Chatbot(
                        label="Conversation History",
                        type="messages",
                        height=400,
                        show_label=True,
                        value=[]
                    )
                    # Text input
                    text_input = gr.Textbox(
                        label="Type your message",
                        placeholder="Press Enter to send",
                        show_label=True,
                        lines=2
                    )
                    # Audio output for agent responses
                    audio_output = gr.Audio(
                        label="AI Response",
                        autoplay=True,
                        show_label=True,
                        type="numpy",
                        streaming=False,  # Set to False for complete audio chunks
                        interactive=False,
                        elem_id="audio-output",
                        format="wav"
                    )
            
            # Configure stream components
            self.stream.additional_inputs = [user_id]
            self.stream.additional_outputs = [audio_output, self.messages]
            self.stream.additional_outputs_handler = self._update_messages
            
            # Configure audio output
            audio_output.autoplay = True
            audio_output.show_download = False
            
            # Get the ReplyOnPause handler from the stream
            if not self.stream.event_handler:
                raise RuntimeError("Stream event_handler not initialized")
            
            # Create a wrapper function to handle the audio stream
            async def handle_audio_stream(audio_data, user):
                try:
                    if audio_data is None or not isinstance(audio_data, tuple) or len(audio_data) != 2:
                        print("Invalid or no audio data received")
                        yield None, self.messages.value
                        return
                    
                    try:
                        sample_rate, audio_samples = audio_data
                        if not isinstance(audio_samples, np.ndarray):
                            print(f"Invalid audio format: {type(audio_samples)}")
                            yield None, self.messages.value
                            return
                    except Exception as e:
                        print(f"Error unpacking audio data: {e}")
                        yield None, self.messages.value
                        return

                    print(f"Received audio input: {type(audio_data)}")
                    
                    # Process audio through AudioGroupChat
                    await self.audio_chat._handle_audio_input(audio_data, user)
                    
                    # Get audio response from AudioGroupChat
                    audio_response = await self.audio_chat.handle_audio_output()
                    
                    # Return the audio response and updated messages
                    if audio_response is not None:
                        # Normalize audio to [-1, 1] range before converting to int16
                        max_val = np.max(np.abs(audio_response))
                        if max_val > 0:
                            normalized_audio = audio_response / max_val
                        else:
                            normalized_audio = audio_response
                            
                        # Convert to int16 range [-32768, 32767]
                        int16_audio = (normalized_audio * 32767).astype(np.int16)
                        
                        # Convert sample rate from 24kHz to 48kHz
                        resampled_audio = np.repeat(int16_audio, 2)  # Simple resampling by repeating samples
                        
                        print(f"Audio output stats - min: {np.min(resampled_audio)}, max: {np.max(resampled_audio)}, mean: {np.mean(resampled_audio)}")
                        yield (48000, resampled_audio), self.audio_chat.messages
                    else:
                        yield None, self.audio_chat.messages
                    
                except Exception as e:
                    self._log_error("Error in handle_audio_stream", e)
                    yield None, self.messages.value

            # Create a wrapper function to handle text input
            async def handle_text_input(text, user):
                try:
                    if not text or not text.strip():
                        yield None, self.messages.value
                        return
                    
                    # Process text through AudioGroupChat
                    await self.audio_chat._handle_chat_message(user, {
                        "type": "chat",
                        "text": text,
                        "sender": user,
                        "channel": "text"
                    })
                    
                    # Get audio response from AudioGroupChat
                    audio_response = await self.audio_chat.handle_audio_output()
                    
                    # Return the audio response and updated messages
                    if audio_response is not None:
                        # Normalize audio to [-1, 1] range before converting to int16
                        max_val = np.max(np.abs(audio_response))
                        if max_val > 0:
                            normalized_audio = audio_response / max_val
                        else:
                            normalized_audio = audio_response
                            
                        # Convert to int16 range [-32768, 32767]
                        int16_audio = (normalized_audio * 32767).astype(np.int16)
                        
                        # Convert sample rate from 24kHz to 48kHz
                        resampled_audio = np.repeat(int16_audio, 2)  # Simple resampling by repeating samples
                        
                        print(f"Audio output stats - min: {np.min(resampled_audio)}, max: {np.max(resampled_audio)}, mean: {np.mean(resampled_audio)}")
                        yield (48000, resampled_audio), self.audio_chat.messages
                    else:
                        yield None, self.audio_chat.messages
                    
                except Exception as e:
                    self._log_error("Error in handle_text_input", e)
                    yield None, self.messages.value
            
            # Attach components to handlers
            audio_input.stream(
                fn=handle_audio_stream,
                inputs=[audio_input, user_id],
                outputs=[audio_output, self.messages],
                show_progress=False,
                queue=True,  # Enable queueing for async
                batch=False,  # Process each audio chunk immediately
                max_batch_size=1  # Process one chunk at a time
            )
            
            text_input.submit(
                fn=handle_text_input,
                inputs=[text_input, user_id],
                outputs=[audio_output, self.messages],
                show_progress=False,
                queue=True
            )
            
            # Queue for processing
            demo.queue()  # Ensure sequential processing
            
            return demo

    async def cleanup(self):
        """Clean up resources when shutting down."""
        try:
            # Stop any ongoing audio sessions
            if self.current_user_id in self.audio_chat.human_participants:
                participant = self.audio_chat.human_participants[self.current_user_id]
                participant["active"] = False
                participant["stream"] = None
            
            # Clean up audio chat resources
            if hasattr(self.audio_chat, '_tasks'):
                for task in self.audio_chat._tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*self.audio_chat._tasks, return_exceptions=True)
            
            print("Resources cleaned up successfully")
            
        except Exception as e:
            self._log_error("Error during cleanup", e)
    
    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        try:
            demo = self.create_interface()
            print("Interface created successfully!")
            
            print("7. Launching web interface...")
            demo.launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=True,
                show_error=True,
                quiet=False,
                **kwargs
            )
            print("Interface launched successfully!")
            
        except Exception as e:
            self._log_error("Error launching interface", e)
            raise
        finally:
            # Ensure cleanup runs on shutdown
            import atexit
            import asyncio
            atexit.register(lambda: asyncio.run(self.cleanup()))

if __name__ == "__main__":
    # Create UI and launch
    ui = GradioUI()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        quiet=False
    )