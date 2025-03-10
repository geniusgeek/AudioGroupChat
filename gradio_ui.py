import asyncio
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
        # If new_messages is a list of message dicts, use it directly
        if isinstance(new_messages, list) and all(isinstance(msg, dict) for msg in new_messages):
            return new_messages
            
        # If new_messages is a list of tuples, convert to dict format
        if isinstance(new_messages, list) and all(isinstance(msg, (list, tuple)) for msg in new_messages):
            converted_messages = []
            for msg in new_messages:
                if len(msg) == 2:
                    user_msg, assistant_msg = msg
                    if user_msg:
                        converted_messages.append({"role": "user", "content": user_msg})
                    if assistant_msg:
                        converted_messages.append({"role": "assistant", "content": assistant_msg})
            return converted_messages
        
        # If it's a single message dict, append it
        if isinstance(new_messages, dict):
            role = new_messages.get("role", "")
            content = new_messages.get("content", "")
            if role and content:
                prev_messages.append(new_messages)
                
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
                    # Audio output
                    audio_output = gr.Audio(
                        label="AI Response",
                        autoplay=True,
                        show_label=True,
                        type="numpy",
                        interactive=False,
                        elem_id="audio-output",
                        format="wav",
                     )
            
            # Configure stream components
            self.stream.additional_inputs = [user_id]
            self.stream.additional_outputs = [self.messages]
            self.stream.additional_outputs_handler = self._update_messages
            
            # Get the ReplyOnPause handler from the stream
            if not self.stream.event_handler:
                raise RuntimeError("Stream event_handler not initialized")
            
            # Create a wrapper function to handle the audio stream
            async def handle_audio_stream(audio_data, user):
                try:
                    # Check if we have valid audio data
                    if audio_data is None:
                        print("No audio data received")
                        return None, self.messages.value
                        
                    # Print debug info
                    print(f"Received audio input: {type(audio_data)}")
                    if isinstance(audio_data, tuple) and len(audio_data) == 2:
                        sr, data = audio_data
                        print(f"Processing audio: sr={sr}, shape={data.shape}")
                    
                    # Process audio through AudioGroupChat
                    response = await self.audio_chat._handle_audio_input(audio_data, user)
                    
                    # Update chat messages from AudioGroupChat history
                    messages = []
                    
                    # Check voice queue first
                    audio_response = None
                    try:
                        while True:  # Process all available voice messages
                            voice_msg = self.audio_chat.voice_queue.get_nowait()
                            if isinstance(voice_msg, dict) and voice_msg.get("type") == "chat":
                                text = voice_msg.get("text")
                                sender = voice_msg.get("sender")
                                if text:
                                    # Add message in dict format
                                    msg = {
                                        "role": "user" if sender == user else "assistant",
                                        "content": text,
                                        "name": sender
                                    }
                                    messages.append(msg)
                                    # Convert to speech if it's an assistant message
                                    if sender != user:
                                        audio_response = await self.audio_chat.text_to_speech(text)
                    except asyncio.QueueEmpty:
                        pass
                        
                    # Check text queue
                    try:
                        while True:  # Process all available text messages
                            text_msg = self.audio_chat.text_queue.get_nowait()
                            if isinstance(text_msg, dict) and text_msg.get("type") == "chat":
                                text = text_msg.get("text")
                                sender = text_msg.get("sender")
                                if text:
                                    # Add message in dict format
                                    msg = {
                                        "role": "user" if sender == user else "assistant",
                                        "content": text,
                                        "name": sender
                                    }
                                    messages.append(msg)
                    except asyncio.QueueEmpty:
                        pass
                    
                    return audio_response, messages
                except Exception as e:
                    self._log_error("Error in handle_audio_stream", e)
                    return None, self.messages.value

            # Create a wrapper function to handle text input
            async def handle_text_input(text, user):
                if not text or not text.strip():
                    return None, self.messages.value
                    
                # Create chat message
                message = {
                    "role": "user",
                    "content": text,
                    "name": user
                }
                
                # Add message to chat history
                self.audio_chat.messages.append(message)
                
                # Process through AudioGroupChat
                await self.audio_chat._handle_chat_message(user, {
                    "type": "chat",
                    "text": text,
                    "sender": user,
                    "channel": "text"
                })
                
                # Return current chat history
                return None, self.audio_chat.messages
            
            # Attach components to handlers
            audio_input.stream(
                fn=handle_audio_stream,
                inputs=[audio_input, user_id],
                outputs=[audio_output, self.messages],
                show_progress=False,
                queue=True,  # Enable queueing for async
                batch=False   # Process each audio chunk immediately
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