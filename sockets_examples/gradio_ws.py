import gradio as gr
import websockets
import asyncio
import json
import numpy as np
from fastrtc import get_stt_model, get_tts_model, ReplyOnPause, AdditionalOutputs
from fastrtc.stream import Stream as FastRTCStream
from aiortc.mediastreams import AudioStreamTrack as FastRTCAudioStreamTrack
from fastrtc.tracks import AudioCallback
from audio_groupchat import AudioStreamHandler

class GradioUI:
    def __init__(self, ws_url="ws://localhost:8000/ws/audio"):
        self.ws_url = ws_url
        self._setup_pipelines()
        self.current_user_id = "user_123"
        # Initialize stream handler with WebRTC standard parameters
        self.stream_handler = AudioStreamHandler()
        self.stream_handler.sample_rate = 48000  # Standard WebRTC sample rate
        self.stream_handler.frame_size = 960  # 20ms frame at 48kHz
        self.stream_handler.num_channels = 1  # Mono audio
        self.stream_handler.dtype = np.int16  # Standard audio format
        self.stream_handler.expected_layout = "mono"
        
        # Initialize FastRTC stream with configured handler
        self.stream = FastRTCStream(
            handler=self.stream_handler,
            mode="send-receive",
            modality="audio"
        )

    def _setup_pipelines(self):
        self.stt_model = get_stt_model()  # Use default STT model
        self.tts_model = get_tts_model("kokoro")

    async def audio_input_handler(self, audio_data, user_id):
        try:
            # Ensure audio data is in correct format for STT
            if isinstance(audio_data, tuple):
                sample_rate, audio_array = audio_data
                if len(audio_array.shape) > 1:
                    # Convert stereo to mono by averaging channels
                    audio_array = audio_array.mean(axis=1).astype(np.float32)
                # Convert to numpy array if needed
                if not isinstance(audio_array, np.ndarray):
                    audio_array = np.array(audio_array)
                # Normalize if needed
                if audio_array.dtype == np.int16:
                    audio_array = audio_array.astype(np.float32) / 32768.0
                    
                # Process audio through stream handler
                self.stream_handler.receive(audio_array)
                processed_frame = await self.stream_handler.next_frame()
                
                # Standardize audio format for STT following FastRTC patterns
                if processed_frame is not None:
                    if isinstance(processed_frame, tuple):
                        text = self.stt_model.stt(processed_frame[1])
                    else:
                        text = self.stt_model.stt(processed_frame)
                else:
                    text = ""
            else:
                print("Invalid audio data format")
                return
        except Exception as e:
            print(f"STT failed: {e}")
            text = ""
            
        try:
            async with websockets.connect(f"{self.ws_url}?user_id={user_id}") as ws:
                await ws.send(json.dumps({
                    "type": "human_input",
                    "text": text,
                    "user_id": user_id
                }))
        except Exception as e:
            print(f"WebSocket connection error: {e}")

    async def audio_output_handler(self):
        while True:
            try:
                print(f"Connecting to WebSocket at {self.ws_url}?user_id={self.current_user_id}")
                async with websockets.connect(f"{self.ws_url}?user_id={self.current_user_id}") as ws:       
                    print("WebSocket connection established")
                    while True:
                        try:
                            msg = await ws.recv()
                            data = json.loads(msg)
                            print(f"Received message type: {data.get('type')}")
                            if data["type"] == "agent_response":
                                try:
                                    print(f"Converting text to speech: {data['text'][:50]}...")
                                    audio_data = self.tts_model.tts(data["text"])
                                    if audio_data is None or len(audio_data) == 0:
                                        print("TTS returned empty audio data")
                                        audio_data = np.zeros(24000, dtype=np.int16)
                                except Exception as e:
                                    print(f"TTS failed: {e}")
                                    import traceback
                                    traceback.print_exc()
                                    audio_data = np.zeros(24000, dtype=np.int16)
                                
                                # Process audio through stream handler
                                self.stream_handler.receive(audio_data)
                                processed_frame = await self.stream_handler.next_frame()
                                
                                # Ensure audio data is in the correct format for Gradio
                                if processed_frame.dtype != np.int16:
                                    processed_frame = (processed_frame * 32768).astype(np.int16)
                                
                                # Reshape for Gradio audio output
                                if len(processed_frame.shape) == 1:
                                    processed_frame = processed_frame.reshape(-1, 1)
                                
                                print(f"Yielding audio data with shape {processed_frame.shape} and dtype {processed_frame.dtype}")
                                yield (48000, processed_frame)  # Use WebRTC standard sample rate
                        except websockets.exceptions.ConnectionClosed:
                            print("WebSocket connection closed, reconnecting...")
                            break
                        except Exception as e:
                            print(f"WebSocket error: {e}")
                            import traceback
                            traceback.print_exc()
                            await asyncio.sleep(1)
            except Exception as e:
                print(f"WebSocket connection error: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(3)

    def create_interface(self):
        with gr.Blocks(title="Huddle Audio Chat") as demo:
            with gr.Row():
                with gr.Column():
                    user_id = gr.Textbox(label="User ID", value=self.current_user_id)
                    audio_input = gr.Audio(sources=["microphone"], streaming=True)
                with gr.Column():
                    chat_history = gr.Chatbot(label="Conversation History", type="messages")
                    audio_output = gr.Audio(autoplay=True)
                # Initialize audio track with proper configuration
                self.stream.track = AudioCallback(
                    track=FastRTCAudioStreamTrack(),
                    event_handler=self.stream_handler,
                    channel=0
                )
            
            # Set up audio input handling without streaming
            audio_input.change(
                fn=self.audio_input_handler,
                inputs=[audio_input, user_id],
                outputs=[]
            )
            
            # Set up audio output handling without streaming
            demo.load(
                fn=self.audio_output_handler,
                inputs=None,
                outputs=audio_output,
                every=0.1  # Poll every 100ms
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
    # Create UI
    ui = GradioUI()
    
    # Create and launch interface
    demo = ui.create_interface()
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        quiet=False
    )