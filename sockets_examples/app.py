import asyncio
import os
import json
import traceback
from autogen.agentchat import AssistantAgent, GroupChatManager, UserProxyAgent
from fastapi import FastAPI, WebSocket, status
from fastapi.responses import HTMLResponse
from fastrtc import Stream, get_twilio_turn_credentials
import gradio as gr
from audio_groupchat import AudioGroupChat
from gradio_ui import GradioUI
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import parse_qs
from starlette.websockets import WebSocketState
from aiortc import RTCSessionDescription, RTCIceCandidate
import numpy as np

# Define system prompts for agents
assistant1_system_prompt = """
You are Agent1, an AI assistant specialized in audio group conversations.
Your role is to engage in meaningful dialogue, provide helpful information, and collaborate with other agents.
You should respond in a conversational, friendly tone and keep responses concise for and sound natural.
"""

assistant2_system_prompt = """
You are Agent2, an AI assistant specialized in audio group conversations.
Your expertise is in providing detailed explanations and alternative perspectives to complement Agent1.
You should respond in a conversational, friendly tone and keep responses concise and sound natural.
"""

# Configure LLM settings with proper parameters
llm_config = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "model": "gpt-4",
    "temperature": 0.7,
    "seed": 42,  # For reproducibility
    "config_list": [{
        "model": "gpt-4",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "timeout": 120  # Using timeout instead of request_timeout
    }],
}

# Create properly configured agents with system prompts
agent1 = AssistantAgent(
    name="Agent1", 
    system_message=assistant1_system_prompt,
    llm_config=llm_config
)

agent2 = AssistantAgent(
    name="Agent2", 
    system_message=assistant2_system_prompt,
    llm_config=llm_config
)

# Create a UserProxyAgent for the human user with proper configuration
human_system_prompt = """
You represent a human user in this conversation. Your inputs will come from actual human speech
that has been transcribed to text. You will relay these messages to the AI assistants.
"""

human_agent = UserProxyAgent(
    name="user_123", 
    system_message=human_system_prompt,
    human_input_mode="NEVER", 
    code_execution_config={"use_docker": False}
)

# Create audio-enabled group chat with manager
audio_chat = AudioGroupChat(
    messages=[],
    agents=[agent1, agent2, human_agent],
    max_round=5,
    speaker_selection_method="round_robin",  # Ensure all agents get a turn
    allow_repeat_speaker=True  # Allow agents to speak multiple times
)

# Create and set the manager (chat will start via audio_session callback)
manager = GroupChatManager(
    groupchat=audio_chat,
    llm_config=llm_config,
)
audio_chat.manager = manager

async def main():
    app = FastAPI()
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Define WebSocket route for audio handling
    @app.websocket("/ws/audio")
    async def handle_audio_stream(websocket: WebSocket):
        try:
            # Get query parameters directly from the WebSocket object
            query_params = dict(websocket.query_params)
            user_id = query_params.get('user_id', ['unknown'])[0]
            
            if user_id == 'unknown':
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return
            
            # Add human participant if not exists
            if user_id not in audio_chat.human_participants:
                audio_chat.add_human_participant(user_id)
                await audio_chat.start_audio_session(user_id)
            
            await websocket.accept()
            
            pc = audio_chat.human_participants[user_id]['pc']
            
            @pc.onicecandidate
            async def on_ice_candidate(candidate):
                if candidate:
                    await websocket.send_json({
                        'type': 'ice_candidate',
                        'candidate': candidate.to_json(),
                        'user_id': user_id
                    })
            
            # Replace async for with a while loop using receive()
            while True:
                try:
                    message = await websocket.receive()
                    
                    # Check if this is a disconnect message
                    if message["type"] == "websocket.disconnect":
                        break
                        
                    # Handle data messages
                    if message["type"] == "websocket.receive":
                        if "bytes" in message:
                            # Handle binary data with correct user ID
                            track_id = audio_chat.human_participants[user_id]['audio_track'].id
                            await audio_chat._handle_audio_input((track_id, np.frombuffer(message["bytes"], dtype=np.int16)))
                            await audio_chat.broadcast_audio(message["bytes"], user_id)
                        elif "text" in message:
                            # Handle text data
                            data = json.loads(message["text"])
                            print(f"Processing message type: {data.get('type')}")
                            
                            if data['type'] == 'offer':
                                pc = audio_chat.human_participants[data['user_id']]['pc']
                                await pc.setRemoteDescription(RTCSessionDescription(
                                    sdp=data['offer']['sdp'],
                                    type=data['offer']['type']
                                ))
                                answer = await pc.createAnswer()
                                await pc.setLocalDescription(answer)
                                await websocket.send_json({
                                    'type': 'answer',
                                    'answer': {
                                        'sdp': answer.sdp,
                                        'type': answer.type
                                    }
                                })
                            
                            elif data['type'] == 'ice_candidate':
                                pc = audio_chat.human_participants[data['user_id']]['pc']
                                candidate = RTCIceCandidate(
                                    candidate=data['candidate']['candidate'],
                                    sdpMid=data['candidate']['sdpMid'],
                                    sdpMLineIndex=data['candidate']['sdpMLineIndex']
                                )
                                await pc.addIceCandidate(candidate)
                            
                            elif data['type'] == 'human_input':
                                response = await audio_chat.send_message({
                                    "content": data['text'],
                                    "sender": data['user_id'],
                                    "recipients": [a.name for a in audio_chat.agents],
                                    "channel": "both"
                                })
                                
                                if response:
                                    await websocket.send_json({
                                        "type": "agent_response",
                                        "text": response
                                    })
                except Exception as e:
                    print(f"WebSocket error: {e}")
                    traceback.print_exc()
                    await asyncio.sleep(1)
        except Exception as e:
            print(f"WebSocket connection error: {e}")
            traceback.print_exc()
            await asyncio.sleep(3)
        finally:
            if websocket.client_state != WebSocketState.DISCONNECTED:
                await websocket.close()
    
    # Add websocket route for text messages
    @app.websocket("/ui/text")
    async def handle_text_messages(websocket: WebSocket):
        await websocket.accept()
        async for message in websocket:
            try:
                data = json.loads(message)
                if data['type'] == 'text_message':
                    # Determine if any recipients are human
                    has_human_recipients = any(
                        recipient in audio_chat.human_participants
                        for recipient in data['recipients']
                    )
                    
                    await audio_chat.send_message({
                        "content": data['text'],
                        "sender": data['sender'],
                        "recipients": data['recipients'],
                        "channel": "both" if has_human_recipients else "text"
                    })
            except Exception as e:
                print(f"Text message handling error: {e}")
                traceback.print_exc()
     
    
    # Add human participant with audio session
    human_user_id = "user_123"
    audio_chat.add_human_participant(human_user_id)
    
    # Start audio session and initialize chat
    await audio_chat.start_audio_session(human_user_id)
    
    # Setup negotiation for the human participant
    if hasattr(audio_chat, '_setup_negotiation'):
        await audio_chat._setup_negotiation(human_user_id)
    
    # Launch Gradio UI concurrently
    ui = GradioUI(ws_url="ws://localhost:8000/ws/audio")
    
    @app.get("/")
    async def read_root():
        return {"message": "Huddle Audio Chat API"}
    
    # Create and configure Gradio interface
    demo = ui.create_interface()
    
    # Mount Gradio app with proper path
    app = gr.mount_gradio_app(app, demo, path="/ui")
    
    # Start the server
    import uvicorn
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False  # Disable auto-reload to prevent WebSocket issues
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())