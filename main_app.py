import asyncio
import os
import json
import traceback
from autogen.agentchat import AssistantAgent, GroupChatManager, UserProxyAgent
from audio_groupchat import AudioGroupChat
from gradio_ui import GradioUI

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

# Configure Ollama settings
config_list = [{
    "model": "llama3.2", #    "model": "deepseek-r1:1.5b",
    "base_url": "http://localhost:11434/v1",
    "price": [0.00, 0.00],
}]

# Configure LLM settings
llm_config = {
    "temperature": 0.7,
    "config_list": config_list,
}

# Create properly configured agents with system prompts
agent1 = AssistantAgent(
    name="Agent1", 
    system_message=assistant1_system_prompt,
    llm_config=llm_config,
    code_execution_config=False  # Disable code execution
)

agent2 = AssistantAgent(
    name="Agent2", 
    system_message=assistant2_system_prompt,
    llm_config=llm_config,
    code_execution_config=False  # Disable code execution
)

# Create a UserProxyAgent for the human user with proper configuration
human_system_prompt = """
You represent a human user in this conversation. Your inputs will come from actual human speech
that has been transcribed to text. You will relay these messages to the AI assistants.
Do not generate responses on your own - only relay the actual transcribed human speech.
"""

human_agent = UserProxyAgent(
    name="user_123", 
    system_message=human_system_prompt,
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False}
)

print("\nInitializing Audio Group Chat...")
print("1. Creating agents:", ["Agent1", "Agent2", "user_123"])

# Create audio-enabled group chat with manager
audio_chat = AudioGroupChat(
    messages=[],
    agents=[agent1, agent2, human_agent],
    max_round=5,
    speaker_selection_method="round_robin",  # Ensure all agents get a turn
    allow_repeat_speaker=True  # Allow agents to speak multiple times
)

print("2. Setting up GroupChatManager...")
manager = GroupChatManager(
    groupchat=audio_chat,
    llm_config=llm_config,
)
audio_chat.manager = manager

print("3. Verifying setup:")
print("   - Agents configured:", [agent.name for agent in audio_chat.agents])
print("   - Manager attached:", audio_chat.manager is not None)
 

async def main():
    try:
        print("\n4. Creating Gradio interface...")
        # Create and launch the Gradio UI with our audio chat instance
        ui = GradioUI(audio_chat=audio_chat)
        demo = ui.create_interface()
        
        print("5. Initializing audio chat components...")
        # Initialize audio chat
        await audio_chat.initialize()
        
        print("6. Starting audio session...")
        # Audio session and participant addition will be handled by Gradio UI
        
        print("7. Launching web interface...")
        # Launch with specific configurations
        await demo.launch(
            server_name="0.0.0.0",  # Listen on all network interfaces
            server_port=7860,       # Default Gradio port
            share=True,             # Create a public link
            show_error=True,        # Show any errors that occur
            quiet=False             # Show all output
        )
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # Create and run event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()