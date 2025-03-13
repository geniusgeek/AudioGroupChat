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
You should respond in a conversational tone and keep responses concise for and sound natural.
"""

assistant2_system_prompt = """
You are Agent2, an AI assistant specialized in audio group conversations.
Your expertise is in providing detailed explanations and alternative perspectives to complement Agent1.
You should respond in a conversationa tone and keep responses concise and sound natural.
"""

# Configure Ollama settings
config_list = [{
    "model": "gemma3:1b", #"llama3.2","gemma3:1b", #    "model": "deepseek-r1:1.5b",
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

print("\nInitializing Audio Group Chat...")
print("1. Creating agents:", ["Agent1", "Agent2"])

# Create audio-enabled group chat with manager
audio_chat = AudioGroupChat(
    messages=[],
    agents=[agent1, agent2],
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
        # Initialize audio chat and wait for it to complete before starting Gradio
        try:
            print("\n=== Starting Audio Chat Initialization ===\n")
            await audio_chat.initialize()
            print("\n=== Audio Chat Initialization Complete ===\n")
        except Exception as e:
            print(f"\n=== Audio Chat Initialization Failed ===\n")
            print(f"Error: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        print("6. Starting audio session...")
        # Audio session and participant addition will be handled by Gradio UI
        
        print("7. Launching web interface...")
        # Launch Gradio with minimal console output
        gradio_task = asyncio.create_task(demo.launch(
            server_name="0.0.0.0",  # Listen on all network interfaces
            server_port=7860,       # Default Gradio port
            share=True,             # Create a public link
            show_error=True,        # Show any errors that occur
            quiet=True              # Minimize Gradio output
        ))
        
        # Wait for initialization to complete first
        try:
            await init_task
            print("Audio chat initialization completed successfully")
        except Exception as e:
            print(f"Error during audio chat initialization: {e}")
            raise

        # Now wait for both tasks to complete or handle failure
        try:
            # Wait for either task to complete or fail
            done, pending = await asyncio.wait(
                [gradio_task], 
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Check for exceptions
            for task in done:
                if task.exception():
                    raise task.exception()
                
            # Keep running until interrupted
            await asyncio.Event().wait()
            
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
        except Exception as e:
            print(f"Error in main loop: {e}")
            raise
        finally:
            # Cancel any pending tasks
            for task in [init_task, gradio_task]:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete cancellation
            await asyncio.gather(*[init_task, gradio_task], return_exceptions=True)
            
    except Exception as e:
        print(f"Fatal error in main: {e}")
        raise

if __name__ == "__main__":
    # Create and run event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"Error in event loop: {e}")
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        loop.close()