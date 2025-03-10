import asyncio
import numpy as np
from autogen import ConversableAgent, GroupChatManager
from audio_groupchat import AudioGroupChat
import os

# Configure OpenAI
config_list = [
    {
        'model': 'gpt-4',
        'api_key': os.getenv('OPENAI_API_KEY')
    }
]

async def main():
    # Create agents with more specific personalities and responses
    researcher = ConversableAgent(
        name="Researcher",
        system_message="""You are a curious AI safety researcher who asks probing questions and analyzes risks.
        When discussing AI safety, focus on concrete technical challenges and potential solutions.""",
        llm_config={"config_list": config_list, "temperature": 0.7}
    )
    
    expert = ConversableAgent(
        name="Expert",
        system_message="""You are an AI safety expert with deep technical knowledge.
        Provide detailed, technical explanations about AI safety mechanisms and potential failure modes.""",
        llm_config={"config_list": config_list, "temperature": 0.7}
    )
    
    moderator = ConversableAgent(
        name="Moderator",
        system_message="""You are a skilled discussion moderator who keeps conversations productive and on-topic.
        Help synthesize different viewpoints and maintain focus on key technical aspects.""",
        llm_config={"config_list": config_list, "temperature": 0.7}
    )
    
    # Create audio group chat
    group_chat = AudioGroupChat(
        agents=[researcher, expert, moderator],
        messages=[],
        max_round=10,
        speaker_selection_method="round_robin"
    )
    
    # Create chat manager
    manager = GroupChatManager(group_chat)
    group_chat.manager = manager
    
    # Add a human participant
    user_id = "human_user"
    session_id = group_chat.add_human_participant(user_id)
    print(f"Added human participant with session ID: {session_id}")
    
    # Start audio session for human
    await group_chat.start_audio_session(user_id)
    
    # Configure channels
    group_chat.set_channel_config(
        voice_enabled=True,
        text_enabled=True,
        agent_voice_enabled=True
    )
    
    # Test Cases
    
    # Test Case 1: Human Voice Input
    print("\nTest Case 1: Human Voice Input")
    # Create test audio data (sine wave at 440Hz)
    sample_rate = 16000  # Standard sample rate for speech
    duration = 1  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # Hz
    audio_data = np.sin(2 * np.pi * frequency * t)
    # Reshape to match model input requirements (sequence_length,)
    audio_data = audio_data.reshape(-1)
    await group_chat._handle_audio_input(user_id, (sample_rate, audio_data))
    await asyncio.sleep(2)  # Wait for responses
    
    # Test Case 2: Agent-to-agent interaction
    print("\nTest Case 2: Agent Interaction")
    message = {
        "type": "chat",
        "text": "What are the key concerns in AI safety that we should discuss?",
        "sender": user_id,
        "channel": "text"
    }
    await group_chat.voice_queue.put(message)
    await asyncio.sleep(2)  # Wait for responses
    
    # Test Case 3: Simultaneous text and voice
    print("\nTest Case 3: Simultaneous Text and Voice")
    # Send text message
    text_message = {
        "type": "chat",
        "text": "I'll type this message while speaking",
        "sender": user_id,
        "channel": "text"
    }
    voice_message = {
        "type": "chat",
        "text": "And here's what I'm saying out loud",
        "sender": user_id,
        "channel": "voice"
    }
    
    # Send both messages concurrently
    await asyncio.gather(
        group_chat.text_queue.put(text_message),
        group_chat.voice_queue.put(voice_message)
    )
    await asyncio.sleep(2)  # Wait for responses
    
    # Test Case 4: Multiple agents responding
    print("\nTest Case 4: Multiple Agent Response")
    message = {
        "type": "chat",
        "text": "Let's all share our thoughts on AI safety.",
        "sender": user_id,
        "channel": "both"
    }
    await group_chat.voice_queue.put(message)
    await asyncio.sleep(5)  # Wait for multiple responses
    
    print("\nTest complete! Check that:")
    print("1. You heard voice from both human and agents")
    print("2. Text messages appeared in chat")
    print("3. Agents responded to both voice and text")
    print("4. Multiple agents were able to converse")
    print("5. Voice and text channels operated independently")

if __name__ == "__main__":
    asyncio.run(main())
