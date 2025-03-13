# Audio Group Chat

A real-time audio group chat implementation enabling voice and text communication between humans and AI agents. This project combines WebRTC, speech-to-text, text-to-speech, and LLM capabilities to create interactive conversations with AI agents.

## Features

- Real-time audio communication using WebRTC
- Multiple AI agents with distinct voices and personalities
- Text-to-Speech (TTS) with customizable voice options
- Speech-to-Text (STT) for human voice input
- Round-robin speaker selection for balanced conversations
- Gradio-based web interface for easy interaction
- Support for both voice and text channels

## Prerequisites

- Python 3.8+
- Node.js (for frontend components)
- Ollama (for local LLM support)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd AudioGroupChat
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Configure Ollama settings in `main_app.py`:
```python
config_list = [{
    "model": "gemma3:1b",  # or other supported models
    "base_url": "http://localhost:11434/v1",
    "price": [0.00, 0.00],
}]
```

2. (Optional) Set up Twilio TURN server credentials for improved WebRTC connectivity:
```bash
export TWILIO_ACCOUNT_SID=your_account_sid
export TWILIO_AUTH_TOKEN=your_auth_token
```

## Usage

1. Start the application:
```bash
python main_app.py
```

2. Open the provided Gradio interface URL in your browser (typically http://localhost:7860)

3. Start a conversation by:
   - Speaking into your microphone
   - Typing text messages
   - Using the provided UI controls

## Project Structure

- `main_app.py`: Main application entry point
- `audio_groupchat.py`: Core audio group chat implementation
- `gradio_ui.py`: Gradio web interface components
- `test_group_chat.py`: Test cases and examples

## Voice Configuration

The system supports multiple voice options for AI agents:
- Energetic (fast, US English)
- Calm (slower, US English)
- British (UK English)
- Authoritative (moderate speed, US English)
- Default (standard US English)

## API Documentation

### AudioGroupChat Class

```python
class AudioGroupChat(GroupChat):
    def __init__(self, agents=None, messages=None, max_round=10,
                 speaker_selection_method="round_robin",
                 allow_repeat_speaker=False)
```

Key methods:
- `initialize()`: Set up audio processing components
- `add_human_participant(user_id)`: Add a human participant
- `start_audio_session(user_id)`: Start an audio session

### GradioUI Class

```python
class GradioUI:
    def __init__(self, audio_chat: AudioGroupChat)
    def create_interface(self) -> gr.Blocks
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Autogen](https://github.com/microsoft/autogen)
- Uses [FastRTC](https://github.com/yourusername/fastrtc) for WebRTC functionality
- Powered by [Gradio](https://gradio.app/) for the web interface