import numpy as np
import soundfile as sf
from fastrtc import get_tts_model, KokoroTTSOptions

def test_voice(name, options, text="Hello, this is a test of the voice settings."):
    """Test a voice configuration and save the output."""
    print(f"\nTesting voice: {name}")
    print(f"Settings: speed={options.speed}, lang={options.lang}")
    
    try:
        # Get TTS model
        tts_model = get_tts_model("kokoro")
        
        # Generate speech
        sample_rate, audio_data = tts_model.tts(text, options=options)
        
        # Ensure audio is in float32 format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize audio
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
            
        # Save audio file
        filename = f"test_voice_{name.lower()}.wav"
        sf.write(filename, audio_data, sample_rate)
        print(f"✓ Successfully saved to {filename}")
        print(f"Audio stats - min: {np.min(audio_data):.3f}, max: {np.max(audio_data):.3f}, mean: {np.mean(audio_data):.3f}")
        
    except Exception as e:
        print(f"✗ Error testing voice {name}: {str(e)}")

def main():
    # Test extreme speed variations
    print("\nTesting speed variations...")
    speeds = [
        ("very_slow", KokoroTTSOptions(speed=0.5, lang="en-us")),
        ("slow", KokoroTTSOptions(speed=0.75, lang="en-us")),
        ("normal", KokoroTTSOptions(speed=1.0, lang="en-us")),
        ("fast", KokoroTTSOptions(speed=1.5, lang="en-us")),
        ("very_fast", KokoroTTSOptions(speed=2.0, lang="en-us")),
    ]
    for name, options in speeds:
        test_voice(name, options)
    
    # Test different languages/accents
    print("\nTesting different languages...")
    languages = [
        ("us_english", KokoroTTSOptions(speed=1.0, lang="en-us")),
        ("british", KokoroTTSOptions(speed=1.0, lang="en-gb")),
        ("australian", KokoroTTSOptions(speed=1.0, lang="en-au")),
    ]
    for name, options in languages:
        test_voice(name, options)

if __name__ == "__main__":
    main()
