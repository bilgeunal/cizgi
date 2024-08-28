from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import librosa
import jax.numpy as jnp

# Initialize the model and the processor
model_name = "tgrhn/wav2vec2-turkish-300m-8"
model = Wav2Vec2ForCTC.from_pretrained(model_name, torch_dtype=torch.float32).to("cuda")
processor = Wav2Vec2Processor.from_pretrained(model_name)

def load_mp3(mp3_file_path, sr=16000):
    # Load the MP3 file with librosa
    audio, _ = librosa.load(mp3_file_path, sr=sr)
    return audio


def process_audio_chunks(audio, model, processor, sr=16000, device='cuda'):


        audio_inputs = processor(audio, return_tensors="pt", sampling_rate=sr).input_values.to("cuda")

        # Perform inference with the model
        with torch.no_grad():
            outputs = model(audio_inputs, output_hidden_states=True)
            print(outputs.keys())
            logits = outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1)
        last_hidden_states = outputs.hidden_states
        
        
        translated_text_from_audio = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        for i, state in enumerate(last_hidden_states):
            print(f"Shape of hidden state {i}: {state.shape}")
        return translated_text_from_audio


mp3_file_path = "/home/han/Desktop/seamless/clip.mp3.mp3"

# Load and process the audio file
audio = load_mp3(mp3_file_path, sr=16000)

# Process all chunks
translated_text_from_audio = process_audio_chunks(audio, model, processor, device='cuda')
print(translated_text_from_audio)
