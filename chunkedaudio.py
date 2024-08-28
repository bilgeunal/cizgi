from transformers import AutoProcessor, SeamlessM4Tv2Model
import librosa
import torch
import time

def load_mp3(mp3_file_path, sr=16000):
    # Load the MP3 file with librosa
    audio, _ = librosa.load(mp3_file_path, sr=sr)
    return audio

def chunk_audio(audio, chunk_size_sec, sr=16000):
    chunk_size_samples = int(chunk_size_sec * sr)
    return [audio[i:i + chunk_size_samples] for i in range(0, len(audio), chunk_size_samples)]

def process_audio_chunks(chunks, processor, model, tgt_lang="tur", sr=16000, device='cpu'):
    full_text = []
    for chunk in chunks:
        # Convert to tensor and move to GPU
        audio_tensor = torch.tensor(chunk).float().unsqueeze(0).to(device)
        
        # Move tensor back to CPU for processing with the processor
        audio_inputs = processor(audios=audio_tensor.cpu(), return_tensors="pt", sampling_rate=sr).to(device)
        
        # Generate output tokens using the model on GPU
        output_tokens = model.generate(**audio_inputs, tgt_lang=tgt_lang, generate_speech=False)
        
        # Decode the output tokens to text
        text_from_chunk = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
        full_text.append(text_from_chunk)
    
    return " ".join(full_text)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2Model.from_pretrained("/home/han/Desktop/seamless/seamless_turkish/checkpoint-139215").to(device)
    mp3_file_path = "/home/han/Desktop/seamless/audio_medikal.mp3"
    
    start_time = time.time()

    # Load and process the audio file
    audio = load_mp3(mp3_file_path, sr=16000)
    chunks = chunk_audio(audio, chunk_size_sec=4, sr=16000)  # Chunk into 4-second segments
    
    # Process all chunks
    translated_text_from_audio = process_audio_chunks(chunks, processor, model, device=device)
    print(translated_text_from_audio)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time} seconds")

if __name__ == "__main__":
    main()
