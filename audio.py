from transformers import AutoProcessor, SeamlessM4Tv2Model
import torchaudio
import librosa
import torch
def load_mp3(mp3_file_path):
    # Load the MP3 file with librosa
    audio, sample_rate = librosa.load(mp3_file_path, sr=16000)  # Load audio with 16 kHz sample rate
    return audio

processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
mp3_file_path = "/home/han/Desktop/seamless/audio/audio.mp3"
audio_array = load_mp3(mp3_file_path)
audio_tensor = torch.tensor(audio_array).float().unsqueeze(0)
# now, process it
audio_inputs = processor(audios=audio_tensor, return_tensors="pt")
output_tokens = model.generate(**audio_inputs, tgt_lang="eng", generate_speech=False)
translated_text_from_audio = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
print(translated_text_from_audio)
