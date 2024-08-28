from transformers import AutoProcessor, SeamlessM4Tv2Model
from datasets import Audio

processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large").to("cuda")
# let's load an audio sample from an Arabic speech corpus
from datasets import load_dataset
dataset = load_dataset("mozilla-foundation/common_voice_17_0", "tr", split="test")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
#dataset = dataset.select(range(2))
audio_sample = next(iter(dataset))["audio"]
label = next(iter(dataset))["sentence"]

# now, process it
audio_inputs = processor(audios=audio_sample["array"], return_tensors="pt", padding=True).to("cuda")



# from audio
output_tokens = model.generate(**audio_inputs, tgt_lang="tur", generate_speech=False)
translated_text_from_audio = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)


#print(output_tokens[0].tolist()[0])
print(translated_text_from_audio)
print(label)
from transformers import SeamlessM4Tv2Model, SeamlessM4Tv2Config

# Initializing a SeamlessM4Tv2 "" style configuration
configuration = SeamlessM4Tv2Config()

# Initializing a model from the "" style configuration
model = SeamlessM4Tv2Model(configuration)

# Accessing the model configuration
configuration = model.config
