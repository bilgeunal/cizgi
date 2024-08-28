from datasets import load_dataset, Audio
from transformers import SeamlessM4Tv2Model, AutoProcessor, SeamlessM4TProcessor, SeamlessM4TFeatureExtractor
import torch
import re
from evaluate import load

dataset = load_dataset("mozilla-foundation/common_voice_17_0","tr",
                                 split="test+validation", 

                                 trust_remote_code=True, 

                                 keep_in_memory=True)


dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

dataset = dataset.remove_columns(["accent", "age", "client_id", 
                                  "down_votes", "gender", "locale", 
                                  "segment", "up_votes", "path", "variant"])

chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\']'
def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower()
    return batch

dataset = dataset.map(remove_special_characters)

def replace_hatted_characters(batch):
    batch["sentence"] = re.sub('[â]', 'a', batch["sentence"])
    batch["sentence"] = re.sub('[î]', 'i', batch["sentence"])
    batch["sentence"] = re.sub('[ô]', 'o', batch["sentence"])
    batch["sentence"] = re.sub('[û]', 'u', batch["sentence"])
    return batch

dataset = dataset.map(replace_hatted_characters)
from transformers import AutoTokenizer

tokenizer =  AutoTokenizer.from_pretrained("facebook/seamless-m4t-v2-large")
processor = SeamlessM4TProcessor.from_pretrained("facebook/seamless-m4t-v2-large", language="tur", task="transcribe")
model = SeamlessM4Tv2Model.from_pretrained("/home/han/Desktop/seamless/seamless_turkish/checkpoint-139215").to("cuda")

def map_to_pred(batch):
    audio_tensor = torch.tensor(batch["audio"]["array"] ).float().unsqueeze(0).to("cuda")
        
    audio_inputs = processor(audios=audio_tensor.cpu(), return_tensors="pt", sampling_rate=16000).to("cuda")
        
    output_tokens = model.generate(**audio_inputs, tgt_lang="tur", generate_speech=False)
        
    batch["prediction"]  = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
    
    batch["sentence"] = batch["sentence"].lower()
    batch["prediction"] = re.sub(chars_to_remove_regex, '', batch["prediction"]).lower()
    
    return batch


result = dataset.map(map_to_pred)
wer_metric = load("wer")
references = result["sentence"]
predictions = result["prediction"]
wer_score = 100*wer_metric.compute(predictions=predictions, references=references)

print(f"Word Error Rate (WER): {wer_score}")
