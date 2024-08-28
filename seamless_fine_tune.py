import torch
from datasets import load_dataset, DatasetDict, Audio
from transformers import SeamlessM4Tv2Model,SeamlessM4TFeatureExtractor
import re

LANGUAGE_ABV = "tr"

LANGUAGE = "Turkish"

TASK = "transcribe"

 
common_voice = DatasetDict()

common_voice["test"] = load_dataset("mozilla-foundation/common_voice_17_0", "tr", split="test")
common_voice["train"] = load_dataset("mozilla-foundation/common_voice_17_0", "tr", split="train+validation")


common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
common_voice = common_voice.remove_columns(["accent", "age",

                                                 "client_id", "down_votes",

                                                 "gender", "locale",

                                                 "segment", "up_votes",

                                                 "path","variant"])

feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/seamless-m4t-v2-large", language="tur", task="transcribe")

from transformers import AutoTokenizer

tokenizer =  AutoTokenizer.from_pretrained("facebook/seamless-m4t-v2-large")



from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large", language="tur", task="transcribe")

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

def remove_special_characters(batch):
    chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\']'
    batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower()
    return batch

common_voice["test"] = common_voice["test"].map(remove_special_characters)
common_voice["train"] = common_voice["train"].map(remove_special_characters)

def replace_hatted_characters(batch):
    batch["sentence"]= re.sub('[â]', 'a', batch["sentence"])
    batch["sentence"] = re.sub('[î]', 'i', batch["sentence"])
    batch["sentence"] = re.sub('[ô]', 'o', batch["sentence"])
    batch["sentence"] = re.sub('[û]', 'u', batch["sentence"])
    return batch

common_voice["test"] = common_voice["test"].map(replace_hatted_characters)
common_voice["train"] = common_voice["train"].map(remove_special_characters)

common_voice  = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=4)



model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
model.generation_config.language = "tur"
model.generation_config.task = "transcribe"

model.generation_config.forced_decoder_ids = None
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
processor=processor,
decoder_start_token_id=model.config.decoder_start_token_id,)

for param in model.speech_encoder.parameters():
    param.requires_grad = False

model.config.use_cache = False


from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./seamless_turkish", 
    group_by_length=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1, 
    per_device_eval_batch_size=1,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=1000,
    eval_steps=1000,
    logging_steps=1000,
    fp16=True,
    learning_rate=1e-5,
    warmup_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    lr_scheduler_type="linear",
    push_to_hub=True,
    weight_decay=0.005
)
from transformers import Trainer

trainer = Trainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
)

trainer.train()
trainer.push_to_hub()




