import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import Dataset, DataLoader


os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

def remove_unwanted_spaces(text):
    text = text.strip()
    text = ' '.join(text.split())
    return text

class TextDataset(Dataset):
    def __init__(self, encoded_texts):
        self.input_ids = encoded_texts["input_ids"]
        self.attention_mask = encoded_texts["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.input_ids[idx].detach().clone(),
        }
        return item

def train():
    xls = pd.ExcelFile('../Data/小红书/小红书上海POI.xlsx')
    sheet_to_df_map = {}

    for sheet_name in xls.sheet_names:
        sheet_to_df_map[sheet_name] = xls.parse(sheet_name)

    contents = []
    for region, sheet in sheet_to_df_map.items():
        contents.extend(sheet['POI描述'].tolist())
    contents = [remove_unwanted_spaces(content) for content in contents]

    tokenizer = AutoTokenizer.from_pretrained("fnlp/moss-moon-003-sft", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    def chunk_and_overlap(texts, tokenizer, max_length, overlap):
        chunked_texts = []

        # Tokenize all texts at once
        encoded_dicts = tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=None,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
            
        step = max_length - overlap

        # Iterate through each encoded text
        for encoded_dict in encoded_dicts["input_ids"]:
            for i in range(0, (encoded_dict != tokenizer.pad_token_id).sum().item(), step):
                chunk = encoded_dict[i:i+max_length]

                # If the chunk is shorter than max_length, pad it
                if len(chunk) < max_length:
                    padding = max_length - len(chunk)
                    chunk = F.pad(chunk, (0, padding), value=tokenizer.pad_token_id)

                chunked_texts.append(chunk)
        # Combine all chunks and create corresponding attention masks
        input_ids = torch.stack(chunked_texts)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    chunked_encoded_texts = chunk_and_overlap(contents, tokenizer, max_length=128, overlap=32)


    # Split the dataset into training and validation sets
    train_texts, valid_texts, train_masks, valid_masks = train_test_split(
        chunked_encoded_texts["input_ids"], 
        chunked_encoded_texts["attention_mask"], 
        test_size=.2, 
        random_state=42)
    train_dataset = TextDataset({"input_ids": train_texts, "attention_mask": train_masks})
    valid_dataset = TextDataset({"input_ids": valid_texts, "attention_mask": valid_masks})


    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        eval_steps=2_000,
        logging_steps=2_000,  
        num_train_epochs=3,            
        gradient_accumulation_steps=8,
        warmup_steps=500,                
        weight_decay=0.01,              
        learning_rate=1e-5,
        fp16=True,
        logging_dir="./logs",
    )


    # Initialize the model
    model = AutoModelForCausalLM.from_pretrained("fnlp/moss-moon-003-sft", trust_remote_code=True, use_cache=False)
    print('model loaded')

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer, 
        args=training_args,             
        train_dataset=train_dataset,     
        eval_dataset=valid_dataset,     
        data_collator=data_collator,
    )
    print('trainer initialized')


    # Train the model
    print('start training')
    trainer.train()

if __name__ == '__main__':
    train()