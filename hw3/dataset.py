from transformers import BertTokenizer
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, raw_dataset, tokenizer=None, max_length=1024) -> None:
        super().__init__()
        self.raw_dataset = raw_dataset['text']
        if not tokenizer:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length=max_length
        self.max_length = max_length

    
    def __len__(self):
        return len(self.raw_dataset)
    
    def __getitem__(self, index):
        item = self.raw_dataset[index]
        tokens = self.tokenizer(item, padding='max_length', truncation=True, return_tensors='pt')
        
        for k in tokens.keys():
            tokens[k] = tokens[k].squeeze()

        mask_index = np.random.randint(self.max_length)
        masked_word = tokens['input_ids'][mask_index].detach().clone()
        tokens['input_ids'][mask_index] = self.tokenizer.mask_token_id
        
        assert(len(tokens['input_ids']) == self.max_length, 'wrong size of tokenized data')
        return tokens, masked_word, mask_index