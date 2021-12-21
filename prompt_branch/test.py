import torch
from transformers import AutoConfig, AutoModelForMaskedLM,AutoTokenizer
# Download model and configuration from huggingface.co and cache.

m = '/home/zy/data2/yyr/codes/ATLOP/pretrain_models/bert-base-cased'
model = AutoModelForMaskedLM.from_pretrained(m)
# Update configuration during loading
model = AutoModelForMaskedLM.from_pretrained(m, output_attentions=True, output_hidden_states=True)
model.config.output_attentions

config = AutoConfig.from_pretrained(m)
# model = AutoModelForMaskedLM.from_pretrained(m, config=config)
tokenizer = AutoTokenizer.from_pretrained(m)
input = tokenizer(['The model is set [MASK] evaluation mode by [MASK] using model.eval', 'To [MASK] train [MASK] line'], padding=True,truncation=True,max_length=512,return_tensors="pt")
a=model(**input)#odict_keys(['logits', 'attentions'])
mask_index = torch.where(input["input_ids"]== tokenizer.mask_token_id)
predict_token_id = [1,2,3,4,5]
prob = a.logits[mask_index][predict_token_id]
