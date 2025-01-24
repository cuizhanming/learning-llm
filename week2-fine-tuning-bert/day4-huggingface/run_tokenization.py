from transformers import AutoTokenizer,BertTokenizer

token = BertTokenizer.from_pretrained(
    r"../local_models/google-bert/bert-base-chinese/models--google-bert--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")

sents = ["my dog is cute", "Hello, my cat is cute too"]
out = token.batch_encode_plus(
    batch_text_or_text_pairs=sents, # list of sentences
    add_special_tokens=True, # add [CLS], [SEP]
    truncation=True, # truncate to the maximum length the model can accept
    #TODO !!Make all sentences the same length
    max_length=20, # maximum length per sentence
    padding="max_length", # pad to the maximum length for all sentences
    return_tensors=None, # tf for TensorFlow, pt for PyTorch, np for NumPy, default is Python lists
    return_attention_mask=True, # return attention mask
    return_token_type_ids=True, # return token type IDs
    return_special_tokens_mask=True, # return special tokens mask
    return_length=True, # return lengths of the encoded inputs
) # return PyTorch tensors

for k,v in out.items():
    print(k, ":", v)
'''
input_ids : [[101, 8422, 13030, 8310, 12322, 8154, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 100, 117, 8422, 10165, 8310, 12322, 8154, 11928, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
token_type_ids : [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
special_tokens_mask : [[1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
length : [7, 10]
attention_mask : [[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
'''

"""
`input_ids` are the encoded input sentences
`special_tokens_mask` are the special tokens, like [CLS], [SEP], [PAD]
"""
print(token.decode(out["input_ids"][0]))
print(token.decode(out["input_ids"][1]))
'''
[CLS] my dog is cute [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
[CLS] [UNK], my cat is cute too [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
'''