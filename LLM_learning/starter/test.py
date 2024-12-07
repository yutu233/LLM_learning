import tiktoken
from GPT.multihead_attention import MultiHeadAttention
text = "Akwirw ier"
tokenizer = tiktoken.get_encoding("gpt2")
integers = tokenizer.encode(text)
strings = tokenizer.decode(integers)

print(strings)

d_in, d_out = 768, 768
num_heads = 12
context_length = 1024
dropout = 0.0

mha = MultiHeadAttention(d_in, d_out, context_length, dropout, num_heads)

print(sum(p.numel() for p in mha.parameters() if p.requires_grad))