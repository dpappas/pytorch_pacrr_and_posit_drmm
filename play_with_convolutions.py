
import torch

emb_size    = 100
ngram       = 3
input       = torch.autograd.Variable(torch.randn(20, 1, 50, emb_size))
m           = torch.nn.Conv2d(1, emb_size, kernel_size=(ngram, emb_size), stride=(1, 1), padding=(ngram-1, 0))
output      = m(input).transpose(1, 3)[:,:,:input.size(-2),:]
print(input.size())
print(output.size())

print(20 * '-')

emb_size    = 100
input       = input.squeeze(1).transpose(-1,-2)
print(input.size())
m           = torch.nn.Conv1d(emb_size, emb_size, ngram, padding=ngram-1, bias=True)
output      = m(input)[:,:,:input.size(-1)]
print(output.size())




