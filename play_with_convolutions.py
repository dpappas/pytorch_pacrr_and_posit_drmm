
import torch

emb_size    = 100
ngram       = 3
input       = torch.autograd.Variable(torch.randn(20, 1, 50, emb_size))
m           = torch.nn.Conv2d(1, emb_size, kernel_size=(ngram, emb_size), stride=(1, 1), padding=(ngram-1, 0))
output      = m(input)
print(input.size())
print(output.size())

print(20 * '-')

emb_size    = 100
input       = input.squeeze(1)
print(input.size())
m           = torch.nn.Conv1d(1, emb_size, ngram, padding=ngram-1, bias=True)
output      = m(input)
print(output.size())




