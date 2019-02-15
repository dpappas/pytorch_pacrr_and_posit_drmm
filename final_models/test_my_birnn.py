
import torch
import  torch.autograd as autograd

birnn               = torch.nn.LSTM(
    input_size      = 30,
    hidden_size     = 30,
    num_layers      = 2,
    bias            = True,
    batch_first     = False,
    dropout         = 0.3,
    bidirectional   = True
)

# h0 : num_layers * num_directions,  batch, hidden_size
h0                  = autograd.Variable(torch.randn(2*2, 1, 30))
# c0 : num_layers * num_directions,  batch,  hidden_size
c0                  = autograd.Variable(torch.randn(2*2, 1, 30))
# input: seq_len,   batch,   input_size
input               = autograd.Variable(torch.randn(3, 1, 30))
# output(seq_len, batch, hidden_size * num_directions):
# h_n(num_layers * num_directions, batch, hidden_size):
# c_n(num_layers * num_directions, batch, hidden_size):
output, (hn, cn)    = birnn(input, (h0, c0))
