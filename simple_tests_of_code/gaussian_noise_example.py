
import torch

use_cuda    = torch.cuda.is_available()
device      = torch.device("cuda") if(use_cuda) else torch.device("cpu")
mean, std   = 0., 0.1

embeddings  = torch.zeros(5, 10).float()
embeddings  = embeddings.to(device)
noise       = torch.zeros(1, embeddings.size(-1)).data.normal_(mean, std)
noise       = noise.to(device)
print(embeddings)
print(noise)

print(embeddings.size())
print(noise.size())

noisy_embeds = embeddings + noise.expand_as(embeddings)
print(noisy_embeds)
print(noisy_embeds.size())









