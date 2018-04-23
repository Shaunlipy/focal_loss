import torch
import numpy as np

# 4D
inputs = torch.autograd.Variable(torch.randn(2, 2, 2, 3))
target = torch.autograd.Variable(torch.ones(2, 2, 3)).long()
target[1,:,:] = 0
print (inputs)
print (target)
weight = torch.Tensor([5, 1])
# 4D - method 1
ce = torch.nn.CrossEntropyLoss()
# -logp
out = ce(inputs, target) 
# p
out_ = torch.exp(-out)
# (1-p)*logp
loss = (1 - out_)**2 * out
print (loss)
ce = torch.nn.CrossEntropyLoss(weight = weight)
# -logp
out = ce(inputs, target) 
# p
out_ = torch.exp(-out)
# (1-p)*logp
loss = (1 - out_)**2 * out
print (loss)

# # 4D - method 2 Not Good
# ce = torch.nn.CrossEntropyLoss(reduce = False, weight=weight)
# # -logp
# out = ce(inputs, target) 
# # p
# out_ = torch.exp(out)
# # (1-p)*logp
# loss = (1 - out_)**2 * out
# print (loss)
# print (loss.mean())

# # 4D - method 3 not good
# N, w, h = target.size()
# target_ = target.view(N, 1, w, h)
# logsoftmax = torch.nn.LogSoftmax(dim=1)
# softmax = torch.nn.Softmax(dim=1)
# out3 = -logsoftmax(inputs)
# out4 = softmax(inputs)
# print ('^^^^^', out3)
# print ('*****', out4)
# out3 = out3.gather(1, target_)
# out4 = (1 - out4.gather(1, target_))

# print ('###',out3)
# print ('$$$',out4)

# print ('()()()', (out4**2 * out3))
# print ('!!!!()()', (out4**2 * out3).mean())
