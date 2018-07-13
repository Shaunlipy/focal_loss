import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import ipdb

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
    
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        # ipdb.set_trace()
        N = inputs.size(0) # 4
        print(N)
        C = inputs.size(1) # 5
        P = F.softmax(inputs) # [4, 5]

        class_mask = inputs.data.new(N, C).fill_(0) #[4, 5]
        class_mask = Variable(class_mask) 
        ids = targets.view(-1, 1) # [4, 1]
        class_mask.scatter_(1, ids.data, 1.) # fill mask row index with 1
        #print(class_mask)
        

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)] # get weights for corresponding class
        # [4, 1]
        probs = (P*class_mask).sum(1).view(-1,1) # [4, 1]

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        print('Probs: {}'.format(probs))

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        print('Batch Loss: {}'.format(batch_loss))
        
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

        

if __name__ == "__main__":
    ipdb.set_trace()
    alpha = torch.rand(21, 1)
    print(alpha)
    FL = FocalLoss(class_num=5, gamma=2 )
    CE = nn.CrossEntropyLoss(reduce = False)
    N = 4
    C = 5
    inputs = torch.rand(N, C)
    targets = torch.LongTensor(N).random_(C)
    inputs_fl = Variable(inputs.clone(), requires_grad=True)
    targets_fl = Variable(targets.clone())

    inputs_ce = Variable(inputs.clone(), requires_grad=True)
    targets_ce = Variable(targets.clone())
    print('----inputs----')
    print(inputs)
    print('---target-----')
    print(targets)

    fl_loss = FL(inputs_fl, targets_fl)
    ce_loss = CE(inputs_ce, targets_ce)
    ce_loss_ = torch.exp(-ce_loss)
    final_ce_loss = (1 - ce_loss_)**2 * ce_loss

    print (final_ce_loss)

    print('ce = {}, fl ={}'.format(ce_loss.data[0], fl_loss.data[0]))
    # fl_loss.backward()
    # ce_loss.backward()
    #print(inputs_fl.grad.data)
    # print(inputs_ce.grad.data)