import torch

class Cross_Entropy(torch.nn.Module):
    """docstring for Cross_Entropy"""

    def __init__(self, args):
        super().__init__()
        self.weights = torch.tensor(args.cross_entr_weights)

    def logsumexp(self, logits):
        m, _ = torch.max(logits, dim=1)
        print('m shape',m.shape)
        print('_: ', _.shape)
        m = m.view(-1, 1)
        print('m viewed: ', m.shape)
        sum_exp = torch.sum(torch.exp(logits-m), dim=1, keepdim=True)
        return m + torch.log(sum_exp)

    def forward(self, logits, labels):
        '''
        logits is a matrix M by C where m is the number of classifications and C are the number of classes
        labels is a integer tensor of size M where each element corresponds to the class that prediction i
        should be matching to
        '''
        print('logits shape: ',logits.shape)
        print('labels shape: ', labels.shape)
        labels = labels.view(-1, 1)
        print('labels shape after view: ', labels.shape)
        print('Labels: ',labels)
        labels = labels.to('cpu')
        logits = logits.to('cpu')
        alpha = self.weights[labels].view(-1, 1)
        print('alpha shape: ',alpha.shape)
        print('alphas: ', alpha)
        print('Logits: ', logits)
        temp = logits.gather(-1,labels)
        print('Gathered logits according to labels shape: ', temp.shape, 'content: ', temp)
        
        loss = alpha * (- logits.gather(-1, labels) + self.logsumexp(logits))
        print('Loss shape: ', loss.shape)
        raise NotImplementedError
        return loss.mean()