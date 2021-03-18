import torch

def get_rank(y_predict, y_thres):
    great = torch.sum(y_predict > y_thres, dim=1)
    equal = torch.sum(y_predict == y_thres, dim=1) / 2
    return great + equal + 1

def mrr(rank):
    return torch.mean(torch.reciprocal(rank.type(torch.float)))

def hit3(rank):
    return torch.mean((rank < 3).type(torch.float))

def hit10(rank):
    return torch.mean((rank < 10).type(torch.float))