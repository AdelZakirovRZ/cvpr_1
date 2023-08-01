import torch


def top_k_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k
    [top_a, top_b, top_c, ...], each element is an accuracy over a batch.

    output: (B, n_cls)
    target: (n_cls)
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)  # (B, maxk), (B, maxk)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # (maxk, B)

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))  # 100.0 -> %
        return res