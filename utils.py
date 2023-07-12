def get_network(args):
    if args.model == 'resnet18':
        from resnet import resnet18
        return resnet18()
    elif args.model == 'resnet34':
        from resnet import resnet34
        return resnet34()
    elif args.model == 'resnet50':
        from resnet import resnet50
        return resnet50()
    elif args.model == 'resnet101':
        from resnet import resnet101
        return resnet101()
    elif args.model == 'resnet152':
        from resnet import resnet152
        return resnet152()

def get_optimizer(args, model):
    if args.optim == 'sgd':
        import torch.optim as optim
        return optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.w_decay)
    if args.optim == 'adam':
        import torch.optim as optim
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    if args.optim == 'adamw':
        import torch.optim as optim
        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
