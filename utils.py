class OneHotLayer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        # Convert the integer state to one-hot encoded vector
        x_onehot = F.one_hot(x.to(torch.int64), num_classes=self.num_classes).float()
        return x_onehot
