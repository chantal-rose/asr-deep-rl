import torch.nn as nn

class ClsHead(nn.Module):
    """
        Final Linear & Classification Module
    """
    def __init__(
            self,
            dim_in: int=256,
            dims: list=[512],
            num_labels: int=43,
            dropout: float=0.2
        ):
        super().__init__()
        # bookkeeping
        self.dims = [dim_in] + dims + [num_labels]
        self.linears = nn.ModuleList()
        # linear(s)
        for i, dim in enumerate(self.dims[:-1]):
            self.linears.append(
                nn.Linear(
                    in_features=dim,
                    out_features=self.dims[i + 1]
                )
            )
            if i < len(self.dims) - 2:
                self.linears.append(nn.Dropout(dropout))
                self.linears.append(nn.GELU())
        assert len(self.linears) == 3 * len(self.dims) - 5
        # logsoftmax
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        for layer in self.linears:
            x = layer(x)
        x = self.softmax(x)
        return x


class Baseline(nn.Module):
    """
        Partial network from BiLSTM to Classification.
    """

    def __init__(
            self,
            input_dim: int,
            lstm_cfgs: dict,
            cls_cfgs: dict
    ):
        super().__init__()
        self.configs = {
            'mfcc_dim': input_dim,
            'lstm': lstm_cfgs,
            'cls': cls_cfgs
        }

        # dim_in for lstm is the original MFCC dimensions
        self.lstm = LSTMLayers(dim_in=input_dim, **lstm_cfgs)

        # dim_in for cls is the final hidden dimension of LSTMs
        cls_dim_in = lstm_cfgs['hidden_dims'][-1]

        # alter the dimension based on whether bidirectional is enabled
        cls_dim_in *= 2 if lstm_cfgs['bidirectionals'][-1] else 1
        self.cls = ClsHead(dim_in=cls_dim_in, **cls_cfgs)

    def forward(self, x, lx):
        # (bi)lstm layers
        xx, lx = self.lstm(x, lx)
        # cls layers
        xx = self.cls(xx)
        return xx, lx
