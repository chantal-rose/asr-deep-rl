class LSTMLayers(nn.Module):
    """
        (Bidirectional) LSTM Module
    """

    def __init__(
            self,
            dim_in: int = 512,
            hidden_dims: list = [256, 256],
            num_layers: list = [4, 4],
            bidirectionals: list = [True, True],
            dropouts: list = [0.3, 0.3]
    ):
        """
            Note: locked dropout only applied once to concat layers at the end
                    to apply one locked dropout per lstm, use LockedGroupedLSTM instead
        """
        super().__init__()
        self.dims = [dim_in] + hidden_dims
        self.num_layers = num_layers
        self.bidirectionals = bidirectionals
        self.dropouts = dropouts
        assert (
                len(self.dims)
                == len(self.num_layers) + 1
                == len(self.bidirectionals) + 1
        )

        self.streamline = nn.ModuleList()
        for l, dim in enumerate(self.dims[:-1]):
            input_dim = (
                dim * 2 if l > 0 and self.bidirectionals[l]
                else dim
            )
            lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=self.dims[l + 1],
                num_layers=self.num_layers[l],
                bidirectional=self.bidirectionals[l],
                dropout=self.dropouts[l],
                batch_first=True
            )
            self.streamline.append(lstm)

    def forward(self, x, lx):
        xx = pack_padded_sequence(
            x, lengths=lx, batch_first=True, enforce_sorted=False
        )
        for layer in self.streamline:
            xx, _ = layer(xx)
        xx, lx = pad_packed_sequence(
            xx, batch_first=True
        )
        return xx, lx
