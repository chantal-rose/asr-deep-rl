import torch
from dataloading import mapping
from dataloading import PHONEMES

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MultinomialDecode:
    def __init__(self, num_samples=1):
        self.num_samples = num_samples

    def decode(self, logits, debug=False):

        decoded = torch.argmax(logits, axis=1)

        # print("output of the decoder is: ",decoded)
        output = (torch.arange(logits.size(0)).to(device) * logits.size(1) + decoded.to(device))[:, None]
        # print("output is: ", output)

        pred = ""
        prev = ""
        for t in range(decoded.shape[0]):
            curr = mapping[PHONEMES[decoded[t].item()]]

            if t == 0:
                pred += curr
                prev = curr
            else:
                if curr != prev and curr != " ":
                    pred += curr
                    prev = curr
                if curr == " ":
                    prev = ""
        predicted = []
        predicted.append(pred)

        return predicted, torch.take(logits, output)