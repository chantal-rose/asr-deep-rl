import torch
from dataloading import mapping
from dataloading import PHONEMES
import Levenshtein
from utils import calculate_levenshtein


class RLLoss:
    def __init__(self, decoder):
        self.decoder = decoder

    def __call__(self, logits, target, input_lengths, target_lengths, return_dist=False, debug=False):
        return self.forward(logits, target, input_lengths, target_lengths, return_dist, debug=debug)

    def forward(self, logits, target, input_lengths, target_lengths, return_dist=False, debug=False):
        batch_size, timesteps, _ = logits.shape

        total_loss = 0.0

        total_dist = 0.0

        for i in range(batch_size):
            probs = logits[i, :input_lengths[i], :]
            predicted, probs = self.decoder.decode(probs)
            actual = "".join([mapping[PHONEMES[j]] for j in target[i, :target_lengths[i]]])

            # print(actual)
            # print(predicted[0])

            sample_distances = []

            for sample in range(len(predicted)):
                sample_distances.append(Levenshtein.distance(predicted[sample], actual))
                """
                if i % 25 == 0 and debug:
                    print("Batch", i+1, "sample", sample + 1, "Levenshtein Distance:", sample_distances[-1])
                if sample == (len(predicted)-1) and debug:
                  print()
                """

            total_dist += sum(sample_distances) / len(predicted)
            sample_distances = torch.tensor(sample_distances, dtype=torch.float, requires_grad=True).to(device)
            sample_distances = sample_distances - torch.mean(sample_distances)
            total_loss += torch.sum(sample_distances * torch.sum(torch.log(probs), axis=0)) / len(predicted)

        total_loss = total_loss / batch_size
        total_dist = total_dist / batch_size
        if return_dist:
            return total_loss, total_dist

        return total_loss