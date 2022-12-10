#code for dataloader
import torch

CMUdict_ARPAbet = {
    "" : " ",
    "[SIL]": "-", "NG": "G", "F" : "f", "M" : "m", "AE": "@",
    "R"    : "r", "UW": "u", "N" : "n", "IY": "i", "AW": "W",
    "V"    : "v", "UH": "U", "OW": "o", "AA": "a", "ER": "R",
    "HH"   : "h", "Z" : "z", "K" : "k", "CH": "C", "W" : "w",
    "EY"   : "e", "ZH": "Z", "T" : "t", "EH": "E", "Y" : "y",
    "AH"   : "A", "B" : "b", "P" : "p", "TH": "T", "DH": "D",
    "AO"   : "c", "G" : "g", "L" : "l", "JH": "j", "OY": "O",
    "SH"   : "S", "D" : "d", "AY": "Y", "S" : "s", "IH": "I",
    "[SOS]": "[SOS]", "[EOS]": "[EOS]"
}

CMUdict = list(CMUdict_ARPAbet.keys())
ARPAbet = list(CMUdict_ARPAbet.values())

PHONEMES = CMUdict
mapping = CMUdict_ARPAbet
LABELS = ARPAbet

BATCH_SIZE = 64
root = 'f0176/half2'

phoneme_index_mapping = dict()
for i, phoneme in enumerate(PHONEMES):
  phoneme_index_mapping[phoneme] = i


class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, subset="train"):
        '''
        Initializes the dataset.
        '''
        self.root_dir = root_dir
        self.subset = subset
        mfcc_dir = f"{root_dir}/{subset}.npy"
        trans_dir = f"{root_dir}/{subset}_labels.npy"

        self.mfccs = torch.from_numpy(np.load(mfcc_dir))
        self.transcripts = torch.from_numpy(np.load(trans_dir))

        self.size = len(self.mfccs)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.mfccs[index], self.transcripts[index]

    def collate_fn(self, batch):
        mfccs = [u[0] for u in batch]
        transcripts = [u[1] for u in batch]

        mfccLens = torch.tensor([len(m) for m in mfccs])
        transcriptLens = torch.tensor([len(t) for t in transcripts])

        mfccs = pad_sequence(
            mfccs, batch_first=True
        )
        transcripts = pad_sequence(
            transcripts, batch_first=True
        )
        return mfccs, transcripts, mfccLens, transcriptLens
