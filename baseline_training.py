import torch
from baseline_modules import Baseline
from dataloading import AudioDataset
from tqdm import tqdm
from utils import calculate_levenshtein
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from torch.nn.modules.loss import CTCLoss
import ctcdecode
from ctcdecode import CTCBeamDecoder
from dataloading import LABELS

torch.manual_seed(0)
MFCC_DIM = 26
HIDDEN_DIMS = [128, 128]
NUM_LAYERS = [2, 2]
BIDIRECTIONALS = [True, True]
LINEAR_DIMS = [256]
DROPOUTS = [0.2, 0.2]
NUM_PHONEMES = 43


LSTM_CFGS = {
    'hidden_dims': HIDDEN_DIMS,
    'num_layers': NUM_LAYERS,
    'bidirectionals': BIDIRECTIONALS,
    'dropouts': DROPOUTS
}

CLS_CFGS = {
    'dims': LINEAR_DIMS,
    'num_labels': NUM_PHONEMES
}

model = Baseline(
    input_dim=MFCC_DIM,
    lstm_cfgs=LSTM_CFGS,
    cls_cfgs=CLS_CFGS
)

model = model.to(device)

baseline_config = {
    "beam_width" : 2,
    "lr" : 1e-3,
    "epochs" : 50,
    "scheduler": "ReduceLR",
    "layers": "LSTM(128, 128) with dropout, Linear(256)",
    }

optimizer =  torch.optim.AdamW(model.parameters(), lr=baseline_config['lr'], weight_decay=1e-5) # What goes in here?
criterion = CTCLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 5, factor = 0.2, min_lr = 1e-7, mode = "min")
scaler = torch.cuda.amp.GradScaler()

decoder = CTCBeamDecoder(
    LABELS,
    beam_width = baseline_config["beam_width"],
    num_processes = 4,
    log_probs_input = True
)


def train_step(train_loader, model, optimizer, criterion, scheduler, scaler):
    model.train()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')
    train_loss = 0

    for i, data in enumerate(train_loader):
        mfccs, phonemes, len_mfccs, len_phonemes = data
        mfccs, phonemes = mfccs.to(device), phonemes.to(device)

        with torch.cuda.amp.autocast():
            out, lens = model(mfccs, len_mfccs)

        loss = criterion(torch.log(out).permute(1, 0, 2), phonemes, lens, len_phonemes)

        batch_bar.set_postfix(
            loss=f"{train_loss / (i + 1):.4f}",
            lr=f"{optimizer.param_groups[0]['lr']}"
        )

        train_loss += loss
        batch_bar.update()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    batch_bar.close()
    train_loss /= len(train_loader)

    return train_loss


def evaluate(data_loader, model, epoch):
    model.eval()
    dist = 0
    loss = 0
    batch_bar = tqdm(total=len(data_loader), dynamic_ncols=True, leave=False, position=0, desc='Val')

    phone_true_list = []
    phone_pred_list = []

    for i, data in enumerate(data_loader):
        mfccs, phonemes, len_mfccs, len_phonemes = data
        mfccs, phonemes = mfccs.to(device), phonemes.to(device)

        with torch.inference_mode():
            out, lens = model(mfccs, len_mfccs)

        loss = criterion(torch.log(out).permute(1, 0, 2), phonemes, lens, len_phonemes)

        dist += calculate_levenshtein(torch.log(out), phonemes, lens, len_phonemes, decoder, LABELS)

        del mfccs, phonemes, out
        torch.cuda.empty_cache()

    dist /= len(data_loader)

    return loss, dist

BATCH_SIZE = 64

root = 'f0176/half1'

train_data = AudioDataset(root, subset="train")
val_data = AudioDataset(root, subset="dev")

train_loader = torch.utils.data.DataLoader(train_data, num_workers = 4,
                                           batch_size = BATCH_SIZE, pin_memory = True,
                                           shuffle = True, collate_fn = train_data.collate_fn)

val_loader = torch.utils.data.DataLoader(val_data, num_workers = 2,
                                         batch_size = BATCH_SIZE, pin_memory = True,
                                         shuffle = False, collate_fn = val_data.collate_fn)

print("Batch size: ", BATCH_SIZE)
print("Train dataset samples = {}, batches = {}".format(train_data.__len__(), len(train_loader)))
print("Val dataset samples = {}, batches = {}".format(val_data.__len__(), len(val_loader)))

best_val_dist = float('inf')

name = "baseline"

train_losses = []
validation_losses = []

for epoch in range(baseline_config['epochs']):
    print("\nEpoch {}/{}".format(epoch + 1, baseline_config['epochs']))
    train_loss = train_step(train_loader, model, optimizer, criterion, scheduler, scaler)
    print("\tTrain Loss: {:.4f}".format(train_loss))

    validation_loss, levenshtein_distance = evaluate(val_loader, model, epoch)
    print("\tLevenshtein Distance: {}".format(levenshtein_distance))

    validation_losses.append(validation_loss)

    scheduler.step(levenshtein_distance)

    if levenshtein_distance < best_val_dist:
        path = "/content/drive/MyDrive/checkpoint_{}.pth".format(name)
        print("Saving model")
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_dist': levenshtein_distance,
                    'epoch': epoch}, path)

        best_val_dist = levenshtein_distance


