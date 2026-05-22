# E:/restormer+volterra/Restormer + Volterra/tasks/dehazing/train_sots.py
import os, sys, csv
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, '..', '..')))  # models

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.amp import autocast, GradScaler

from models.restormer_volterra import RestormerVolterra

BASE_DIR  = r"E:/restormer+volterra/data/SOTS"
SPLIT_DIR = os.path.join(BASE_DIR, "splits")
SAVE_DIR  = r"E:/restormer+volterra/checkpoints/sots_volterra"; os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH  = 2
EPOCHS = 100
LR     = 2e-4

resize_schedule = {0:128, 30:192, 60:256}
def get_transform(epoch):
    size = max(v for k,v in resize_schedule.items() if epoch>=k)
    return transforms.Compose([
        transforms.Resize((size,size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])

def read_pairs_csv(path):
    rows=[]
    with open(path, newline="", encoding="utf-8") as f:
        r=csv.DictReader(f)
        a=next((c for c in r.fieldnames if c.lower().startswith("dist")), None)
        b=next((c for c in r.fieldnames if c.lower().startswith("ref")),  None)
        for row in r:
            rows.append((row[a], row[b]))
    return rows

class PairDS(Dataset):
    def __init__(self, rows, tfm):
        self.rows, self.t = rows, tfm
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        a,b=self.rows[i]
        return self.t(Image.open(a).convert("RGB")), self.t(Image.open(b).convert("RGB"))

@torch.no_grad()
def evaluate(model, dl):
    model.eval(); P=S=0.0; n=0
    for x,y in dl:
        x,y=x.to(DEVICE), y.to(DEVICE)
        with autocast(device_type="cuda", enabled=(DEVICE.type=="cuda")):
            o=model(x).clamp(0,1)
        o=o.cpu().numpy(); g=y.cpu().numpy()
        for i in range(o.shape[0]):
            O=np.transpose(o[i],(1,2,0)); G=np.transpose(g[i],(1,2,0))
            P+=psnr(G,O,data_range=1.0); S+=ssim(G,O,channel_axis=2,data_range=1.0); n+=1
    return (P/n if n else 0.0, S/n if n else 0.0)

def main():
    ind_tr = read_pairs_csv(os.path.join(SPLIT_DIR,"indoor_train.csv"))
    ind_te = read_pairs_csv(os.path.join(SPLIT_DIR,"indoor_test.csv"))
    out_tr = read_pairs_csv(os.path.join(SPLIT_DIR,"outdoor_train.csv"))
    out_te = read_pairs_csv(os.path.join(SPLIT_DIR,"outdoor_test.csv"))

    train_rows = ind_tr + out_tr
    test_rows  = ind_te + out_te
    if len(train_rows)==0:
        raise RuntimeError("❌ No training pairs. 먼저 make_sots_splits.py를 실행해 splits를 만드세요.")

    model = RestormerVolterra().to(DEVICE)
    opt   = optim.AdamW(model.parameters(), lr=LR)
    crit  = nn.L1Loss()
    scaler= GradScaler()

    for e in range(1,EPOCHS+1):
        tfm = get_transform(e)
        tr_ds = PairDS(train_rows, tfm)
        tr_dl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)

        model.train(); tot=0.0
        loop=tqdm(tr_dl, desc=f"[Train] {e}/{EPOCHS}")
        opt.zero_grad(set_to_none=True)
        for x,y in loop:
            x,y=x.to(DEVICE), y.to(DEVICE)
            with autocast(device_type="cuda", enabled=(DEVICE.type=="cuda")):
                o=model(x); loss=crit(o,y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True); tot+=loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg= tot/max(1,len(tr_dl))
        # Validation at fixed 256
        val_ds = PairDS(test_rows, transforms.Compose([
            transforms.Resize((256,256), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ]))
        val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        vp,vs = evaluate(model, val_dl)
        print(f"📣 Epoch {e:03d} | TrainLoss {avg:.5f} | Val PSNR {vp:.2f} | Val SSIM {vs:.4f}")

        ckpt = os.path.join(SAVE_DIR, f"epoch_{e}_valssim{vs:.4f}_valpsnr{vp:.2f}.pth")
        torch.save(model.state_dict(), ckpt)
        print(f"💾 Saved: {ckpt}")

if __name__ == "__main__":
    main()
