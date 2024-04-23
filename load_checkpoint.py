from src.module import MNISTModule
from src.dm import MNISTDataModule
import wandb
import torch


wandb.init(mode="offline")
module = MNISTModule.load_from_checkpoint('checkpoints/008-val_loss=0.19057-epoch=5.ckpt')
dm = MNISTDataModule(**module.hparams['datamodule'])
dm.setup()

module.eval()
with torch.no_grad():
    preds, labels = torch.tensor([]), torch.tensor([])
    for imgs, _labels in dm.val_dataloader():
        outputs = module.predict(imgs.to(module.device)) > 0.5
        preds = torch.cat([preds, outputs.cpu().long()])
        labels = torch.cat([labels, _labels])

acc = (preds == labels).float().mean()

print(acc.item()), print(module.hparams)

wandb.finish()



