import torch
import os
import json
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .. import utils

class SegmentationLightningModule(pl.LightningModule):
    """
    Lightning Module class for training segmentation U-Net
    """
    def __init__(self, config, datasets, model, log_dir):
        super().__init__()
        self.config = config
        self.datasets = datasets
        self.model = model
        self.log_dir = log_dir
        self.parse_config()
        self.collate_fn = None
        self.outputs = {
            "train": [],
            "val": [],
            "test": []
        }
        self.metrics = {}
        self.loss_fn = torch.nn.CrossEntropyLoss(self.class_weights)
    
    def parse_config(self):
        self.batch_size = int(self.config["batch_size"])
        self.lr = float(self.config["lr"])
        self.weight_decay = float(self.config["weight_decay"])
        self.class_weights = self.config["class_weights"]
        if len(self.class_weights) == 0:
            self.class_weights = [1.] * self.model.n_output_channels
        elif len(self.class_weights) == self.model.n_output_channels:
            try:
                self.class_weights = [float(w) for w in self.class_weights]
            except:
                raise TypeError("Not all class weights are numerical")
        self.class_weights = torch.tensor(self.class_weights)
        self.dice_weight = float(self.config["dice_weight"])
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    def step(self, batch, stage):
        images = batch['image'] # [B, 1, H, W, D]
        masks = batch['label'] # [B, 1, H, W, D]
        preds = self.model(images) # [B, n_classes+1, H, W, D]
        
        # class encoding
        class_labels = torch.cat([masks==c for c in range(len(self.class_weights))], dim=1).float() # [B, n_classes+1, H, W, D], first class is backgroyund
        
        loss = self.loss_fn(preds, class_labels)
        
        _, dice_loss = utils.dice(class_labels[:,1:], torch.softmax(preds, dim=1)[:,1:])
        dice_loss = torch.einsum('bc,c->b', dice_loss, self.class_weights[1:].to(dice_loss.device).float()) #weighted dice loss for non-background classes
        loss += self.dice_weight * dice_loss.mean()
        
        output_dict = {'loss': loss.mean()}
        
        with torch.no_grad():
            preds_postproc = torch.argmax(preds,dim=1)
            preds_postproc = torch.stack([preds_postproc==c for c in range(1, len(self.class_weights))], dim=1).float()
            coef = utils.dice_coef(class_labels[:,1:], preds_postproc)

        for c in range(len(self.class_weights)-1):
            output_dict[f'dice_class_{c+1}'] = coef[:,c].mean().detach()

        self.outputs[stage].append(output_dict)
        return output_dict
    
    def training_step(self, batch, _):
        return self.step(batch, "train")
    
    def validation_step(self, batch, _):
        return self.step(batch, "val")
    
    def test_step(self, batch, _):
        return self.step(batch, "test")
    
    def predict_step(self, batch, _):
        return self.step(batch)
    
    def epoch_end(self, stage):
        outputs = self.outputs[stage]
        metrics_to_log = list(outputs[0].keys())
        for m in metrics_to_log:
            val = torch.stack([x[m] for x in outputs]).mean()
            self.log_metric(f"{stage}/{m}", val)
            self.log(f"{stage}_{m}", val)
        self.save_metrics()
        self.outputs[stage] = []
    
    def on_train_epoch_end(self):
        self.epoch_end("train")
    
    def on_validation_epoch_end(self):
        self.epoch_end("val")
    
    def on_test_epoch_end(self):
        self.epoch_end("test")
    
    def log_metric(self, metric_name, metric_value):
        self.log(metric_name, metric_value, on_epoch=True, prog_bar=True)
        if metric_name in self.metrics:
            self.metrics[metric_name].append(metric_value.item())
        else:
            self.metrics[metric_name] = [metric_value.item()]
    
    def load_metrics(self):
        filename = os.path.join(self.log_dir, "metrics.json")
        with open(filename, "r") as f:
            self.metrics = json.load(f)
    
    def save_metrics(self):
        filename = os.path.join(self.log_dir, "metrics.json")
        with open(filename, "w") as f:
            json.dump(self.metrics, f)
    
    def train_dataloader(self):
        train_dataset = self.datasets["train"]
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader
    
    def val_dataloader(self):
        eval_dataset = self.datasets["val"]
        eval_loader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False)
        return eval_loader
    
    def test_dataloader(self):
        test_dataset = self.datasets["test"]
        test_loader = self.dataloader_fn(test_dataset, batch_size=self.batch_size, shuffle=False)
        return test_loader
