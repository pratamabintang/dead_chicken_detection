###-----    Library & Depedency     -----###

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T

import random
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
from timeit import default_timer as timer
import matplotlib.patches as patches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Model run by {device}")

CLASS_NAMES = ["__background__", "chicken"]

###-----            Main            -----###

def main():
  data_root = Path("/content/drive/MyDrive/dataset_chicken")                                    ### DATA PATH
  img_dir = data_root / "images"
  ann_dir = data_root / "annotations"

  TRAIN_SPLIT = 0.85                                                                            ### PARAMETER
  BATCH_SIZE = 4
  NUM_WORKERS = os.cpu_count()                           # os.cpu_count()
  LEARNING_RATE = 0.005
  EPOCHS = 5
  SCALER_TOGGLE = False                      # "store_true"
  MOMENTUM = 0.9
  WEIGHT_DECAY = 5e-4
  MAX_NORM = 0.0
  RESUME = ""       #"/content/runs/checkpoints/best.pt"                               # path to resume

  ann_stems = {p.stem for p in ann_dir.glob("*.xml")}
  img_stems = {p.stem for p in img_dir.iterdir() if p.is_file()}
  ids = sorted(list(ann_stems.intersection(img_stems)))

  print(f"Found {len(ids)} matching image and annotation pairs.")

  random.shuffle(ids)
  n_train = int(len(ids) * TRAIN_SPLIT)
  train_ids = ids[:n_train]
  test_ids = ids[n_train:]

  train_loader = DataLoader(VOCDataset(data_root, train_ids, augment=True),
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=NUM_WORKERS,
                            collate_fn=collate_fn)
  test_loader = DataLoader(VOCDataset(data_root, test_ids, augment=False),
                           batch_size=BATCH_SIZE,
                           shuffle=False,
                           num_workers=NUM_WORKERS,
                           collate_fn=collate_fn)

  print(f"Length of train dataloader: {len(train_loader)} batches of {BATCH_SIZE}")
  print(f"Length of test dataloader: {len(test_loader)} batches of {BATCH_SIZE}")

  model = create_model(num_classes=len(CLASS_NAMES))
  model.to(device)

  params = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
  lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, EPOCHS // 3), gamma=0.1)
  scaler = torch.cuda.amp.GradScaler() if (SCALER_TOGGLE and device.type == 'cuda') else None

  start_epoch = 0
  best_val = float("inf")

  if RESUME and os.path.isfile(RESUME):
    ckpt = torch.load(RESUME, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    lr_sched.load_state_dict(ckpt["lr_sched"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val = ckpt.get("best_val", best_val)
    print(f"Resumed from {RESUME} at epoch {start_epoch}")

  ckpt_dir = Path("model")

  train_time_start = timer()

  train_loss_history = []
  train_acc_history = []
  test_loss_history = []
  test_acc_history = []

  for epoch in tqdm(range(start_epoch, EPOCHS)):
    print(f"Epoch: {epoch + 1}\n---------")
    train_loss, train_acc = train_model(model, optimizer, train_loader, device, scaler, max_norm=MAX_NORM)
    test_loss, test_acc = eval_loss(model, test_loader, device)
    lr_sched.step()

    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)
    test_loss_history.append(test_loss)
    test_acc_history.append(test_acc)
    print(f"Epoch {epoch+1}/{EPOCHS} | train loss : {train_loss:.4f} | train accuracy : {train_acc:.4f} | val loss : {test_loss:.4f} | test accuracy : {test_acc:.4f}")

    # Save latest
    save_checkpoint({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_sched": lr_sched.state_dict(),
        "best_val": best_val,
    }, ckpt_dir / "last.pt")

    # Save best by val loss
    if test_loss < best_val:
      best_val = test_loss
      save_checkpoint({
          "epoch": epoch,
          "model": model.state_dict(),
          "optimizer": optimizer.state_dict(),
          "lr_sched": lr_sched.state_dict(),
          "best_val": best_val,
      }, ckpt_dir / "best.pt")
      print(f"Saved new best checkpoint: val_loss={best_val:.4f}")

  train_time_end = timer()
  model_time = print_train_time(start=train_time_start, end=train_time_end, device=device)

  # show_predictions_and_count(model, VOCDataset(data_root, test_ids, augment=False), device)         ### output inference test

  history = {
      "train_loss": train_loss_history,
      "train_acc": train_acc_history,
      "test_loss": test_loss_history,
      "test_acc": test_acc_history,
  }
  torch.save(history, ckpt_dir / "training_history.pt")
  print("History saved!")



###-----            Utils           -----###

def collate_fn(batch):
  return tuple(zip(*batch))

def set_seed(seed: int = 42):
  random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

def print_train_time(start, end, device: torch.device = device):
  print(f"Train time on {device} : {end - start} sec")
  return end - start



###-----        Datasets            -----###

def parse_xml(file_path: Path) -> tuple[list[list[int]], list[int]]:
  tree = ET.parse(str(file_path))
  root = tree.getroot()
  boxes: list[list[int]] = []
  labels: list[int] = []

  for obj in root.findall("object"):
    bnd = obj.find("bndbox")
    xmin = int(float(bnd.find("xmin").text))
    ymin = int(float(bnd.find("ymin").text))
    xmax = int(float(bnd.find("xmax").text))
    ymax = int(float(bnd.find("ymax").text))
    boxes.append([xmin, ymin, xmax, ymax])
    labels.append(1)                        ### Class chicken (1)

  return boxes, labels

class VOCDataset(Dataset):
  def __init__(self, root: str, ids: list[str], augment: bool = False):
    self.root = Path(root)
    self.ids = ids
    self.augment = augment
    self.img_dir = self.root / "images"
    self.ann_dir = self.root / "annotations"
    self.img_paths = {p.stem: p for p in self.img_dir.iterdir() if p.is_file()}

    if augment:
      self.tf = T.Compose([T.ToTensor()])
      self.hflip_p = 0.3
      self.jitter_p = 0.3
      self.jitter = T.ColorJitter(brightness=0.5, contrast=0.5)
    else:
      self.tf = T.Compose([T.ToTensor()])
      self.hflip_p = 0.0
      self.jitter_p = 0.0
      self.jitter = None

  def __len__(self):
    return len(self.ids)

  def __getitem__(self, idx: int):
    base = self.ids[idx]
    stem = Path(base).stem
    img_path = self.img_paths.get(stem)

    if img_path is None:
      raise FileNotFoundError(f"Gambar tidak ditemukan: {stem}")

    ann_path = self.ann_dir / f"{stem}.xml"
    if not ann_path.exists():
      raise FileNotFoundError(f"Label tidak ditemukan: {ann_path}")

    img = Image.open(img_path).convert("RGB")
    boxes, labels = parse_xml(ann_path)
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.int64)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    label: dict[str, torch.Tensor] = {
      "boxes": boxes,
      "labels": labels,
      "image_id": torch.tensor([idx]),
      "area": area,
      "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.uint8),
    }

    if self.jitter is not None and random.random() < self.jitter_p:
      img = self.jitter(img)

    img = self.tf(img)

    if boxes.numel() > 0:
      if random.random() < self.hflip_p:
        _, h, w = img.shape
        if random.random() < 0.5:       ### flip horizontal
          img = torch.flip(img, dims=[2])
          x_min = w - boxes[:, 2]
          x_max = w - boxes[:, 0]
          boxes[:, 0] = x_min
          boxes[:, 2] = x_max
        else:                           ### flip vertical
          img = torch.flip(img, dims=[1])
          y_min = h - boxes[:, 3]
          y_max = h - boxes[:, 1]
          boxes[:, 1] = y_min
          boxes[:, 3] = y_max
        label["boxes"] = boxes

    return img, label



###-----            Model           -----###

def create_model(num_classes: int = 2):
  model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
  return model



###-----            Train           -----###

def iou(box1, box2):
  x1 = max(box1[0], box2[0])
  y1 = max(box1[1], box2[1])
  x2 = min(box1[2], box2[2])
  y2 = min(box1[3], box2[3])

  inter_area = max(0, x2 - x1) * max(0, y2 - y1)

  box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
  box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

  union_area = box1_area + box2_area - inter_area
  return inter_area / union_area if union_area > 0 else 0

def detection_accuracy(pred_boxes, gt_boxes, iou_threshold=0.5, verbose=True):
  if len(gt_boxes) == 0:
    acc = 1.0 if len(pred_boxes) == 0 else 0.0
    if verbose:
      print(f"Benar: {0}/{0} | Accuracy: {acc:.2f}")
    return acc

  correct = 0
  matched_gt = set()

  for pred in pred_boxes:
    for i, gt in enumerate(gt_boxes):
      if i in matched_gt:
        continue
      if pred[4] == gt[4] and iou(pred[:4], gt[:4]) >= iou_threshold:
        correct += 1
        matched_gt.add(i)
        break

  total = len(gt_boxes)
  acc = correct / total

  if verbose:
    print(f"Benar: {correct}/{total} | Accuracy: {acc:.2f}")

  return acc

@torch.inference_mode()
def evaluate_batch(model, images, targets, device, iou_threshold=0.5):
  model.eval()
  preds = model(images)
  accs = []
  for pred, gt in zip(preds, targets):
    pred_boxes = []
    for box, label in zip(pred["boxes"], pred["labels"]):
      x1, y1, x2, y2 = box.tolist()
      pred_boxes.append([x1, y1, x2, y2, label.item()])

    gt_boxes = []
    for box, label in zip(gt["boxes"], gt["labels"]):
      x1, y1, x2, y2 = box.tolist()
      gt_boxes.append([x1, y1, x2, y2, label.item()])

    acc = detection_accuracy(pred_boxes, gt_boxes, iou_threshold=iou_threshold, verbose=False)
    accs.append(acc)
  return sum(accs) / len(accs) if accs else 0.0

def train_model(model: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                loader: torch.utils.data.DataLoader,
                device: torch.device,
                scaler=None,
                max_norm: float = 0.0):
  total_loss = 0.0
  total_acc = 0.0

  for _, (images, targets) in enumerate(loader):
    model.train()
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    optimizer.zero_grad(set_to_none=True)
    if scaler is not None:
      with torch.cuda.amp.autocast():
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
      scaler.scale(losses).backward()
      if max_norm > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
      scaler.step(optimizer)
      scaler.update()
    else:
      loss_dict = model(images, targets)
      losses = sum(loss for loss in loss_dict.values())
      losses.backward()
      if max_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
      optimizer.step()

    total_loss += losses.item()
    total_acc += evaluate_batch(model, images, targets, device)

  avg_loss = total_loss / max(1, len(loader))
  avg_acc = total_acc / max(1, len(loader))

  return avg_loss, avg_acc

@torch.no_grad()
def eval_loss(model, loader, device, iou_threshold=0.5):
  model.train()
  total_loss = 0.0
  accs = []

  for images, targets in loader:
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    total_loss += losses.item()

    model.eval()
    acc = evaluate_batch(model, images, targets, device, iou_threshold=iou_threshold)
    model.train()
    accs.append(acc)

  avg_loss = total_loss / max(1, len(loader))
  avg_acc = sum(accs) / len(accs) if accs else 0.0

  return avg_loss, avg_acc

def save_checkpoint(state: dict, path: Path):
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(state, str(path))



###-----    Eval Model     -----###

def apply_nms(outputs, iou_threshold=0.5):
  if isinstance(outputs, list):
    outputs = outputs[0]

  boxes = outputs['boxes']
  scores = outputs['scores']
  labels = outputs['labels']

  keep = torchvision.ops.nms(boxes, scores, iou_threshold)
  return {
    'boxes': boxes[keep],
    'scores': scores[keep],
    'labels': labels[keep]
  }

def show_predictions_and_count(model, dataset, device, num_images=5, score_threshold=0.75):
  model.eval()
  indices = random.sample(range(len(dataset)), num_images)

  for idx in indices:
    img, target = dataset[idx]
    img_tensor = img.to(device).unsqueeze(0)

    with torch.inference_mode():
      outputs = model(img_tensor)
      outputs = apply_nms(outputs, iou_threshold=0.5)

    pred_boxes = outputs["boxes"].cpu().numpy()
    pred_scores = outputs["scores"].cpu().numpy()
    pred_labels = outputs["labels"].cpu().numpy()

    keep = pred_scores >= score_threshold
    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]
    pred_labels = pred_labels[keep]

    total_detected = len(pred_boxes)

    img_np = img.permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img_np)

    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
      x1, y1, x2, y2 = box
      rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                linewidth=2, edgecolor="r", facecolor="none")
      ax.add_patch(rect)
      ax.text(x1, y1 - 5, f"{label}:{score:.2f}", color="yellow", fontsize=10, bbox=dict(facecolor="red", alpha=0.5))

    ax.set_title(f"Objects detected: {total_detected}", fontsize=14, color="blue")

    plt.show()
    print(f"Gambar {idx} → Terdeteksi {total_detected} objek (threshold={score_threshold})")



if __name__ == "__main__":
  main()
