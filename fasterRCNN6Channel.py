###-----    Library & Depedency     -----###

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import torchvision.transforms as T

import random
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
from timeit import default_timer as timer
import matplotlib.patches as patches

CLASS_NAMES = ["__background__", "chicken"]

###-----            Main            -----###

def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Model run by {device}")

  data_root = Path("/content/drive/MyDrive/dataset6c") ########## CHANGE PATH ############
  nimg_dir = data_root / "images/normal"
  timg_dir = data_root / "images/thermal"
  ann_dir = data_root / "annotations"

  TRAIN_SPLIT = 0.8
  BATCH_SIZE = 4
  NUM_WORKERS = os.cpu_count()                           # os.cpu_count()
  LEARNING_RATE = 0.005
  EPOCHS = 1
  SCALER_TOGGLE = False                      # "store_true"
  MOMENTUM = 0.9
  WEIGHT_DECAY = 5e-4
  MAX_NORM = 0.0
  RESUME = ""       #"/content/runs/checkpoints/best.pt"                               # path to resume

  ann_stems = {p.stem for p in ann_dir.glob("*.xml")}
  nimg_stems = {p.stem for p in nimg_dir.iterdir() if p.is_file()}
  timg_stems = {p.stem for p in timg_dir.iterdir() if p.is_file()}

  ids = sorted(list(ann_stems.intersection(nimg_stems)))

  random.shuffle(ids)
  n_train = int(len(ids) * TRAIN_SPLIT)
  train_ids = ids[:n_train]
  test_ids = ids[n_train:]

  train_set = VOCDataset(data_root, train_ids, augment=True)
  test_set = VOCDataset(data_root, test_ids, augment=False)

  train_loader = DataLoader(train_set,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=NUM_WORKERS,
                            collate_fn=collate_fn)
  test_loader = DataLoader(test_set,
                          batch_size=BATCH_SIZE,
                          shuffle=False,
                          num_workers=NUM_WORKERS,
                          collate_fn=collate_fn)

  print(f"Length of train dataloader: {len(train_loader)} batches of {BATCH_SIZE}")
  print(f"Length of test dataloader: {len(test_loader)} batches of {BATCH_SIZE}")

  model = create_model(num_classes=2)
  model = adjust_model_for_six_channels(model)
  model.to(device)

  for name, param in model.backbone.body.named_parameters():                          ### freeze some backbone layer except conv1 layer 1
    if not name.startswith("conv1") and not name.startswith("layer1"):
      param.requires_grad = False

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
  ckpt_dir.mkdir(parents=True, exist_ok=True)

  train_time_start = timer()

  train_loss_history = []
  val_loss_history, val_precision_history, val_recall_history, val_f1_history = [], [], [], []

  for epoch in tqdm(range(start_epoch, EPOCHS)):
    print(f"Epoch: {epoch + 1}\n---------")
    train_loss = train_model(
      model, optimizer, train_loader, device, scaler, max_norm=MAX_NORM
    )

    val_loss, val_prec, val_rec, val_f1 = validate_model(
      model, test_loader, device
    )
    lr_sched.step()

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    val_precision_history.append(val_prec)
    val_recall_history.append(val_rec)
    val_f1_history.append(val_f1)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"train loss: {train_loss:.4f} | "
          f"val loss: {val_loss:.4f} | P: {val_prec:.3f} | R: {val_rec:.3f} | F1: {val_f1:.3f}")

    # Save latest
    save_checkpoint({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_sched": lr_sched.state_dict(),
        "best_val": best_val,
    }, ckpt_dir / "last.pt")

    if val_loss < best_val:
      best_val = val_loss
      save_checkpoint({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_sched": lr_sched.state_dict(),
        "best_val": best_val,
      }, ckpt_dir / "best.pt")
      print(f"✅ Saved new best checkpoint: val_loss={best_val:.4f}")

  train_time_end = timer()
  model_time = print_train_time(start=train_time_start, end=train_time_end)

  # show_predictions_and_count(model, train_set, device, num_images=3, mode="thermal")         ### output inference test

  history = {
    "train_loss": train_loss_history,
    "val_loss": val_loss_history,
    "val_precision": val_precision_history,
    "val_recall": val_recall_history,
    "val_f1": val_f1_history
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

def print_train_time(start, end):
  print(f"Train time : {end - start} sec")
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
  def __init__(self, root: str, ids: list[str], augment: bool = False, img_size=(720, 1280)):
    self.root = Path(root)
    self.ids = ids
    self.augment = augment
    self.nimg_dir = self.root / "images/normal"
    self.timg_dir = self.root / "images/thermal"
    self.ann_dir = self.root / "annotations"
    self.img_size = img_size
    self.tf_resize = T.Resize(self.img_size)

    if augment:
      self.tf = T.Compose([
        T.ToTensor(),
      ])
      self.hflip_p = 0.5
      self.jitter_p = 0.5
      self.jitter = T.ColorJitter(brightness=0.5, contrast=0.5)
    else:
      self.tf = T.Compose([
        T.ToTensor(),
      ])
      self.hflip_p = 0.0
      self.jitter = None

  def __len__(self):
    return len(self.ids)

  def __getitem__(self, idx: int):
    base = self.ids[idx]
    stem = Path(base).stem

    nimg_path = self.nimg_dir / f"{stem}.jpg"
    timg_path = self.timg_dir / f"{stem}.jpg"
    ann_path = self.ann_dir / f"{stem}.xml"

    if not nimg_path.exists():
      raise FileNotFoundError(f"Gambar normal tidak ditemukan: {nimg_path}")
    if not timg_path.exists():
      raise FileNotFoundError(f"Gambar thermal tidak ditemukan: {timg_path}")
    if not ann_path.exists():
      raise FileNotFoundError(f"Annotation tidak ditemukan: {ann_path}")

    nimg = Image.open(nimg_path).convert("RGB")
    timg = Image.open(timg_path).convert("RGB")

    nimg = self.tf_resize(nimg)
    timg = self.tf_resize(timg)

    boxes, labels = parse_xml(ann_path)

    if len(boxes) == 0:
      boxes = torch.zeros((0, 4), dtype=torch.float32)
      labels = torch.zeros((0,), dtype=torch.int64)
    else:
      boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
      labels = torch.as_tensor(labels, dtype=torch.int64)

    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    target: dict[str, torch.Tensor] = {
      "boxes": boxes,
      "labels": labels,
      "image_id": torch.tensor([idx]),
      "area": area,
      "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.uint8),
    }

    if self.jitter is not None and random.random() < self.jitter_p:
      nimg = self.jitter(nimg)
      timg = self.jitter(timg)

    nimg = self.tf(nimg)
    timg = self.tf(timg)

    if boxes.numel() > 0:
      if random.random() < self.hflip_p:
        _, h, w = nimg.shape
        if random.random() < 0.5:
          nimg = torch.flip(nimg, dims=[2])
          timg = torch.flip(timg, dims=[2])
          x_min = w - boxes[:, 2]
          x_max = w - boxes[:, 0]
          boxes[:, 0] = x_min
          boxes[:, 2] = x_max
        else:
          nimg = torch.flip(nimg, dims=[1])
          timg = torch.flip(timg, dims=[1])
          y_min = h - boxes[:, 3]
          y_max = h - boxes[:, 1]
          boxes[:, 1] = y_min
          boxes[:, 3] = y_max
        target["boxes"] = boxes

    merged_img = torch.cat([nimg, timg], dim=0)
    return merged_img, target



###-----            Model           -----###

def create_model(num_classes: int = 2):                                             ### change backbone input channel
  model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone="DEFAULT")

  old_conv = model.backbone.body.conv1
  new_conv = nn.Conv2d(
    in_channels=6,
    out_channels=old_conv.out_channels,
    kernel_size=old_conv.kernel_size,
    stride=old_conv.stride,
    padding=old_conv.padding,
    bias=old_conv.bias is not None
  )

  with torch.no_grad():
    new_conv.weight[:, :3] = old_conv.weight
    new_conv.weight[:, 3:] = old_conv.weight

  model.backbone.body.conv1 = new_conv

  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

  return model

def adjust_model_for_six_channels(model):                                           ### adjust data mean std
  old_transform = model.transform

  new_mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
  new_std  = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225]

  model.transform = GeneralizedRCNNTransform(
    min_size=old_transform.min_size,
    max_size=old_transform.max_size,
    image_mean=new_mean,
    image_std=new_std
  )
  return model


###-----            Train           -----###

def train_model(model: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                loader: torch.utils.data.DataLoader,
                device: torch.device,
                scaler=None,
                max_norm: float = 0.0,
                log_interval: int = 10):
  model.train()
  total_loss = 0.0

  for batch, (images, targets) in enumerate(loader):
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

    if batch % log_interval == 0:
      print(f"Batch {batch}/{len(loader)} | Loss: {losses.item():.4f} | "
            f"Cls: {loss_dict['loss_classifier'].item():.4f}, "
            f"Box: {loss_dict['loss_box_reg'].item():.4f}")

  avg_loss = total_loss / max(1, len(loader))
  return avg_loss

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

@torch.inference_mode()
def validate_model(model: torch.nn.Module,
                   loader: torch.utils.data.DataLoader,
                   device: torch.device,
                   iou_threshold: float = 0.5,
                   score_threshold: float = 0.5):
  model.eval()
  total_loss = 0.0
  all_tp, all_fp, all_fn = 0, 0, 0
  n_batches = 0

  for images, targets in loader:
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    if any(len(t["boxes"]) > 0 for t in targets):
      loss_dict = model(images, targets)
      losses = sum(loss for loss in loss_dict.values())
      total_loss += losses.item()
      n_batches += 1

    preds = model(images)

    for pred, gt in zip(preds, targets):
      pred_boxes = []
      for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
        if score < score_threshold:
          continue
        pred_boxes.append([*box.tolist(), label.item()])

      gt_boxes = [[*box.tolist(), label.item()] for box, label in zip(gt["boxes"], gt["labels"])]

      matched_gt = set()
      for pb in pred_boxes:
        found_match = False
        for i, gb in enumerate(gt_boxes):
          if i in matched_gt:
            continue
          if pb[4] == gb[4] and iou(pb[:4], gb[:4]) >= iou_threshold:
            matched_gt.add(i)
            found_match = True
            break
        if found_match:
          all_tp += 1
        else:
          all_fp += 1
      all_fn += (len(gt_boxes) - len(matched_gt))

  precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
  recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
  f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
  avg_loss = total_loss / max(1, n_batches)

  print(f"[VAL] Loss: {avg_loss:.4f} | P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f}")
  return avg_loss, precision, recall, f1

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


def show_predictions_and_count(model, dataset, device, num_images=5, score_threshold=0.75, mode="normal"):
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

    if mode == "normal":
      img_np = img[:3].permute(1, 2, 0).cpu().numpy()
    elif mode == "thermal":
      img_np = img[3:6].permute(1, 2, 0).cpu().numpy()
    else:
      raise ValueError("mode harus 'normal' atau 'thermal'")

    _, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img_np)

    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
      x1, y1, x2, y2 = box
      rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2, edgecolor="r", facecolor="none"
      )
      ax.add_patch(rect)
      ax.text(x1, y1 - 5, f"{label}:{score:.2f}", color="yellow", fontsize=10, bbox=dict(facecolor="red", alpha=0.5))

    ax.set_title(f"Objects detected: {total_detected}", fontsize=14, color="blue")
    plt.show()

    print(f"Gambar {idx} → Terdeteksi {total_detected} objek (threshold={score_threshold})")



if __name__ == "__main__":
  main()
