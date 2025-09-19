import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as T

def create_model(num_classes: int = 2):                                             ### change backbone input channel
  model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone="DEFAULT")

  old_conv = model.backbone.body.conv1
  model.backbone.body.conv1 = nn.Conv2d(
    in_channels=6,
    out_channels=old_conv.out_channels,
    kernel_size=old_conv.kernel_size,
    stride=old_conv.stride,
    padding=old_conv.padding,
    bias=old_conv.bias is not None
  )

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

model = create_model(num_classes=2)
model = adjust_model_for_six_channels(model)
model.to("cuda")

checkpoint = torch.load("model/best.pt", map_location=torch.device("cuda"))
model.load_state_dict(checkpoint["model"])
model.eval()

nimg_path = "/content/drive/MyDrive/dataset6c/images/normal/1035s.jpg"
timg_path = "/content/drive/MyDrive/dataset6c/images/thermal/1035s.jpg"
nimg = Image.open(nimg_path).convert("RGB")
timg = Image.open(timg_path).convert("RGB")
transform = T.Compose([T.ToTensor()])
nimg_tensor = transform(nimg)
timg_tensor = transform(timg)

img_tensor_6ch = torch.cat([nimg_tensor, timg_tensor], dim=0).to("cuda")  # (6,H,W)

with torch.inference_mode():
  prediction = model([img_tensor_6ch])

boxes = prediction[0]["boxes"]
labels = prediction[0]["labels"]
scores = prediction[0]["scores"]

threshold = 0.5
keep = scores >= threshold
boxes = boxes[keep]
labels = labels[keep]
scores = scores[keep]

nms_iou_threshold = 0.3
keep_idx = nms(boxes, scores, nms_iou_threshold)

boxes = boxes[keep_idx]
labels = labels[keep_idx]
scores = scores[keep_idx]

fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(timg)

for box, label, score in zip(boxes, labels, scores):
  x1, y1, x2, y2 = box
  rect = patches.Rectangle(
    (x1, y1), x2 - x1, y2 - y1,
    linewidth=2, edgecolor="red", facecolor="none"
  )
  ax.add_patch(rect)
  ax.text(
    x1, y1, f"{label.item()} ({score:.2f})",
    bbox=dict(facecolor="yellow", alpha=0.5),
    fontsize=10, color="black"
  )

plt.title("Hasil Deteksi Faster R-CNN 6 Channel")
plt.show()
