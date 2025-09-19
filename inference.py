import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as T

model = fasterrcnn_resnet50_fpn_v2(weights=None)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

checkpoint = torch.load("model/best.pt", map_location=torch.device("cpu"))                          ### Model locatin & device
model.load_state_dict(checkpoint["model"])
model.eval()

img_path = "/content/drive/MyDrive/dataset_chicken/images/images (1).bmp"                           ### path gambar test
img = Image.open(img_path).convert("RGB")
transform = T.Compose([T.ToTensor()])
img_tensor = transform(img)

with torch.inference_mode():
  prediction = model([img_tensor])

boxes = prediction[0]["boxes"]
labels = prediction[0]["labels"]
scores = prediction[0]["scores"]

threshold = 0.5                                 ### threshold
keep = scores >= threshold
boxes = boxes[keep]
labels = labels[keep]
scores = scores[keep]

nms_iou_threshold = 0.3                         ### nms threshold
keep_idx = nms(boxes, scores, nms_iou_threshold)

boxes = boxes[keep_idx]
labels = labels[keep_idx]
scores = scores[keep_idx]

fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(img)

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

plt.title("Hasil Deteksi Faster R-CNN")
plt.show()
