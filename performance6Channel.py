import matplotlib.pyplot as plt
import torch

def plot_training_history(train_loss_history,
                          val_loss_history,
                          val_precision_history,
                          val_recall_history,
                          val_f1_history):
  epochs = range(1, len(train_loss_history) + 1)

  plt.figure(figsize=(16, 5))

  plt.subplot(1, 2, 1)
  plt.plot(epochs, train_loss_history, label="Train Loss", marker="o")
  plt.plot(epochs, val_loss_history, label="Val Loss", marker="o")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.title("Training & Validation Loss")
  plt.legend()
  plt.grid(True, linestyle="--", alpha=0.6)

  plt.subplot(1, 2, 2)
  plt.plot(epochs, val_precision_history, label="Precision", marker="o")
  plt.plot(epochs, val_recall_history, label="Recall", marker="o")
  plt.plot(epochs, val_f1_history, label="F1 Score", marker="o")
  plt.xlabel("Epoch")
  plt.ylabel("Score")
  plt.title("Validation Metrics")
  plt.ylim(0, 1.05)
  plt.legend()
  plt.grid(True, linestyle="--", alpha=0.6)

  plt.tight_layout()
  plt.show()

history_load = torch.load("model/training_history.pt")

train_loss_history = history_load["train_loss"]
val_loss_history = history_load["val_loss"]
val_precision_history = history_load["val_precision"]
val_recall_history = history_load["val_recall"]
val_f1_history = history_load["val_f1"]

plot_training_history(
  train_loss_history,
  val_loss_history,
  val_precision_history,
  val_recall_history,
  val_f1_history
)
