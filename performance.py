import matplotlib.pyplot as plt
import torch

history_load = torch.load("model/training_history.pt")                                            ### Training history location

train_loss_history = history_load["train_loss"]
train_acc_history  = history_load["train_acc"]
test_loss_history  = history_load["test_loss"]
test_acc_history   = history_load["test_acc"]

epochs = range(1, len(train_loss_history) + 1)
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(epochs, train_loss_history, label="Train Loss", marker="o")
plt.plot(epochs, test_loss_history, label="Test Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Test Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, train_acc_history, label="Train Acc", marker="o")
plt.plot(epochs, test_acc_history, label="Test Acc", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Test Accuracy")
plt.legend()

plt.show()
