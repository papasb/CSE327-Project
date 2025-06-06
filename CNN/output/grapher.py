import re
import matplotlib.pyplot as plt

log_path = '/home/mjkim2/Documents/Spike_Driven/CNN/output/0603/0603acc.txt'  # ← 네 로그 텍스트 파일 이름

# 저장할 리스트 초기화
epochs = []
losses = []
accuracies = []

with open(log_path, 'r') as f:
    for line in f:
        match = re.match(r"Epoch (\d+): Loss = ([\d.]+), Avg Class Accuracy = ([\d.]+)", line.strip())
        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))
            acc = float(match.group(3)) * 100  # 퍼센트로 변환 (선택적)

            epochs.append(epoch)
            losses.append(loss)
            accuracies.append(acc)

# 📉 Loss Plot
plt.figure(figsize=(10, 4))
plt.plot(epochs, losses, marker='o', label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('loss_from_txt_plot.png')
plt.show()

# 📈 Accuracy Plot
plt.figure(figsize=(10, 4))
plt.plot(epochs, accuracies, marker='o', color='green', label='Avg Class Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Average Class Accuracy per Epoch')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('accuracy_from_txt_plot.png')
plt.show()
