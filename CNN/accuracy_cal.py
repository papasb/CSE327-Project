import re

input_path = '/home/mjkim2/Documents/Spike_Driven/CNN/output/0604x/log.txt'             # 입력 파일
output_path = '/home/mjkim2/Documents/Spike_Driven/CNN/output/0604x/0604acc.txt'   # 출력 파일

epoch_results = []  # (epoch, loss, avg_class_acc) 저장 리스트
current_epoch = None
current_class_accs = []
current_loss = None

with open(input_path, 'r') as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()

    # Epoch 시작 (예: Epoch 260/300)
    epoch_match = re.match(r"Epoch (\d+)/", line)
    if epoch_match:
        # 이전 에폭 결과 저장
        if current_epoch is not None and current_class_accs and current_loss is not None:
            avg_acc = sum(current_class_accs) / len(current_class_accs)
            epoch_results.append((current_epoch, current_loss, avg_acc))

        # 새로운 에폭 초기화
        current_epoch = int(epoch_match.group(1))
        current_class_accs = []
        current_loss = None

    # Loss 포함 줄 (예: Epoch 260: Loss=0.0188, Top1=...)
    loss_match = re.search(r"Loss=([\d.]+)", line)
    if loss_match:
        current_loss = float(loss_match.group(1))

    # 클래스별 정확도 줄 (예: Class 0 Acc: 0.6070)
    class_match = re.match(r"Class \d+ Acc: ([\d.]+)", line)
    if class_match:
        acc = float(class_match.group(1))
        current_class_accs.append(acc)

# 마지막 에폭 저장
if current_epoch is not None and current_class_accs and current_loss is not None:
    avg_acc = sum(current_class_accs) / len(current_class_accs)
    epoch_results.append((current_epoch, current_loss, avg_acc))

# 최고 평균 정확도 에폭 찾기
best_epoch, _, best_acc = max(epoch_results, key=lambda x: x[2])

# 파일 출력
with open(output_path, 'w') as f:
    for epoch, loss, acc in epoch_results:
        f.write(f"Epoch {epoch}: Loss = {loss:.4f}, Avg Class Accuracy = {acc:.4f}\n")
    f.write("\n")
    f.write(f"Best Epoch: {best_epoch}\n")
    f.write(f"Best Avg Class Accuracy: {best_acc:.4f}\n")

print(f"✅ 결과 저장 완료: {output_path}")
