from spikeformer import SpikeDrivenTransformer

# depths를 리스트로 지정 (예: [4,4,4,4] 또는 네트워크에 맞게)
depths = [4, 4, 4, 4]  # 예시: 4개의 블록, 각 4층

model = SpikeDrivenTransformer(
    img_size_h=128,
    img_size_w=128,
    in_channels=2,
    num_classes=11,
    depths=depths
)

params = sum(p.numel() for p in model.parameters())
print(f"Spike-Driven Transformer Parameter Count: {params}")
