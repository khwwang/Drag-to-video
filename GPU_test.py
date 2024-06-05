import torch
import torchaudio
import torchvision
import torch.nn.functional as F


############## cuda & cudatoolkit설치 확인 부분################
print(torch.__version__)
print(torchaudio.__version__)
print(torchvision.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())


############## cuDNN설치 확인 위한 부분################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 입력 이미지와 커널 생성
src = torch.randn(1, 1, 32, 32).to(device)
kernel = torch.tensor([[-1, -2, -1], [-1, 10, -1], [-1, -2, -1]], dtype=torch.float).unsqueeze(0).unsqueeze(0).to(device)

# 컨볼루션 연산 수행
out = F.conv2d(src, kernel, padding=1)

print(out.shape)