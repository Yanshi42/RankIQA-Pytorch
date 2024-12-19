import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from rankiqa.rankiqa import RankIQA
from dataset import RankIQADataset
from torchvision import transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='./runs/RankIQA')  # 日志文件会保存在 './runs/RankIQA'

class Config:
    def __init__(self):
        self.gpu = torch.cuda.is_available()
        self.batch_size = 32
        self.num_epochs = 50
        self.learning_rate = 1e-4
        self.pretrained_model_path = './pretrain/Rank_live.caffemodel.pt'
        self.train_data = './train_data_a.json'

opt = Config()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = RankIQADataset(json_file=opt.train_data, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)

model = RankIQA(opt)
print("model loaded successfully.")
if opt.gpu:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)

for epoch in range(opt.num_epochs):
    model.train()
    total_loss = 0

    # 使用 tqdm 显示进度条
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{opt.num_epochs}", unit="batch") as pbar:
        for img1, img2, labels in pbar:
            if opt.gpu:
                img1, img2, labels = img1.cuda(), img2.cuda(), labels.cuda()

            feat1, feat2 = model(img1, img2)

            loss = model.ranking_loss(feat1, feat2, labels)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新进度条的描述
            pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    writer.add_scalar('Loss/train', avg_loss, epoch)  # 记录损失

    print(f"Epoch {epoch+1}/{opt.num_epochs}, Loss: {avg_loss}")

    # model.eval()  # 评估模式
    # with torch.no_grad():
    #     val_loss = 0
    #     # val_loss = ...

torch.save(model.state_dict(), 'rankiqa_model_pre.pth')

writer.close()
