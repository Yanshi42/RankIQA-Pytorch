import torch
from rankiqa.rankiqa import RankIQA
from torchvision import transforms
from PIL import Image

# 加载配置
class Config:
    def __init__(self):
        self.gpu = torch.cuda.is_available()
        self.batch_size = 32
        self.num_epochs = 50
        self.learning_rate = 1e-4
        self.pretrained_model_path = '/home/color/myRankIQA/rankiqa_model.pth'
        self.train_data = './train_data.json'

opt = Config()

# 初始化模型
model = RankIQA(opt)

# 加载模型权重
model.load_state_dict(torch.load('rankiqa_model.pth'))

# 设置为评估模式（避免 Dropout 等操作）
model.eval()

# 如果有 GPU，转移到 GPU 上
if opt.gpu:
    model.cuda()

# 定义预处理（与训练时相同）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 加载图片
img_path1 = '/home/color/COCO/data_generation/saturation/saturation-global-0.7-0.5-7-0.8-2/pic/000000000009.jpg'  # 替换为你图片的路径
img_path2 = '/home/color/COCO/data_generation/saturation/saturation-global-0.3-0.5-7-0.8-2/pic/000000000009.jpg'  # 替换为你图片的路径
image1 = Image.open(img_path1).convert('RGB')  # 加载图片并转换为 RGB 格式
image2 = Image.open(img_path2).convert('RGB')  # 加载图片并转换为 RGB 格式

# 应用预处理
input_image1 = transform(image1).unsqueeze(0)
input_image2 = transform(image2).unsqueeze(0)
if opt.gpu:
    input_image1 = input_image1.cuda()
    input_image2 = input_image2.cuda()

# 进行预测
with torch.no_grad():  # 关闭梯度计算（用于推理）
    feat1, feat2 = model(input_image1, input_image2)  # 使用两张相同的图片进行比较（示例）
    print(feat1, feat2)  # 输出特征

    # 计算欧式距离
    distance = feat1 - feat2
    print(f"Distance between the two features: {distance.item()}")
