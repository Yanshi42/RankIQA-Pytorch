import os
import json

# 定义图片所在文件夹路径
image_folder = '/home/color/COCO/test2017/'

# 获取文件夹下所有图片文件（假设图片以 .jpg 结尾）
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

# 创建一个列表来存储所有的字典数据
data = []

# 遍历图片文件，生成每张图片对应的四行数据
for img in image_files:
    base_name = img.split('.')[0]  # 获取图片文件名（不包含后缀）

    # 对每个图片生成四行数据
    data.append({
        "img1_path": f"/home/color/COCO/data_generation/artifact/artifact-5-100-130-3-10000-5/pic/{base_name}_superpixel_1.jpg",
        "img2_path": f"/home/color/COCO/data_generation/artifact/artifact-15-100-130-3-10000-5/pic/{base_name}_superpixel_1.jpg",
        "label": 1
    })
    data.append({
        "img1_path": f"/home/color/COCO/data_generation/artifact/artifact-15-100-130-3-10000-5/pic/{base_name}_superpixel_1.jpg",
        "img2_path": f"/home/color/COCO/data_generation/artifact/artifact-25-100-130-3-10000-5/pic/{base_name}_superpixel_1.jpg",
        "label": 1
    })
    data.append({
        "img1_path": f"/home/color/COCO/data_generation/artifact/artifact-25-100-130-3-10000-5/pic/{base_name}_superpixel_1.jpg",
        "img2_path": f"/home/color/COCO/data_generation/artifact/artifact-35-100-130-3-10000-5/pic/{base_name}_superpixel_1.jpg",
        "label": 1
    })
    data.append({
        "img1_path": f"/home/color/COCO/data_generation/artifact/artifact-35-100-130-3-10000-5/pic/{base_name}_superpixel_1.jpg",
        "img2_path": f"/home/color/COCO/data_generation/artifact/artifact-45-100-130-3-10000-5/pic/{base_name}_superpixel_1.jpg",
        "label": 1
    })

# 将数据保存为JSON格式
output_file = 'train_data_a.json'
with open(output_file, 'w') as f:
    json.dump(data, f, indent=4)

print(f"JSON file '{output_file}' has been created successfully.")
