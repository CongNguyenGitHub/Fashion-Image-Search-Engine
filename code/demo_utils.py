import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from torchvision.models import vgg16, resnet50, inception_v3
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import cv2
from random import sample
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import random
import warnings
import time
import faiss

warnings.filterwarnings("ignore")


class CustomVGG(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=1000):
        """
        Tạo mô hình VGG16 tùy chỉnh trong PyTorch.

        Args:
        - input_shape: Kích thước đầu vào của ảnh (mặc định: (3, 224, 224)).
        - num_classes: Số lớp đầu ra cho bài toán phân loại.

        Returns:
        - model: Mô hình VGG16 tùy chỉnh.
        """
        super(CustomVGG, self).__init__()

        # Tải mô hình VGG16 có pretrained trên ImageNet
        self.base_model = vgg16(pretrained=True)

        # Freeze các lớp trong base model
        for param in self.base_model.features.parameters():
            param.requires_grad = False

        # Thay thế phần classifier bằng các lớp tùy chỉnh
        self.base_model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1028),  # Lớp đầu tiên
            nn.ReLU(inplace=True),  # Hàm kích hoạt ReLU
            nn.Dropout(p=0.75, inplace=False),
            nn.Linear(in_features=1028, out_features=1028, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7, inplace=False),
            nn.Linear(1028, num_classes),  # Lớp đầu ra
        )

    def forward(self, x):
        """
        Định nghĩa forward pass qua mạng.

        Args:
        - x: Tensor đầu vào.

        Returns:
        - out: Tensor đầu ra.
        """
        out = self.base_model(x)
        return out


class CustomResNet50(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=1000):
        """
        Tạo mô hình ResNet50 tùy chỉnh trong PyTorch.

        Args:
        - input_shape: Kích thước đầu vào của ảnh (mặc định: (3, 224, 224)).
        - num_classes: Số lớp đầu ra cho bài toán phân loại.

        Returns:
        - model: Mô hình ResNet50 tùy chỉnh.
        """
        super(CustomResNet50, self).__init__()

        # Tải mô hình ResNet50 đã pretrained trên ImageNet
        self.base_model = resnet50(pretrained=True)

        # Freeze từng lớp cụ thể
        for name, child in self.base_model.named_children():
            if name in ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2"]:
                for param in child.parameters():
                    param.requires_grad = False  # Đóng băng các lớp này

            if name in ["layer4", "avgpool", "fc"]:
                for param in child.parameters():
                    param.requires_grad = True  # Các lớp này được train lại

        # Sửa đổi lớp fully connected (fc)
        self.base_model.fc = nn.Sequential(
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        """
        Định nghĩa forward pass qua mạng.

        Args:
        - x: Tensor đầu vào.

        Returns:
        - out: Tensor đầu ra.
        """
        out = self.base_model(x)
        return out


class CustomInceptionV3(nn.Module):
    def __init__(
        self,
        input_shape=(3, 299, 299),
        num_classes=1000,
        fine_tune_layers=["Mixed_7b", "Mixed_7c"],
    ):
        """
        Tạo mô hình Inception_v3 tùy chỉnh với khả năng fine-tuning.

        Args:
        - input_shape: Kích thước đầu vào của ảnh (mặc định: (3, 299, 299)).
        - num_classes: Số lớp đầu ra cho bài toán phân loại.
        - fine_tune_layers: Danh sách các layer cần fine-tuning.

        Returns:
        - model: Mô hình Inception_v3 tùy chỉnh.
        """
        super(CustomInceptionV3, self).__init__()

        # Tải mô hình Inception_v3 pretrained trên ImageNet
        self.base_model = inception_v3(pretrained=True, transform_input=False)

        # Freeze tất cả các lớp
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Cho phép fine-tuning các lớp được chỉ định
        for name, module in self.base_model.named_modules():
            if name in fine_tune_layers:
                for param in module.parameters():
                    param.requires_grad = True

        # Thay thế lớp `avgpool` và `fc`
        self.base_model.fc = nn.Sequential(nn.Linear(2048, num_classes))

        # Lưu kích thước đầu vào
        self.input_shape = input_shape

    def forward(self, x):
        """
        Định nghĩa forward pass qua mạng.

        Args:
        - x: Tensor đầu vào.

        Returns:
        - out: Tensor đầu ra.
        """
        # Resize ảnh về kích thước yêu cầu (299x299)
        x = F.interpolate(
            x, size=self.input_shape[1:], mode="bilinear", align_corners=False
        )
        # Forward qua mạng
        out = self.base_model(x)
        return out


vgg_model = CustomVGG(num_classes=37)
resnet_model = CustomResNet50(num_classes=37)
inception_model = CustomInceptionV3(num_classes=37)

vgg_model.load_state_dict(
    torch.load(
        "models/vgg_model/best_model.pt",
        weights_only=True,
        map_location=torch.device("cpu"),
    )
)
resnet_model.load_state_dict(
    torch.load(
        "models/resnet_model/best_model.pt",
        weights_only=True,
        map_location=torch.device("cpu"),
    )
)
inception_model.load_state_dict(
    torch.load(
        "models/inception_model/best_model.pt",
        weights_only=True,
        map_location=torch.device("cpu"),
    )
)


def load_features(file_path):
    """
    Hàm để đọc đặc trưng, nhãn và đường dẫn từ file .npz.

    Args:
        file_path (str): Đường dẫn tới file .npz đã lưu.

    Returns:
        features (numpy.ndarray): Mảng chứa đặc trưng của ảnh.
        labels (numpy.ndarray): Mảng chứa nhãn của ảnh.
        paths (numpy.ndarray): Mảng chứa đường dẫn tới ảnh.
    """
    try:
        # Tải dữ liệu từ file .npz
        data = np.load(file_path)

        # Lấy các thành phần
        features = data["features"]
        labels = data["labels"]
        paths = data["paths"]

        print(f"Loaded data from {file_path}")
        print(f"Features shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Paths shape: {paths.shape}")

        return features, labels, paths

    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None, None, None


# Đường dẫn tới các file đặc trưng
vgg_file = "features/cleaned_vgg_features.npz"
inception_file = "features/cleaned_inception_features.npz"
resnet_file = "features/cleaned_resnet_features.npz"
# Đọc file đặc trưng cho VGG
vgg_features, vgg_labels, vgg_paths = load_features(vgg_file)

# Đọc file đặc trưng cho Inception
inception_features, inception_labels, inception_paths = load_features(inception_file)

# Đọc file đặc trưng cho ResNet
resnet_features, resnet_labels, resnet_paths = load_features(resnet_file)


def find_top_k_images_vgg_faiss(
    pil_image,
    model,
    gallery_features,
    gallery_labels,
    gallery_paths,
    top_k=5,
    device="cpu",
):
    """
    Tìm top-k ảnh giống nhất từ một ảnh đầu vào sử dụng FAISS.

    Args:
        pil_image: Ảnh đầu vào dạng PIL Image.
        model: Mô hình trích xuất đặc trưng (VGG16 hoặc tương tự).
        gallery_features: Đặc trưng của gallery đã được trích xuất trước.
        gallery_labels: Nhãn tương ứng với gallery_features.
        gallery_paths: Đường dẫn tương ứng với gallery_features.
        top_k: Số lượng kết quả cần tìm.
        device: Thiết bị sử dụng (mặc định là 'cpu').

    Returns:
        top_k_images: Danh sách các ảnh trong top-k (dạng PIL Image).
        top_k_labels: Nhãn tương ứng với các ảnh trong top-k.
        top_k_paths: Đường dẫn tương ứng với các ảnh trong top-k.
    """
    # Biến đổi ảnh PIL thành tensor phù hợp với mô hình
    transform_data = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize về kích thước tiêu chuẩn của VGG
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = transform_data(pil_image).unsqueeze(0).to(device)

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        # Forward qua mô hình để trích xuất đặc trưng ảnh đầu vào
        feature_maps = model.base_model.features(input_tensor)
        pooled_features = model.base_model.avgpool(feature_maps)
        query_features = torch.flatten(pooled_features, 1)
        query_features = model.base_model.classifier[:6](query_features).cpu().numpy()

    # FAISS: Khởi tạo chỉ mục FAISS cho phép tìm kiếm
    index = faiss.IndexFlatL2(query_features.shape[1])  # L2 distance (Euclidean)
    index.add(gallery_features.astype(np.float32))  # Thêm gallery features vào index

    # Sử dụng FAISS để tìm top-k ảnh gần nhất
    D, I = index.search(query_features.astype(np.float32), top_k)

    # Lấy các nhãn và đường dẫn tương ứng với top-k kết quả
    top_k_labels = gallery_labels[I[0]]
    top_k_paths = gallery_paths[I[0]]

    # Lấy ảnh từ các đường dẫn trong gallery_paths
    top_k_images = [Image.open(path).convert("RGB") for path in top_k_paths]

    return top_k_images, top_k_labels


def find_top_k_images_resnet_faiss(
    pil_image,
    model,
    gallery_features,
    gallery_labels,
    gallery_paths,
    top_k=5,
    device="cpu",
):
    """
    Tìm top-k ảnh giống nhất từ một ảnh đầu vào sử dụng FAISS.

    Args:
        pil_image: Ảnh đầu vào dạng PIL Image.
        model: Mô hình trích xuất đặc trưng (ResNet50 hoặc tương tự).
        gallery_features: Đặc trưng của gallery đã được trích xuất trước.
        gallery_labels: Nhãn tương ứng với gallery_features.
        gallery_paths: Đường dẫn tương ứng với gallery_features.
        top_k: Số lượng kết quả cần tìm.
        device: Thiết bị sử dụng (mặc định là 'cpu').

    Returns:
        top_k_images: Danh sách các ảnh trong top-k (dạng PIL Image).
        top_k_labels: Nhãn tương ứng với các ảnh trong top-k.
        top_k_paths: Đường dẫn tương ứng với các ảnh trong top-k.
    """
    # Biến đổi ảnh PIL thành tensor phù hợp với mô hình
    transform_data = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Kích thước chuẩn cho ResNet
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = transform_data(pil_image).unsqueeze(0).to(device)

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        # Forward pass qua mô hình để trích xuất đặc trưng
        x = model.base_model.conv1(input_tensor)  # Lớp convolution đầu tiên
        x = model.base_model.bn1(x)  # BatchNorm đầu tiên
        x = model.base_model.relu(x)  # ReLU activation
        x = model.base_model.maxpool(x)  # Maxpooling

        # Các ResNet blocks (layer1, layer2, layer3, layer4)
        x = model.base_model.layer1(x)
        x = model.base_model.layer2(x)
        x = model.base_model.layer3(x)
        x = model.base_model.layer4(x)

        # Global Average Pooling
        x = model.base_model.avgpool(x)
        x = torch.flatten(x, 1)  # Chuyển đổi thành vector 1D

        query_features = x.cpu().numpy()  # Chuyển sang numpy

    # FAISS: Khởi tạo chỉ mục FAISS cho phép tìm kiếm
    index = faiss.IndexFlatL2(query_features.shape[1])  # L2 distance (Euclidean)
    index.add(gallery_features.astype(np.float32))  # Thêm gallery features vào index

    # Sử dụng FAISS để tìm top-k ảnh gần nhất
    D, I = index.search(query_features.astype(np.float32), top_k)

    # Lấy các nhãn và đường dẫn tương ứng với top-k kết quả
    top_k_labels = gallery_labels[I[0]]
    top_k_paths = gallery_paths[I[0]]

    # Lấy ảnh từ các đường dẫn trong gallery_paths
    top_k_images = [Image.open(path).convert("RGB") for path in top_k_paths]

    return top_k_images, top_k_labels


def find_top_k_images_inception_faiss(
    pil_image,
    model,
    gallery_features,
    gallery_labels,
    gallery_paths,
    top_k=5,
    device="cpu",
):
    """
    Tìm top-k ảnh giống nhất từ một ảnh đầu vào sử dụng FAISS.

    Args:
        pil_image: Ảnh đầu vào dạng PIL Image.
        model: Mô hình trích xuất đặc trưng (InceptionV3 hoặc tương tự).
        gallery_features: Đặc trưng của gallery đã được trích xuất trước.
        gallery_labels: Nhãn tương ứng với gallery_features.
        gallery_paths: Đường dẫn tương ứng với gallery_features.
        top_k: Số lượng kết quả cần tìm.
        device: Thiết bị sử dụng (mặc định là 'cpu').

    Returns:
        top_k_images: Danh sách các ảnh trong top-k (dạng PIL Image).
        top_k_labels: Nhãn tương ứng với các ảnh trong top-k.
        top_k_paths: Đường dẫn tương ứng với các ảnh trong top-k.
    """
    # Biến đổi ảnh PIL thành tensor phù hợp với mô hình
    transform_data = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Kích thước chuẩn cho ResNet
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = transform_data(pil_image).unsqueeze(0).to(device)

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        # Resize ảnh nếu cần (Inception yêu cầu đầu vào 299x299)
        inputs = F.interpolate(
            input_tensor, size=(299, 299), mode="bilinear", align_corners=False
        )

        # Forward qua các lớp ban đầu của Inception
        x = model.base_model.Conv2d_1a_3x3(inputs)
        x = model.base_model.Conv2d_2a_3x3(x)
        x = model.base_model.Conv2d_2b_3x3(x)
        x = model.base_model.maxpool1(x)

        x = model.base_model.Conv2d_3b_1x1(x)
        x = model.base_model.Conv2d_4a_3x3(x)
        x = model.base_model.maxpool2(x)

        # Qua các khối Inception
        x = model.base_model.Mixed_5b(x)
        x = model.base_model.Mixed_5c(x)
        x = model.base_model.Mixed_5d(x)
        x = model.base_model.Mixed_6a(x)
        x = model.base_model.Mixed_6b(x)
        x = model.base_model.Mixed_6c(x)
        x = model.base_model.Mixed_6d(x)
        x = model.base_model.Mixed_6e(x)
        x = model.base_model.Mixed_7a(x)
        x = model.base_model.Mixed_7b(x)
        x = model.base_model.Mixed_7c(x)

        # Global average pooling
        x = model.base_model.avgpool(x)
        x = torch.flatten(x, 1)  # Chuyển đổi thành vector 1D

        query_features = x.cpu().numpy()  # Chuyển sang numpy

    # FAISS: Khởi tạo chỉ mục FAISS cho phép tìm kiếm
    index = faiss.IndexFlatL2(query_features.shape[1])  # L2 distance (Euclidean)
    index.add(gallery_features.astype(np.float32))  # Thêm gallery features vào index

    # Sử dụng FAISS để tìm top-k ảnh gần nhất
    D, I = index.search(query_features.astype(np.float32), top_k)

    # Lấy các nhãn và đường dẫn tương ứng với top-k kết quả
    top_k_labels = gallery_labels[I[0]]
    top_k_paths = gallery_paths[I[0]]

    # Lấy ảnh từ các đường dẫn trong gallery_paths
    top_k_images = [Image.open(path).convert("RGB") for path in top_k_paths]

    return top_k_images, top_k_labels


def plot_top_k_images(top_k_images, top_k_labels, top_k=10):
    """
    Hiển thị top-k ảnh với nhãn và đường dẫn tương ứng.

    Args:
        top_k_images: Danh sách các ảnh (dạng PIL Image).
        top_k_labels: Danh sách các nhãn tương ứng với các ảnh.
        top_k: Số lượng ảnh cần hiển thị (mặc định là 10).

    Returns:
        None
    """
    # Tạo một figure với lưới 2x5 (10 ảnh)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()

    for i in range(top_k):
        # Lấy ảnh, nhãn và đường dẫn
        img = top_k_images[i]
        label = top_k_labels[i]

        # Hiển thị ảnh
        axes[i].imshow(img)
        axes[i].set_title(f"Label: {label}")
        axes[i].axis("off")  # Tắt trục

    # Hiển thị tất cả ảnh
    plt.tight_layout()
    plt.show()


pil_image = Image.open(vgg_paths[1])
pil_image

top_k_images, top_k_labels = find_top_k_images_vgg_faiss(
    pil_image, vgg_model, vgg_features, vgg_labels, vgg_paths, top_k=10
)

plot_top_k_images(top_k_images, top_k_labels)

top_k_images, top_k_labels = find_top_k_images_resnet_faiss(
    pil_image, resnet_model, resnet_features, resnet_labels, resnet_paths, top_k=10
)

plot_top_k_images(top_k_images, top_k_labels)

top_k_images, top_k_labels = find_top_k_images_inception_faiss(
    pil_image,
    inception_model,
    inception_features,
    inception_labels,
    inception_paths,
    top_k=10,
)

plot_top_k_images(top_k_images, top_k_labels)
