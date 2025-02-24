## 🛍️ Fashion Image Search Engine

🚀 *A deep learning-powered search engine that finds similar fashion items based on an input image.*

---

### 📌 1. Giới thiệu

Trong thời đại thương mại điện tử phát triển mạnh mẽ, người dùng thường gặp khó khăn khi tìm kiếm quần áo theo hình ảnh. Dự án này xây dựng một hệ thống **Fashion Image Search Engine** sử dụng **deep learning** để tìm kiếm các sản phẩm thời trang tương tự từ một kho dữ liệu lớn.

**🔹 Công nghệ chính sử dụng:**

- **Feature Extraction:** Sử dụng **ResNet / VGG / EfficientNet** để trích xuất đặc trưng hình ảnh.
- **Image Retrieval:** Dùng **cosine similarity** hoặc **FAISS (Facebook AI Similarity Search)** để tìm kiếm ảnh tương tự.
- **Web App:** Giao diện demo trực quan với **Streamlit**.

---

### 👥 2. Thành viên nhóm

- 🧑‍💻 **Trần Lê Bảo Trung** - *21521598*
- 🧑‍💻 **Nguyễn Công Nguyên** - *21521200*

---

### 📂 3. Cấu trúc thư mục

```
Fashion-Image-Search-Engine/
│── docs/               # Chứa tài liệu dự án (Slide + Report)
│── code/               # Source code chính
│   │── app.py          # Giao diện Streamlit
│   │── demo_utils.py   # Các hàm hỗ trợ demo
│   │── requirements.txt # Danh sách thư viện cần cài đặt
│── README.md           # Giới thiệu và hướng dẫn sử dụng
│── data.txt            # Link tải dataset
```

---

### 📦 4. Cài đặt & Chạy Demo

#### 🔹 Bước 1: Cài đặt thư viện

```bash
cd ./code
pip install -r requirements.txt
```

#### 🔹 Bước 2: Tải dataset & models

📥 **Tải và giải nén 3 thư mục** từ Google Drive:
[📌 Link tải models & features](https://drive.google.com/file/d/1rCl1C3zzfj_hWOtDCxE93r2U-aqoKhhV/view)

Sau khi tải về, **giải nén** và đặt 3 thư mục sau vào `code/`:

- `features/`
- `models/`
- `input/` (Chứa dataset Farfetch Listings)

#### 🔹 Bước 3: Chạy demo trên Streamlit

```bash
cd ./code
streamlit run app.py
```

📌 **Demo giao diện tìm kiếm thời trang:**\


---

### 📚 5. Tài liệu tham khảo

- [Paper về Image Retrieval](https://arxiv.org/pdf/2003.13035.pdf)
- [FAISS - Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)

---

💡 **Ghi chú:**

- Nếu dataset từ Kaggle không thể tải, bạn có thể sử dụng file `input.zip` có sẵn trong link drive.
- Model đã được train trước, không cần train lại.

🔗 **Github Project:** *[Link repo khi bạn push xong]*

---

💬 **Nếu bạn cần thêm thông tin hoặc có góp ý, hãy liên hệ với chúng tôi qua GitHub Issues! 🚀**
