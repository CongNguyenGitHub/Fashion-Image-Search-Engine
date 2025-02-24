## 🛍️ Fashion Image Search Engine  

🚀 *A deep learning-powered search engine that finds similar fashion items based on an input image.*

---

### 📌 1. Introduction  

In the rapidly growing era of e-commerce, users often struggle to find clothing items based on images. This project builds a **Fashion Image Search Engine** using **deep learning** to search for similar fashion products from a large dataset.

**🔹 Key Technologies Used:**

- **Feature Extraction:** Utilizes **ResNet50 / VGG16 / InceptionV3** for extracting image features.
- **Image Retrieval:** Implements **FAISS (Facebook AI Similarity Search)** for fast and efficient similarity search.
- **Web App:** User-friendly demo interface powered by **Streamlit**.

---

### 👥 2. Team Members  

- 🧑‍💻 **Trần Lê Bảo Trung** - *21521598*  
- 🧑‍💻 **Nguyễn Công Nguyên** - *21521200*  

---

### 📂 3. Directory Structure  

```
Fashion-Image-Search-Engine/
│── docs/               # Project documentation (Slides + Report)
│── code/               # Main source code
│   │── app.py          # Streamlit interface
│   │── demo_utils.py   # Utility functions for demo
│   │── requirements.txt # Required libraries
│── README.md           # Project overview & instructions
│── data.txt            # Dataset download link
│── models.txt          # Pretrained model download link
│── features.txt        # Feature embeddings download link
```

---

### 📦 4. Installation & Running the Demo  

#### 🔹 Step 1: Install Dependencies  

```bash
cd ./code
pip install -r requirements.txt
```

#### 🔹 Step 2: Download Dataset, Models, and Features  

📌 The datasets are too large to be stored on GitHub. Please download them from the following links:

- 📥 **Dataset:** [Farfetch Listings Dataset](https://www.kaggle.com/datasets/alvations/farfetch-listings) *(or use `input.zip` in the provided Google Drive link)*
- 📥 **Pretrained Models:** Refer to `models.txt` for the download link.
- 📥 **Feature Embeddings:** Refer to `features.txt` for the download link.

After downloading, extract and place the following folders inside `code/`:

- `features/`
- `models/`
- `input/` (Contains the Farfetch Listings dataset)

#### 🔹 Step 3: Run the Demo on Streamlit  

```bash
cd ./code
streamlit run app.py
```

📌 **Demo Interface Preview:**  
*(Insert an example image of the search interface here)*

---

### 📚 5. References  

[1] K. He, X. Zhang, S. Ren, and J. Sun, **"Deep Residual Learning for Image Recognition,"** *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 770–778, 2016.  

[2] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, et al., **"ImageNet Large Scale Visual Recognition Challenge,"** *International Journal of Computer Vision*, vol. 115, no. 3, pp. 211–252, 2015.  

[3] K. Simonyan and A. Zisserman, **"Very Deep Convolutional Networks for Large-Scale Image Recognition,"** *arXiv preprint arXiv:1409.1556*, 2015.  

[4] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, **"Rethinking the Inception Architecture for Computer Vision,"** *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2818–2826, 2016.  

[5] L. Tan, **"Farfetch Listings Dataset,"** *Kaggle*, 2019. Accessed: May 20, 2024.  
🔗 [Dataset Link](https://www.kaggle.com/datasets/alvations/farfetch-listings)  

---

💡 **Notes:**  
- The pretrained model and feature embeddings are provided to avoid retraining from scratch.
- If the Kaggle dataset is unavailable, you can use the `input.zip` file from the provided Google Drive link.

🔗 **GitHub Repository:** *[Insert repository link here]*  

---

💬 **For any questions or suggestions, feel free to open an issue on GitHub! 🚀**

