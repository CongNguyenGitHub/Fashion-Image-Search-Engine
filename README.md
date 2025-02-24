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

- 📥 **Dataset:** Refer to `data.txt` for the download link.
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
Below is the demo interface of the **Fashion Image Search Engine**, where users can upload an image and receive visually similar results.  

![Fashion Image Search Demo](images/demo.png)

---

💡 **Notes:**  
- The pretrained model and feature embeddings are provided to avoid retraining from scratch.

---

💬 **For any questions or suggestions, feel free to open an issue on GitHub! 🚀**

