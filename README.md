## 🛍️ Fashion Image Search Engine

🚀 *A deep learning-powered search engine that finds similar fashion items based on an input image.*  

---

### 📌 1. Introduction  

With the rapid growth of e-commerce, users often struggle to find clothing items based on images. This project builds a **Fashion Image Search Engine** using **deep learning** to retrieve visually similar fashion products from a large dataset.  

**🔹 Key Technologies Used:**  

- **Feature Extraction:** Pretrained deep learning models (**ResNet50 / VGG16 / InceptionV3**) extract meaningful representations of images.  
- **Image Retrieval:** Uses **FAISS (Facebook AI Similarity Search)** for fast and efficient similarity search.  
- **Web Application:** An intuitive **Streamlit** interface for user-friendly image-based search.  

---

### 👥 2. Team Members  

- 🧑‍💻 **Trần Lê Bảo Trung** - *21521598*  
- 🧑‍💻 **Nguyễn Công Nguyên** - *21521200*  

---

### 📂 3. Project Structure  

```
Fashion-Image-Search-Engine/
│── docs/               # Project documentation (Slides + Report)
│── code/               # Source code
│   │── app.py          # Streamlit web application
│   │── demo_utils.py   # Helper functions for the demo
│   │── requirements.txt # Required dependencies
│── README.md           # Project overview and setup guide
│── data.txt            # Link to dataset
│── models.txt          # Link to pretrained models
│── features.txt        # Link to extracted feature embeddings
```

---

### 📦 4. Setup & Running the Demo  

#### 🔹 Step 1: Install Dependencies  

```bash
cd ./code
pip install -r requirements.txt
```

#### 🔹 Step 2: Download Dataset, Models & Features  

The necessary files for running the project are available in text files:  
- **Dataset:** See `data.txt` for the download link.  
- **Pretrained Models:** See `models.txt` for the download link.  
- **Feature Embeddings:** See `features.txt` for the download link.  

After downloading, extract and place the respective folders into `code/`:  

```
code/
├── input/    # Dataset
├── models/   # Pretrained deep learning models
└── features/ # Precomputed feature vectors for faster retrieval
```

#### 🔹 Step 3: Run the Fashion Search Engine  

```bash
cd ./code
streamlit run app.py
```

---

### 📸 5. Demo Interface  

Example search result:  

📌 **Demo Screenshot Here** *(Insert an example image showcasing the search results.)*  

---

### 📚 6. References  

- [FAISS - Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)  

---

💡 **Notes:**  
- Pretrained models and extracted features are provided, so no additional training is required.  

🔗 **GitHub Repository:** *[Add link after pushing to GitHub]*  

---

💬 **For any inquiries or feedback, feel free to reach out via GitHub Issues! 🚀**

