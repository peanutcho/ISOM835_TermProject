## 🎓 Intelligent Email Classifier for College Admissions
*Automated Multi-label Email Classification & Response System for International Admissions*

This project develops a machine learning system to automatically classify and respond to international student admissions emails using both traditional predictive models and fine-tuned BERT. It includes multi-label classification, auto-reply generation, and zero-shot classifier with BART. The entire analysis is performed in Google Colab for reproducibility and accessibility.

---

## 🧠 Project Overview & Objectives

International admissions offices receive a high volume of repetitive, time-sensitive emails from international applicants. This project presents an intelligent, end-to-end system to:

- Build a robust **multi-label classification model** to categorize emails into multiple relevant topics
- Automatically generate **contextual replies** based on predicted labels
- Evaluate and compare performance between **traditional models and fine-tuned BERT**
- Explore **zero-shot learning (BART)** for flexible, prompt-based classification without retraining
- Run entirely in **Google Colab** for accessibility and reproducibility

---
## 🗃️ Dataset Description

A synthetic dataset of 1,000 realistic emails was created using admission-related templates. Each email is randomly labeled with up to 3 of the following 10 categories:
`ENGLISH`, `DOCS`, `VISA`, `DEADLINE`, `FINANCE`, `STATUS`, `PROGRAM`, `TECH`, `CASCADE`, `COMPLEX`


Example entry:

| email_text                                                                                      | labels               |
|--------------------------------------------------------------------------------------------------|----------------------|
| Hello, I submitted my financial documents for my I-20. When will I receive an update?           | [VISA, DOCS, STATUS] |


---
 ## ✉️ Example of Automated Response Generator

Each category is mapped to a pre-written professional response. The system uses predicted categories to generate contextual replies.

---

## 🧰 Tools & Libraries Used

- **Language Models**: `bert-base-uncased` (BERT), `facebook/bart-large-mnli` (BART for zero-shot)
- **Libraries**:  
  - `Scikit-learn`: Logistic Regression, Random Forest, SVM, Naive Bayes  
  - `PyTorch`: Model training and evaluation  
  - `Transformers`: BERT/BART via Hugging Face  
  - `Pandas`, `NumPy`: Data handling  
  - `Matplotlib`, `Seaborn`: Visualization
- **Platform**: Google Colab (cloud-based, no setup required)

---

### ▶️ How to Run This Project in Google Colab

1. **Open the notebook in Google Colab**  
   👉 [Open the Full Project in Colab](https://colab.research.google.com/drive/1hPm27eTfAE1ev-B7vueoqfqERvWCVYIl)

2. **Option A** – For fastest setup (recommended):  
   **Upload the following files** from your local machine to skip training:  
   - `email_dataset.csv`  
   - `Y.npy`  
   - `X_test_texts.csv`  
   - `y_test.npy`  
   - `bert_classifier_sd.pt` (fine-tuned BERT model weights)

   ✅ This allows you to **skip data generation and model training** steps and start directly from evaluation and testing.  
   ➤ **Skip Steps 1 through 5**, and begin at **Step 5-2**.

3. Once setup is complete, continue with:  
   - **Step 6**: Model evaluation  
   - **Step 7**: Auto-reply generation  
   - **Step 8–10**: Real-time predictions, zero-shot classification, and visualizations

4. **Option B** – If you want to generate everything from scratch:  
   You can run **Step 1 through Step 10** to create synthetic data, train traditional models and BERT, and evaluate all results.  
   ⏳ This will take longer, especially the BERT fine-tuning step.
   ➤ In this case, **skip Step 5-1 and Step 5-2** (which are only for model/data restoration).

---

## 📊 Visualizations

All visual outputs are generated directly in the notebook:
- Category frequency distribution  
- Accuracy comparison across ML & BERT models  
- Precision/Recall/F1 evaluation for multi-label classification  
- Auto-reply examples with category tags  

Example chart:  

<p align="center">
  <img src="https://github.com/peanutcho/ISOM835_TermProject/blob/main/category_frequency_distribution.png" alt="Model Accuracy Comparison" width="600"/>
</p>

<p align="center">
  <img src="https://github.com/peanutcho/ISOM835_TermProject/blob/main/model_accuracy_comparison.png" alt="Model Accuracy Comparison" width="600"/>
</p>

--- 

## 🧪 Highlights

| Feature                          | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| 🔤 Multi-label Classification    | Emails can belong to multiple categories simultaneously                     |
| 🤖 BERT Fine-tuning              | Custom classifier trained with `BertForSequenceClassification`              |
| ✉️ Auto-response Generator      | Templates mapped to each category to provide reply suggestions              |
| 🔍 Zero-shot Classification      | No retraining needed — BART model dynamically classifies novel queries      |

---

## 📁 Directory Structure

├── /visualizations/ # All plots (saved manually from Colab)
├── /notebooks/ # Google Colab .ipynb notebook
├── bert_classifier_sd.pt # Saved model weights (not uploaded to GitHub)
├── README.md # Project documentation
└── .gitignore

> 📌 Note: Model weights and dataset files are excluded from the repo due to size. Instructions are provided to upload these when running the notebook.

---

## 🔚 Final Notes

✅ Fully modular codebase
✅ Easily adaptable to new email categories
✅ Supports real-time or batch email analysis
✅ Ideal for real-time admissions chatbot integration
✅ Can be expanded with real-world datasets and multilingual support

---

## 📬 Contact

If you’d like to collaborate, adapt this for your institution, or learn more about the modeling process, feel free to reach out!





