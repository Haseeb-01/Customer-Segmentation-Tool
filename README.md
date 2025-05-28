# 📊 Customer Segmentation Tool

Welcome to **Customer Segmentation Tool**, an interactive Streamlit web app that helps you analyze customer data using **K-Means Clustering**. Whether you're a business owner, marketer, or data enthusiast, this tool empowers you to find meaningful patterns in your data—**with zero coding required**! 🚀

---

## ✨ Features

* 📂 **Upload Your Data**: Easily upload `.csv` files to explore customer behavior.
* 🛠️ **Smart Feature Selection**: Automatically recommends the best features based on variance.
* 📉 **Elbow Method**: Visualizes optimal number of clusters.
* 🤖 **K-Means Clustering**: Customize cluster count (2–10) and get real-time results.
* 🔬 **2D Cluster Visualization**: PCA-based scatter plot with clear color coding.
* 🧼 **Robust Preprocessing**: Automatically fills missing values with column medians.
* 📋 **Cluster Summary**: Understand your clusters through descriptive statistics.
* ⬇️ **Download Results**: Export clustered data and summaries as CSV.
* ⚠️ **Error Handling**: Friendly warnings for missing files or invalid input.
* 🏃 **Fast & Interactive**: Real-time processing in a sleek interface.

---

## 🎯 Use Cases

* 🛒 **E-commerce**: Segment users by browsing or buying behavior.
* 🏬 **Retail**: Understand customer demographics and purchasing habits.
* 🚀 **Startups**: Analyze without needing a dedicated data team.
* 📚 **Education**: Hands-on tool for teaching clustering concepts.

---

## 🚀 Getting Started

### ✅ Prerequisites

* Python 3.8–3.10 🐍
* Git
* GitHub account
* Optional: [Streamlit Community Cloud](https://streamlit.io/cloud) for free deployment

### 📦 Installation

```bash
git clone https://github.com/your-username/CustomerSegmentationTool.git
cd CustomerSegmentationTool
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### ▶️ Run the App

```bash
streamlit run app.py
```

Then open: [http://localhost:8501](http://localhost:8501)

---

## 🌸 Try It Now

Use the [Iris dataset](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv) to explore features instantly!

---

## 🌐 Deployment

### Option 1: Streamlit Cloud (Recommended)

1. Sign up at [streamlit.io](https://streamlit.io/cloud)
2. Connect your GitHub repo
3. Set main file to `app.py`
4. Deploy — done! 🎉

### Option 2: Screenshots

<img width="897" alt="62" src="https://github.com/user-attachments/assets/7ece9b41-2d9f-464b-bcf9-5f517d08457c" />
<img width="929" alt="63" src="https://github.com/user-attachments/assets/550a1772-864b-4ba0-b5bc-8255d6d456fe" />





---

## 📖 Usage

1. 📥 Upload a CSV file
2. ✅ Select numeric features (at least two)
3. 📉 Analyze clusters using the Elbow plot
4. 🔧 Adjust number of clusters via slider
5. 📈 View PCA scatter plot and summary table
6. ⬇️ Download results as CSV

---

## 🔒 Data Privacy

* Data is processed in-memory and **never stored** permanently
* Deploy over HTTPS for secure transfers

---

## 🤝 Contributing

Love the project? Want to make it better?

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push: `git push origin feature/your-feature`
5. Submit a pull request

---

## 📬 Contact

For feedback, feature requests or bug reports, open an issue or reach out at "haseebcheema397@gmail.com". 💬

---

## 🌟 Acknowledgments

* Built with ❤️ using **Streamlit**, **pandas**, **scikit-learn**, and **matplotlib**
* Inspired by the need for accessible data analytics in 🇵🇰 Pakistan and beyond
* Thanks to the open-source community! 🙏

---
##  Graphs
![cluster_boxplots](https://github.com/user-attachments/assets/2b2c89d9-be70-431e-bfc4-730295460f3d)
![cluster_scatter](https://github.com/user-attachments/assets/dfccfc57-8182-4067-81b0-b36524a6a722)
![elbow_plot](https://github.com/user-attachments/assets/af3c8019-abc9-4b23-ae93-efe9fbbd1086)





## 🎉 Happy Clustering!

