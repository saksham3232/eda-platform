# 📊 EDA Platform with ML Preprocessing

An interactive **Exploratory Data Analysis** web app built with [Streamlit](https://streamlit.io/). Upload any CSV or Excel dataset and instantly explore it through smart filters, rich interactive visualisations, and a one-click ML preprocessing pipeline.

---

## ✨ Features

### 🔍 Data Exploration
- Upload **CSV, XLSX, or XLS** files directly from the sidebar
- Auto-detects numeric and categorical columns
- Dataset overview: row/column counts, data types, missing value summary, and descriptive statistics

### 🎯 Smart Filtering
- **Categorical filters** default to empty — all rows are shown until you actively filter
- **Numeric range sliders** for precise row selection
- Live row-count feedback showing how many rows match active filters

### 📈 Interactive Visualisations (7+ chart types)
| Chart | Use Case |
|---|---|
| Histogram | Column distributions |
| Scatter Plot | Relationship between two numeric columns |
| Bar Chart | Aggregated totals by category |
| Line Chart | Trends over a numeric or time axis |
| Box Plot | Distribution spread and outliers by group |
| Violin Plot | Full distribution shape by group |
| Correlation Heatmap | Pairwise numeric correlations |
| Pie / Donut Chart | Proportional breakdowns |

### 🔬 Advanced Analysis
- **Distribution Analysis** — histogram with KDE-style binning
- **Correlation Analysis** — interactive heatmap with coefficient labels
- **Group Comparison** — compare average and total metrics across categories
- **Outlier Detection** — IQR method with visual fence lines on box plots

### 🤖 ML Preprocessing Pipeline
One-click pipeline to produce ML-ready data:
- **Missing value imputation** — median for numeric columns, mode for categorical
- **Feature scaling** — choose from `StandardScaler`, `MinMaxScaler`, or `RobustScaler`
- **Label encoding** — converts all categorical columns to numeric
- Download the preprocessed dataset as **CSV or Excel**

### 🌙 UI & Theming
- Full **dark mode / light mode** toggle
- Wide layout with responsive multi-column design
- Custom CSS cards for metrics and pipeline status

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/saksham3232/eda-platform.git
cd eda-platform

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📦 Dependencies

```
streamlit
pandas
numpy
plotly
scikit-learn
openpyxl
```

Create a `requirements.txt` with:

```bash
pip freeze > requirements.txt
```

Or install manually:

```bash
pip install streamlit pandas numpy plotly scikit-learn openpyxl
```

---

## 🖥️ Usage

1. **Upload** a CSV or Excel file using the sidebar uploader.
2. Navigate the **tabs** to explore your data:
   - `📊 Overview` — dataset shape, column info, missing values, summary stats
   - `🔍 Exploration` — filter and browse column statistics
   - `📈 Visualisations` — create charts from your columns
   - `🔬 Advanced` — correlation, group comparison, outlier detection
   - `📋 Data` — view, sort, filter, and export raw data
   - `🤖 ML Pipeline` — preprocess and download ML-ready data
3. **Export** filtered data or preprocessed data as CSV or Excel at any time.

---

## 📁 Project Structure

```
eda-platform/
├── app.py          # Main Streamlit application
├── requirements.txt
└── README.md
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

> Built with ❤️ using [Streamlit](https://streamlit.io/), [Plotly](https://plotly.com/), and [scikit-learn](https://scikit-learn.org/).