# Twitter Bot Detection Pipeline
### End-to-End Machine Learning & Behavioral Analytics

This project implements a robust machine learning pipeline to distinguish between human users and automated bots on Twitter using the **TwiBot-20** research dataset.

## 🚀 Key Features
- **Big Data Handling**: Processed nested JSON raw data into a structured feature matrix.
- **Custom Feature Engineering**: Developed 18+ behavioral metrics including Reputation Score, Engagement Ratios, and mass-following heuristics.
- **Bias Mitigation**: Conducted an ablation study to remove biased "shortcut" features (Verified Status), ensuring the model learns genuine behavioral patterns.
- **Interactive Dashboard**: Built a Streamlit-based web application for real-time account auditing and model visualization.

## 📊 Model Performance
The pipeline compares 6 different classifiers. The **Gradient Boosting** model achieved the best results:
- **F1-Score**: 80.4%
- **Bot Recall**: ~90% (Catches 9 out of 10 bots)
- **Top Features**: Followers/Friends Ratio, Reputation Score, Listed Count.

## 📁 Project Structure
- `feature_extraction.py`: ETL script for parsing raw JSON and engineering features.
- `train_model.py`: Machine learning pipeline (Scaling, Training, Comparison, Serialization).
- `dashboard.py`: Streamlit web application for live predictions.
- `predict.py`: CLI tool for individual account auditing.
- `Twitter_Bot_Detection.ipynb`: Exploratory Data Analysis (EDA) and project walkthrough.

## 🛠️ Installation & Usage
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install pandas scikit-learn streamlit plotly
   ```
3. Run the Dashboard:
   ```bash
   streamlit run dashboard.py
   ```

