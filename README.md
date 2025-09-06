# Customer Segmentation Project

This project performs **customer segmentation** for a mall based on **Annual Income** and **Spending Score** using **K-Means Clustering**. 
It identifies meaningful customer segments such as **Premium, Impulsive, Budget, Careful, and Average** and visualizes insights through an interactive, portfolio-ready dashboard.

## 🛠️ Skills and Tools

- **Python**: pandas, matplotlib, seaborn, scikit-learn  
- **Machine Learning**: Unsupervised Learning (K-Means Clustering)  
- **Data Visualization**: Scatter plots, bar charts, combined dashboard  
- **Business Understanding**: Customer segmentation, spending patterns, income analysis  

## 📂 Dataset

- **Mall Customers Dataset** (`data/Mall_Customers.csv`)  
- Contains 200 customers with the following columns:  
  - `CustomerID`  
  - `Gender`  
  - `Age`  
  - `Annual Income (k$)`  
  - `Spending Score (1-100)`  

## ⚡ Features

1. **Customer Clustering**  
   - K-Means clustering based on Annual Income and Spending Score  
   - Clusters mapped to meaningful names: Premium, Impulsive, Budget, Careful, Average  

2. **Visual Dashboard**  
   - Scatter plot with cluster names  
   - Number of customers per segment  
   - Average Spending Score per segment  
   - Average Annual Income per segment  

3. **Insights**  
   - Average income and spending score per segment  
   - Count of customers in each segment  

## 📊 How to Run

1. Clone the repository:

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd CustomerSegmentation
