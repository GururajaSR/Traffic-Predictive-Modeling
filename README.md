# **Traffic Predictive Modeling Using Machine Learning**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange.svg)](https://www.tensorflow.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  

This project predicts travel times and congestion levels using advanced machine learning techniques. By analyzing historical traffic data from San Francisco, the models help improve commute planning and optimize urban traffic management.

---

## **📑 Table of Contents**
1. [**Project Overview**](#project-overview)  
2. [**Dataset**](#dataset)  
3. [**Key Features**](#key-features)  
4. [**Models and Techniques**](#models-and-techniques)  
5. [**Results and Visualizations**](#results-and-visualizations)  
6. [**Installation and Usage**](#installation-and-usage)  
7. [**Future Work**](#future-work)  
8. [**License**](#license)  

---

## **🚀 Project Overview**
- **Objective**: Build machine learning models to forecast travel times based on historical and contextual factors like time, weather, and holidays.  
- **Purpose**: Help commuters and urban planners optimize travel by providing actionable insights into traffic trends.  

---

## **📊 Dataset**
- **Source**: [San Francisco Caltrain and Uber Movement Data (Kaggle)](https://www.kaggle.com/vaishalij/san-francisco-caltrain-uber-movement-data).  
- **Key Features**:  
  - **Origin and Destination IDs**: Unique identifiers for travel locations.  
  - **Travel Time**: Mean, upper, and lower bounds in seconds.  
  - **Weather Data**: Simulated data for temperature, precipitation, and wind.  
  - **Holiday Indicator**: Boolean flag for special days.  
  - **Temporal Data**: Day of the week, hour of the day.  

---

## **🔑 Key Features**
- **Data Preprocessing**:  
  - Handled missing values and inconsistencies.  
  - Normalized numerical features for improved model performance.  

- **Feature Engineering**:  
  - Extracted temporal features like day of the week and hour of the day.  
  - Incorporated simulated weather and holiday data for context.  

---

## **🤖 Models and Techniques**
### **1. Linear Regression**  
- Baseline model to compare performance.  

### **2. Random Forest Regressor**  
- Captured non-linear relationships effectively.  

### **3. Neural Networks (Deep Learning)**  
- Multi-layer perceptron model for complex feature interactions.  

**Evaluation Metrics**:  
- **Mean Squared Error (MSE)**  
- **R² Score**  

---

## **📈 Results and Visualizations**

### **Model Performance**:
| **Model**              | **MSE**      | **R² Score** |  
|-------------------------|--------------|--------------|  
| Linear Regression       | 586,222      | 0.07         |  
| Random Forest Regressor | 151,337      | 0.76         |  
| Neural Network          | 184,656      | 0.71         |  

### **Key Visualizations**:
1. **Predicted vs Actual Travel Times**  
   - Shows alignment of model predictions with real data, particularly during peak traffic hours.  
2. **3D Visualization**:  
   - Highlights travel times based on time of day and day of the week, showcasing temporal patterns.

---

## **💻 Installation and Usage**  

### **1. Clone the Repository**:
   ```bash
   git clone https://github.com/GururajaSR/Traffic-Predictive-Modeling.git
   cd Traffic-Predictive-Modeling
   ```

### **2. Install Dependencies**:
   ```bash
   pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
   ```

### **3. Run the Jupyter Notebook**:
   Open `BigDataForCAV.ipynb` in Jupyter Notebook or Google Colab:
   ```bash
   jupyter notebook BigDataForCAV.ipynb
   ```

### **4. View Results**:
   Follow the step-by-step analysis and visualizations in the notebook.

---

## **🔮 Future Work**
- **Real-Time Data**: Incorporate live traffic data for dynamic predictions.  
- **Advanced Models**: Implement time-series models like RNNs or LSTMs for sequential data.  
- **Scalability**: Extend the solution to include multiple cities and regions.  

---

## **📜 License**  
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.  

---

## **🤝 Acknowledgments**  
- **Dataset**: [San Francisco Caltrain Data (Kaggle)](https://www.kaggle.com/vaishalij/san-francisco-caltrain-uber-movement-data).  
- **Libraries Used**: TensorFlow, Scikit-learn, Pandas, Seaborn, and Matplotlib.  
