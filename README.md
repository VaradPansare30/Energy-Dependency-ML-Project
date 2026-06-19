# Energy Dependency ML Project

An easy-to-use Machine Learning website that checks how much different districts in Maharashtra rely on the traditional power grid versus renewable energy (Solar, Wind, Biomass and Hydro). It uses an unsupervised K-Means clustering model to automatically group regions based on their energy habits.

## 🚀 Project Overview

This project helps people see which areas use clean energy and which areas rely too much on the regular power grid. 

By combining energy data from different sources, the project calculates a **Grid Dependency Ratio** for each district. A Machine Learning model then automatically groups these districts into three clear categories. This helps policymakers quickly see where more green energy setup is needed.

---

## ✨ Features

*   **Combines 5 Data Sources:** Drops Solar, Wind, Biomass, Hydro and Grid Power data into one clean dataset.
*   **Real-World Energy Math:** Uses realistic efficiency percentages (Capacity Factors) to calculate actual power generated:
    *   ☀️ **Solar:** 20%
    *   💨 **Wind:** 30%
    *   🪵 **Biomass:** 80%
    *   🌊 **Hydro:** 40%
*   **Smart Machine Learning:** Uses a Scikit-Learn K-Means model to sort districts into 3 simple groups.
*   **Automatic Labels:** The code automatically reads model outputs and gives them easy-to-understand names:
    *   🟢 *Renewable Source Dependent* (Uses mostly clean energy)
    *   🟡 *Moderately Conventional Source Dependent* (Uses a mix of both)
    *   🔴 *Highly Conventional Source Dependent* (Relies heavily on the regular grid)
*   **Interactive Live Map:** Shows the final results on a beautiful, interactive map of Maharashtra.

---

## 🔧 Tech Stack

*   **Web Dashboard:** Streamlit, Plotly Express (for the map view)
*   **Data & Machine Learning:** Python, Pandas, Scikit-Learn (K-Means, StandardScaler)
*   **Model Saving:** Joblib (to save and load .pkl files)

---

## 📁 Project Structure

```text
├── Solar_energy.xlsx                             # Solar power data by district
├── wind_energy.xlsx                              # Wind power data by district
├── biomass.xlsx                                  # Biomass power data by district
├── hydro_energy.xlsx                             # Hydro power data by district
├── maharashtra_area_average_load_MW.xlsx         # Regular grid power use data
│
├── ideathon.ipynb                                # Jupyter Notebook where the model is built
├── energy_dependency_scaler.pkl                  # Saved data scaler file for the website
├── energy_dependency_kmeans.pkl                  # Saved Machine Learning model file for the website
│
├── app.py                                        # Main Streamlit website code
├── requirements.txt                              # List of required Python packages
└── README.md                                     # Project documentation
```

## **🧠 Technical Highlights**
▪️ Custom Feature Formula: 
Created a custom math formula to measure grid reliance:

$$\text{Grid Dependency Ratio} = \frac{\text{Grid Demand}}{\text{Grid Demand} + \text{Effective Renewable Power}}$$

▪️ Fast Loading (Decoupled Architecture): The model is trained inside the notebook and saved as a .pkl file. The website (app.py) loads this file instantly. It does not waste time re-training the model when a user visits the site.

▪️ Live Data Scaling: The app uses a saved StandardScaler to clean and prepare user inputs on-the-fly, preventing any data errors.

## **⚙️ How to Setup and Run Locally**
1. Clone the Project

Click here to view the repository: [GitHub Repository Link](https://github.com/VaradPansare30/Energy-Dependency-ML-Project.git)
  
3. Install Requirements
  ```
  pip install -r requirements.txt
  ```
4. Run the Website
  ```
  streamlit run app.py
  ```
## **🌐 Live Website Link**
Click the link below to see the project running live on the web:

🚀 Live Web App: [Maharashtra Energy Dependency Dashboard](https://energy-dependency-ml-project-arrtn7g7jjmyc2bb2yfapph.streamlit.app/)

## **📌 Important Notes**
▪️Data Cleaning: The Python pipeline automatically fixes text bugs, removes extra commas and fills missing data cells with 0 so the app never crashes.

▪️Free Map Integration: The map runs on free OpenStreetMap layers, so no expensive or private API setup keys are needed.

⭐ If you like this project, please give it a star on GitHub!
