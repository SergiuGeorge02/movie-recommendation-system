
# Movie Recommendation System

## Overview

This repository contains a Movie Recommendation System built using Python. The system aims to recommend movies or TV shows to users based on their preferences and viewing history. It leverages machine learning techniques and the MovieLens dataset for content-based and collaborative filtering approaches.

## Files & Folders

### 1. **Data Preprocessing**

- `data.py`: Handles data loading and preprocessing.
- `dataset_enhance.py`: Enhances the dataset by adding extra features or cleaning the data.
- `merging_datasets.py`: Merges multiple datasets to create a unified dataset for analysis.
- `movielens_data.py`: Processes and works with the MovieLens dataset.
- `Data_Analysis.ipynb`: Jupyter notebook for exploratory data analysis and visualizing the data.

### 2. **Model Development**

- `prediction_model.ipynb`: Jupyter notebook for building and evaluating the movie recommendation model.

### 3. **Web Application**

- The repository may contain a folder for deploying the recommendation system as a web application. (This is still under development, so specific details on deployment are not provided.)

### 4. **Dataset**

- `cleaned_movies.csv`: Contains information about movies (e.g., genres, descriptions).
- `cleaned_tv_shows.csv`: Contains information about TV shows.

### 5. **Environment Setup**

The recommendation system uses Python along with several common libraries. To get started, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/SergiuGeorge02/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. Install required dependencies:
   You can create a virtual environment and install the necessary libraries using:
   ```bash
   pip install -r requirements.txt
   ```

   If a `requirements.txt` file is not provided, manually install the necessary packages such as:
   - pandas
   - numpy
   - scikit-learn
   - matplotlib
   - seaborn
   - flask (for web deployment)

3. Run the Jupyter Notebooks:
   - To run the data analysis and model building, open the notebooks in Jupyter:
     ```bash
     jupyter notebook
     ```

### 6. **Recommendation Approach**

This project uses a combination of **content-based filtering** and **collaborative filtering** techniques to recommend movies and TV shows to users. The MovieLens dataset, a popular choice for recommendation system development, is used to train and test the model.

### 7. **Usage**

Once the model is trained, you can use it to make recommendations. Details for usage might depend on the web application implementation or specific functions in the Jupyter notebooks.

### 8. **Deployment**

If you want to deploy the recommendation system as a web application, check the `Web application` folder. The web app allows users to input their preferences and receive movie recommendations.


