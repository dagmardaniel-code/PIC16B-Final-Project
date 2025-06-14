## PIC16B-Final-Project

#### The Project
In this project, we aim to predict flight prices using this dataset: https://www.kaggle.com/datasets/dilwong/flightprices with machine learning algorithms. 

In this project, we did data cleaning and preprocessing, feature engineering, exploratory analysis, dimension reduction (PCA), feature selection (SelectKBest), and modeling. For our models, we used Linear Regression, XgBoost, Random Forest, and Neural Networks. 
In the end, our best model was X, which yielded an R^2 of 0.85, RMSE of 63.18, and MAE of 44.38,  

In the future, we plan on simplifying the model to reduce the runtime. Additionally, we plan on finding more dataset that captures flight trends across multiple periods so that we are able to study the yearly, monthly, and seasonal flight trends in flight prices.  

### Technicalities
Our project code was implemented on a MacBook Air with an Apple M2 chip and 8 GB of RAM. The code was developed using Python 3.11.7 in a Jupyter Lab Notebook and Google Colab on macOS 14.2.1. We used Pandas, Numpy, Scikit-learn, Matplotlib, and Seaborn to clean and preprocess our data, construct and evaluate our machine learning models, and visualize results. 

### Files 
- `project.py` - contains functions written for this project
- ‘flight_fare.ipynb` - contains the implementation of the project 
- ‘flight_data_sampled.csv` - random subset of 1 million rows form dataset (too big to upload to github)

### Dataset
Our dataset is sourced from Kaggle, it has 80 million rows, 27 columns, and over 30 GB. For time efficiency, we read a random subset of 1 million rows and exported it as a new CSV 'flight_data_sampled.csv.' 


Documentation on the existing columns: https://www.kaggle.com/datasets/dilwong/flightprices. 

Engineered Features: 

- `daysInAdvance`
- `numStops` - number of stops a flight makes
- `isRedEyeFlight` - does flight depart between 10PM to 2AM (red-eye hours) 
- `departsOnHoliday` - does the flight depart on a U.S. federal holiday 
- `departsThreeDaysBeforeHoliday` - does the flight depart within three days before on a U.S. federal holiday 
- `depatureSeason` - season of flight departure (e.g. winter, spring, summer, fall) 
- `flightRouteCount` - the number of times a flight route appears in the dataset 
- `flightsOnRouteSameDay` - number of flights on the same route on the same day 
- `flightsPerAirlineCount`- number of flights per airline in the dataset 
- `flightsPerAirlineOnRouteSdCount`- number of flights on the same route on the same day from the same airline  





#### Group Contribution Statement
Dagmar worked on reading the data, data cleaning, and the Neural Network model. Ari worked on feature engineering, exploratory analysis, and the Random Forest model. Jocelynne worked on feature selection, dimension reduction, Linear Regression, and the XGBoost model. For the GitHub repo, Dagmar committed the README file, Ari committed the .ipynb file, and Jocelynne committed the .py file. 

