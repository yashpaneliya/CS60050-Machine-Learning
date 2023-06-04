# Airline Passenger Segmentation

## Problem Statement

An Indian airline is conducting a customer segmentation analysis to understand their customer's behavior better. For that, they have created a dataset of their customers.
Given the dataset, your task is to cluster the dataset into an optimal number of clusters using K-means and Single linkage divisive hierarchical clustering algorithms.

## Dataset

Dataset contains 3000 rows and 23 columns. Each row represents a customer and each column contains customer’s attributes described on the column Metadata.

Features:

`'id', 'Gender', 'Customer Type', 'Age', 'Type of Travel',
       'Class', 'Flight Distance', 'Inflight wifi service',
       'Departure/Arrival time convenient', 'Ease of Online booking',
       'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
       'Inflight entertainment', 'On-board service', 'Leg room service',
       'Baggage handling', 'Checkin service', 'Inflight service',
       'Cleanliness', 'Departure Delay in Minutes',
       'Arrival Delay in Minutes'`

## Code walkthrough

1. Importing libraries
2. Loading dataset
3. Data preprocessing
    - Null checking and filling
    - Dropping irrelevant columns
    - Encoding categorical columns
4. Model building (k-Means)
    - Class named `CustomKMeans` for KMeans clustering
    - Distance metrics: `cosine similarity`
    - Methods:
        - `fit(x)` : Compute centroids and update according to data points
        - `save_final_clusters(x)` : Save final clusters in a file with indices of data points
5. Model evaluation
    - Silhouette score
6. Plotting clusters using scatter plot
7. Model building (Heirarchical clustering)
    - Class named `SingleLinkageDivisiveCLustering` for Heirarchical clustering
    - Distance metrics: `cosine similarity`
    - Methods:
        - `fit(x)` : Compute clusters and update according to data points
        - `dfs(start)` : Depth first search to find connected data points in a cluster
        - `generate_proximity(x)` : Generate proximity matrix
        - `remove_min_edge()` : Remove minimum edge from MST
    - Save clusters in a file with indices of data points
8. Model output comparison (k-Means vs Heirarchical)
    - Jaccard similarity
        
## Steps to run:

1. Install Jupyter Notebook or use Google Colab.
2. Open the file `ML_Assignment_3.ipynb` in Jupyter Notebook or Google Colab.
3. Run all the cells.

## Libraries used:

- pandas
- numpy
- matplotlib
- sklearn
- seaborn

*Detailed report can be found [here](/Clustering/22CS60R70_A3_REPORT.pdf)*