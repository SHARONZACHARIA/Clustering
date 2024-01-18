# Importing Libarraies 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

#loading the dataset
data = pd.read_csv("Live.csv")

#Display basic statistics and information about the dataset
print(data.describe())
print(data.info())

# Missing values 
print(data.isnull().sum())

# Explore unique values in categorical columns
print(data['status_type'].value_counts())

# Convert 'status_published' column to datetime
data['status_published'] = pd.to_datetime(data['status_published'])

# Plot 1: Line plot of the number of reactions over time
def PlotLine_Reactions(data):
    plt.figure(figsize=(8, 6))
    plt.plot(data['status_type'], data['num_reactions'], label='Number of Reactions', marker='o')
    plt.title('Number of Reactions Over Time')
    plt.xlabel('Time')
    plt.ylabel('Number of Reactions')
    plt.legend()
    plt.show()


# Plot 2: Bar plot of the average number of reactions by status type
def plotBar_ReactionsByStatus(data):
    average_reactions_by_type = data.groupby('num_shares')['num_reactions'].mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    average_reactions_by_type.plot(kind='bar', color='tomato')
    plt.title('Average Number of Reactions by Status Type')
    plt.xlabel('number of shares')
    plt.ylabel('Average Number of Reactions')
    plt.show()

# Select relevant columns for clustering
features = data[['num_reactions', 'num_comments', 'num_shares']]
normalized_features = (features - features.mean()) / features.std()

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster_label'] = kmeans.fit_predict(normalized_features)


# Plot cluster membership
def PlotScatter_Membership(data):
    plt.scatter(data['num_reactions'], data['num_comments'], c=data['cluster_label'], cmap='viridis', alpha=0.5)
    plt.title('Clustering Results')
    plt.xlabel('Number of Reactions')
    plt.ylabel('Number of Comments')
    plt.show()


# Plot cluster centers
def PlotScatter_Centers(data):
    plt.scatter(data['num_reactions'], data['num_comments'], c=data['cluster_label'], cmap='viridis', alpha=0.5)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200)
    plt.title('Clustering Results with Centers')
    plt.xlabel('Number of Reactions')
    plt.ylabel('Number of Comments')
    plt.show()



data.drop(['Column1', 'Column2', 'Column3', 'Column4'], axis=1, inplace=True)
data.info()
data.describe()

# view  how many different types of variables are there
data['status_id'].unique()
len(data['status_id'].unique())
data['status_published'].unique()
len(data['status_published'].unique())
data['status_type'].unique()
len(data['status_type'].unique())


# Data preprocessing For model fitting 
data.drop(['status_id', 'status_published'], axis=1, inplace=True)
data.info()
data.head()

# Feature vector and Targtet Variable 

X = data
y = data['status_type']

# Convert Categorical variable into integers
le = LabelEncoder()
X['status_type'] = le.fit_transform(X['status_type'])
y = le.transform(y)
X.info()
X.head()


# Feature Scaling 
cols = X.columns
ms = MinMaxScaler()
X = ms.fit_transform(X)
X = pd.DataFrame(X, columns=[cols])
X.head()


# K-MEANS Clustering for Classification
kmeans = KMeans(n_clusters=2, random_state=0) 
kmeans.fit(X)
# K-MEANS Models Paratmerters
kmeans.cluster_centers_
kmeans.inertia_

#Check quality of weak Classification by the model 
labels = kmeans.labels_

# check how many of the samples were correctly labeled
correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))


#Using elbow method to find optimal number of clusters 
cs = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    cs.append(kmeans.inertia_)
plt.plot(range(1, 11), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()

# Result with Two clusters 
kmeans = KMeans(n_clusters=2,random_state=0)
kmeans.fit(X)
labels = kmeans.labels_

# check how many of the samples were correctly labeled
correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))

# Result with Three Clusters 
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

# check how many of the samples were correctly labeled
labels = kmeans.labels_
correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))

# Resukt with 4 Clusters 
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X)

# check how many of the samples were correctly labeled
labels = kmeans.labels_
correct_labels = sum(y == labels)
print("Result: %d out of %d samples were correctly labeled." % (correct_labels, y.size))
print('Accuracy score: {0:0.2f}'. format(correct_labels/float(y.size)))

