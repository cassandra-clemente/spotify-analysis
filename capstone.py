#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 22:46:33 2023

@author: cassyclemente
"""

'''
Data Specs
This data is stored in the file “spotify52kData.csv”, as follows:
Row 1: Column headers
Row 2-52001: Specific individual songs
Column 1: songNumber – the track ID of the song, from 0 to 51999.
Column 2: artist(s) – the artist(s) who are credited with creating the song.
Column 3: album_name – the name of the album
Column 4: track_name – the title of the specific track corresponding to the track ID
Column 5: popularity – this is an important metric provided by spotify, an integer from 0 to 100,
where a higher number corresponds to a higher number of plays on spotify.
Column 6: duration – this is the duration of the song in ms. A ms is a millisecond. There are a
thousand milliseconds in a second and 60 seconds in a minute.
Column 7: explicit – this is a binary (Boolean) categorical variable. If it is true, the lyrics of the track
contain explicit language, e.g. foul language, swear words or otherwise content that some consider to
be indecent.
Column 8: danceability – this is an audio feature provided by the Spotify API. It tries to quantify how
easy it is to dance to the song (presumably capturing tempo and beat), and varies from 0 to 1.
Column 9: energy - this is an audio feature provided by the Spotify API. It tries to quantify how “hard”
a song goes. Intense songs have more energy, softer/melodic songs lower energy, it varies from 0 to 1
Column 10: key – what is the key of the song, from A to G# (mapped to categories 0 to 11).
Column 11: loudness – average loudness of a track in dB (decibels)
Column 12: mode – this is a binary categorical variable. 1 = song is in major, 0 – song is in minor
Column 13: speechiness – quantifies how much of the song is spoken, varying from 0 (fully
instrumental songs) to 1 (songs that consist entirely of spoken words).
Column 14: acousticness – varies from 0 (song contains exclusively synthesized sounds) to 1 (song
features exclusively acoustic instruments like acoustic guitars, pianos or orchestral instruments)
Column 15: instrumentalness – basically the inverse of speechiness, varying from 1 (for songs
without any vocals) to 0.
Column 16: liveness - this is an audio feature provided by the Spotify API. It tries to quantify how
likely the recording was live in front of an audience (values close to 1) vs. how likely it was recorded in
a studio without a live audience (values close to 0).
Column 17: valence - this is an audio feature provided by the Spotify API. It tries to quantify how
uplifting a song is. Songs with a positive mood =close to 1 and songs with a negative mood =close to 0
Column 18: tempo – speed of the song in beats per minute (BPM)
Column 19: time_signature – how many beats there are in a measure (usually 4 or 3)
Column 20: track_genre – genre assigned by spotify, e.g. “blues” or “classical”
'''

import pandas as pd
df = pd.read_csv('spotify52kData.csv', dtype=None, delimiter=',') 
data = pd.read_csv('spotify52kData.csv', dtype=None, delimiter=',')
from scipy import stats
from scipy.stats import normaltest
from scipy.stats import zscore
from scipy.stats import ttest_ind
from scipy.stats import t
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, auc
import random

import numpy as np




random.seed(16320569)
random_seed = 16320569

#%% Preprocessing

# Sort DataFrame by 'Popularity' in descending order
df.sort_values('popularity', ascending=False, inplace=True)

#Drop duplicates based on 'TrackName', 'artists'  and all other numerical values other than popularity and keep the first occurrence (highest popularity)
df.drop_duplicates(subset = ['track_name', 'energy', 'liveness', 'danceability', 'duration', 'explicit', 'key', 'loudness', 'mode', 
                            'speechiness', 'acousticness', 'instrumentalness', 'valence', 'tempo', 'time_signature', 'artists'], keep='first', inplace=True)

## decided to delete entries with different genres, because the highest popularity entry is most likely the most relevant genre to the song.

#keeps duplicates with different genres
#df.drop_duplicates(subset = ['track_genre', 'track_name', 'energy', 'liveness', 'danceability', 'duration', 'explicit', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'valence', 'tempo', 'time_signature', 'artists'], keep='first', inplace=True)


# Reset the index
df.reset_index(drop=True, inplace=True)

# Print or use the modified DataFrame


'''
duplicates = data[data.duplicated('track_name', keep=False)]

duplicates_sorted = duplicates.sort_values(by='track_name')

# 'duplicates' now contains all rows with duplicate values in the 'title' column
print(duplicates)
'''

#%%

'''
Q1
Consider the 10 song features duration, danceability, energy, loudness, speechiness,
acousticness, instrumentalness, liveness, valence and tempo. Is any of these features
reasonably distributed normally? If so, which one? [Suggestion: Include a 2x5 figure with
histograms for each feature) 
'''

#Q1 - Plots

for column_index in range(5, 18):
    if ((column_index != 6) and (column_index != 9) and (column_index != 11)):  # Skip column 7 (explicit) and column 
        column_name = df.columns[column_index]
    
        # Plot histogram for the current column
        plt.figure(figsize=(9, 6))
        plt.hist(df.iloc[:, column_index], bins='auto', color = 'lightcoral', edgecolor='black', lw = 0.1)  # 'auto' determines the number of bins automatically
        plt.title(f'{column_name}' )
        plt.xlabel(column_name)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
        
        plt.figure(figsize=(8, 8))
        stats.probplot(df.iloc[:, column_index], plot=plt)
        plt.title(f'Probability Plot of {column_name}')
        plt.show()
        
        '''
        skewness = stats.skew(df.iloc[:, column_index])
        kurt = stats.kurtosis(df.iloc[:, column_index])
        print(f'{column_name} - Skewness: {skewness}, Kurtosis: {kurt}')
        
        
        k2_stat, k2_p_value = normaltest(df[column_name])
        print(f'D\'Agostino and Pearson\'s Test - Statistic: {k2_stat}, p-value: {k2_p_value}')
        '''

#%%
        
'''
Q2
Is there a relationship between song length and popularity of a song? If so, if the relationship
positive or negative? [Suggestion: Include a scatterplot]
'''
#correlation


'''

plt.figure(figsize=(8, 6))
plt.scatter(df['duration'], df['popularity'])
plt.title('Relationship between Duration and Popularity')
plt.xlabel('Duration (seconds)')
plt.ylabel('Popularity')
plt.grid(True)
plt.show()

'''

X = df['duration'].values.reshape(-1, 1)
y = df['popularity'].values

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions using the model
predictions = model.predict(X)

# Plot scatter plot and regression line
plt.figure(figsize=(8, 6))
plt.scatter(X, y, label='Data points', color = 'skyblue')
plt.plot(X, predictions, color='red', label='Regression line')
plt.title('Relationship between Duration and Popularity with Regression Line')
plt.xlabel('Duration (ms)')
plt.ylabel('Popularity')
plt.xlim(0, 1000000)
plt.legend()
plt.grid(True)
plt.show()

# Calculate correlation coefficient
correlation = df['duration'].corr(df['popularity'])
print(f"Correlation coefficient: {correlation}")


#%%
'''
Q3
Are explicitly rated songs more popular than songs that are not explicit? [Suggestion: Do a
suitable significance test, be it parametric, non-parametric or permutation]

plot popularity of explicit songs and of non explicit, compare with t test
'''      
X1 = df.loc[df['explicit'] == True, 'popularity']
X2 = df.loc[df['explicit'] == False, 'popularity']

std_X1 = np.std(X1)
std_X2 = np.std(X2)

print(f'Standard Deviation of X1: {std_X1:.2f}')
print(f'Standard Deviation of X2: {std_X2:.2f}')


t_stat, p_value = ttest_ind(X1, X2)

# Print t-statistic and p-value
print(f'T-statistic: {t_stat}')
print(f'P-value: {p_value}')

# Plotting
plt.figure(figsize=(8, 6))
plt.boxplot([X1, X2], labels=['Explicit', 'Non-Explicit'])
plt.title('Popularity Comparison: Explicit vs. Non-Explicit Songs')
plt.ylabel('Popularity Score')
plt.show()

plt.figure(figsize=(8, 6))
plt.hist([X1, X2], bins=20, alpha=0.5, label=['Explicit', 'Non-Explicit'], density = True, color = ['salmon', 'purple'], edgecolor = 'black', lw = 0.5)
plt.axvline(X1.mean(), color='purple', linestyle='dashed', linewidth=2, label='Mean Explicit')
plt.axvline(X2.mean(), color='red', linestyle='dashed', linewidth=2, label='Mean Non-Explicit')

plt.text(X1.mean(), 0.025, f'Mean Explicit: {X1.mean():.2f}', color='purple', fontsize=14, ha='left')
plt.text(X2.mean(), 0.028, f'Mean Non-Explicit: {X2.mean():.2f}', color='red', fontsize=14, ha='right')

plt.title('Histogram of Popularity for Explicit vs. Non-Explicit Songs')
plt.xlabel('Popularity Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()

#%%       
'''
Q4
Are songs in major key more popular than songs in minor key? [Suggestion: Do a suitable
significance test, be it parametric, non-parametric or permutation]
t test same as above
'''
X1 = df.loc[df['mode'] == 1, 'popularity']
X2 = df.loc[df['mode'] == 0, 'popularity']

std_X1 = np.std(X1)
std_X2 = np.std(X2)

print(f'Standard Deviation of X1: {std_X1:.2f}')
print(f'Standard Deviation of X2: {std_X2:.2f}')

t_stat, p_value = ttest_ind(X1, X2)

d_f = 42000
alpha = 0.05
critical_t = t.ppf(1 - alpha, d_f)

print(f"Critical T-value for a right-tailed test: {critical_t:.4f}")

# Print t-statistic and p-value
print(f'T-statistic: {t_stat}')
print(f'P-value: {p_value}')

# Plotting
plt.figure(figsize=(8, 6))
plt.boxplot([X1, X2], labels=['Major', 'Minor'])
plt.title('Popularity Comparison: Major vs. Minor Songs')
plt.ylabel('Popularity Score')
plt.show()

plt.figure(figsize=(8, 6))
plt.hist([X1, X2], bins=20, alpha=0.5, label=['Major', 'Minor'], density=True, color = ['blue', 'skyblue'], edgecolor = 'black', lw = 0.5)
plt.axvline(X1.mean(), color='maroon', linestyle='dashed', linewidth=2, label='Mean Major')
plt.axvline(X2.mean(), color='red', linestyle='dashed', linewidth=2, label='Mean Minor')


plt.text(X1.mean(), 0.022, f'Mean Major: {X1.mean():.2f}', color='blue', fontsize=14, ha='right')
plt.text(X2.mean(), 0.025, f'Mean Minor: {X2.mean():.2f}', color='black', fontsize=14, ha='left')


plt.title('Histogram of Popularity for Major vs. Minor Songs')
plt.xlabel('Popularity Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()


#%%
      
''' 
Q5
Energy is believed to largely reflect the “loudness” of a song. Can you substantiate (or refute)
that this is the case? [Suggestion: Include a scatterplot]
'''  
#correlations?    


df['loudness_zscore'] = (df['loudness'] - df['loudness'].mean()) / df['loudness'].std()
df['energy_zscore'] = (df['energy'] - df['energy'].mean()) / df['energy'].std()

slope, intercept = np.polyfit(df['loudness_zscore'], df['energy_zscore'], 1)
regression_line = slope * df['loudness_zscore'] + intercept

print("Slope:", slope)
print("Intercept:", intercept)

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df['loudness_zscore'], df['energy_zscore'], color = 'skyblue')
plt.plot(df['loudness_zscore'], regression_line, color='red', label='Regression line')

#plt.plot(df['loudness_zscore'], df['energy_zscore'], color='red', label='Regression line')
plt.title(f'Relationship between Loudness and Energy\nCorrelation: {df["loudness_zscore"].corr(df["energy_zscore"]):.2f}')
plt.xlabel('Z-Scored Loudness (dBFS)')
plt.ylabel('Z-Scored Energy')
plt.ylim(df['energy_zscore'].min(), df['energy_zscore'].max())
plt.show()





#%%

'''
Q6
Which of the 10 song features in question 1 predicts popularity best? How good is this model?
'''

excluded_columns = [6, 9, 11]
predictor_columns = [col for col in range(5, 18) if col not in excluded_columns]

# Perform linear regression for each predictor column
for column_index in predictor_columns:
    X = df.iloc[:, column_index].values.reshape(-1, 1)
    y = df['popularity']
    column_name = df.columns[column_index]
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Make predictions using the model
    predictions = model.predict(X)

    rSqr = model.score(X, y)
    # Plot the data and regression line
    plt.figure(figsize=(8, 6))
              
    plt.scatter(X, y, label='Actual Data', color = 'cornflowerblue')
    plt.plot(X, predictions, color='red', label='Regression Line')
    plt.title(f'Linear Regression: Column {column_name} vs Popularity: R^2 = {rSqr:.3f}')
    #plt.title('R^2 = {:.3f}'.format(rSqrTest))
    plt.xlabel(f'Column {column_index}')
    plt.ylabel('Popularity')
    plt.legend()
    plt.show()

    # Print the regression coefficients, R-squared value, and RMSE
    
    rmse = mean_squared_error(y, predictions, squared=False)

    print(f"Column {column_name}:")
    print(f"  Intercept: {model.intercept_}")
    print(f"  Coefficient (slope): {model.coef_[0]}")
    print(f"  R-squared: {model.score(X, y)}")
    print("   RMSE:", rmse)
    print("\n")

#%%

'''
Q7
Building a model that uses *all* of the song features in question 1, how well can you predict
popularity? How much (if at all) is this model improved compared to the model in question 7).
How do you account for this?

# need to plot on a 2d model

'''
'''
# Separate features (X) and target variable (y)
X = df.drop(['track_name', 'popularity', 'track_genre', 'key', 'mode', 'explicit', 'artists', 'album_name', 'time_signature'], axis=1)
y = df['popularity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

# Standardize features (optional but often recommended)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and fit a linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

'''

X = df.iloc[:,5:18].drop(columns = ['key' , 'mode', 'explicit'])
y = df.iloc[:, 4]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

# Fit the model on the training data
fullModel = LinearRegression().fit(X_train, y_train)

# Calculate R-squared on the test set
rSqrTest = fullModel.score(X_test, y_test)

b0, b1 = fullModel.intercept_, fullModel.coef_

# Calculate predicted values
yHat = b0 + np.dot(X_test, b1)

# Plotting the predicted vs actual values
plt.scatter(yHat, y_test, marker='o', s=25, color='darkseagreen', alpha=0.7)
plt.plot([min(yHat), max(yHat)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2)
plt.xlabel('Predicted Popularity')
plt.ylabel('Actual Popularity')
plt.title('R^2 = {:.3f}'.format(rSqrTest))
plt.show()

# Evaluate the model

print(f'R^2 Score: {rSqrTest}')

#ks test

'''
fullModel = LinearRegression().fit(X,y)
rSqrFull = fullModel.score(X,y) #All literally same syntax as before
print(rSqrFull)


b0, b1 = fullModel.intercept_, fullModel.coef_ 

#yHat = b1[0]*X[:,0] + b1[1]*X[:,1] + b1[2]*X[:,2] + b1[3]*X[:,3] + b1[4]*X[:,4] + b1[5]*X[:,5] + b1[6]*X[:,6] + b1[7]*X[:,7] + b1[8]*X[:,8] + b1[9]*X[:,9] + b0 #Evaluating the model: First coefficient times IQ value + 2nd coefficient * hours worked and so on, plus the intercept (offset)
yHat = b0 + np.dot(X, b1)
plt.plot(yHat,y,'o',markersize=4) 
plt.xlabel('Prediction from model') 
plt.ylabel('Actual ipopularity')  
plt.title('R^2 = {:.3f}'.format(rSqrFull))

'''

#%%

'''
Q8
When considering the 10 song features above, how many meaningful principal components
can you extract? What proportion of the variance do these principal components account for?
Using the principal components, how many clusters can you identify? 
'''

numericData = df.iloc[:,5:18].drop(columns = ['key' , 'mode', 'explicit'])

zscoredData = stats.zscore(numericData)
pca = PCA().fit(zscoredData)
eigVals = pca.explained_variance_
loadings = pca.components_
rotatedData = pca.fit_transform(zscoredData)

# 4. For the purposes of this, you can think of eigenvalues in terms of 
# variance explained:
varExplained = eigVals/sum(eigVals)*100

# Now let's display this for each factor:
for ii in range(len(varExplained)):
    print(varExplained[ii].round(3))
    
numQuestions = len(eigVals)
x = np.linspace(1,numQuestions,numQuestions)
plt.bar(x, eigVals, color='lightsalmon')
plt.plot([0,numQuestions],[1,1],color='black') # Orange Kaiser criterion line for the fox
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.show()

for i in range(1, 7):
    whichPrincipalComponent = i # Select and look at one factor at a time, in Python indexing
    plt.bar(x,loadings[whichPrincipalComponent,:]*-1, color = 'purple') # note: eigVecs multiplied by -1 because the direction is arbitrary
    #and Python reliably picks the wrong one. So we flip it.
    plt.xlabel('Question')
    plt.ylabel('Loading')
    plt.title(f'Principal Component {i}')
    plt.show() # Show bar plot
    
# PC1: upbeatness - valence and dancability
# PC2: !speechiness and !liveness
# PC3: !duration
# PC4: !tempo
# PC5: instrumentalness
# PC6: 
    '''
plt.plot(rotatedData[:,0]*-1,rotatedData[:,1]*-1,'o',markersize=5) #Again the -1 is for polarity
#good vs. bad, easy vs. hard
plt.xlabel('Mood and Energy Lifting')
plt.ylabel('Wordiness')
plt.show()
'''
X_pca = rotatedData[:, :2]

# Specify the number of clusters (you can choose an appropriate value)
n_clusters = 3

# Create KMeans instance and fit the data
kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed)
kmeans.fit(X_pca)

# Get cluster labels and centroids
cluster_labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Visualize the clusters on the PCA plot
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', edgecolors='k', s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.xlabel('Upbeatness')
plt.ylabel('Wordiness')
plt.title('K-Means Clustering on PCA Plot')
plt.legend()
plt.show()




#%%

'''
Q9
Can you predict whether a song is in major or minor key from valence? If so, how good is this
prediction? If not, is there a better predictor? [Suggestion: It might be nice to show the logistic
regression once you are done building the model]
'''
X = df[['valence']]
y = df['mode']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

# Standardize features (optional but often recommended)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and fit a logistic regression model
logreg_model = LogisticRegression(random_state=random_seed)
logreg_model.fit(X_train_scaled, y_train)

y_pred = logreg_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(classification_rep)

# Plot the decision boundary
plt.figure(figsize=(8, 6))



# Plot data points
plt.scatter(X['valence'], y, color=['blue' if val == 1 else 'red' for val in y], label='Actual Data')

x_values = np.linspace(X['valence'].min() - 0.1, X['valence'].max() + 0.1, 100).reshape(-1, 1)
x_values=x_values.reshape(100,1)
y_probabilities = logreg_model.predict_proba(scaler.transform(x_values))[:, 1]
plt.plot(x_values, y_probabilities, color='orange', label='Sigmoid Function')

# Plot decision boundary
x_min, x_max = X['valence'].min() - 0.1, X['valence'].max() + 0.1
x_values = np.linspace(x_min, x_max, 100)
y_values = 1 / (1 + np.exp(-logreg_model.coef_ * (x_values - logreg_model.intercept_)))
plt.plot(x_values, y_values[0], color='green', label='Decision Boundary')

# Add labels and legend
plt.title('Logistic Regression: Decision Boundary')
plt.xlabel('Valence')
plt.ylabel('Mode (Major: 1, Minor: 0)')
plt.legend()
plt.show()


y_probabilities = logreg_model.predict_proba(X_test_scaled)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_probabilities)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

X = df[['tempo']]
y = df['mode']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

# Standardize features (optional but often recommended)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and fit a logistic regression model
logreg_model = LogisticRegression(random_state=random_seed)
logreg_model.fit(X_train_scaled, y_train)

y_pred = logreg_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(classification_rep)

# Plot the decision boundary
plt.figure(figsize=(8, 6))



# Plot data points
plt.scatter(X['tempo'], y, color=['blue' if val == 1 else 'red' for val in y], label='Actual Data')

x_values = np.linspace(X['tempo'].min() - 0.1, X['tempo'].max() + 0.1, 100).reshape(-1, 1)
x_values=x_values.reshape(100,1)
y_probabilities = logreg_model.predict_proba(scaler.transform(x_values))[:, 1]
plt.plot(x_values, y_probabilities, color='orange', label='Sigmoid Function')

# Plot decision boundary
x_min, x_max = X['tempo'].min() - 0.1, X['tempo'].max() + 0.1
x_values = np.linspace(x_min, x_max, 100)
y_values = 1 / (1 + np.exp(-logreg_model.coef_ * (x_values - logreg_model.intercept_)))
plt.plot(x_values, y_values[0], color='green', label='Decision Boundary')

# Add labels and legend
plt.title('Logistic Regression: Decision Boundary')
plt.xlabel('tempo')
plt.ylabel('Mode (Major: 1, Minor: 0)')
plt.legend()
plt.show()


y_probabilities = logreg_model.predict_proba(X_test_scaled)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_probabilities)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


#%%
'''
Q10
Can you predict the genre, either from the 10 song features from question 1 directly or the
principal components you extracted in question 8? [Suggestion: Use a classification tree, but
you might have to map the qualitative genre labels to numerical labels first]

#df.drop_duplicates(subset = ['track_genre', 'track_name', 'energy', 'liveness', 'danceability', 'duration', 'explicit', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'valence', 'tempo', 'time_signature', 'artists'], keep='first', inplace=True)

'''
data.drop_duplicates(subset = ['track_genre', 'track_name', 'energy', 'liveness', 'danceability', 'duration', 'explicit', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'valence', 'tempo', 'time_signature', 'artists'], keep='first', inplace=True)

X = numericData  # Use the 10 song features
y = df['track_genre']

# Use LabelEncoder to convert string labels to numerical labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Add a new column 'genre_label' to the DataFrame with numerical labels
df['genre_label'] = y_encoded

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=random_seed)

# Create a decision tree classifier
clf = DecisionTreeClassifier(random_state=random_seed)

# Fit the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

# Print the results
print(f'Accuracy: {accuracy:.2f}')
print('\nClassification Report:')
print(classification_rep)

X = data.iloc[:,5:18].drop(columns = ['key' , 'mode', 'explicit'])  # Use the 10 song features
y = data['track_genre']

# Use LabelEncoder to convert string labels to numerical labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Add a new column 'genre_label' to the DataFrame with numerical labels
data['genre_label'] = y_encoded

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=random_seed)

# Create a decision tree classifier
clf = DecisionTreeClassifier(random_state=random_seed)

# Fit the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

# Print the results
print(f'Accuracy: {accuracy:.2f}')
print('\nClassification Report:')
print(classification_rep)


#%%

'''
Extra Credit Ideas
#find accuracy of classification ML and genre
# find relationship between key and genre and mode and genre and time_signature and genre
#

'''

X = df.iloc[:, [9, 10, 18]] # Use the 10 song features
y = df['track_genre']

# Use LabelEncoder to convert string labels to numerical labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Add a new column 'genre_label' to the DataFrame with numerical labels
df['genre_label'] = y_encoded

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=random_seed)

# Create a decision tree classifier
clf = DecisionTreeClassifier(random_state=random_seed)

# Fit the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

# Print the results
print(f'Accuracy: {accuracy:.2f}')
print('\nClassification Report:')
print(classification_rep)


#%%

'''
code sessions notes

# correlation code session

use np.corrcoef() to make correlation matrix from class
    advantage, don't need to loop to calculate all possible combinations'

use plt.subplots (5, 2)

diff distribution can give same summary statistics. must do EDA

when building a model with multiple predictors, the strongest predictor, would be the one with the strongest 
(positive or negative) correlation with the outcome 
'''













        
        
        
        
        
        
        