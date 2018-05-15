import numpy as np
import pandas as pd
# load datasets
header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=header)

# Exploratory data analysis 
df.describe() # Each row is not unique
user_list= df.user_id.unique()
users= len(df.user_id.unique())
movies= len(df.item_id.unique())
print('unique users:'+ str(users) + '\nunique movies:'+  str(movies))

# Divide data into train and test 
from sklearn.cross_validation import train_test_split
train_data, test_data = train_test_split(df, test_size=0.25)

# Create user movie matrix, Ratings 1-5 if user rated the movie else zero
train_user_matrix = np.zeros((users,movies))
for i in train_data.itertuples():
    train_user_matrix[i[1]-1, i[2]-1] = i[3] 
      
test_user_matrix = np.zeros((users,movies))
for i in test_data.itertuples():
    test_user_matrix[i[1]-1, i[2]-1] = i[3]

# Similarity matrices ( u*u- user similarity array of array, i*i item similarity) 
from sklearn.metrics.pairwise import cosine_similarity
user_similarity = cosine_similarity(train_user_matrix) # Both users have to rate same set of movies to be considered for measuring similarity 
item_similarity = cosine_similarity(train_user_matrix.T) # A user has to rate both the movies to be considered for similarity between two movies

# Scoring movie recommendations based on cf type(score:dot product of similarity and ratings)  based on similarity measure
def predict(ratings, similarity, type='user'):
    if type == 'user': # normalize to keep the range of ratings from 1-5
        mean_user_rating = ratings.mean(axis=1)
        #You use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)+1e-9])
    return pred

item_prediction = predict(train_user_matrix, item_similarity, type='item')  
user_prediction = predict(train_user_matrix, user_similarity, type='user')

### Use MSE as the Validation measure 
from sklearn.metrics import mean_squared_error
def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

print ('User based CF MSE: ' + str(get_mse(user_prediction, test_user_matrix)))
print ('Item based CF MSE: ' + str(get_mse(item_prediction, test_user_matrix)))