 ## Movie Recommendation System: Collaborative Filtering 
  
This repository implements a movie recommendation system using a collaborative filtering approach. Collabrorative filtering is one of most exicting topics in personalized recommendation systems. 
 
 ### Data
The data for this project is online movie ratings given by users on [MovieLens](https://movielens.org/) website. GroupLens Research collates this dat for research purposes and cand be downloaded [here](https://grouplens.org/datasets/movielens/) website. 
 
 ### Methodology
The project implements memory based collaborative filtering approaches to make movie recommendation. The central similarity measure for both cases is cosine similarity and evaluation metric is root mean square error. Both the approaches are explained below.
 
 * **User-based Collaborative filtering** : For each user, movies with high ratings from similar users are recommended. For example 
    > Users similar to you also liked………….
 
 * **Item-based Collaborative filtering**
 Will find users who like a movie, and recommend movies like by these users and other similar users. For example 
      > Users who liked this item also liked…
