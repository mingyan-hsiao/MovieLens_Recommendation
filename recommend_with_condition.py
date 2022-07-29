### Recommend manually
### Read data
df_ex = read("CollaborativeFiltering.genre_explode")
df_ratings = read("CollaborativeFiltering.ratings")

### Get user's favorite movie genre
def fav_genre(user:int): 
	user_rating = df_ratings.filter(df_ratings.userId==user)
	df = df_ex.join(user_rating,df_ex.movieId==user_rating.movieId,"inner").drop("timestamp")
	df.createOrReplaceTempView("joindf")
	user_genre = spark.sql("""
		SELECT genres, ROUND(AVG(rating),2) as avg_rating 
		FROM joindf
		GROUP BY genres
		ORDER BY avg_rating DESC
		""")
	return user_genre.collect()[0][0] # return their favorite genre

### Commonly liked movies
from pyspark.sql.functions import *
def common_like(): 
	# add a new column to tell if low rated
	df_low = df_ratings.withColumn("low_rating", when(df_ratings.rating < 3, 1).otherwise(0))
	df_movie_rating = df_low.groupBy("movieId").agg(sum("low_rating").alias("low_rated"),\
		                                            count("rating").alias("num_of_rating"),\
		                                            round(avg("rating"),2).alias("avg_rating"))
	# add a new column to calculate low ratio	
	df_movie_rating = df_movie_rating.withColumn("low_ratio", round(df_movie_rating.low_rated/df_movie_rating.num_of_rating,2))
	# add filter: liked by most users
	df_movie_rating = df_movie_rating.filter(df_movie_rating.num_of_rating>30)\
	                                 .filter(df_movie_rating.low_ratio<0.2)
	return df_movie_rating

### Classic movies in the genre
def classic_movie(genre): 
	movie_avg_rating = common_like()
 	# use inner join to skip obscure movies in this genre
 	df = df_ex.join(movie_avg_rating,df_ex.movieId==movie_avg_rating.movieId,"inner").drop(movie_avg_rating.movieId)\
	          .filter(df_ex.genres==genre)
	# sorting
	df = df.sort(df.avg_rating.desc(), df.num_of_rating.desc()).drop("low_rated")
	return df

### Integated
def main(user):
	movies_to_recom = classic_movie(fav_genre(user))
	return movies_to_recom


# we can compare with the recommendation using ALS




	
