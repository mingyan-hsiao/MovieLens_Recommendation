# Import the required functions
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import *

df_ratings = read("CollaborativeFiltering.ratings")
df_movies = read("CollaborativeFiltering.movies")
df_ratings.show(5)
df_movies.show(5)

### count missing for each column
from pyspark.sql.functions import col, isnan, when, count
df_miss = df_ratings.select([count(when(isnan(c)|col(c).isNull(),c)).alias(c) for c in df_ratings.columns])
df_miss.show(5)
df_miss2 = df_movies.select([count(when(isnan(c)|col(c).isNull(),c)).alias(c) for c in df_movies.columns])
df_miss2.show(5)

### Data sparsity
# Count the total number of ratings in the dataset
numerator = df_ratings.select("rating").count()
# Count the number of distinct userIds and distinct movieIds
num_users = df_ratings.select("userId").distinct().count()
num_movies = df_ratings.select("movieId").distinct().count()
# Set the denominator equal to the number of users multiplied by the number of movies
denominator = num_users * num_movies
# Divide the numerator by the denominator
sparsity = (1.0 - (numerator *1.0)/denominator)*100
print("The ratings dataframe is ", "%.2f" % sparsity + "% empty.")


# Movies with highest average score and at least 50 ratings
from pyspark.sql.functions import *
top_rated = df_ratings\
.groupBy("movieId")\
.agg(count("userId"), avg(col("rating")))\
.withColumnRenamed("count(userId)", "num_ratings")\
.withColumnRenamed("avg(rating)", "avg_rating")\

top_rated_movies = top_rated.join(df_movies, top_rated.movieId == df_movies.movieId)\
                            .drop(df_movies.movieId).filter("num_ratings >= 50")\
                            .sort(desc("avg_rating"), desc("num_ratings"))
top_rated_movies.show(5)


# Most popular movies
most_popular_movies = top_rated_movies.sort(desc("num_ratings"), desc("avg_rating"))
most_popular_movies.show(5, truncate=False)
df = df_ratings


# Split data
# Smaller dataset so we will use 0.8 / 0.2
(train_data, test_data) = df.randomSplit([0.8, 0.2], seed=42)

# Create ALS model
als = ALS(
         userCol="userId", 
         itemCol="movieId",
         ratingCol="rating", 
         nonnegative = True, 
         implicitPrefs = False,
         coldStartStrategy="drop"
)


# Add hyperparameters and their respective values to param_grid
param_grid = ParamGridBuilder() \
            .addGrid(als.rank, [10, 50, 100, 150]) \
            .addGrid(als.regParam, [.01, .05, .1, .15]) \
            .build()

# Define evaluator as RMSE and print length of evaluator
evaluator = RegressionEvaluator(
           metricName="rmse", 
           labelCol="rating", 
           predictionCol="prediction") 
print ("Num models to be tested: ", len(param_grid))


# Build cross validation using CrossValidator
cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)
#Fit cross validator to the 'train' dataset
model = cv.fit(train_data) # 9 min?????
#Extract best model from the cv model above
best_model = model.bestModel
# View the predictions
test_predictions = best_model.transform(test_data)
RMSE = evaluator.evaluate(test_predictions)
print(RMSE)
print("**Best Model**")
# Print "Rank"
print("  Rank:", best_model._java_obj.parent().getRank())
# Print "MaxIter"
print("  MaxIter:", best_model._java_obj.parent().getMaxIter())
# Print "RegParam"
print("  RegParam:", best_model._java_obj.parent().getRegParam())

# Build the recommendation model using ALS on the training data
als = ALS(maxIter=5, regParam=0.01, 
          userCol="userId", itemCol="movieId", 
          ratingCol="rating", coldStartStrategy="drop")
model = als.fit(train_data)

# save model
temp_path = "/incorta/IncortaAnalytics/Tenants/ebs_cloud/incorta.ml/models"
modelPath = temp_path + "/als-model"
model.write().overwrite().save(modelPath)
# load the model
model2 = ALSModel.load("/incorta/IncortaAnalytics/Tenants/ebs_cloud/incorta.ml/models"+ "/als-model")


# Evaluate model performance
predictions = model.transform(test_data)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("RMSE = " + str(rmse))


def recommendForUser(user, n): # recommend "n" movies for "user"
  userRecs = model.recommendForAllUsers(n)
  nrecommendations = userRecs\
                  .withColumn("rec_exp", explode("recommendations"))\
                  .select("userId", "rec_exp.movieId", "rec_exp.rating")
  df_out = nrecommendations.where(nrecommendations.userId == user)
  df_out = df_out.join(df_movies, df_out.movieId == df_movies.movieId, "left").drop(df_movies.movieId).sort(desc("rating"))
  return df_out

# test
recommendForUser(5, 3).show()


def recommendForMovie(movie, n):
  movieRecs = model.recommendForAllItems(n)
  nrecommendations = movieRecs\
                  .withColumn("rec_exp", explode("recommendations"))\
                  .select("movieId", "rec_exp.userId", "rec_exp.rating")
  df_out = nrecommendations.where(nrecommendations.movieId == movie)
  df_out = df_out.join(df_movies, df_out.movieId == df_movies.movieId, "left").drop(df_movies.movieId).sort(desc("rating"))
  return df_out

# test
recommendForMovie(movie=5, n=3).show()
df_output = recommendForMovie(movie=5, n=3)

save(df_output)