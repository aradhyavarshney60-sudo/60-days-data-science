# ============================================================
# DAY 16 - MOVIE RECOMMENDATION USING K-NEAREST NEIGHBORS
# ============================================================

import os
import ssl
import zipfile
import urllib.request
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

ZIP_FILE = "ml-latest-small.zip"
DATA_FOLDER = "ml-latest-small"

MOVIES_FILE = os.path.join(DATA_FOLDER, "movies.csv")
RATINGS_FILE = os.path.join(DATA_FOLDER, "ratings.csv")

K_VALUES = [3, 5, 7, 10]

# Movie for recommendation
DEFAULT_MOVIE = "Toy Story (1995)"

print("=" * 60)
print("DAY 16 - MOVIE RECOMMENDATION USING KNN")
print("=" * 60)


# ============================================================
# 2. DOWNLOAD MOVIELENS DATASET
# ============================================================

def download_movielens():
    """
    Download MovieLens dataset.
    SSL verification is disabled only to handle
    certificate issues on some Windows/Python setups.
    """

    if os.path.exists(MOVIES_FILE) and os.path.exists(RATINGS_FILE):
        print("\nMovieLens dataset already exists.")
        return

    print("\nDownloading MovieLens dataset...")

    try:
        # SSL context to avoid certificate verification issue
        ssl_context = ssl._create_unverified_context()

        request = urllib.request.Request(
            DATA_URL,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        with urllib.request.urlopen(
            request,
            context=ssl_context
        ) as response:

            with open(ZIP_FILE, "wb") as file:
                file.write(response.read())

        print("Dataset downloaded successfully.")

        # Extract ZIP
        with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall(".")

        print("Dataset extracted successfully.")

    except Exception as error:

        print("\nDataset download failed.")
        print("Error:", error)

        print("\nPlease download MovieLens manually from:")
        print(DATA_URL)

        print("\nAfter downloading:")
        print("1. Extract the ZIP file.")
        print("2. Keep the 'ml-latest-small' folder inside this project.")
        print("3. Make sure movies.csv and ratings.csv are inside it.")

        return


# ============================================================
# 3. LOAD DATASET
# ============================================================

download_movielens()

if not os.path.exists(MOVIES_FILE):
    raise FileNotFoundError(
        "\nmovies.csv not found.\n"
        "Please put the MovieLens 'ml-latest-small' folder "
        "inside the project folder."
    )

if not os.path.exists(RATINGS_FILE):
    raise FileNotFoundError(
        "\nratings.csv not found.\n"
        "Please put ratings.csv inside ml-latest-small folder."
    )


movies = pd.read_csv(MOVIES_FILE)
ratings = pd.read_csv(RATINGS_FILE)

print("\nMovieLens dataset loaded successfully.")

print("\nMovies shape:", movies.shape)
print("Ratings shape:", ratings.shape)


# ============================================================
# 4. BASIC DATA ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nMovies:")
print(movies.head())

print("\nRatings:")
print(ratings.head())

print("\nMissing values in movies:")
print(movies.isnull().sum())

print("\nMissing values in ratings:")
print(ratings.isnull().sum())


# ============================================================
# 5. CREATE MOVIE FEATURES
# ============================================================

print("\nCreating movie features...")

# Convert genres into lists
movies["genre_list"] = movies["genres"].apply(
    lambda x: x.split("|") if isinstance(x, str) else []
)

# One-hot encode genres
mlb = MultiLabelBinarizer()

genre_features = mlb.fit_transform(
    movies["genre_list"]
)

genre_df = pd.DataFrame(
    genre_features,
    columns=mlb.classes_,
    index=movies.index
)

# Combine features
movie_features = pd.concat(
    [
        movies[["movieId", "title"]],
        genre_df
    ],
    axis=1
)

print("Movie features created successfully.")

print("\nFeature shape:", genre_df.shape)


# ============================================================
# 6. CALCULATE AVERAGE RATINGS
# ============================================================

average_ratings = (
    ratings
    .groupby("movieId")["rating"]
    .agg(["mean", "count"])
    .reset_index()
)

average_ratings.columns = [
    "movieId",
    "average_rating",
    "rating_count"
]

movie_features = movie_features.merge(
    average_ratings,
    on="movieId",
    how="left"
)

movie_features["average_rating"] = (
    movie_features["average_rating"]
    .fillna(0)
)

movie_features["rating_count"] = (
    movie_features["rating_count"]
    .fillna(0)
)


# ============================================================
# 7. PREPARE KNN FEATURES
# ============================================================

# Genre features are used for similarity
X = genre_df.values.astype(float)

print("\nKNN feature matrix created successfully.")
print("Matrix shape:", X.shape)


# ============================================================
# 8. TRAIN KNN MODEL
# ============================================================

print("\nTraining KNN model...")

knn = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

knn.fit(X)

print("KNN model trained successfully.")


# ============================================================
# 9. RECOMMENDATION FUNCTION
# ============================================================

def recommend_movies(movie_name, k=5):

    # Find movie
    matches = movies[
        movies["title"].str.lower()
        == movie_name.lower()
    ]

    # If exact match doesn't exist
    if matches.empty:

        partial_matches = movies[
            movies["title"]
            .str.lower()
            .str.contains(movie_name.lower(), na=False)
        ]

        if partial_matches.empty:
            print(
                f"\nMovie '{movie_name}' not found."
            )
            return pd.DataFrame()

        movie_index = partial_matches.index[0]

        movie_name = partial_matches.iloc[0]["title"]

    else:
        movie_index = matches.index[0]
        movie_name = matches.iloc[0]["title"]

    # Number of neighbors
    n_neighbors = min(
        k + 1,
        len(movies)
    )

    distances, indices = knn.kneighbors(
        X[movie_index].reshape(1, -1),
        n_neighbors=n_neighbors
    )

    recommendations = []

    for distance, index in zip(
        distances[0],
        indices[0]
    ):

        # Don't recommend same movie
        if index == movie_index:
            continue

        similarity = 1 - distance

        recommendations.append(
            {
                "Input_Movie": movie_name,
                "Recommended_Movie": movies.iloc[index]["title"],
                "Similarity": round(
                    similarity,
                    4
                ),
                "Average_Rating": round(
                    movie_features.iloc[index]["average_rating"],
                    2
                ),
                "Rating_Count": int(
                    movie_features.iloc[index]["rating_count"]
                )
            }
        )

    return pd.DataFrame(
        recommendations[:k]
    )


# ============================================================
# 10. TEST RECOMMENDATION
# ============================================================

print("\n" + "=" * 60)
print("MOVIE RECOMMENDATION")
print("=" * 60)

recommendation_result = recommend_movies(
    DEFAULT_MOVIE,
    k=5
)

if not recommendation_result.empty:

    print(
        f"\nRecommendations for: {DEFAULT_MOVIE}"
    )

    print(
        recommendation_result.to_string(
            index=False
        )
    )

else:

    # Use first movie if Toy Story is unavailable
    fallback_movie = movies.iloc[0]["title"]

    recommendation_result = recommend_movies(
        fallback_movie,
        k=5
    )

    print(
        f"\nRecommendations for: {fallback_movie}"
    )

    print(
        recommendation_result.to_string(
            index=False
        )
    )


# ============================================================
# 11. COMPARE DIFFERENT K VALUES
# ============================================================

print("\n" + "=" * 60)
print("COMPARING DIFFERENT K VALUES")
print("=" * 60)

comparison_results = []

comparison_movie = DEFAULT_MOVIE

# Check if movie exists
if not (
    movies["title"]
    .str.lower()
    .eq(DEFAULT_MOVIE.lower())
    .any()
):
    comparison_movie = movies.iloc[0]["title"]


for k in K_VALUES:

    result = recommend_movies(
        comparison_movie,
        k=k
    )

    if result.empty:
        continue

    average_similarity = (
        result["Similarity"].mean()
    )

    average_rating = (
        result["Average_Rating"].mean()
    )

    comparison_results.append(
        {
            "K": k,
            "Number_of_Recommendations": len(result),
            "Average_Similarity": round(
                average_similarity,
                4
            ),
            "Average_Recommended_Rating": round(
                average_rating,
                4
            )
        }
    )


comparison_df = pd.DataFrame(
    comparison_results
)

print("\nK comparison:")
print(
    comparison_df.to_string(
        index=False
    )
)


# ============================================================
# 12. IDENTIFY BEST K
# ============================================================

if not comparison_df.empty:

    best_row = comparison_df.loc[
        comparison_df["Average_Similarity"].idxmax()
    ]

    best_k = int(best_row["K"])

    best_similarity = best_row[
        "Average_Similarity"
    ]

    print(
        f"\nBest K identified: {best_k}"
    )

    print(
        "Best average similarity:",
        best_similarity
    )

else:

    best_k = 5

    print(
        "\nUnable to calculate best K."
    )
    print(
        "Using K = 5 as default."
    )


# ============================================================
# 13. GENERATE FINAL RECOMMENDATIONS
# ============================================================

final_recommendations = recommend_movies(
    comparison_movie,
    k=best_k
)

print("\n" + "=" * 60)
print("FINAL RECOMMENDATIONS")
print("=" * 60)

if not final_recommendations.empty:

    print(
        final_recommendations.to_string(
            index=False
        )
    )


# ============================================================
# 14. SAVE RECOMMENDATIONS
# ============================================================

final_recommendations.to_csv(
    "day16_recommendations.csv",
    index=False
)

print(
    "\nRecommendation results saved successfully."
)


# ============================================================
# 15. SAVE K COMPARISON
# ============================================================

comparison_df.to_csv(
    "day16_k_comparison.csv",
    index=False
)

print(
    "K comparison analysis saved successfully."
)


# ============================================================
# 16. CREATE K COMPARISON VISUALIZATION
# ============================================================

if not comparison_df.empty:

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        comparison_df["K"],
        comparison_df["Average_Similarity"],
        marker="o"
    )

    plt.xlabel(
        "K Value"
    )

    plt.ylabel(
        "Average Similarity"
    )

    plt.title(
        "KNN Recommendation Quality Across K Values"
    )

    plt.xticks(
        comparison_df["K"]
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        "day16_k_comparison.png",
        dpi=300
    )

    plt.show()

    print(
        "Visualization saved successfully."
    )


# ============================================================
# 17. MODEL RESULTS
# ============================================================

model_results = pd.DataFrame(
    {
        "Metric": [
            "Algorithm",
            "Distance Metric",
            "Best K",
            "Average Similarity",
            "Number of Movies",
            "Number of Ratings"
        ],
        "Value": [
            "K-Nearest Neighbors",
            "Cosine Distance",
            best_k,
            (
                round(
                    best_similarity,
                    4
                )
                if not comparison_df.empty
                else "N/A"
            ),
            len(movies),
            len(ratings)
        ]
    }
)

model_results.to_csv(
    "day16_model_results.csv",
    index=False
)

print(
    "Model results saved successfully."
)


# ============================================================
# 18. BUSINESS INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("BUSINESS INTERPRETATION")
print("=" * 60)

print(
    "\n1. KNN recommends movies based on similarity."
)

print(
    "2. Cosine distance is used to compare movie genres."
)

print(
    "3. Smaller K gives fewer and more closely related recommendations."
)

print(
    "4. Larger K provides more recommendations but may reduce similarity."
)

print(
    f"5. Based on the analysis, K = {best_k} performed best "
    "according to average similarity."
)

print(
    "6. Recommendation systems can improve user engagement "
    "by suggesting relevant movies."
)


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DAY 16 SUMMARY")
print("=" * 60)

print(
    "Movie recommendation dataset loaded successfully."
)

print(
    "Movie features created successfully."
)

print(
    "KNN model trained successfully."
)

print(
    "Similarity-based recommendations generated."
)

print(
    "Different K values compared."
)

print(
    "Best K identified."
)

print(
    "Recommendation results saved."
)

print(
    "K comparison analysis saved."
)

print(
    "Visualization saved."
)

print(
    "Model results saved."
)

print(
    "\nDay 16 Movie Recommendation using K-Nearest Neighbors "
    "completed successfully! 🎬"
)

print("=" * 60)