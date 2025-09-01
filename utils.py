
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class LearningPathRecommender:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df.fillna("", inplace=True)
        self.df["combined"] = self.df["current_skill"] + " " + self.df["career_goal"]
        self.vectorizer = TfidfVectorizer()
        self.vectors = self.vectorizer.fit_transform(self.df["combined"])

    def recommend(self, current_skills, career_goal, top_n=5):
        query = " ".join(current_skills) + " " + career_goal
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.vectors).flatten()
        idx = sims.argsort()[::-1]
        recs = self.df.iloc[idx][["recommended_skill", "why", "resources"]].drop_duplicates()
        return recs.head(top_n)
