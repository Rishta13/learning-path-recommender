
import streamlit as st
import pandas as pd
from utils import LearningPathRecommender

st.set_page_config(page_title="Personalized Learning Path Recommender", page_icon="🎓")

st.title("🎓 Personalized Learning Path Recommender")
st.write("Suggests the **next best skills** based on your current skills and career goal.")

recommender = LearningPathRecommender("learning_paths.csv")

with st.sidebar:
    st.header("About this project")
    st.markdown('''
    **Built by:** Rishta Raj  
    **Goal:** Help learners pick their next skill step using a content-based recommendation approach (TF-IDF + cosine).
    ''')
    st.markdown('---')
    st.subheader("How it works")
    st.markdown('''
    We compute similarity between your selections and known skill→goal mappings,
    then rank possible next skills. We also include quick resource pointers.
    ''')

df = pd.read_csv("learning_paths.csv")
all_skills = sorted(set(df["current_skill"]).union(set(df["recommended_skill"])))
goals = sorted(df["career_goal"].unique())

current = st.multiselect("Select your current skills", options=all_skills, default=["Python"])
goal = st.selectbox("Select your career goal", options=goals, index=0)

if st.button("Get Recommendations"):
    if not current:
        st.warning("Please select at least one current skill.")
    else:
        recs = recommender.recommend(current, goal)
        if recs.empty:
            st.info("No recommendations found for your selection. Try another goal.")
        else:
            st.success(f"Recommended skills to pursue for {goal}:")
            for _, row in recs.iterrows():
                st.markdown(f"- **{row['recommended_skill']}** → {row['why']}")
                st.caption(f"Resources: {row['resources']}")
