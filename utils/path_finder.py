import re
import pandas as pd
import openai
from typing import List
from utils.config import get_env_variable
from utils.data_loader import DataLoader
from utils.embeddings_calculator import EmbeddingsCalculator
from scipy.spatial.distance import cosine

openai.api_key = get_env_variable('OPENAI_API_KEY')

# Default values for embedding size, maximum recommendations, and dataset path
DEFAULT_EMBEDDING_SIZE = get_env_variable('DEFAULT_EMBEDDING_SIZE', 1536, int)
MAX_RECOMMENDATIONS = get_env_variable('MAX_RECOMMENDATIONS', 5, int)
DATASET = get_env_variable('DATASET', 'data/job_listings.csv')


class PathFinder:
    def __init__(self):
        self.embedding_calculator = EmbeddingsCalculator()
        self.data_loader = DataLoader()

    def load_data(self, file_path):
        """
        Load data from a CSV file and preprocess the 'description' column.
        Generate embeddings for the preprocessed descriptions.
        Returns the processed DataFrame.
        """
        df = self.data_loader.load_csv(file_path)
        df['description'] = df['description'].fillna('').apply(self.preprocess_text)
        df['embeddings'] = self.generate_embeddings(df['description'].tolist())
        return df

    def preprocess_text(self, text):
        """
        Preprocesses the given text by removing email addresses, multiple spaces,
        non-ASCII characters, problematic characters, short words, and converting
        to lowercase. Returns the preprocessed text.
        """
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'[*]', '', text)
        text = re.sub(r'\b\w{1,2}\b', '', text)
        text = text.strip().lower()
        return text

    def generate_embeddings(self, texts):
        """
        Generates embeddings for a list of texts using the embedding calculator.
        Returns a list of embeddings.
        """
        return [self.get_embedding_with_error_handling(text) for text in texts]

    def get_embedding_with_error_handling(self, text: str) -> List[float]:
        """
        Gets embeddings for a text using the embedding calculator.
        If an error occurs, handles the error and returns a default embedding.
        Returns the embedding as a list of floats.
        """
        try:
            embedding = self.embedding_calculator.get_embeddings(text)
            return embedding
        except Exception as e:
            print(f"Error generating embedding for text: {text}. Error: {e}")
            return [0] * DEFAULT_EMBEDDING_SIZE

    def get_user_input(self):
        """
        Prompts the user for a job description or skill set and returns it as a string.
        """
        user_input = input("Enter a job description, skills, or keywords to find similar jobs: ")
        return self.preprocess_text(user_input)

    def calculate_similarity(self):
        """
        Calculates the similarity between the user input and job listings.
        Returns a tuple containing the similarity scores and the job listings DataFrame.
        """
        user_description = self.get_user_input()
        user_embedding = self.get_embedding_with_error_handling(user_description)
        job_listings = self.load_data(DATASET)
        job_listing_embeddings = job_listings['embeddings'].tolist()
        similarities = [1 - cosine(user_embedding, job_embedding) for job_embedding in job_listing_embeddings]
        return similarities, job_listings

    def get_results(self, similarities, job_listings):
        """
        Prints the top recommendations based on similarity scores.
        """
        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:MAX_RECOMMENDATIONS]
        for i, idx in enumerate(top_indices):
            print(f"Recommendation {i + 1}:")
            print(f"Title: {job_listings.iloc[idx]['title']}")
            print(f"Company: {job_listings.iloc[idx]['company']}")
            print(f"Location: {job_listings.iloc[idx]['location']}")
            print(f"Description: {job_listings.iloc[idx]['description'][:200]}...")
            print(f"Similarity Score: {similarities[idx]:.4f}")
            print()
