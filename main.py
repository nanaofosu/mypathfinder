from utils.path_finder import PathFinder

if __name__ == "__main__":
    job_recommender = PathFinder()
    similarities, job_listings = job_recommender.calculate_similarity()
    job_recommender.get_results(similarities, job_listings)
