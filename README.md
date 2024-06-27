# Job Recommendation System

This project implements a job recommendation system that utilizes text embeddings to find the most relevant job listings based on user input. It leverages OpenAI's embedding models to convert text data into high-dimensional vectors and calculates the similarity between user queries and job descriptions.

## Features

- **Text Embedding Generation**: Utilizes the `text-embedding-3-small` model to convert text data into high-dimensional vectors.
- **Dataset Specific**: Tailored to work with the `data/job_listings.csv` dataset, providing insights into various job listings.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.6 or higher
- Pip for Python package installation
- OpenAI API key

## Installation

To install the necessary libraries, run the following command:

```bash
pip install -r requirements.txt
```

## Configuration

The project uses environment variables for configuration. Copy the `.env.sample` file to a new file named `.env` and update the variables to match your setup.

The key variables include:

- OPENAI_API_KEY: Your OpenAI API key.
- DEFAULT_EMBEDDING_SIZE: Specifies the embedding size. Default is 1536.
- MAX_RECOMMENDATIONS: The number of top recommendations to return. Default is 5.
- DATASET: The path to the dataset file. Default is data/job_listings.csv.

## Usage

To run the project, execute the following command:

```
python main.py
```

This will prompt you to enter a job description, skills, or keywords, process the dataset using the specified model, and output the most similar job listings.

Example

**Job Description**

```
Senior Frontend Developer

Company: Tech Innovators Inc.
Location: Remote

Job Description:
Tech Innovators Inc. is seeking a highly skilled Senior Frontend Developer with extensive experience in JavaScript and React. The ideal candidate will have a strong background in building scalable web applications and a passion for creating beautiful and user-friendly interfaces.

Responsibilities:
- Develop and maintain web applications using JavaScript, React, and Redux.
- Collaborate with backend developers to integrate APIs and ensure seamless user experiences.
- Implement responsive designs and ensure cross-browser compatibility.
- Optimize applications for performance and scalability.
- Conduct code reviews and mentor junior developers.
- Stay up-to-date with the latest industry trends and best practices.

Requirements:
- Bachelor's degree in Computer Science or a related field.
- 5+ years of experience in frontend development.
- Expertise in JavaScript, React, and Redux.
- Strong understanding of HTML, CSS, and JavaScript frameworks.
- Experience with version control systems such as Git.
- Excellent problem-solving skills and attention to detail.
- Ability to work independently and in a team environment.
- Strong communication skills.

Preferred Qualifications:
- Experience with TypeScript and Next.js.
- Familiarity with RESTful APIs and GraphQL.
- Knowledge of testing frameworks such as Jest and Cypress.
- Understanding of CI/CD pipelines and DevOps practices.

Benefits:
- Competitive salary and benefits package.
- Flexible working hours and remote work options.
- Opportunities for professional development and growth.
- Collaborative and innovative work environment.
```

**Query**

```
I am looking for a remote job as a Senior Frontend Developer with expertise in JavaScript, React, and Redux. I have over 5 years of experience in building scalable web applications and creating user-friendly interfaces. I am proficient in HTML, CSS, and Git. I have a background in collaborating with backend developers to integrate APIs, implementing responsive designs, and optimizing applications for performance. I also have experience mentoring junior developers and conducting code reviews. Flexible working hours and opportunities for professional development are important to me.
```

**Expected Output**

```
Recommendation 1:
Title: Senior Frontend Developer
Company: Tech Innovators Inc.
Location: Remote
Description: Tech Innovators Inc. is seeking a highly skilled Senior Frontend Developer with extensive experience in JavaScript and React. The ideal candidate will have a strong background in building scalable web applications and a passion for creating beautiful and user-friendly interfaces...
Similarity Score: 0.9800

Recommendation 2:
...

Recommendation 3:
...
```

## Contributing

Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License