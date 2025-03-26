# fake_review_detection

In this repository you can find the code for fake review detection from the Amazon site. This project was part of my master thesis, where the main aim of the project was to utilize the Large Language Models for extracting context-rich features from E-commerce sites.

# Prerequisite to start with this repository

Start installing the required dependencies from the requirements.txt file, and get started with cloning this repository.

#clone the repo

To clone the repository, go to the project directory and open the command prompt.

git clone https://github.com/PragnyaPataskar/fake_review_detection.git

# install the dependencies

pip install -r requirements.txt

# Data

The data used for this project is from the Kaggle site; the link to the data can be found here: https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset.

But for the original project, the data was used from E-commerce sites such as Amazon, Otto and Trustpilot by scraping it from September 2001 to July 2023 to get the overall view of the use from this timeframe to just check over the years how the opinions have changed.

# The main idea

After many experiment and discussion, the idea was to cross-check whether the traditional way of features exaction way is better than LLM's feature extraction. For LLM's the approach used are prompt engineering, and OpenAI's gpt 3.5-turbo, gpt-4 and gpt 4o for extracting features. This project also made use of Langchain's agents, tools for the structured output and for calculations. From the traditional way NLP techniques were used such as TF-IDF, word- embedding.

# Models performed

After extracting features both using traditional and LLM-based, the next step was to check how well these features act with the AI models. So for training these features, the models used are Supervised Learning models like Logistic Regression, Support Vector Machine, Random Forest, Decision Tree, and Naive Bayes.

# The Conclusion

The results obtained were such as the traditional features showed good results with these models compared to LLM extracted features.






