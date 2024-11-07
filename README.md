# Fake-News-FactCheck-AI
This project applies natural language processing (NLP) and machine learning (ML) algorithms to classify news articles as true or false using Python's scikit-learn library.

Getting Started
Follow these instructions to set up and run this project on your local machine for development and testing. Deployment notes for live implementation are also included.

Prerequisites
To run this project, you will need the following:

Python 3.6
Install Python 3.6 from python.org. Once installed, set up the PATH variables if you want to run Python programs directly. Detailed instructions are available here.

Alternatively, you can use Anaconda for easier setup and package management. Download Anaconda from Anaconda’s website.

Required Packages
After setting up Python or Anaconda, install the following packages:

For Python:
bash
Copy code
pip install -U scikit-learn
pip install numpy
pip install scipy
For Anaconda:
bash
Copy code
conda install -c scikit-learn
conda install -c anaconda numpy
conda install -c anaconda scipy
Dataset Information
This project uses the LIAR dataset, which includes training, test, and validation sets in .tsv format.

Dataset Details:
The original LIAR dataset provides 14 columns for each entry, covering information like statement, label, speaker, and truth counts. For simplicity, we reduced the dataset to two columns:

Statement: News headline or text.

Label: Binary classification with two possible values, True and False, mapped as follows:

Original Label	New Label
True, Mostly-true, Half-true	True
Barely-true, False, Pants-fire	False
The .csv files used in this project are named train.csv, test.csv, and valid.csv, while the original .tsv files remain in the liar folder.

Project Files
DataPrep.py
This script handles all data preprocessing, including loading data files (train, test, validation), and applying tokenization, stemming, and other text processing techniques. It also includes exploratory data analysis for distribution checks and quality assessments.

FeatureSelection.py
This script performs feature extraction and selection using scikit-learn techniques such as bag-of-words, n-grams, and term frequency-inverse document frequency (TF-IDF) weighting. Preliminary work with POS tagging and Word2Vec is also included but not fully implemented in this version.

classifier.py
In this file, several classifiers are built, including Naive Bayes, Logistic Regression, Linear SVM, Stochastic Gradient Descent, and Random Forest, all using scikit-learn. Models are evaluated using F1 score and confusion matrices. GridSearchCV is applied for hyperparameter tuning on the top-performing models. The final selected model includes feature importance analysis using TF-IDF.

prediction.py
This script uses the final saved classifier (Logistic Regression) to predict the truthfulness of user-input statements. The model is saved as final_model.sav and loaded by prediction.py to classify statements as True or False with an associated probability.

Process Flow
Data Preprocessing
Feature Selection
Model Training and Evaluation
Prediction with Probability Score
Model Performance
The project includes learning curves for top-performing models such as Logistic Regression and Random Forest. The current F1 score is in the 70% range, which may be improved by increasing dataset size and implementing additional feature extraction techniques like Word2Vec and topic modeling.

Installation and Usage
Clone the Project

bash
Copy code
git clone https://github.com/nishitpatel01/Fake_News_Detection.git
Run with Anaconda
Open Anaconda Prompt, navigate to the project folder, and run:

bash
Copy code
cd /path/to/project
python prediction.py
Enter a news headline for verification. The model will classify the statement as True or False with a probability of truth.

Run with Python
If not using PATH variables, locate python.exe and use the full path to run prediction.py:

bash
Copy code
c:/path/to/python.exe /path/to/project/prediction.py
Then enter a news headline for classification.

For those who have set up PATH, simply navigate to the project folder and use:

bash
Copy code
python prediction.py
The model will provide both the classification result and a probability score indicating the truthfulness of the statement.
