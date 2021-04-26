# Importing sklearn models
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# Importing data cleaning and feature extraction functions
from data_cleaning import clean_words
from feature_extraction import get_features

# Datasets path
true_dataset_path = 'dataset/True.csv'
fake_dataset_path = 'dataset/Fake.csv'

# Data pre-processing
df = get_features(true_dataset_path, fake_dataset_path)

# Cleaning text Data
df['total'] = df['total'].apply(clean_words)

# Defining x and y as feature and label respectively
x = df['total']
y = df['label']

# Train-Test splitting
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.25)


# Logistic Regression
LR = Pipeline([('TFIDF_Vectorizer', TfidfVectorizer()),
               ('Logistic_Regression', LogisticRegression(n_jobs=-1))])
LR.fit(x_train,y_train)
LRscore = LR.score(x_test,y_test)
print('LR score',LRscore)

# Saving Logistic Regression model pipeline
filename = 'saved_models/logistic_regression_fnd.pkl'
joblib.dump(LR, filename)


# Decision Tree Classifier
DTC = Pipeline([('TFIDF_Vectorizer', TfidfVectorizer()),
                ('Logistic_Regression', DecisionTreeClassifier())])
DTC.fit(x_train,y_train)
DTCscore = DTC.score(x_test,y_test)
print('DTC score',DTCscore)

# Saving Decision Tree Classifier model pipeline
filename = 'saved_models/decision_tree_classifier_fnd.pkl'
joblib.dump(DTC, filename)


# Gradient Boosting Classifier
GBC = Pipeline([('TFIDF_Vectorizer', TfidfVectorizer()),
                ('Gradient_Boosting_Classifier', GradientBoostingClassifier(random_state=0, learning_rate=0.001))])
GBC.fit(x_train,y_train)
GBCscore = GBC.score(x_test,y_test)
print('GBC score',GBCscore)

# Saving Gradient Boosting Classifier model pipeline
filename = 'saved_models/gradient_boosting_classifier_fnd.pkl'
joblib.dump(GBC, filename)


# Random Forest Classifier
RFC = Pipeline([('TFIDF_Vectorizer', TfidfVectorizer()),
                ('Random_Forest_Classifier', RandomForestClassifier(random_state=0, n_jobs=-1))])
RFC.fit(x_train,y_train)
RFCscore = RFC.score(x_test,y_test)
print('RFC score',RFCscore)

# Saving Random Forest Classifier pipeline
filename = 'saved_models/random_forest_classifier_fnd.pkl'
joblib.dump(RFC, filename)