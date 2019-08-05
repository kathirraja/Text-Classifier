from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


class DataPreProcessor(object):


	__CountVectorizer_params = {
		'stop_words': stop_words,
	}
	__TfidfTransformer_params = {
		'use_idf':False
	}

	def __init__(self):
		self.count_vectorizer = CountVectorizer(**DataPreProcessor.__CountVectorizer_params)
		self.tfidf_transformer = TfidfTransformer(**DataPreProcessor.__TfidfTransformer_params)



	def pre_process(data):
		pass 


class TextClassifier(DataPreProcessor):

	"""
	This class takes cteates text classifier handler object by taking 
	input perameters four parameters (dataset, classifier, features, label)

	Perameters
	----------

	1) dataset_file_path : <type:string> (mandatory)
		Absolute path of the .csv file 
	
	2) classifier : <type: Any Classifier Object> (mandatory)
		sklearn classifier objects 
	3) features : <type: List> (mandatory)
		List of strings, The strings must be a column name of the given csv file. This will be eveluvated for 
		classification from given dataset
	4) Label : <type: String> (mandatory)
		It reprecents the column name from the csv file to be predicted.

	Returns
	-------
		<TextClassifier: objects>

	Methods
	-------
	1)	read() 
		operation => takes self objects and reads the csv file and Maka a data Frame Using only Features and lebel columns,
					 removes all rows which contains NaN values and stores the dataframe to 'self.dataframe' variable for further use.

		returns : pandas DataFrame with Featues+label Column

	2) train()
		operation => Fits the Features column in the DataFrame and stores the classifier object in classifier variable and changes the
					 status of variable is_first to false. This funtion will be called at the first time of the training only.

		returns: classifier object

	3) predict()
		operation => Takes list of Features as an input and predicts the labels.

		Returns: List of labels.

	4) update(): 
		operation => Takes aditional dataset and do partial_fit on existing classifier.

		returns : updated classifier object

	Example:
	--------
	>>> dataset = "classifier_data_set.csv"
	>>> classifier = MultinomialNB()
	>>> features = ['text']
	>>> label = 'label_column'
	>>> tch = TextClassifier(dataset_file_path=dataset,
	    classifier=classifier,
	    features=features,
	    label=label,
	    is_first=True)
	>>>
	>>> tch.read()
	>>> tch.train()
	>>> tch.predict(['Document to be classify'])

	"""


	__CountVectorizer_params = {
		'stop_words': stop_words,
	}
	__TfidfTransformer_params = {
		'use_idf':False
	}


	def __init__(self, dataset_file_path, features, label, classifier=LogisticRegression(), is_first=True):
		self.count_vectorizer = CountVectorizer(**TextClassifier.__CountVectorizer_params)
		self.tfidf_transformer = TfidfTransformer(**TextClassifier.__TfidfTransformer_params)
		if is_first:
			self.features = features
			self.label = label
			self.is_first = True
			self.dataset_file_path = dataset_file_path
			self.dataframe = None
			self.classifier = classifier
			self.X_train_counts = None
			self.X_train_tfidf = None

	def text_preprocessor(self, texts):
		self.X_train_counts = self.count_vectorizer.fit_transform(texts)
		self.X_train_tfidf = self.tfidf_transformer.fit_transform(self.X_train_counts)
		return self.X_train_tfidf

	def read(self, dataset_file_path=None):
		self.dataframe = pd.read_csv(dataset_file_path if dataset_file_path else self.dataset_file_path)
		self.dataframe = self.dataframe[self.features+[self.label]]
		self.dataframe.dropna(inplace=True)
		self.dataframe = self.dataframe.sample(frac=1.0)
		return self.dataframe

	def train(self):
		self.read()
		texts = np.array(self.dataframe[self.features].astype(str)).ravel()
		self.text_preprocessor(texts)
		if self.is_first:
			self.classifier = self.classifier.fit(self.X_train_tfidf, self.dataframe[self.label].astype(str))
			self.is_first = False
		return self.classifier

	def update(self, dataset_file_path):

		if not self.is_first:
			dataframe = self.read(dataset_file_path)
			X_train_counts = self.count_vectorizer.fit_transform(dataframe[self.features])
			X_train_tfidf = self.tfidf_transformer.fit_transform(X_train_counts)
			self.classifier.partial_fit(X_train_tfidf, dataframe[self.label])
		else:
			raise "Call Train Method befour calling update"

	def transform(self, features):
		X_new_counts = self.count_vectorizer.transform(features)
		X_new_tfidf = self.tfidf_transformer.transform(X_new_counts)
		return X_new_tfidf

	def predict(self, features):
		X_new_tfidf = self.transform(features)
		predicted = self.classifier.predict(X_new_tfidf)
		return predicted
    