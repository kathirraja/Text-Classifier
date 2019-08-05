# Text-Classifier

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