SMS dataset that contains spam and ham labelled sms messages. 
Substitute the spam as 1 and ham as 0 in the labels from the first column. We want to consider spam as the positive class for which we will fill the answers in the table.

Perform tf and tf-idf vectorizations using CountVectorizer() and TfidfVectorizer() in scikit-learn. Use default values for all vectorizer parameters.

Use 5-fold cross-validation with SVC() classifier in stratified manner. Report results for tf and tf-idf vectors using scikit-learn with the following combinations of parameters (of SVC()):
• linear, rbf and sigmoid kernel
• C parameter value from the list [0.1, 1, 10]
• random_state=13.
• Other parameters should have default values.
