# **Swahili News Classification**
## **Team;** George Tido, Miriam Ongare, John Nkakuyia, Shalom Irungu, Mercy Ronoh
## Project Overview
### Business Problem
Swahili serves as a vital language for communication, education, and cultural expression in Tanzania and across East Africa. With the increasing dominance of English in online spaces, there's a risk of losing the representation of Swahili, especially in digital media such as news platforms. We strive to address this challenge by developing a multi-class classification model to automatically categorize Swahili news articles into specific categories. By doing so, online news platforms can enhance user experience by providing readers with easy access to news content relevant to their interests, while also contributing to the preservation and promotion of the Swahili language in the digital age.
#### Objectives
* To **Develop a Multi-Class Classification Model that utilizes machine learning techniques to categorize Swahili News.**
* To **Enhance User Experience by improving the accessibility of Swahili news content by enabling automated categorization on online news platforms.**
* To **Promote Swahili Language by contributing to the representation and preservation of Swahili in digital media by ensuring its inclusion and visibility in online products and services.**
## Data
The data used is from Zindi Africa and has 5151 Swahili articles and 3 features.
## Data Preparation
During this process, we checked for null values, the shape of the database, and investigated the distribution of the Swahili news categories.
## Exploratory Data Analysis
Here, we label-encoded the categories, corrected punctuations where necessary, tokenized our dataset, and created subplots for our tokens.
## Modelling
We performed the following machine learning models:
* **a baseline model without balancing the categories**
* **Baseline model with balanced categories**
* **MultinomialNB**
* **Logistic Regression**	
*	**Decision Tree Classifier**	
*	**RandomForestClassifier**	
*	**XGBClassifier**	
* **LGBMClassifier**	
* **CatBoostClassifier**

## Evaluation
We evaluated the precision, accuracy, recall, f1 scores and log loss for these models.
## Modelling Results
**baseline model**: The baseline performance of the Multinomial Naive Bayes classifier was evaluated using cross-validation. It had an overall accuracy of approximately 39.2%. This indicates that the classifier's predictive ability is slightly better than random chance. However, it's essential to consider the class balance within the dataset, as it significantly influences the interpretation of the results. The class distribution reveals that the majority class, 'Kitaifa', comprises approximately 39.2% of the training data, while the other classes, such as 'michezo' and 'Biashara', are less represented, with proportions of around 32.9% and 26.4%, respectively. The imbalance could affect the classifier's performance, potentially leading to biased predictions favoring the majority class. Thus, while the baseline accuracy provides a starting point for model evaluation, it's crucial to employ additional performance metrics and techniques, such as class weighting or resampling methods, to address the class imbalance and improve the classifier's effectiveness.

We used random oversampler which works by adding more instances on the minority classes and removed stopwords then ran the baseline again. The model performed way better as it yielded a substantially higher mean cross-validated accuracy score of about 0.626.

We performed domain feature engineering and  we did subplots to visualize the distribution of sentences by category. We iteratively ran seven models in order for us to examine which models performed the best in terms of accuracy and log loss.

**MultinomialNB**: This model has an accuracy of 0.821429 and log loss of 0.482447. 

**Logistic Regression**: The logistic regression model has an accuracy of 0.836957 and a log loss of 0.438784.

**Decision Tree Classifier**: The Decision Tree Classifier has an accuracy of 0.712733 and log loss of	10.354155.

**RandomForestClassifier**: The RandomForest Classifier has an accuracy of 0.840839 and log loss of	0.575659.

**XGBClassifier**: The XGBClassifier has an accuracy of 0.856366	and log loss of 0.457004.

**LGBMClassifier**: The LGBMClassifier has an accuracy of 0.858696 and log loss of 0.464706.

**CatBoostClassifier**: The CatBoostClassifier has an accuracy of 0.847826 and log loss of 0.420255.

We decided to dig deeper into  the LGBMClassifier and CatBoostClassifier since they had the highest accuracies to determine their precision, accuracy, recall, f1 score. We also did confusion matrices for them.




  ## Conclusion and Recommendations 



  ## Necessary Links
* **Jupyter Notebook** [Notebook]()
* **Presentation** [Powerpoint Presentation]()
