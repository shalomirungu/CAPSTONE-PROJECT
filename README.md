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

We also did confusion matrices for them. According to the results, the LGBMClassifier achieved the highest accuracy of 0.859, closely followed by the CatBoostClassifier with an accuracy of 0.848. However, considering both accuracy and log loss, it's noteworthy that the CatBoostClassifier attained the lowest log loss of 0.420 among all models. We decided to dig deeper into  the Logistic regression model, LGBMClassifier and CatBoostClassifier since they had the highest accuracies to determine their precision, accuracy, recall, f1 score.



Examining the confusion matrices across the three classifications, we observe varying degrees of mislabeled posts. The MultinomialNB shows the highest number of mislabeled posts at 131, followed by 93 in LogisticRegression, and finally, 75 under CatBoostClassifier. This indicates disparities in the models' abilities to accurately classify instances, with MultinomialNB exhibiting the greatest challenge in correctly labeling posts.

* **The Logistic regression model** has Precision of 0.836954, Recall of 0.836957, F1-Score of 0.835880, and Accuracy of 0.836957.
* **MultinomialNB** has Precision of 0.848309, Recall of 0.821429, F1-Score of 0.823609 and Accuracy of 0.821429.
* **CatBoostClassifier** has Precision of 0.851815, Recall of 0.847826, F1-Score of 0.849121 and Accuracy of 0.847826.

While MultinomialNB exhibits a lower recall, indicating potential limitations in capturing all positive instances, CatBoostClassifier stands out with its high precision, showcasing its proficiency in making accurate positive predictions while minimizing false positives. We performed Stacking to combine multiple classification or regression models via a meta-classifier or meta-regressor.

Rare classes like Burudani and Kimataifa in the dataset pose challenges for the model in correctly classifying instances belonging to these classes. Consequently, the confusion matrix below shows lower values for true positives for these rare classes.





On the stacking model, we achieved an accuracy of 0.8524, implying that approximately 85.24% of the model's predictions were correct. Additionally, with a log loss of 0.3984, it suggests that the model's predictions are closely aligned with the actual probabilities.

## Conclusion
Upon examining the confusion matrix, the model exhibits robustness in its predictions, effectively capturing the underlying patterns and features of the data. There are no instances of mislabeled posts, indicating consistent and precise predictions across all classes. This reliability suggests that the model can be trusted for accurate classifications. Given its strong performance, the model may be suitable for deployment in real-world applications where accurate classification is crucial.

While the absence of mislabeled posts is encouraging, further analysis is warranted to ensure the model's robustness across diverse datasets or conditions and to identify any potential biases or limitations.








  ## Recommendations 



  ## Necessary Links
* **Jupyter Notebook** [Notebook]()
* **Presentation** [Powerpoint Presentation]()
