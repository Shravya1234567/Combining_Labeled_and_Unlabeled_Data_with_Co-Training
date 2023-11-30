## Implementation of Cotrain algorithm for text classification
This project is the implementation of the cotrain paradigm mentioned in the below paper:

[Blum, Avrim, and Tom Mitchell. "Combining labeled and unlabeled data with co-training." Proceedings of the eleventh annual conference on Computational learning theory. 1998.](https://www.cs.cmu.edu/~avrim/Papers/cotrain.pdf)

The project is implemented in python using the scikit-learn library.

### Introduction
There are some algorithms which require a large amount of labeled data to train the classifier. But, in some cases, it is difficult to get a large amount of labeled data. In such cases, the unlabeled data can be used to improve the performance of the classifier. One such algorithm is the cotrain algorithm which is mentioned in the above paper.

The cotrain algorithm is a semi-supervised learning algorithm which uses the unlabeled data to improve the performance of the classifier. The algorithm uses two classifiers and two views of the data to train the classifiers. The algorithm is implemented for text classification using the bag of words model. The algorithm is tested on the WebKB dataset and the News Category dataset. The algorithm is also tested on the Sentence Polarity dataset to check the performance of the algorithm on a binary classification problem.

### Project Structure


### Requirements
- The requirements for the project are listed in the requirements.txt file.
- They could be installed using the following command:

```bash
pip install -r requirements.txt
```
- The nltk package is used for the wordnet lemmatizer.
- The following commands are used to download the required packages:

```bash
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
```

### Datasets
The datasets used in the project are the following:
- WebKB dataset(binary): [http://www.cs.cmu.edu/~webkb/](http://www.cs.cmu.edu/~webkb/) (mentioned in the paper)

- WebKB dataset(multiclass): [https://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/](https://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/) (Implemented for multiclass classification)

- News Category Dataset: [https://www.kaggle.com/datasets/rmisra/news-category-dataset](https://www.kaggle.com/rmisra/news-category-dataset) (Additional dataset to test the algorithm)

- Sentence Polarity Dataset: [https://www.kaggle.com/datasets/nltkdata/sentence-polarity] (https://www.kaggle.com/datasets/nltkdata/sentence-polarity) (Additional dataset to test the algorithm)

### Instructions to run the models

- The models can be run using the following command:

```bash
python3 inference.py filepath_of_view1_classifier filepath_of_view2_classifier X_test_filepath y_test_filepath
```

### Results

- The project report has all the information about the results of the project.






