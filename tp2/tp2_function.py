import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.metrics import f1_score, zero_one_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string


def clean_data(data):
    data = data.apply(remove_stopwords)
    data = data.apply(preprocess_string)
    data = data.apply(lambda x: " ".join(x))
    return data

def get_tfidf(corpus):
    tfidf = TfidfVectorizer()
    tfidf.fit_transform(corpus)
    tfidf_dict = {}
    for ele1, ele2 in zip(tfidf.get_feature_names_out(), tfidf.idf_):
        tfidf_dict[ele1] = ele2
    return tfidf, tfidf_dict

def encode_tfidf(sentence: str, tfidf_dict: dict, vector_size: int):
	vec = np.zeros(vector_size)
	for idx, word in enumerate(sentence.split(" ")):
		try:
			if idx >= vector_size:
				break
			vec[idx] = tfidf_dict[word]
		except:
			pass
	return vec.tolist()

def fit_predict_custom(
    X_train, y_train, X_test, base_estimator, model, *args, **kwargs
):
    m = model(base_estimator, *args, **kwargs)
    m.fit(X_train, y_train)
    m = m.predict(X_test)
    return m


def run_model(data, labels, base_estimator):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.5)
    X_train = np.array(X_train.tolist())
    X_test = np.array(X_test.tolist())

    for key, value in base_estimator.items():
        print("##################", key, "##################")
        moc = fit_predict_custom(X_train, y_train, X_test, value, MultiOutputClassifier)
        chain = fit_predict_custom(
            X_train,
            y_train,
            X_test,
            value,
            ClassifierChain,
            order="random",
            random_state=42,
        )

        print(
            "----------MultiOutputClassifier----------",
        )
        print(f1_score(y_test, moc, average="micro"))
        print(f1_score(y_test, moc, average="macro"))
        print(zero_one_loss(y_test, moc))

        print(
            "----------ClassifierChain----------",
        )
        print(f1_score(y_test, chain, average="micro"))
        print(f1_score(y_test, chain, average="macro"))
        print(zero_one_loss(y_test, chain))
        
def encode_sentence(sentence: str, model, tfidf_dict):
    vec = np.zeros(model.vector_size)
    for word in sentence.split(" "):
        try:
            if word in tfidf_dict.keys():
                vec += tfidf_dict[word] * model.wv[word]
            else:
                vec += model.wv[word]
        except:
            pass
    return vec.tolist()
