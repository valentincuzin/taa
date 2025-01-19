from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
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
        chains = []
        for x in range(2):
            chains.append(fit_predict_custom(
                X_train,
                y_train,
                X_test,
                value,
                ClassifierChain,
                order="random",
                random_state=42,
            ))

        print(
            "----------MultiOutputClassifier----------",
        )
        print('f1 micro', f1_score(y_test, moc, average="micro"))
        print('f1 macro', f1_score(y_test, moc, average="macro"))
        print('0/1 loss', zero_one_loss(y_test, moc))

        print(
            "----------ClassifierChain----------",
        )
        f1_score_micro = []
        f1_score_macro = []
        zero_one_losses = []
        for chain in chains:
            f1_score_micro.append(f1_score(y_test, chain, average="micro"))
            f1_score_macro.append(f1_score(y_test, chain, average="macro"))
            zero_one_losses.append(zero_one_loss(y_test, chain))
        print('f1 micro', np.mean(f1_score_micro))
        print('f1 macro', np.mean(f1_score_macro))
        print('0/1 loss', np.mean(zero_one_losses))

def plot_word_vec(model, word, topn=5):	
	final_tag = [(word, 1)]
	total_encode = []
	final_tag.extend(model.wv.most_similar(positive=word, topn=topn))
	for tag in final_tag:
		try:
			total_encode.append(model.wv[tag[0]])
		except Exception as e:
			print(str(e))
	total_encode = pd.DataFrame(total_encode)

	fig, ax = plt.subplots()
	ax.quiver([0 for _ in range(len(total_encode))], [0 for _ in range(len(total_encode))], total_encode[0], total_encode[1], angles='xy', scale_units='xy', scale=1)

	# Ajouter les labels
	for i, tag in enumerate(final_tag):
		ax.annotate(tag[0], (total_encode[0][i], total_encode[1][i]))
	# Définir les limites du plot pour inclure toutes les flèches
	ax.set_xlim([min(total_encode[0]) - 1, max(total_encode[0]) + 1])
	ax.set_ylim([min(total_encode[1]) - 1, max(total_encode[1]) + 1])
	plt.show()

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


def encode_sentence2(sentence: str, model):
    vec = np.zeros(model.vector_size)
    nb_word = 0
    for word in sentence.split(" "):
        try:
            vec += model.wv[word]
            nb_word += 1
        except:
            pass
    nbword = 1 if nb_word == 0 else nb_word
    vec = vec / nb_word
    return vec.tolist()