
# Python Core Imports
import re
import pickle
import time

# Third Party Imports
from flask import current_app as app
from nltk import word_tokenize
from nltk.corpus import stopwords
import gensim
from gensim.models import doc2vec

# Application imports
from document import Document
from accuracy import Accuracy


TaggedDocument = gensim.models.doc2vec.TaggedDocument


def clean_text(text):
    norm_text = text.lower()
    # Replace breaks with spaces
    norm_text = re.sub("[^a-zA-Z0-9]", " ", norm_text)
    # Word Tokenize
    words = word_tokenize(norm_text)

    # Remove stopwords
    stops = set(stopwords.words("english"))
    words = [w for w in words if not w in stops]
    return words


def vec_for_learning(doc2vec_model, tagged_docs, step_count=0):
    """
        Given the document text returns the vector representation using the doc2vec model
    """
    docs = tagged_docs
    targets, regressors = zip(*[(doc.tags[0], doc2vec_model.infer_vector(doc.words, steps=step_count)) for doc in docs])
    return targets, regressors


def load_d2v(path):
    """
    loads doc2vec model
    """
    return doc2vec.Doc2Vec.load(path)


def load_model(name):
    """
    loads pickled model file
    """
    data = None
    with open(name, 'rb') as handle:
        data = pickle.load(handle)
    return data


def label_sentences(X, y, label_type):
    """
    Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
    We do this by using the TaggedDocument method. where first parameter is the cleaned text and the second
    is the document class number
    """
    labeled = []
    for i, v in enumerate(X):
        labeled.append(TaggedDocument(clean_text(str(v)), [y[i]]))
    return labeled


def test_model(d2v, model, testing_vectors, testing_labels):
    """
    give model results for given text
    """
    start_time = time.time()
    test_My, test_Mx = vec_for_learning(d2v, testing_vectors, step_count=20)
    vec_t = time.time() - start_time
    start_time = time.time()
    testing_predictions = model.predict(test_Mx)
    class_t = time.time() - start_time
    start_time = time.time()
    testing_predictions_prob = model.predict_proba(test_Mx)
    prob_t = time.time() - start_time
    return test_My, testing_predictions, testing_predictions_prob, vec_t, class_t, prob_t


def page_confidence(pred_prob, index):
    """
    converts confidence into percentage
    """
    return round(max(pred_prob[index]) * 100, 2)


def page_confidence_list(pred_prob):
    """
    list of page confidence
    """
    res = []
    length = len(pred_prob)
    i = 0
    while i < length:
        confidence = page_confidence(pred_prob, i)
        res.append(confidence)
        i = i + 1
    return res


def convert_to_confident_doc_list(arr, confidence_list):
    """
    converts list of confident documents
    """
    res = []
    length = len(arr)
    i = 0
    cur_item = None
    item_type = None

    while i < length:
        cur_item = arr[i].split('-')
        if len(cur_item) > 1:
            item_type = 'last'
        elif cur_item == 'Other':
            item_type = 'mid'
        else:
            item_type = 'start'

        if ((item_type == 'start' or item_type == 'last') and confidence_list[i] < app.config['FAIR_ACCURACY_SCORE']):
            res.append('Other')
        else:
            res.append(arr[i])

        i = i + 1
    return res


def average_prob(page_from, page_to, pred_prob):
    """
    average probability
    """
    i = page_from - 1
    total_pages = page_to - page_from + 1
    sum = 0

    while i < page_to:
        sum += max(pred_prob[i])
        i = i + 1

    return sum / total_pages


def result_accuracy(page_from, page_to, doc_id, pred_prob):
    """
    returns accuracy of document as defined in config
    """
    if doc_id == app.config['UNKNOWN_DOCUMENT']:
        return Accuracy.bad.value

    avg_prob = average_prob(page_from, page_to, pred_prob) * 100

    if avg_prob >= app.config['GOOD_ACCURACY_SCORE']:
        return Accuracy.good.value

    if avg_prob >= app.config['FAIR_ACCURACY_SCORE']:
        return Accuracy.fair.value

    return Accuracy.bad.value


def translated_predictions(arr):
    """
    translates predictions
    """
    res = []
    length = len(arr)
    i = 0
    while i < length:
        splitted_Item = arr[i].split('-')
        if splitted_Item[0] == 'Other':
            if i - 1 >= 0:
                res.append(res[i - 1])
            else:
                res.append(app.config['UNKNOWN_DOCUMENT'])
        else:
            res.append(splitted_Item[0])
        i = i + 1
    return res


def final_documents(docs, arr, ml_doc_ids, document_names, pred_prob):
    """
    converts to documents based upon confidence and page number
    """
    res = []
    length = len(arr)
    i = 0
    page_from = 1
    page_to = 1

    if (length == 1):
        if docs[i] == app.config['UNKNOWN_DOCUMENT'] or docs[i] not in ml_doc_ids:
            name = 'Unknown'
            docs[i] = app.config['UNKNOWN_DOCUMENT']
        else:
            name = document_names[ml_doc_ids[docs[i]]]
            docs[i] = ml_doc_ids[docs[i]]
        doc = Document(docs[i], name, page_from, page_to, page_to - page_from + 1,
                       result_accuracy(page_from, page_from, docs[i], pred_prob),
                       average_prob(page_from, page_from, pred_prob) * 100)
        res.append(doc)
        return res

    while i < length:
        if (i + 1 < length and arr[i] >= arr[i + 1]) or i + 1 == length:
            if docs[i] == app.config['UNKNOWN_DOCUMENT'] or docs[i] not in ml_doc_ids:
                name = 'Unknown'
                docs[i] = app.config['UNKNOWN_DOCUMENT']
            else:
                name = document_names[ml_doc_ids[docs[i]]]
                docs[i] = ml_doc_ids[docs[i]]
            page_to = i + 1
            doc = Document(docs[i], name, page_from, page_to, page_to - page_from + 1,
                           result_accuracy(page_from, page_from, docs[i], pred_prob),
                           average_prob(page_from, page_from, pred_prob) * 100)
            res.append(doc)
            page_from = page_to + 1
        i = i + 1
    return res


def translated_predicted_docs_pages(arr):
    """
    translates docs list of ML into list of pages
    """
    pages = []
    docs = []
    length = len(arr)
    if length <= 0:
        return docs, pages

    i = 0
    page = 0
    doc_start = None
    if length > 0 and arr[0].split('-')[0] != 'Other':
        doc_start = arr[0].split('-')[0]

    while i < length:
        cur_item = arr[i].split('-')

        if len(cur_item) == 2:
            if ((i - 1 >= 0 and (len(arr[i - 1].split('-')) == 2 or docs[i - 1] == app.config['UNKNOWN_DOCUMENT']) or
                 (doc_start is not None and doc_start != cur_item[0]))):
                page = 1
                pages.append(page)
                docs.append(cur_item[0])
            else:
                page += 1
                pages.append(page)
                docs.append(cur_item[0])
                page = 1
        elif cur_item[0] == 'Other':
            if i - 1 >= 0:
                if len(arr[i - 1].split('-')) == 1:
                    page += 1
                    pages.append(page)
                    docs.append(docs[i - 1])
                else:
                    page = 1
                    pages.append(page)
                    docs.append(app.config['UNKNOWN_DOCUMENT'])
            else:
                page = 1
                pages.append(page)
                docs.append(app.config['UNKNOWN_DOCUMENT'])
        else:
            page = 1
            pages.append(page)
            docs.append(cur_item[0])

        if page == 1 and cur_item[0] != 'Other':
            doc_start = cur_item[0]
        i = i + 1

    return docs, pages


def translate_problematic_docs(pdocs, docs):
    """
    logic to deal with problematic documents
    """
    p_length = len(pdocs)
    length = len(docs)
    if p_length <= 0 or length <= 0:
        return
    has_history = False
    in_history = None
    i = 0
    while i < length:
        cur_item = docs[i].split('-')
        if docs[i] in pdocs:
            if i > 0 and has_history == True and docs[i - 1] == 'Other' and in_history == docs[i]:
                docs[i] = docs[i] + '-last'
                has_history = False
                in_history = None
            elif has_history == True and cur_item[0] != 'Other' and in_history != docs[i]:
                has_history = True
                in_history = docs[i]
            elif has_history == False:
                has_history = True
                in_history = docs[i]
        i = i + 1


def filter_unknown(docs):
    """
    filters unknown documents from result
    """
    res = []
    if len(docs) > 0:
        res = [doc for doc in docs if doc.Document_Id != app.config['UNKNOWN_DOCUMENT'] and doc.Accuracy != Accuracy.bad.value]
    return res

