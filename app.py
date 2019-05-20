"""
end point for indexing through ML
"""

# Python Core Imports
import logging
from logging import Formatter, FileHandler
from io import StringIO
import json
import time
import re

# Third Party Imports
from flask import request, g
from flask_api import FlaskAPI, status

# Application Imports
from db import file_order_info, file_page_text_info, client_document_names, insert_indexing_result
from db import update_indexed_document_counts_in_file_order, update_file_page_text, insert_file_Order_log
from db import update_file_page_text_with_customer_order_id, insert_into_processed_order_logs
from utils import load_d2v, load_model, test_model, translate_problematic_docs, page_confidence_list
from utils import convert_to_confident_doc_list, translated_predicted_docs_pages, filter_unknown, final_documents


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def perform_indexing(file_order_id):
    start_time = time.time()
    customer_id, customer_order_id, file_name = file_order_info(file_order_id)
    if customer_id is None or customer_order_id is None or file_name is None:
        app.logger.info("No file Order exists against fileOrderId: {0}".format(file_order_id))
        return
    
    ml_doc_ids, key_val_docs = client_document_names(customer_id)
    if key_val_docs is None:
        app.logger.info("No documents available against customer ID: {0}".format(customer_id))
        return
    
    rds, new_y_test = file_page_text_info(file_order_id)
    if rds is None:
        app.logger.info("No file Page Text exists against fileOrderId: {0}".format(file_order_id))
        return
    
    Y , pred_y, pred_prob_y, vec_t, class_t, prob_t = test_model(d2v_model, classifier_model, rds, new_y_test)
    pred_y = pred_y.tolist()
    pred_prob_y = pred_prob_y.tolist()
    translate_problematic_docs(app.config['PROBLEMATIC_DOCUMENTS'], pred_y)
    confidence_list = page_confidence_list(pred_prob_y)
    confident_doc_list = convert_to_confident_doc_list(pred_y, confidence_list)
    docs, pages = translated_predicted_docs_pages(confident_doc_list)

    all_docs = []
    if len(docs) > 0 and len(pages) > 0:
        all_docs = final_documents(docs, pages, ml_doc_ids, key_val_docs, pred_prob_y)

    res = filter_unknown(all_docs)
    indexed_docs = len(res)
    if indexed_docs > 0:
        insert_indexing_result(res, file_order_id)
        update_indexed_document_counts_in_file_order(file_order_id, indexed_docs)
        
        if re.match("^\d+?$", customer_order_id) is None:
            update_file_page_text(res, file_order_id, customer_id)
        else:
            update_file_page_text_with_customer_order_id(res, file_order_id, customer_id, customer_order_id)
    indexing_time = time.time() - start_time
    insert_into_processed_order_logs(file_order_id, prob_t, indexing_time)
    in_mem_file = StringIO()
    
    in_mem_file.write("\nCustomer Order ID: {0}".format(customer_order_id))
    in_mem_file.write("\nFile Order ID: {0}".format(file_order_id))
    #in_mem_file.write("\npredicted docs\n")
    #in_mem_file.write(', '.join(pred_y))
    #in_mem_file.write("\nconverted docs\n")
    #in_mem_file.write(', '.join(docs))
    #in_mem_file.write("\nconverted Pages\n")
    #in_mem_file.write(', '.join([str(p) for p in pages]))
    #in_mem_file.write("\nTrained Classes\n")
    #in_mem_file.write(', '.join(classifier_model.classes_))

    if pred_y is not None and len(pred_y) > 0:
        #app.logger.info("Predicted Classes With Maximum Confidence")
        in_mem_file.write("\nPredicted Classes With Maximum Confidence\n")

        index = 0
        length = len(pred_y) 
        model_length = len(classifier_model.classes_)
        k = 0 
        while index < length:
            k = 0
            in_mem_file.write("\nPage# {0} Predicted Document Class: {1} Max Confidence: {2}%".format(index+1, classifier_model.classes_[pred_prob_y[index].index(max(pred_prob_y[index]))], round(max(pred_prob_y[index])*100, 2)))
            in_mem_file.write("\n")
            index = index + 1

        in_mem_file.write("\n")
        in_mem_file.write("\nAll Converted Documents\n")
        index=1
        for r in all_docs:
            in_mem_file.write("\nSr# {6} Document ID: {0} Page From: {1} Page To: {2} Page Count: {3} Accuracy: {4} Confidence: {5}%".format(r.Document_Id, r.Page_From, r.Page_To, r.Page_Count, r.Accuracy, round(r.Average_Prob, 2), index))
            index = index + 1

        in_mem_file.write("\n")
        in_mem_file.write("\nIndexed Documents\n")
        index=1
        for r in res:
            in_mem_file.write("\nSr# {6} Document ID: {0} Page From: {1} Page To: {2} Page Count: {3} Accuracy: {4} Confidence: {5}%".format(r.Document_Id, r.Page_From, r.Page_To, r.Page_Count, r.Accuracy, round(r.Average_Prob, 2), index))
            index = index + 1
        
        in_mem_file.write("\nTotal Predicted Pages: {0}".format(length))
        in_mem_file.write("\nTotal Converted Documents: {0}".format(len(all_docs)))
        in_mem_file.write("\nTotal Indexed Documents: {0}".format(indexed_docs))

    in_mem_file.write("\nTime Taken by ML model: {0}".format(prob_t))
    in_mem_file.write("\nTotal Indexing Time by API: {0}".format(indexing_time))
    log_for_db = in_mem_file.getvalue()
    in_mem_file.close()
    insert_file_Order_log(file_order_id, log_for_db)





with open('config.json', 'r') as f:
    config = json.load(f)

doc2vec_model_path = config['MODELS']['DOC2VEC']
random_forest_model_path =  config['MODELS']['RANDOM_FOREST']

d2v_model = load_d2v(doc2vec_model_path)
classifier_model = load_model(random_forest_model_path)

file_handler = FileHandler('app.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
        Formatter('%(asctime)s : %(levelname)s : %(message)s'))


app = FlaskAPI(__name__)
app.logger.addHandler(file_handler)
app.config.update(
    DATABASE_DRIVER = config['DATABASE_CONFIG']['DRIVER'],
    DATABASE_HOST = config['DATABASE_CONFIG']['HOST'],
    DATABASE_NAME = config['DATABASE_CONFIG']['DATABASE_NAME'],
    UID = config['DATABASE_CONFIG']['USERNAME'],
    PWD = config['DATABASE_CONFIG']['PASSWORD'],
    GOOD_ACCURACY_SCORE = config['APP_SETTINGS']['GOOD_ACCURACY_SCORE'],
    FAIR_ACCURACY_SCORE = config['APP_SETTINGS']['FAIR_ACCURACY_SCORE'],
    UNKNOWN_DOCUMENT = config['APP_SETTINGS']['UNKNOWN_DOCUMENT'],
    PROBLEMATIC_DOCUMENTS = config['APP_SETTINGS']['PROBLEMATIC_DOCUMENTS']
)


def handle_bad_request(e):
    return 'Bad Request', status.HTTP_400_BAD_REQUEST


def handle_internal_server_error(e):
    return 'Internal Server Error', status.HTTP_500_INTERNAL_SERVER_ERROR


app.register_error_handler(status.HTTP_400_BAD_REQUEST, handle_bad_request)
app.register_error_handler(status.HTTP_500_INTERNAL_SERVER_ERROR, handle_internal_server_error)


@app.teardown_appcontext
def teardown_db(self):
    db = g.pop('db', None)

    if db is not None:
        db.close()


@app.route("/", methods=['GET'])
def index():
    file_order_id = None
    try:
        file_order_id = str(request.args.get('q')).strip()
        if file_order_id is not None and len(file_order_id) > 0:
            perform_indexing(file_order_id)
        else:
            return "Invalid Request parameters.", status.HTTP_400_BAD_REQUEST
        return "Indexing Completed.", status.HTTP_200_OK
    except Exception as ex:
        app.logger.info('Exception: {}'.format(str(ex)))