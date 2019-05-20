"""
database functionality
"""

# Python Core Imports
import datetime
import pandas as pd

# Third Party Imports
import pyodbc
from flask import g, current_app as app

# Application Imports
from utils import label_sentences


def create_connection():
    connection = pyodbc.connect(Driver= app.config['DATABASE_DRIVER'], Server= app.config['DATABASE_HOST'], Database= app.config['DATABASE_NAME'], uid= app.config['UID'], pwd= app.config['PWD'])
    return connection


def get_db():
    if 'db' not in g:
        g.db = create_connection()

    return g.db


def file_order_info(file_Order_Id):
    """
    returns file order info for given file order id
    """
    conn = get_db()
    customer_id, customer_order_id, file_name = None, None, None
    query = 'EXEC [dbo].[SP_GetFileOrder] @FileOrderID = {0}'.format(file_Order_Id)
    file_order = pd.read_sql(query, conn)
    if not file_order.empty:
        customer_id, customer_order_id, file_name = str(file_order.CustomerID.values[0]), str(file_order.CustomerOrderID.values[0]), str(file_order.FileName.values[0])
    return customer_id, customer_order_id, file_name


def file_page_text_info(file_order_id):
    """
    returns file page text for given file order id
    """
    conn = get_db()
    x_test, y_test, data = None, None, None
    query = 'EXEC [dbo].[GET_FilePageTextByOrderId] @FileOrderID = {0}'.format(file_order_id)
    dataset = pd.read_sql(query, conn)
    if not dataset.empty:
        x_test, y_test =  dataset.PageText.values, dataset.DocumentIdentifierID.values
        data = label_sentences(x_test, y_test,'Test')
    return data, y_test


def client_document_names(customer_id):
    """
    mapping for ML trained and document identifier ids and names dictionary for given client id
    """
    conn = get_db()
    ml_docs_ids = None
    docs_names = None
    query = 'EXEC [dbo].[Get_MLSpecificDocumentsByClient] @CustomerID = {0}'.format(customer_id)
    dataset = pd.read_sql(query, conn)
    if not dataset.empty:
        ml_docs_ids = dataset.loc[:, ['MLDocumentID','DocumentIdentifierID']].set_index('MLDocumentID').to_dict()['DocumentIdentifierID']
        docs_names = dataset.loc[:, ['DocumentIdentifierID','DocumentName']].set_index('DocumentIdentifierID').to_dict()['DocumentName']
    return ml_docs_ids, docs_names


def insert_indexing_result(res, file_Order_Id):
    """
    inserts resultant documents into indexing result
    """
    cnxn = get_db()
    crsr = cnxn.cursor()
    crsr.fast_executemany = True
    sql = "INSERT INTO IndexingResult(FileOrderID, DocumentIdentifierID, DocumentName, PageFrom, PageTo, PageCount, Accuracy_Level) VALUES(?,?,?, ?,?,?,?)"
    params = [(file_Order_Id, r.Document_Id, r.Document_Name, r.Page_From, r.Page_To, r.Page_Count, r.Accuracy,) for r
              in res]
    crsr.executemany(sql, params)
    cnxn.commit()


def update_file_page_text_with_customer_order_id(res, file_Order_Id, customer_id, customer_order_id):
    """
    updates file page text attributes for indexing results
    """
    cnxn = get_db()
    crsr = cnxn.cursor()
    crsr.fast_executemany = True

    sql = "UPDATE FilePageText SET DocumentID = ?, DocumentName=?, CustomerID=?, CustomerOrderID=? Where FileOrderID = ? and PageNumber >= ? and PageNumber <= ?"
    params = [[r.Document_Id, r.Document_Name, customer_id, customer_order_id, file_Order_Id, r.Page_From, r.Page_To]
              for r in res]
    crsr.executemany(sql, params)
    cnxn.commit()


def update_file_page_text(res, file_Order_Id, customer_id):
    """
    updates file page text attributes for indexing results
    """
    cnxn = get_db()
    crsr = cnxn.cursor()
    crsr.fast_executemany = True

    sql = "UPDATE FilePageText SET DocumentID = ?, DocumentName=?, CustomerID=? Where FileOrderID = ? and PageNumber >= ? and PageNumber <= ?"
    params = [[r.Document_Id, r.Document_Name, customer_id, file_Order_Id, r.Page_From, r.Page_To] for r in res]
    crsr.executemany(sql, params)
    cnxn.commit()


def insert_into_logs(file_Order_Id, log_type, message):
    """
    inserts file order log
    """
    cnxn = get_db()
    crsr = cnxn.cursor()
    sql = 'exec [dbo].[SP_Logs_Insert] ?, ?, ?, ?, ?, ?'
    values = (file_Order_Id, datetime.datetime.now(), log_type, message, 0, 0)
    crsr.execute(sql, (values))
    cnxn.commit()


def update_indexed_document_counts_in_file_order(file_Order_Id, indexed_docs):
    """
    count of indexed documets against file order id
    """
    cnxn = get_db()
    crsr = cnxn.cursor()
    sql = 'exec [dbo].[Update_Indexed_Document_Counts] ?, ?'
    values = (file_Order_Id, indexed_docs)
    crsr.execute(sql, (values))
    cnxn.commit()


def insert_into_processed_order_logs(file_Order_Id, ml_model_time, total_indexing_time):
    """
    insert ML model time and total indexing time by ML api
    """
    cnxn = get_db()
    crsr = cnxn.cursor()
    sql = 'exec [dbo].[InsertModelAndIndexingTimeByML] ?, ?, ?'
    values = (file_Order_Id, "{0:.2f}".format(ml_model_time), "{0:.2f}".format(total_indexing_time))
    crsr.execute(sql, (values))
    cnxn.commit()


def delete_existing_file_Order_log(file_order_Id):
    """
    deletes from file order log
    """
    cnxn = get_db()
    crsr = cnxn.cursor()
    sql = "DELETE FROM fileorderlog where FileOrderID = ?"
    params = (file_order_Id,)
    crsr.execute(sql, params)
    cnxn.commit()


def insert_file_Order_log(file_Order_Id, order_log):
    """
    inserts file order log
    """
    delete_existing_file_Order_log(file_Order_Id)
    cnxn = get_db()
    crsr = cnxn.cursor()
    sql = "INSERT INTO fileorderlog(FileOrderID, OrderLog) VALUES(?,?)"
    params = (file_Order_Id, order_log,)
    crsr.execute(sql, params)
    cnxn.commit()
