"""
ML model result is converted and transformed into Document class to save it in indexing result
"""


class Document:
    Document_Id = 0
    Document_Name = ''
    Page_From = 0
    Page_To = 0
    Page_Count = 0
    Accuracy = 0
    Average_Prob = 0

    def __init__(self, Document_Id, Document_Name , Page_From, Page_To, Page_Count, Accuracy, Average_Prob):
        """

        :param Document_Id:
        :param Document_Name:
        :param Page_From:
        :param Page_To:
        :param Page_Count:
        :param Accuracy:
        :param Average_Prob:
        """
        self.Document_Id = Document_Id
        self.Document_Name = Document_Name
        self.Page_From = Page_From
        self.Page_To = Page_To
        self.Page_Count = Page_Count
        self.Accuracy = Accuracy
        self.Average_Prob = Average_Prob