

class Model:
    def __init__(self):
        self.model = self.load_model()

    def train(self):
        return

    def load_model(self):
        #Load in model
        return "Model"

    def predict(self, informationobject):
        #ExtractUserID
        UID = informationobject.getUID()

        #Extract Lat/long
        latlong = informationobject.getLATLONG()

        #Extract category
        category = informationobject.getCATEGORY()

        #scores = PredictOnBacth(Userid, latlong, category)

        bestPOIS = []
        #For score in scores:
            #Get 10 best pois, add them to bestPois

        return bestPOIS

class InformationHolder:
    def __init__(self, userID, latlongtuple, categories):
        self.userID = userID
        self.latlongtuple = latlongtuple
        self.category = categories
        self.numberofpoi = len(self.category)

    def getUID(self):
        UIDs = []

        for i in range(self.numberofpoi):
            UIDs.append(self.userID)
        return UIDs

    def getLATLONG(self):
        return self.latlongtuple

    def getCATEGORY(self):
        return self.category