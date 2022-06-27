from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
import sys

class EmbModel(tf.keras.Model):
    def __init__(self, useridlength, category_length):
        super(EmbModel, self).__init__()
        self.d_steps = 1
        self.useridlength = useridlength
        self.category_length = category_length
        self.model = self.init_model()
        print(self.useridlength)
        
    def call(self, inputs):
        return
    
    def init_model(self):
        poi_latitude_input = keras.layers.Input(shape=(1,), name='poi_latitude')
        poi_longitude_input = keras.layers.Input(shape=(1,), name='poi_longitude')
        poi_concat_input = tf.keras.layers.Concatenate(axis=-1)([poi_latitude_input, poi_longitude_input])
        #input_length:  #This is the length of input sequences, as you would define for any input layer of a Keras model. 
                        #For example, if all of your input documents are comprised of 1000 words, this would be 1000
        #input_dim: 
                        #This is the size of the vocabulary in the text data. 
                        #For example, if your data is integer encoded to values between 0-10, then the size of the vocabulary would be 11 words.
        poi_dense = keras.layers.Dense(256)(poi_concat_input)
        poi_reshape = keras.layers.Reshape((1, 256))(poi_dense)
        
        category_input = keras.layers.Input(shape=(1), name='category_input')
        category_emb = keras.layers.Embedding(self.category_length, 256)(category_input)    
        category_concat = tf.keras.layers.Concatenate(axis=-1)([category_emb, poi_reshape])
    
        user_input = keras.layers.Input(shape=(1,), name='user_id')
        user_emb = keras.layers.Embedding(self.useridlength, 512)(user_input)
        #user_reshape = layers.Reshape((1, 256))(user_emb)
                                    
        dot = keras.layers.Dot(axes=(2))([category_concat, user_emb])
            
        model = Model([category_input, poi_latitude_input, poi_longitude_input, user_input], dot)
        model.summary()
        return model
    
    def compile_model(self, optimizer):
        super(EmbModel, self).compile(run_eagerly=True)
        self.optimizer = optimizer
        
    def train_step(self, data):
        if len(data) == 3:
            real_data, labels, sample_weight = data
        else:
            sample_weight = None
            real_data, labels = data
        cat_data = real_data[0]
        lat_data = real_data[1]
        long_data = real_data[2]
        user_data = real_data[3]

        for i in range(self.d_steps):
            with tf.GradientTape() as tape:
                
                #print(latlong_data[0])
                #print(latlong_data[1])
                #print(user_data)
                
                dotproduct = self.model([cat_data, lat_data, long_data, user_data])
                #print(dotproduct)
                # Loss function = ||S-GroundTruth|| 
                loss = tf.math.abs(tf.subtract(tf.cast(dotproduct, tf.float64), labels))
                #print(loss)
            d_gradient = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(d_gradient, self.model.trainable_variables))
        return {'loss': loss}
    
    def predict_step(self, data):
        sample_weight = None
        cat_data = real_data[0]
        lat_data = real_data[1]
        long_data = real_data[2]
        user_data = real_data[3]
        return self.model([cat_data, lat_data, long_data, user_data])

def one_hot_encode(ground_truth, lst):
    result = []
    for category in lst:
        oh_encoding = np.zeros(len(ground_truth))
        if category in ground_truth:
            print(category)
            index = np.where(ground_truth == category)[0][0]
            
            #Get index og category, and insert 1 into the vector.
            result.append(index)
    return result

print("Loading checkins")
checkin_cols = ['user_id', 'poi_id', 'timestamp', 'timezone']
checkins = pd.read_csv(r'/user/student.aau.dk/lharde18/Data/Den_checkins.csv', sep=',', names=checkin_cols, encoding='latin-1').dropna(axis=1)

print("Loading POIs")
venue_cols = ['poi_id', 'latitude', 'longitude', 'category', 'country_code']
pois = pd.read_csv(r'/user/student.aau.dk/lharde18/Data/Den_pois.csv', sep=',', names=venue_cols, encoding='latin-1')

#One checkin for each user
users = checkins.copy()
users.drop_duplicates(subset="user_id", keep = 'first', inplace = True)
print("Checkins: ", len(checkins))
print("Users: ", len(users))
len_checkins = len(checkins)
len_users = len(users)

#The rest of the checkins and categorie
#This takes checkins - users
checkins_rest = users.merge(checkins, how = 'outer' ,indicator=True).loc[lambda x : x['_merge']=='right_only']
print("Gotten: ", len(checkins_rest))

#One of each category in checkins_rest
categories1 = pd.DataFrame(pois, columns=['poi_id', 'category'])
categories1 = checkins_rest.merge(categories1, on='poi_id')

#Nu da vi har categories for hver af tingene i checkins_rest, tager vi en kopi af det, og dropper duplicates.
users_categories1 = categories1.copy()
users_categories1.drop_duplicates(subset="category", keep = 'first', inplace = True)
print("Unique categories in checkins_rest: ", len(users_categories1))

#Dataframe med ['user_id', 'poi_id', 'timestamp', 'timezone', 'category']
categories_cat_no_cat = pd.DataFrame(users_categories1, columns=['user_id', 'poi_id', 'timestamp', 'timezone', 'category'])

#Dataframe med ['user_id', 'poi_id', 'timestamp', 'timezone']
checkins_rest_no_merge = pd.DataFrame(checkins_rest, columns=['user_id', 'poi_id', 'timestamp', 'timezone'])

#Dataframe med ['poi_id', 'category']
poisandcategories = pd.DataFrame(pois, columns=['poi_id', 'category'])

#Dataframe med ['user_id', 'poi_id', 'timestamp', 'timezone', 'category']
checkins_rest_no_merge = checkins_rest_no_merge.merge(poisandcategories, on='poi_id')
cat_length = len(categories_cat_no_cat)
print("Total categories: ", len(categories_cat_no_cat))

#Udregner checkins_rest - categories
test = categories_cat_no_cat.merge(checkins_rest_no_merge, how = 'outer' ,indicator=True).loc[lambda x : x['_merge']=='right_only']

restset = pd.DataFrame(test, columns=['user_id', 'poi_id', 'timestamp', 'timezone'])
userset = users
categoryset = pd.DataFrame(users_categories1, columns=['user_id', 'poi_id', 'timestamp', 'timezone'])

print("Overview:")
print(userset)
print(categoryset)
print(restset)

encoding = userset.copy()
#print(encoding)
encoding_array = {}
temp = 0
for user in encoding.iterrows():
    user = user[0]
    value = encoding._get_value(user, 'user_id')
    encoding_array[value] = temp
    temp += 1

np.save(r"/user/student.aau.dk/lharde18/Data-output/Den/encoding_array.npy", encoding_array)

###################################
###################################
####### Iteration 1: User #########
###################################
###################################

#Laver incidence matrix af poi_id og user_id
print("Step 1")
checkin_data = userset.merge(pois, on='poi_id')
checkin_data.drop_duplicates(subset="user_id", keep = 'first', inplace = True)
df = checkin_data.set_index('user_id').poi_id.str.get_dummies(',')
df = df.groupby('user_id').max()

#Laver en version af checkin_data, som ikke har nogle duplicates, og extracter category og poi_id fra den.
print("Step 2")
checkin_data_no_duplicates = checkin_data.copy()
checkin_data_no_duplicates.drop_duplicates(subset ="poi_id", keep = 'first', inplace = True)
checkin_data_no_duplicates = pd.DataFrame(checkin_data_no_duplicates, columns = ['poi_id', 'category'])

#Extract categorical data
#Dette gøres ved, at category fra checkin_data bruges, hvortil duplicates bliver droppet. På denne måde har vi en liste af categories.
print("Step 3")
categories = pd.DataFrame(checkin_data, columns=['category'])
categories.drop_duplicates(subset ="category", keep = 'first', inplace = True)
category_length = len(categories)
categories_numpy = categories.to_numpy()


#Extracting all of the users and the pois
#
print("Step 3.5")
#Pga at der i df
listofusers = pd.DataFrame(checkin_data, columns= ['user_id']).groupby('user_id').max().sample(frac=1)
listofpois = pd.DataFrame(checkin_data, columns= ['poi_id', 'latitude', 'longitude']).groupby('poi_id').max().sample(frac=1)
userarray = listofusers.index.to_numpy()
poiarray = listofpois.index.to_numpy()
#Her har vi nu en dataframe fyldt med user_id og et med poi_id, uden duplicates. Så hvis vi tager dot-produktet, så er det muligt at at få en dataframe, som består af rækker af poi_id og user_id.
userdataframe = pd.DataFrame(userarray, columns = ['Users'])
poidataframe = pd.DataFrame(poiarray, columns = ['Poi'])
dot = userdataframe.merge(poidataframe, how='cross')

#Her skabes der en liste af dictionaries, hvor hver dictionary holder latitude/longitude versionen til user_id i dot. (Forhåbentligt i rækkefølge)
print("Step 4")
rows_list = []
for i in range(len(dot)):
    temp = dot.loc[i, "Poi"]
    latitude = listofpois.loc[temp]['latitude']
    longitude = listofpois.loc[temp]['longitude']
    dict1 = {'latitude':latitude, 'longitude':longitude}
    rows_list.append(dict1)
    #latitude = poiarray[i]
latlong = pd.DataFrame(rows_list)

#Her kombineres hver user_id med dens tilsvarende laitude og longitude fra latlong (Oplagt sted at teste)
print("Step 5")
userdot = pd.DataFrame(dot, columns= ['Users'])
latlong['latitude'] = pd.to_numeric(latlong['latitude'])
latlong['longitude'] = pd.to_numeric(latlong['longitude'])
dataset = pd.concat([userdot, latlong], axis=1)

#Her extractes ground_truth på baggrund af dot fra df. Derudover extracted der også en category, ved brug af poi_id fra dot.
print("Step 6")
rows_list = []
category_list = []
for i in range(len(dot)):
    temp = df[dot.loc[i, "Poi"]][dot.loc[i, "Users"]]
    dict1 = {'ground_truth':float(temp)}
    rows_list.append(dict1)
    #Extract category from the list
    category = checkin_data_no_duplicates.loc[checkin_data_no_duplicates['poi_id'] == dot.loc[i, "Poi"]]
    cat = category['category']
    index = np.where(categories_numpy == [cat])[0][0]
    category_list.append(index)
#groundtruth laves til en dataframe.
groundtruth = pd.DataFrame(rows_list)
#result = pd.concat([dot, groundtruth], axis=1)

#Her laves categories til en dataframe, og concatenates med dataset (Samme rækkefølge forhåbentligt)
print("Step 7")
categories = pd.DataFrame(category_list, columns=['category'])
datasetst = pd.concat([dataset, categories], axis=1)

#Hver af vores datasets (latitude, longitude, category) (ground_truth), (categories)
print("Step 8")
dataset_numpy = datasetst.to_numpy()
labels_numpy = groundtruth.to_numpy()
#Redundant
categories_numpy = categories.to_numpy()

print("Step 9")
x_train_df = pd.DataFrame(dataset_numpy, columns=['User','Latitude','Longitude', '0'])
#x_test_df = pd.DataFrame(x_test, columns=['User','Latitude','Longitude', '0'])
y_train_df = pd.DataFrame(labels_numpy)
#y_test_df = pd.DataFrame(y_test)

#Dataset with Users
dataset1_df = pd.DataFrame(x_train_df['User'])

#Her bruges encoding array, til at få dets tilsvarende user_id, som ligger inden for embedding_arrayets vocabulary.
index = 0
for user in dataset1_df.iterrows():
    user = user[0]
    value = dataset1_df._get_value(user, 'User')
    dataset1_df.xs(user)['User']=encoding_array.get(value)
    #encoding.at[index,'user_id']=encoding_array.get(user)
    index += 1
    
#Dataset with Poi's
dataset2_df = pd.DataFrame(x_train_df[['Latitude']])
dataset3_df = pd.DataFrame(x_train_df[['Longitude']])

dataset4_df = pd.DataFrame(x_train_df[['0']])

print("Step 10")
dataset1 = tf.convert_to_tensor(
    dataset1_df, dtype=None, dtype_hint=None, name=None)
dataset2 = tf.convert_to_tensor(
    dataset2_df, dtype=None, dtype_hint=None, name=None)
dataset3 = tf.convert_to_tensor(
    dataset3_df, dtype=None, dtype_hint=None, name=None)
dataset4 = tf.convert_to_tensor(
    dataset4_df, dtype='int64', dtype_hint=None, name=None)
labels = tf.convert_to_tensor(
    y_train_df, dtype=None, dtype_hint=None, name=None)

model = EmbModel(len(users), cat_length)

print("Length prints")
print(len(users))
print(cat_length)

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)

model.compile(
    optimizer
)

#Train_data, [dataset.user_id, dataset.poi_id]. Label: ground_truth
model.fit([dataset4, dataset2, dataset3, dataset1], labels, epochs = 4, batch_size=27)

###################################
###################################
###### Iteration 1: Category ######
###################################
###################################

print("Step 1")
checkin_data = categoryset.merge(pois, on='poi_id')
checkin_data.drop_duplicates(subset="category", keep = 'first', inplace = True)
df = checkin_data.set_index('user_id').poi_id.str.get_dummies(',')
df = df.groupby('user_id').max()

print("Step 2")
checkin_data_no_duplicates = checkin_data.copy()
checkin_data_no_duplicates.drop_duplicates(subset ="poi_id", keep = 'first', inplace = True)
checkin_data_no_duplicates = pd.DataFrame(checkin_data_no_duplicates, columns = ['poi_id', 'category'])

#Extract categorical data
print("Step 3")
categories = pd.DataFrame(checkin_data, columns=['category'])
categories.drop_duplicates(subset ="category", keep = 'first', inplace = True)
category_length = len(categories)
categories_numpy = categories.to_numpy()


#Extracting all of the users and the pois
print("Step 3.5")
listofusers = pd.DataFrame(checkin_data, columns= ['user_id']).groupby('user_id').max().sample(frac=1)
listofpois = pd.DataFrame(checkin_data, columns= ['poi_id', 'latitude', 'longitude']).groupby('poi_id').max().sample(frac=1)
userarray = listofusers.index.to_numpy()
poiarray = listofpois.index.to_numpy()
userdataframe = pd.DataFrame(userarray, columns = ['Users'])
poidataframe = pd.DataFrame(poiarray, columns = ['Poi'])
dot = userdataframe.merge(poidataframe, how='cross')

print("Step 4")
rows_list = []
for i in range(len(dot)):
    temp = dot.loc[i, "Poi"]
    latitude = listofpois.loc[temp]['latitude']
    longitude = listofpois.loc[temp]['longitude']
    dict1 = {'latitude':latitude, 'longitude':longitude}
    rows_list.append(dict1)
    #latitude = poiarray[i]
latlong = pd.DataFrame(rows_list)

#Creating dataset
print("Step 5")
userdot = pd.DataFrame(dot, columns= ['Users'])
latlong['latitude'] = pd.to_numeric(latlong['latitude'])
latlong['longitude'] = pd.to_numeric(latlong['longitude'])
dataset = pd.concat([userdot, latlong], axis=1)

#Extracting ground_truth from incidence matrix
print("Step 6")
rows_list = []
category_list = []
for i in range(len(dot)):
    temp = df[dot.loc[i, "Poi"]][dot.loc[i, "Users"]]
    dict1 = {'ground_truth':float(temp)}
    rows_list.append(dict1)
    #Extract category from the list
    category = checkin_data_no_duplicates.loc[checkin_data_no_duplicates['poi_id'] == dot.loc[i, "Poi"]]
    cat = category['category']
    index = np.where(categories_numpy == [cat])[0][0]
    category_list.append(index)
#category_label = 
groundtruth = pd.DataFrame(rows_list)
#result = pd.concat([dot, groundtruth], axis=1)

print("Step 7")
categories = pd.DataFrame(category_list, columns=['category'])
datasetst = pd.concat([dataset, categories], axis=1)

print("Step 8")
dataset_numpy = datasetst.to_numpy()
labels_numpy = groundtruth.to_numpy()
#Redundant
categories_numpy = categories.to_numpy()

#x_train, x_test, y_train, y_test = train_test_split(dataset_numpy, labels_numpy, test_size=0.05, random_state=0)

print("Step 9")
x_train_df = pd.DataFrame(dataset_numpy, columns=['User','Latitude','Longitude', '0'])
#x_test_df = pd.DataFrame(x_test, columns=['User','Latitude','Longitude', '0'])
y_train_df = pd.DataFrame(labels_numpy)
#y_test_df = pd.DataFrame(y_test)

#Dataset with Users
dataset1_df = pd.DataFrame(x_train_df['User'])

index = 0
for user in dataset1_df.iterrows():
    user = user[0]
    value = dataset1_df._get_value(user, 'User')
    dataset1_df.xs(user)['User']=encoding_array.get(value)
    #encoding.at[index,'user_id']=encoding_array.get(user)
    index += 1
    
#Dataset with Poi's
dataset2_df = pd.DataFrame(x_train_df[['Latitude']])
dataset3_df = pd.DataFrame(x_train_df[['Longitude']])

dataset4_df = pd.DataFrame(x_train_df[['0']])

print("Step 10")
dataset1 = tf.convert_to_tensor(
    dataset1_df, dtype=None, dtype_hint=None, name=None)
dataset2 = tf.convert_to_tensor(
    dataset2_df, dtype=None, dtype_hint=None, name=None)
dataset3 = tf.convert_to_tensor(
    dataset3_df, dtype=None, dtype_hint=None, name=None)
dataset4 = tf.convert_to_tensor(
    dataset4_df, dtype='int64', dtype_hint=None, name=None)
labels = tf.convert_to_tensor(
    y_train_df, dtype=None, dtype_hint=None, name=None)

model.fit([dataset4, dataset2, dataset3, dataset1], labels, epochs = 4, batch_size=27)

###################################
###################################
###### Iteration 1: Restset #######
###################################
###################################

print("Step 1")
checkin_data = restset.merge(pois, on='poi_id')
df = checkin_data.set_index('user_id').poi_id.str.get_dummies(',')
df = df.groupby('user_id').max()

print("Step 2")
checkin_data_no_duplicates = checkin_data.copy()
checkin_data_no_duplicates.drop_duplicates(subset ="poi_id", keep = 'first', inplace = True)
checkin_data_no_duplicates = pd.DataFrame(checkin_data_no_duplicates, columns = ['poi_id', 'category'])

#Extract categorical data
print("Step 3")
categories = pd.DataFrame(checkin_data, columns=['category'])
categories.drop_duplicates(subset ="category", keep = 'first', inplace = True)
category_length = len(categories)
categories_numpy = categories.to_numpy()


#Extracting all of the users and the pois
print("Step 3.5")
listofusers = pd.DataFrame(checkin_data, columns= ['user_id']).groupby('user_id').max().sample(frac=1)
listofpois = pd.DataFrame(checkin_data, columns= ['poi_id', 'latitude', 'longitude']).groupby('poi_id').max().sample(frac=1)
userarray = listofusers.index.to_numpy()
poiarray = listofpois.index.to_numpy()
userdataframe = pd.DataFrame(userarray, columns = ['Users'])
poidataframe = pd.DataFrame(poiarray, columns = ['Poi'])
dot = userdataframe.merge(poidataframe, how='cross')

print("Step 4")
rows_list = []
for i in range(len(dot)):
    temp = dot.loc[i, "Poi"]
    latitude = listofpois.loc[temp]['latitude']
    longitude = listofpois.loc[temp]['longitude']
    dict1 = {'latitude':latitude, 'longitude':longitude}
    rows_list.append(dict1)
    #latitude = poiarray[i]
latlong = pd.DataFrame(rows_list)

#Creating dataset
print("Step 5")
userdot = pd.DataFrame(dot, columns= ['Users'])
latlong['latitude'] = pd.to_numeric(latlong['latitude'])
latlong['longitude'] = pd.to_numeric(latlong['longitude'])
dataset = pd.concat([userdot, latlong], axis=1)

#Extracting ground_truth from incidence matrix
print("Step 6")
rows_list = []
category_list = []
for i in range(len(dot)):
    temp = df[dot.loc[i, "Poi"]][dot.loc[i, "Users"]]
    dict1 = {'ground_truth':float(temp)}
    rows_list.append(dict1)
    #Extract category from the list
    category = checkin_data_no_duplicates.loc[checkin_data_no_duplicates['poi_id'] == dot.loc[i, "Poi"]]
    cat = category['category']
    index = np.where(categories_numpy == [cat])[0][0]
    category_list.append(index)
#category_label = 
groundtruth = pd.DataFrame(rows_list)
#result = pd.concat([dot, groundtruth], axis=1)

print("Step 7")
categories = pd.DataFrame(category_list, columns=['category'])
datasetst = pd.concat([dataset, categories], axis=1)

print("Step 8")
dataset_numpy = datasetst.to_numpy()
labels_numpy = groundtruth.to_numpy()
#Redundant
categories_numpy = categories.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(dataset_numpy, labels_numpy, test_size=0.20, random_state=0)

print("Step 9")
x_train_df = pd.DataFrame(x_train, columns=['User','Latitude','Longitude', '0'])
x_test_df = pd.DataFrame(x_test, columns=['User','Latitude','Longitude', '0'])
y_train_df = pd.DataFrame(y_train)
y_test_df = pd.DataFrame(y_test)

x_test_df.to_csv(r'/user/student.aau.dk/lharde18/Data-output/Den/y_train_df.csv', sep=',', index=['User','Latitude','Longitude', '0'])
y_test_df.to_csv(r'/user/student.aau.dk/lharde18/Data-output/Den/y_test_df.csv', sep=',', index=['0'])

#Dataset with Users
dataset1_df = pd.DataFrame(x_train_df['User'])

index = 0
for user in dataset1_df.iterrows():
    user = user[0]
    value = dataset1_df._get_value(user, 'User')
    dataset1_df.xs(user)['User']=encoding_array.get(value)
    #encoding.at[index,'user_id']=encoding_array.get(user)
    index += 1
    
#Dataset with Poi's
dataset2_df = pd.DataFrame(x_train_df[['Latitude']])
dataset3_df = pd.DataFrame(x_train_df[['Longitude']])

dataset4_df = pd.DataFrame(x_train_df[['0']])

print("Step 10")
dataset1 = tf.convert_to_tensor(
    dataset1_df, dtype=None, dtype_hint=None, name=None)
dataset2 = tf.convert_to_tensor(
    dataset2_df, dtype=None, dtype_hint=None, name=None)
dataset3 = tf.convert_to_tensor(
    dataset3_df, dtype=None, dtype_hint=None, name=None)
dataset4 = tf.convert_to_tensor(
    dataset4_df, dtype='int64', dtype_hint=None, name=None)
labels = tf.convert_to_tensor(
    y_train_df, dtype=None, dtype_hint=None, name=None)

model.fit([dataset4, dataset2, dataset3, dataset1], labels, epochs = 4, batch_size=71)
model.model.save('/user/student.aau.dk/lharde18/Data-output/Den/model')
model.save_weights('/user/student.aau.dk/lharde18/Data-output/Den/my_checkpoint')
