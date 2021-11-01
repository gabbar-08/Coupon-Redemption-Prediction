
#Step 1 :- Importing dependancies and train test data generated
from config import *
train_data = pd.read_csv("data/train_data/train_feature.csv")
test_data = pd.read_csv("data/test_data/test_feature.csv")

#Step 2 :- Getting train data insights and drop unnecessary columns, Splitting data into input and target variable sets.
print(list(train_data['redemption_status']).count(0) * 100 / len(train_data['redemption_status']), "% coupons not redeemed in training data ") 

X = train_data
X.dropna(inplace=True)
X.drop(["id","campaign_id","c_freq_category","c_rare_category","start_date","end_date","duration","age_range","overall_freq_category","overall_rare_category"], axis=1,inplace=True)

y = train_data['redemption_status']
X.drop('redemption_status',axis = 1, inplace = True)

#Step 3 :- Train-test Split for the model 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Step 4 :- Initiate model and fit transform
model = GaussianNB()

model.fit(X_train, y_train)

#Step 5 :- Predict on the test part of the split 
y_pred = model.predict(X_test)


#Step 6 :- Save the model for the inference engine
filename = 'model/finalized_model_2.sav'
pickle.dump(model, open(filename, 'wb'))

#Step 7 :- Calculate Training data accuracy of the model
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#Step 8 :- Use the model on test data to predict the target in test data
Y = test_data
Y.drop(["id","campaign_id","c_freq_category","c_rare_category","start_date","end_date","duration","age_range","overall_freq_category","overall_rare_category"], axis=1,inplace=True)
Y.dropna(inplace = True)
Predictions = model.predict(Y)
# Print results
print(list(Predictions).count(0) * 100 / len(Predictions) , "%  Coupans not redeemed in Test Data" )
