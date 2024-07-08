from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404
import datetime
import openpyxl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.ensemble import VotingClassifier

import warnings
warnings.filterwarnings("ignore")
plt.style.use('ggplot')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score




# Create your views here.
from Remote_User.models import ClientRegister_Model,drug_drug_interactions,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')



def Register1(request):

    if request.method == "POST":
        if request.method == "POST":
            username = request.POST.get('username')
            email = request.POST.get('email')
            password = request.POST.get('password')
            phoneno = request.POST.get('phoneno')
            country = request.POST.get('country')
            state = request.POST.get('state')
            city = request.POST.get('city')
            address = request.POST.get('address')
            gender = request.POST.get('gender')
            ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                                country=country, state=state, city=city, address=address, gender=gender)
            obj = "Registered Successfully"
            return render(request, 'RUser/Register1.html', {'object': obj})
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Drug_To_Drug_Interact_Type(request):
    if request.method == "POST":

        if request.method == "POST":

            uniqueID= request.POST.get('uniqueID')
            First_drugName= request.POST.get('First_drugName')
            condition1= request.POST.get('condition1')
            review= request.POST.get('review')
            Second_drugName= request.POST.get('Second_drugName')
            condition2= request.POST.get('condition2')

            df = pd.read_csv('Drug_Drug_Interactions.csv', encoding='latin-1')

            def apply_recommend(Rating):
                if (Rating <= 3):
                    return 0  # Bad
                elif (Rating > 3 and Rating <= 7):
                    return 1  # Average
                elif (Rating > 7 and Rating <= 10):
                    return 2  # Very Good

            df['Results'] = df['rating'].apply(apply_recommend)

            # cv = CountVectorizer()
            X = df['review']
            y = df['Results']

            cv = CountVectorizer(lowercase=False, strip_accents='unicode', ngram_range=(1, 1))

            X = cv.fit_transform(X)

            models = []
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
            X_train.shape, X_test.shape, y_train.shape

            print("Naive Bayes")
            from sklearn.naive_bayes import MultinomialNB
            NB = MultinomialNB()
            NB.fit(X_train, y_train)
            predict_nb = NB.predict(X_test)
            naivebayes = accuracy_score(y_test, predict_nb) * 100
            print(naivebayes)
            print(confusion_matrix(y_test, predict_nb))
            print(classification_report(y_test, predict_nb))
            models.append(('naive_bayes', NB))
            detection_accuracy.objects.create(names="Naive Bayes", ratio=naivebayes)

            # SVM Model
            print("SVM")
            from sklearn import svm
            lin_clf = svm.LinearSVC()
            lin_clf.fit(X_train, y_train)
            predict_svm = lin_clf.predict(X_test)
            svm_acc = accuracy_score(y_test, predict_svm) * 100
            print(svm_acc)
            print("CLASSIFICATION REPORT")
            print(classification_report(y_test, predict_svm))
            print("CONFUSION MATRIX")
            print(confusion_matrix(y_test, predict_svm))
            models.append(('svm', lin_clf))
            detection_accuracy.objects.create(names="SVM", ratio=svm_acc)

            print("Logistic Regression")
            from sklearn.linear_model import LogisticRegression
            reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            print("ACCURACY")
            print(accuracy_score(y_test, y_pred) * 100)
            print("CLASSIFICATION REPORT")
            print(classification_report(y_test, y_pred))
            print("CONFUSION MATRIX")
            print(confusion_matrix(y_test, y_pred))
            models.append(('logistic', reg))

            from sklearn.tree import DecisionTreeClassifier
            print("Decision Tree Classifier")
            dtc = DecisionTreeClassifier()
            dtc.fit(X_train, y_train)
            dtcpredict = dtc.predict(X_test)
            print("ACCURACY")
            print(accuracy_score(y_test, dtcpredict) * 100)
            print("CLASSIFICATION REPORT")
            print(classification_report(y_test, dtcpredict))
            print("CONFUSION MATRIX")
            print(confusion_matrix(y_test, dtcpredict))
            models.append(('DecisionTreeClassifier', dtc))

            print("KNeighborsClassifier")
            from sklearn.neighbors import KNeighborsClassifier
            kn = KNeighborsClassifier()
            kn.fit(X_train, y_train)
            knpredict = kn.predict(X_test)
            print("ACCURACY")
            print(accuracy_score(y_test, knpredict) * 100)
            print("CLASSIFICATION REPORT")
            print(classification_report(y_test, knpredict))
            print("CONFUSION MATRIX")
            print(confusion_matrix(y_test, knpredict))
            models.append(('KNeighborsClassifier', kn))

            classifier = VotingClassifier(models)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            review1 = [review]
            vector1 = cv.transform(review1).toarray()
            predict_text = classifier.predict(vector1)

            pred = str(predict_text).replace("[", "")
            pred1 = pred.replace("]", "")

            prediction = int(pred1)

            if (prediction== 0):
                val='Bad'
            elif (prediction== 1):
                val='Average'
            elif (prediction== 2):
                val='Very Good'

            print(val)
            print(prediction)

            drug_drug_interactions.objects.create(
            uniqueID=uniqueID,
            First_drugName=First_drugName,
            condition1=condition1,
            review=review,
            Second_drugName=Second_drugName,
            condition2=condition2,
            Prediction=val
            )


        return render(request, 'RUser/Predict_Drug_To_Drug_Interact_Type.html',{'objs': val})
    return render(request, 'RUser/Predict_Drug_To_Drug_Interact_Type.html')



