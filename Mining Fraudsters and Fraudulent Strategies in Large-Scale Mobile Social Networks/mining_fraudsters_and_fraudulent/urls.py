"""mining_fraudsters_and_fraudulent URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from Remote_User import views as remoteuser
from mining_fraudsters_and_fraudulent import settings
from Service_Provider import views as serviceprovider
from django.conf.urls.static import static


urlpatterns = [
    url('admin/', admin.site.urls),
    url(r'^$', remoteuser.login, name="login"),
    url(r'^Register1/$', remoteuser.Register1, name="Register1"),
    url(r'^Predict_fraudsters_and_fraudulent/$', remoteuser.Predict_fraudsters_and_fraudulent, name="Predict_fraudsters_and_fraudulent"),
    url(r'^ratings/(?P<pk>\d+)/$', remoteuser.ratings, name="ratings"),
    url(r'^ViewYourProfile/$', remoteuser.ViewYourProfile, name="ViewYourProfile"),
    url(r'^Add_DataSet_Details/$', remoteuser.Add_DataSet_Details, name="Add_DataSet_Details"),
    url(r'^serviceproviderlogin/$',serviceprovider.serviceproviderlogin, name="serviceproviderlogin"),
    url(r'View_Remote_Users/$',serviceprovider.View_Remote_Users,name="View_Remote_Users"),
    url(r'^charts/(?P<chart_type>\w+)', serviceprovider.charts,name="charts"),
    url(r'^charts1/(?P<chart_type>\w+)', serviceprovider.charts1, name="charts1"),
    url(r'^likeschart/(?P<like_chart>\w+)', serviceprovider.likeschart, name="likeschart"),
    url(r'^train_model/$', serviceprovider.train_model,name="train_model"),
    url(r'^View_All_Fraudsters_and_Fraudulent_PredictionType/$', serviceprovider.View_All_Fraudsters_and_Fraudulent_PredictionType, name="View_All_Fraudsters_and_Fraudulent_PredictionType"),
    url(r'^View_Data_Sets_Details/$', serviceprovider.View_Data_Sets_Details, name="View_Data_Sets_Details"),
    url(r'^Download_Trained_DataSets/$', serviceprovider.Download_Trained_DataSets, name="Download_Trained_DataSets"),
    url(r'^Find_Fraudsters_and_Fraudulent_Prediction_TypeRatio/$', serviceprovider.Find_Fraudsters_and_Fraudulent_Prediction_TypeRatio, name="Find_Fraudsters_and_Fraudulent_Prediction_TypeRatio"),




]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
