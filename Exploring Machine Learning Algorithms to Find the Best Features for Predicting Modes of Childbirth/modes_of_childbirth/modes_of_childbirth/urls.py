"""modes_of_childbirth URL Configuration

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
from modes_of_childbirth import settings
from Service_Provider import views as serviceprovider
from django.conf.urls.static import static


urlpatterns = [
    url('admin/', admin.site.urls),
    url(r'^$', remoteuser.login, name="login"),
    url(r'^Register1/$', remoteuser.Register1, name="Register1"),
    url(r'^Predict_ChildBirth_DataSets/$', remoteuser.Predict_ChildBirth_DataSets, name="Predict_ChildBirth_DataSets"),
    url(r'^ratings/(?P<pk>\d+)/$', remoteuser.ratings, name="ratings"),
    url(r'^ViewYourProfile/$', remoteuser.ViewYourProfile, name="ViewYourProfile"),
    url(r'^Add_DataSet_Details/$', remoteuser.Add_DataSet_Details, name="Add_DataSet_Details"),
    url(r'^serviceproviderlogin/$',serviceprovider.serviceproviderlogin, name="serviceproviderlogin"),
    url(r'View_Remote_Users/$',serviceprovider.View_Remote_Users,name="View_Remote_Users"),
    url(r'^charts/(?P<chart_type>\w+)', serviceprovider.charts,name="charts"),
    url(r'^charts1/(?P<chart_type>\w+)', serviceprovider.charts1, name="charts1"),
    url(r'^likeschart/(?P<like_chart>\w+)', serviceprovider.likeschart, name="likeschart"),
    url(r'^Find_ChildBirth_Prediction_Ratio/$', serviceprovider.Find_ChildBirth_Prediction_Ratio, name="Find_ChildBirth_Prediction_Ratio"),
    url(r'^View_Emergency_Birth_Details/$', serviceprovider.View_Emergency_Birth_Details, name="View_Emergency_Birth_Details"),
    url(r'^Train_View_ChildBirth_DataSets_Details/$', serviceprovider.Train_View_ChildBirth_DataSets_Details, name="Train_View_ChildBirth_DataSets_Details"),
    url(r'^Download_Trained_DataSets/$', serviceprovider.Download_Trained_DataSets, name="Download_Trained_DataSets"),




]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
