from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('search/', views.search),
    path('rerank/', views.rerank),
    path('feedback/', views.feedback),
    path('clear/', views.clear),
    path('displaynext/', views.displaynext),
    path('displayall/', views.displayall),
    path('gptupdate/',views.GPT_QA),
    path('gptsubmit/',views.GPT_describe),
    path('feedbacktime/',views.feedTime),
    path('cut/',views.cut),
    path('finish/',views.finish),
    path('send/',views.send),
    path('submitconfirm/', views.submitconfirm),
]