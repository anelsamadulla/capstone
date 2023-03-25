from django.urls import path 
from . import views 

urlpatterns = [
    path('predictor/', views.predictor, name='predictor'),
    path('result/', views.formInfo, name='result'),
    #path('test/', views.DTmodel, name='result'),
    path('upload/', views.upload_folder, name='upload_folder'),
    path('test_upload/<str:trainerid>/', views.test_upload, name='test_upload'),
    # path('download_joblib_file/<str:filename>/', views.download_joblib_file, name='download_joblib_file'),
    path('get_trainers/', views.get_trainers, name='get_trainers'),
    path('view_trainers/', views.view_trainers, name='view_trainers'),
    path('train_model/', views.train_model, name='train_model'),
    path('download-joblib/', views.download_joblib_file, name='download_joblib_file'),
    path('download/<str:foldername>/<str:filename>/', views.download, name='download'),


    # path('uploadfiles/', views.upload_files, name='upload_files'),
    path('success/', views.success, name='success')
]