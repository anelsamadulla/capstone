from django.shortcuts import render, redirect
from .models import TextFile, Trainer
from .forms import TextFileForm, UploadFolderForm
from joblib import load
import numpy as np
import tensorflow
from tensorflow import keras
from rest_framework.decorators import api_view
from sklearn.tree import DecisionTreeClassifier
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.csrf import csrf_protect
from rest_framework.response import Response
from sklearn.svm import SVC
from django.core.files.uploadedfile import TemporaryUploadedFile
import tempfile
import zipfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
# from rest_framework.response import Response
# from rest_framework import status
import joblib
# from tensorflow.keras.utils import to_categorical  
# import tensorflow.keras.utils as util
import ast
import os
from django.core.files.storage import FileSystemStorage
from django.conf import settings
# model = load('./savedModels/model.joblib') 
import boto3
import uuid
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render
from .serializers import TrainerSerializer
from .models import UploadedFolder
from django.contrib import messages
from django.db import transaction

import zipfile
def view_trainers(request):
    user = request.user
    print(f"USER:{user}")
    trainers = Trainer.objects.filter(user = str(user))
    print(f"trainers:{trainers}")
    serializer = TrainerSerializer(trainers, many = True)
    context = {'uploaded': True, 'data': serializer}
    return render(request, 'trainers.html', {'data': trainers})
def test_upload(request, trainerid):
    return upload_folder(request, trainerid)
def upload_folder(request, trainerid):
    print(f"H1: {trainerid}")
    username = request.user
    session_id =request.session.session_key
    if request.method == 'POST':
        if 'file_folder' not in request.FILES:
            # No file selected, return an error message
            context = {'error': 'No file selected'}
            return render(request, 'upload.html', context)
        uploaded_file = request.FILES['file_folder']
        s3 = boto3.client('s3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
        )
        bucket_name = 'seniorprojectgaitbucket'
        folder = ''
        subfolders = []
        with zipfile.ZipFile(uploaded_file) as zip_file:
            for file_name in zip_file.namelist():
                # Extract the file contents
                file_content = zip_file.read(file_name)

                # Remove the leading directory path to get the relative file path
                relative_file_path = os.path.relpath(file_name, zip_file.namelist()[0])

                print(f"FILE:{relative_file_path}, FILENAEM:{file_name}")
                # Upload the file to S3 with a unique name and folder structure
                if relative_file_path.endswith('.txt'):
                    if relative_file_path.split("/")[-2] not in subfolders:
                        subfolders.append(relative_file_path.split("/")[-2])
                    if folder == '':
                        folder = relative_file_path.split("/")[-3]
                    s3_file_name = f"{username}/{session_id}/{relative_file_path}"
                    print("ADDED s3")
                    s3.put_object(Bucket=bucket_name, Key=s3_file_name, Body=file_content)

        if trainerid:
            trainer_model = Trainer.objects.get(pk=int(trainerid))
        else:
            trainer_model = Trainer.objects.create(user=str(username))
        trainer_model.session_id = str(session_id)
        trainer_model.datafolder = str(f"{username}/{session_id}")
        trainer_model.save()
        # set session variables for the new model instance
        request.session['folder'] = folder
        request.session['subfolders'] = subfolders
        context = {'uploaded': True}
        return render(request, 'upload.html', context)
    return render(request, 'upload.html')

        # print('FOLDER:',folder)
        # print('Subfol:',subfolders)
        # algorithm = request.POST.get('algorithm')
        # print(f"ALGO:{algorithm}")
        # if algorithm == "DT":
        #     try:
        #         max_depth = request.POST.get('max_depth')
        #         min_samples_leaf = request.POST.get('min_samples_leaf')
        #         min_samples_split = request.POST.get('min_samples_split')
        #         clf = DecisionTreeClassifier(max_depth=int(max_depth), min_samples_leaf=int(min_samples_leaf), min_samples_split=int(min_samples_split))
        #         # do something with the values
        #     except ValueError:
        #         context = {'error': 'Please enter integer values for max depth, min samples leaf, and min samples split'}
        #         return render(request, 'upload.html', context)
        # elif algorithm == "KNN":
        #     try:
        #         n_neighbors = request.POST.get('n_neighbors')
        #         algorithm_knn = request.POST.get('algorithm_knn')
        #         leaf_size = request.POST.get('leaf_size')
        #         clf = KNeighborsClassifier(n_neighbors=int(n_neighbors), algorithm=algorithm_knn, leaf_size=int(leaf_size))
        #         # do something with the values
        #     except ValueError:
        #         context = {'error': 'Please enter integer values for n_neighbors, leaf_size, and proper algorithm'}
        #         return render(request, 'upload.html', context)
        # elif algorithm == "SVM":
        #     try:
        #         C = request.POST.get('C')
        #         kernel = request.POST.get('kernel')
        #         gamma = request.POST.get('gamma')
        #         clf = SVC(C=float(C), kernel=kernel, gamma=float(gamma))
        #         # do something with the values
        #     except ValueError:
        #         context = {'error': 'Please enter float values for C, gamma, and proper kernel'}
        #         return render(request, 'upload.html', context)
        # elif algorithm == "RF":
        #     try:
        #         n_estimators = request.POST.get('n_estimators')
        #         max_depth_rf = request.POST.get('max_depth_rf')
        #         min_samples_leaf_rf = request.POST.get('min_samples_leaf_rf')
        #         clf = RandomForestClassifier(n_estimators=int(n_estimators), max_depth=int(max_depth_rf), min_samples_leaf=int(min_samples_leaf_rf))
        #         # do something with the values
        #     except ValueError:
        #         context = {'error': 'Please enter integer values for n_estimators, max_depth, and min_samples_leaf'}
        #         return render(request, 'upload.html', context)
        # elif algorithm == "NB":
        #     try:
        #         priors = request.POST.get('priors').split()
        #         priors = [float(p) for p in priors]
        #         var_smoothing = request.POST.get('var_smoothing')
        #         clf = GaussianNB(priors=priors,var_smoothing=float(var_smoothing))
        #     except ValueError:
        #         context = {'error': 'Please enter float values for var smoothing and separated by commas or space for the priors'}
        #         return render(request, 'upload.html', context)
        # train_model(clf=clf, folder=folder, subfolders=subfolders)
        # return redirect('success')

    
from django.http import FileResponse
def download_joblib_file(request):
    file_path = request.POST.get('filename') # Replace with the actual path to your .joblib file
    print(f"FILE:{file_path}")
    with open(file_path, 'rb') as f:
        response = FileResponse(f)
        response['Content-Type'] = 'application/octet-stream'
        response['Content-Disposition'] = 'attachment; filename="%s"' % os.path.basename(file_path)
        return response
@api_view(['GET'])
def download(request, foldername, filename):
    file_path = "savedModels/"+filename
    print(f"FILE:{file_path}")
    with open(file_path, "rb") as fprb:
        response = HttpResponse(fprb.read())
        response['Content-Disposition'] = 'attachment; filename="%s"' % os.path.basename(file_path)
        return response
@api_view(['GET'])
def get_trainers(request):
    user = request.user
    print(f"USER:{user}")
    trainers = Trainer.objects.filter(user = str(user))
    print(f"trainers:{trainers}")
    serializer = TrainerSerializer(trainers, many = True)
    return Response(serializer.data)
def train_model(request):
    trainer_model = Trainer.objects.last()
    
    folder = request.session.get('folder')
    subfolders = request.session.get('subfolders')
    print('FOLDER:',folder)
    print('Subfol:',subfolders)
    if request.method == "POST":
        algorithm = request.POST.get('algorithm')
        print(f"ALGO:{trainer_model.datafolder}")
        if algorithm == "DT":
            try:
                max_depth = request.POST.get('max_depth')
                min_samples_leaf = request.POST.get('min_samples_leaf')
                min_samples_split = request.POST.get('min_samples_split')
                clf = DecisionTreeClassifier(max_depth=int(max_depth), min_samples_leaf=int(min_samples_leaf), min_samples_split=int(min_samples_split))
                    # do something with the values
            except ValueError:
                context = {'error': 'Please enter integer values for max depth, min samples leaf, and min samples split'}
                return render(request, 'upload.html', context)
        elif algorithm == "KNN":
            try:
                n_neighbors = request.POST.get('n_neighbors')
                algorithm_knn = request.POST.get('algorithm_knn')
                leaf_size = request.POST.get('leaf_size')
                clf = KNeighborsClassifier(n_neighbors=int(n_neighbors), algorithm=algorithm_knn, leaf_size=int(leaf_size))
                    # do something with the values
            except ValueError:
                context = {'error': 'Please enter integer values for n_neighbors, leaf_size, and proper algorithm'}
                return render(request, 'upload.html', context)
        elif algorithm == "SVM":
            try:
                C = request.POST.get('C')
                kernel = request.POST.get('kernel')
                gamma = request.POST.get('gamma')
                clf = SVC(C=float(C), kernel=kernel, gamma=float(gamma))
                    # do something with the values
            except ValueError:
                context = {'error': 'Please enter float values for C, gamma, and proper kernel'}
                return render(request, 'upload.html', context)
        elif algorithm == "RF":
            try:
                n_estimators = request.POST.get('n_estimators')
                max_depth_rf = request.POST.get('max_depth_rf')
                min_samples_leaf_rf = request.POST.get('min_samples_leaf_rf')
                clf = RandomForestClassifier(n_estimators=int(n_estimators), max_depth=int(max_depth_rf), min_samples_leaf=int(min_samples_leaf_rf))
                    # do something with the values
            except ValueError:
                context = {'error': 'Please enter integer values for n_estimators, max_depth, and min_samples_leaf'}
                return render(request, 'upload.html', context)
        elif algorithm == "NB":
            try:
                priors = request.POST.get('priors').split()
                priors = [float(p) for p in priors]
                var_smoothing = request.POST.get('var_smoothing')
                clf = GaussianNB(priors=priors,var_smoothing=float(var_smoothing))
            except ValueError:
                context = {'error': 'Please enter float values for var smoothing and separated by commas or space for the priors'}
                return render(request, 'upload.html', context)
            
        data_per_sequence=[]
        train_ds=[[],[]]
        print(f"MODEL: {clf}")
        s3 = boto3.client('s3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
            )
        bucket_name="seniorprojectgaitbucket"
        seen = []
        print(f"PREFIX:{trainer_model.user}/{trainer_model.session_id}")
        
        params = {'Bucket': bucket_name, 'Prefix': f"{trainer_model.user}/{trainer_model.session_id}"}
        
        response = s3.list_objects_v2(**params)
        if 'Contents' in response:
            for obj in response['Contents']:
                key = obj['Key']
                if key in seen:
                    continue
                else:
                    seen.append(key)
                if key.endswith('.txt'):
                    label = key.split('/')[-2].lstrip('Person')
                    print(f"label: {label} key:{key} ")
                    obj = s3.get_object(Bucket=bucket_name, Key=key)
                    var =obj['Body'].read().decode('utf-8')
                    for line in var.splitlines():
                        # print(f"LINE:{line}", end="")
                        aList = list(ast.literal_eval(line))
                        data_per_sequence.append(aList)
                    f_means= mean_of_geometric(data_per_sequence)
                    for i in range(0, len(f_means)):
                        train_ds[0].append(f_means[i])
                        train_ds[1].append((int(label)-1))

        temp = train_ds[0]
        train_ds[0] = (train_ds[0] - np.average(temp))/(np.std(temp))
        train_data, train_label = shuffle(train_ds[0], train_ds[1])
        #print(train_data)
        nsamples, nx, ny, nz = np.array(train_data).shape
        train_dataset = np.array(train_data).reshape((nsamples,nx*ny*nz))
        
        print("Training model...")
        clf.fit(train_dataset, train_label)
        train_acc = clf.score(train_dataset, train_label)
        
        print("Training accuracy:", train_acc)
        #save model
        filename = f"savedModels/{trainer_model.modelname}_{trainer_model.id}.joblib"
        print(f"{trainer_model.modelname}_{trainer_model.id}.joblib")
        trainer_model.datafolder = str(trainer_model.id)
        trainer_model.modelname = str(clf)
        trainer_model.training = float(train_acc)
        trainer_model.status = 1
        trainer_model.model = filename
        joblib.dump(clf, filename)

        trainer_model.save()
        context = {'train_acc': train_acc}
        return render(request, 'model.html', context)

    return render(request, 'model.html')
    #return redirect('success')




# def train_nb(request):
#     if request.method == 'POST':
#         algorithm = request.POST.get('algorithm')

#         if algorithm == "NB":
#             priors_str = request.POST.get('priors')
#             var_smoothing_str = request.POST.get('var_smoothing')

#             # Validate priors input
#             try:
#                 priors = [float(p) for p in priors_str.split()]
#                 if len(priors) != 2:
#                     raise ValueError("Priors must contain 2 values.")
#             except (ValueError, TypeError):
#                 return HttpResponse("Invalid priors input.")

#             # Validate var_smoothing input
#             try:
#                 var_smoothing = float(var_smoothing_str)
#             except (ValueError, TypeError):
#                 return HttpResponse("Invalid var_smoothing input.")

#             # Train model
#             clf = GaussianNB(priors=priors, var_smoothing=var_smoothing)
#             # rest of the code



def upload_text_file(request):
    if request.method == 'POST':
        file = request.FILES.getlist('file')[0]
        data_per_sequence=[]
        train_ds=[[],[]]
        var =file.read().decode('utf-8')
        for line in var.splitlines():
            aList = list(ast.literal_eval(line))
            data_per_sequence.append(aList)
        f_means= mean_of_geometric(data_per_sequence)
        for i in range(0, len(f_means)):
            train_ds[0].append(f_means[i])
        temp = train_ds[0]
        train_ds[0] = (train_ds[0] - np.average(temp))/(np.std(temp))
        train_data= shuffle(train_ds[0])

        nsamples, nx, ny, nz = np.array(train_data).shape
        train_dataset = np.array(train_data).reshape((nsamples,nx*ny*nz))
        X = train_ds[0]
        knn = joblib.load('./savedModels/trained_model2.joblib')
        y = knn.predict(train_dataset)
        print("Y:", y)
        context = {'y':y}
        return render(request, 'success.html', context)
    return render(request, 'upload.html')

def success(request):
    return render(request, 'success.html')

def predictor(request):
    return render(request, 'post.html')

def formInfo(request):
    sepal_length = request.GET['sepal-length']
    sepal_width = request.GET['sepal-width']
    petal_length = request.GET['petal-length']
    petal_width = request.GET['petal-width']
    y_pred = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    if y_pred[0] == 0:
        y_pred = 'Setosa'
    elif y_pred[0] == 1:
        y_pred = 'Verscicolor'
    else:
        y_pred = 'Virginica'
    return render(request, 'result.html')

def DTmodel(filename):
    path = "C:/Users/hp/Desktop/Senior Project/myproject/GaitSequences1/GaitSequences1/train/Person1"
    # subfolders = [ f.path for f in os.scandir(path) if f.is_dir() ]

    data_per_sequence=[]
    train_ds=[[],[]]
    with open (path+"/seq2.txt") as f1:
        var =f1.read()
        for line in var.splitlines():
            aList = list(ast.literal_eval(line))
            data_per_sequence.append(aList)
        f_means= mean_of_geometric(data_per_sequence)
        for i in range(0, len(f_means)):
            train_ds[0].append(f_means[i])
    temp = train_ds[0]
    train_ds[0] = (train_ds[0] - np.average(temp))/(np.std(temp))
    train_data= shuffle(train_ds[0])

    nsamples, nx, ny, nz = np.array(train_data).shape
    train_dataset = np.array(train_data).reshape((nsamples,nx*ny*nz))
    X = train_ds[0]
    knn = joblib.load('./savedModels/trained_model2.joblib')
    y = knn.predict(train_dataset)
    #print("Y:", y)

    # for directory in subfolders:
    #     label=int(os.path.basename(directory).lstrip('Person'))
    #     for filename in os.listdir(directory):
    #         f = os.path.join(directory, filename)
    #         if os.path.isfile(f):
    #             if f.endswith(".txt"):
    #                 data_per_sequence=[]
    #                 with open(f, 'r') as fl:
    #                     var=fl.read()
    #                     for line in var.splitlines():
    #                         aList = list(ast.literal_eval(line))
    #                         data_per_sequence.append(aList)
    #                     f_means= mean_of_geometric(data_per_sequence)
    #                     for i in range(0, len(f_means)):
    #                         train_ds[0].append(f_means[i])
    #                         train_ds[1].append((label-1))
    # temp=train_ds[0]
    # train_ds[0] = (train_ds[0] - np.average(temp)) / (np.std(temp))
    # nsamples, nx, ny, nz = np.array(train_ds[0]).shape
    # train_dataset = np.array(train_ds[0]).reshape((nsamples,nx*ny*nz))

    # clf = KNeighborsClassifier(n_neighbors=2)
    # clf.fit(train_dataset, train_label)
    # y_pred = clf.predict(test_dataset)
    # print("KNN Accuracy: %.3f" %metrics.accuracy_score(test_ds[1], y_pred))

def mean_of_geometric(data):
    F_lower=[[],[]]
    F=[]
    d=[]
    f=[]

    ox=np.array([1,0,0]).reshape(-1,1)
    oy=np.array([0,1,0]).reshape(-1,1)
    oz=np.array([0,0,1]).reshape(-1,1)

    for N in range(1, (len(data)//20)+1):
        F_lower=[[],[]]
        for i in range((N-1)*20, N*20):
            for j in range(i+1, N*20):
                (xi, yi, zi)=data[i]
                (xj, yj, zj)=data[j]
                ji=np.sqrt((yj-yi)**2+(zj-zi)**2+(xj-xi)**2)

                d.append(np.sqrt((yj-yi)**2+(zj-zi)**2))
                d.append(np.sqrt((xj-xi)**2+(zj-zi)**2))
                d.append(np.sqrt((xj-xi)**2+(yj-yi)**2))
                F_lower[0].append(d)

                if ji!=0.0:
                    t1=np.arccos(np.clip((np.dot(np.array(d), oy)//ji), -1.0, 1.0))
                    f.append(t1[0])
                    t2=np.arccos(np.clip((np.dot(np.array(d), oz)//ji), -1.0, 1.0))
                    f.append(t2[0])
                    t3=np.arccos(np.clip((np.dot(np.array(d), ox)//ji), -1.0, 1.0))
                    f.append(t3[0])
                    F_lower[1].append(f)
                else:
                    f.append(0.0)
                    f.append(0.0)
                    f.append(0.0)
                    F_lower[1].append(f)

                d=[]
                f=[] 
        F.append(F_lower)
        
        F_means=[]
        F_mean=[[], []]
        
        m=[[],[]]
        tem=np.zeros((190, 3)).tolist()
        m[0]=tem
        m[1]=tem

        count=0
        k=0
        while k<len(F):
            if count==30:
                k=k-24
                count=0
                m=(np.array(m)/30).tolist()
                F_mean[0]=m[0]
                F_mean[1]=m[1]
                
                F_means.append(F_mean)
                
                F_mean=[[], []]
                
                m=[[],[]]
                tem=np.zeros((190, 3)).tolist()
                m[0]=tem
                m[1]=tem
                
            for j in range(0, 190): 
                m[0][j]=(np.array(m[0][j])+np.array(F[k][0][j])).tolist()
                m[1][j]=np.array(m[1][j])+np.array(F[k][1][j]).tolist()
                
            
            if k==len(F)-1 and count!=30 and count!=0:
                t=k-(k//30)*30
                if t!=0:
                    m=(np.array(m)/t).tolist()
                    F_mean[0]=m[0]
                    F_mean[1]=m[1]
                    
                    F_means.append(F_mean)
                    
            count+=1
            k+=1

    return F_means

