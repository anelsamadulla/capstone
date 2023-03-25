from django.db import models

class UploadedFolder(models.Model):
    id = models.AutoField(primary_key=True)
    folder_name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    file_folder = models.FileField(upload_to='uploaded_folders/')

class TextFile(models.Model):
    uploaded_folder = models.ForeignKey(UploadedFolder, on_delete=models.CASCADE)
    file = models.FileField(upload_to='text_files/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = "uploadapp"

# class TextFile(models.Model):
#     file = models.FileField(upload_to='text_files/')
#     uploaded_at = models.DateTimeField(auto_now_add=True)
#     created_at = models.DateTimeField(auto_now_add=True)
#     class Meta:
#         app_label = "uploadapp"

class Trainer(models.Model):
    user = models.TextField()
    session_id = models.TextField()
    datafolder = models.TextField(null=True, blank=True)
    modelname = models.TextField(null=True, blank=True)
    model = models.TextField(null=True, blank=True)
    status = models.IntegerField(default=0) # 0 - not trained, 1 - trained
    training = models.FloatField(null=True, blank=True)
    testing = models.FloatField(null=True, blank=True)
