from django import forms

class UploadFolderForm(forms.Form):
    folder = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}))

class TextFileForm(forms.Form):
    file = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}))
