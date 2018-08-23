from crispy_forms.helper import FormHelper
from crispy_forms.layout import Submit
from django import forms

from Algorithm.models import Videos


class VideoProcessForm(forms.ModelForm):
    class Meta:
        model = Videos
        fields = ['ip_link', 'file']

    def __init__(self, *args, **kwargs):
        super(VideoProcessForm, self).__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'post'
        self.helper.form_action = ''
        self.helper.form_class = 'form-horizontal'
        self.helper.label_class = 'col-lg-2'
        self.helper.field_class = 'col-lg-8'
        self.helper.add_input(Submit("submit", "Submit"))
