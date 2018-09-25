# Generated by Django 2.1.1 on 2018-09-20 11:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Algorithm', '0002_videolog'),
    ]

    operations = [
        migrations.AddField(
            model_name='videolog',
            name='moving_avg',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='videolog',
            name='data',
            field=models.CharField(blank=True, max_length=500, null=True),
        ),
    ]