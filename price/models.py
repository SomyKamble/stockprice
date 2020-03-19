from django.db import models
from django.urls import reverse

# Create your models here.

class Product(models.Model):
    tweets_uname=models.CharField(max_length=500)

    tweet_info=models.CharField(max_length=500)

    def __str__(self):
        return self.tweet_info


