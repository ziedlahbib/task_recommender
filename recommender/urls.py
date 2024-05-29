from django.urls import path
from .views import recommend_view

urlpatterns = [
    path('recommend/<int:task_id>/', recommend_view, name='recommend'),
]
