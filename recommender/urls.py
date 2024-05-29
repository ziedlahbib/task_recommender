from django.urls import path
from .views import give_recommendations

urlpatterns = [
    path('recommendations/<int:task_id>/', give_recommendations, name='recommendations'),
]
