from django.urls import path
from . import views



urlpatterns=[
    path('',views.index,name='index'),
    path('square',views.square,name='square'),
    path('Squareprediction',views.Squareprediction,name='Squareprediction'),
    path('squareresult',views.squareresult,name='squareresult'),

    path('Lshape',views.Lshape,name='Lshape'),
    path('Lshapeprediction',views.Lshapeprediction,name='Lshapeprediction'),
    path('Lresult',views.Lresult,name='Lresult'),

    path('Cshape',views.Cshape,name='Cshape'),
    path('Cshapeprediction',views.Cshapeprediction,name='Cshapeprediction'),
    path('Cresult',views.Cresult,name='Cresult'),

    path('Tshape',views.Tshape,name='Tshape'),
    path('Tshapeprediction',views.Tshapeprediction,name='Tshapeprediction'),
    path('Tresult',views.Tresult,name='Tresult'),

    path('Ishape',views.Ishape,name='Ishape'),
    path('Ishapeprediction',views.Ishapeprediction,name='Ishapeprediction'),
    path('Iresult',views.Iresult,name='Iresult'),

 
]