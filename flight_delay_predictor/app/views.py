from django import forms
from django.shortcuts import render
from django.views import View
from .Flight import predict
import pandas as pd


class Homepage(View):

    def get(self, request):
        # airport = request.data['airport']
        # month = request.data['month']
        # carrier = request.data['carrier']
        # weather = request.data['weather']
        #
        # predict(airport,month,carrier,weather)
        form = MyForm()
        context = {
            'form':form,
        }
        return render(request, 'home.html', context)

    def post(self,request):

        form = MyForm(request.POST)
        if form.is_valid():
            airport = request.POST['airport']
            month = request.POST['month']
            carrier = request.POST['carrier']
            weather = request.POST['weather']

            prediction = predict(airport, month, carrier, weather)
            if prediction[0] < 20:
                prediction = "No Delay, ("+str(prediction[0])+" %)"
            elif prediction[0] <30:
                prediction = "Slight Delay, ("+str(prediction[0])+" %)"
            elif prediction[0] < 60:
                prediction = "Delay, (" + str(prediction[0]) + " %)"
            else:
                prediction = "Confirmed Delay, (" + str(prediction[0]) + " %)"

            context = {
                'form': form,
                'result':prediction
            }
            return render(request, 'home.html', context)
        else:
            return request(request,'home.html',{'form':form})


class MyForm(forms.Form):
    dataFrame = pd.read_csv('static/data.csv', header=None)
    columns = ['year', 'month', 'carrier', 'carrier_name', 'airport', 'airport_name', 'arr_flights', 'arr_del15',
               'carrier_ct', 'weather_ct', 'nas_ct', 'security_ct', 'late_aircraft_ct', 'arr_cancelled', 'arr_diverted',
               'arr_delay', 'carrier_delay', 'weather_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay',
               'temp']
    dataFrame.columns = columns

    d1 = dataFrame[['airport','airport_name']]
    d1 = d1.head(100).values

    d2 = dataFrame[['carrier','carrier_name']]

    MONTH_CHOICES = (
        ('1', 'January'),
        ('2', 'February'),
        ('3', 'March'),
        ('4', 'April'),
        ('5', 'May'),
        ('6', 'June'),
        ('7', 'July'),
        ('8', 'August'),
        ('9', 'September'),
        ('10', 'October'),
        ('11', 'November'),
        ('12', 'December')
    )

    CARRIER_CHOICES = (
        ('AA', 'American Airlines Inc'),
        ('AS', 'Alaska Airlines Inc'),
        ('B6', 'JetBlue Airways'),
        ('CO', 'Continental Air Lines Inc.'),
        ('DH', 'Atlantic Coast Airlines'),
        ('DL', 'Delta Air Lines Inc.')
    )

    AIRPORT_CHOICES = (
        ('ABQ','Albuquerque, NM: Albuquerque International Sunport'),
        ('ANC','Anchorage, AK: Ted Stevens Anchorage International')
    )

    airport = forms.ChoiceField(label='Airport',choices=d1)
    carrier = forms.ChoiceField(label='Carrier',choices=CARRIER_CHOICES)
    month = forms.ChoiceField(label='Month',choices=MONTH_CHOICES)
    weather = forms.IntegerField(label='Weather')

    class Meta:
        fields = ('airport', 'month', 'carrier','weather')
