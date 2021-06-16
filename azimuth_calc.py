#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import math
from datetime import datetime
import urllib.request
import requests
import string
import re
import time
import pandas as pd
import pyorbital.orbital
from sgp4_calc import *
from sats_over_gs import *

earthRadius = 6378.137
mu = 398600.5
minutesPerDay = 1440.0
opsmode = 'i';
pi = math.pi
xpdotp = 1440.0 / (2.0 * pi);
xke = 60.0/ math.sqrt(earthRadius * earthRadius/mu)
tumin = 1.0 / xke
ixpdotp = 1440 / (2.0 * 3.141592654)
x2o3 = 2.0/3.0

vkmpersec = earthRadius * xke / 60.0;
tumin = 1.0 / xke;
j2 = 0.00108262998905;
j3 = -0.00000253215306;
j4 = -0.00000161098761;
j3oj2 = j3 / j2;


deg2rad = pi/ 180.0
twoPi = pi * 2
satDf = pd.DataFrame({"minutes":[0],
                         "station":[0],
                        "elevation":[0]})

defaultStationOptions = {
    'orbitMinutes': 0,
    'satelliteSize': 150
}


# In[ ]:




def calcAzimuth(lat, long, alt, time, station):
    ourLat = 78.229989
    ourLong = 15.404913
    ourHeight = 200
    ourAzEl = pyorbital.orbital.get_observer_look(np.atleast_1d(lat), np.atleast_1d(long), np.atleast_1d(alt), datetime.now(), np.atleast_1d(ourLat), np.atleast_1d(ourLong), np.atleast_1d(ourHeight))
    ourDf = pd.DataFrame({"minutes":[time],
                         "station":[station['name']],
                        "elevation":[ourAzEl[1][0]]})
    if ourAzEl[1][0] > 20:
        global satDf
        satDf = satDf.append(ourDf, ignore_index = True)
        print('WE ARE AT A GOOD ANGLE!!!', ourAzEl[1][0], 'minutes', time, 'station', station['name'])
        print(satDf)


# In[11]:


def ecfToLookAngles(observerGeodetic, satelliteEcf, station): 
    topocentricCoords = topocentric(observerGeodetic, satelliteEcf);
    return topocentricToLookAngles(topocentricCoords, station);


# In[12]:


def topocentricToLookAngles(tc, station):
    topS = tc['topS'],
    topE = tc['topE'],
    topZ = tc['topZ'];
#     print('topS', topS, 'topE', topE, 'topZ', topZ)
    rangeSat = math.sqrt(topS[0] * topS[0] + topE[0] * topE[0] + topZ * topZ);
    El = math.asin(topZ / rangeSat);
    Az = math.atan2(-topE[0], topS[0]) + pi;
    if El > 10:
        print('THIS IS A BIG ENOUGH EL', El, 'station', station)
    print(station['name'], {
      'azimuth': Az,
      'elevation': El,
      'rangeSat': rangeSat 

#         // Range in km

    })
    return {
      'azimuth': Az,
      'elevation': El,
      'rangeSat': rangeSat 
#         // Range in km

    };


# In[5]:


def topocentric(observerGeodetic, satelliteEcf):
#     // http'://www.celestrak.com/columns/v02n02/
#     // TS Kelso's method, except I'm using ECF frame
#     // and he uses ECI.
    longitude = observerGeodetic['longitude'],
    latitude = observerGeodetic['latitude'];
    observerEcf = geodeticToEcf(observerGeodetic);
    rx = satelliteEcf['x'] - observerEcf['x'];
    ry = satelliteEcf['y'] - observerEcf['y'];
    rz = satelliteEcf['z'] - observerEcf['z'];
    topS = math.sin(latitude) * math.cos(longitude[0]) * rx + math.sin(latitude) * math.sin(longitude[0]) * ry - math.cos(latitude) * rz;
    topE = -math.sin(longitude[0]) * rx + math.cos(longitude[0]) * ry;
    topZ = math.cos(latitude) * math.cos(longitude[0]) * rx + math.cos(latitude) * math.sin(longitude[0]) * ry + math.sin(latitude) * rz;
    return {
      'topS': topS,
      'topE': topE,
      'topZ': topZ
    };


# In[6]:


def geodeticToEcf(geodetic):
    longitude = geodetic['longitude'],
    latitude = geodetic['latitude'],
    height = geodetic['height'];
    a = 6378.137;
    b = 6356.7523142;
    f = (a - b) / a;
    e2 = 2 * f - f * f;
    normal = a / math.sqrt(1 - e2 * (math.sin(latitude[0]) * math.sin(latitude[0])));
    x = (normal + height) * math.cos(latitude[0]) * math.cos(longitude[0]);
    y = (normal + height) * math.cos(latitude[0]) * math.sin(longitude[0]);
    z = (normal * (1 - e2) + height) * math.sin(latitude[0]);
    return {
    'x': x,
    'y': y,
    'z': z
  };


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




