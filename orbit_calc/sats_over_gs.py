#!/usr/bin/env python
# coding: utf-8

# In[18]:


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
from sgp4_calc1 import *
from azimuth_calc import *

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


# ## LoadLteFileStations
# 
# The function below grabs a url and plugs it into `addTleFileStations`. For our purposes we are using the url of TLE elements of active satellites
# 
# 

# In[19]:


def loadLteFileStations(url):
    val = requests.get(url)
    return _addTleFileStations(val.text);


# ## addTleFileStations
# 
# The function below takes the retrieved url as input and:
# 
# 1. Gathers stations through the `parseTleFile` function
# 2. Gathers the position of each satellite in the stations list through the `addSatellite` function

# In[20]:


def _addTleFileStations(lteFileContent):
        stations = parseTleFile(lteFileContent);
        for s in stations:
            print(s)
            addSatellite(s);


        return stations;


# ## parseTleFile
# 
# The function below parses the TLE lines and creates a list of stations

# In[21]:


def parseTleFile(fileContent):
    result = [];
    lines = fileContent.split("\n");
    current = None;

    for ind,i in enumerate(lines):
        line = lines[ind].strip();
        if (len(line) == 0): 
            continue;

        if (line[0] == '1'):
            current['tle1'] = line;
        
        elif (line[0] == '2'):
            current['tle2'] = line;
        
        else:
            current = { 
                "name": line,
                "orbitMinutes": 0
            };
            result.append(current);
        

    return result;


# ## addSatellite
# 
# This function gets:
# 1. The current position of the satellite with the `_getPositionFromTle` function
# 2. The orbit of the satellite with the `addOrbit` function

# In[22]:


def addSatellite(station):
    initialDate = datetime.utcnow();
    pos = _getPositionFromTle(station, initialDate);
    if (not pos): 
        return;
    addOrbit(station);


# ## _getPositionFromTle
# 
# This function:
# 1. Gets the x, y and z coordinates of both velocity and position of the satellite from the `getSolution` function
# 2. Translates the date into julian time with the `jday` function (found in sgp4_calc.py)
# 3. Translates the julian time to greenwich sidereal time with the `gstime` function (found in sgp4_calc.py)
# 4. Translates position to Geodetic format with the `eciToGeodetic` function
# 5. Translates Geodetic position to ECF with the `geodeticToEcf` function (found in azimuth_calc.py)
# 6. Calculates azimuth and elivation with the `ecfToLookAngles` function (found in azimuth_calc.py)
# 7. Returns latitude, longitude, and height with the `toThree` function

# In[23]:


def _getPositionFromTle(station, dateArg, type = 1):
    if (not station or not dateArg): return None;

    positionVelocity = getSolution(station, dateArg);

    positionEci = positionVelocity['position'];
    if (type == 2): return toThree(positionEci);
    
    if isinstance(dateArg, float) == True:
        date3 = datetime.utcfromtimestamp(dateArg)
        date2 = jday(date3.year, date3.month, date3.day, date3.hour, date3.minute, date3.second)
    else:
        date2 = jday(dateArg.year, dateArg.month, dateArg.day, dateArg.hour, dateArg.minute, dateArg.second)
    gmst = gstime(date2);

    if (not positionEci): return None;  
    positionGeo = eciToGeodetic(positionEci, gmst);
    positionEcf = geodeticToEcf(positionGeo)
    observerGeo = {
        'latitude' : 78.229989,
        'longitude' : 15.404913,
        'height' : 200
    }

    ecfToLookAngles(observerGeo, positionEcf, station)

    return toThree(positionGeo);


# ## getSolution
# 
# This function takes as input:
# 1. the tle elements of a single station
# 2. date
# 
# Then:
# 1. Separates the tle lines for a station
# 2. Returns a dictionary of elements parsed from TLE lines in the `twoline2satrec` function (found in sgp4_calc.py)
# 3. Returns position and velocity with the `propagate` function

# In[24]:


def getSolution(station, date):
    
    if ('satrec' in station.keys()):
        pass
    else:
        print('Here is a station:', station)
        tle1 = station['tle1'];
        tle2 = station['tle2']
        if (not tle1 or not tle2): 
            return None;
        station['satrec'] = twoline2satrec(tle1, tle2);
    return propagate(station['satrec'], date);


# ## propagate
# 
# This function takes as input:
# 1. The dictionary of elements parsed from the TLE lines
# 2. date
# 
# Then:
# 1. translates the date to julian time with the `jday` function (found in sgp4_calc.py)
# 2. returns position and velocity with the `sgp4` function (found in sgp4_calc.py)

# In[25]:


def propagate(satrec,dateArg):
    if type(dateArg) == float:
        date2 = datetime.utcfromtimestamp(dateArg)
        year = date2.year
        mon = date2.month
        day = date2.day
        hr = date2.hour
        minute = date2.minute
        sec = date2.second
    else:
        year = dateArg.year
        mon = dateArg.month
        day = dateArg.day
        hr = dateArg.hour
        minute = dateArg.minute
        sec = dateArg.second
    j = jday(year, mon, day, hr, minute, sec);
    m = (j - satrec['jdsatepoch']) * minutesPerDay;
    return sgp4(satrec, m);


# ## eciToGeodetic
# 
# This function takes as input:
# 1. x, y, and z coordinates of position
# 2. date in greenwich sidereel time
# 
# Then returns latitude longitude and height coordinates

# In[26]:


def eciToGeodetic(eci, gmst):
    a = 6378.137;
    b = 6356.7523142;
    R = math.sqrt(eci['x'] * eci['x'] + eci['y'] * eci['y']);
    f = (a - b) / a;
    e2 = 2 * f - f * f;
    longitude = math.atan2(eci['y'], eci['x']) - gmst;

    while (longitude < -pi):
        longitude += twoPi;

    while (longitude > pi):
        longitude -= twoPi;


    kmax = 20;
    k = 0;
    latitude = math.atan2(eci['z'], math.sqrt(eci['x'] * eci['x'] + eci['y'] * eci['y']));

    while (k < kmax):
        C = 1 / math.sqrt(1 - e2 * (math.sin(latitude) * math.sin(latitude)));
        latitude = math.atan2(eci['z'] + a * C * e2 * math.sin(latitude), R);
        k += 1;


    height = R / math.cos(latitude) - a * C;
    return {
      'longitude': longitude,
      'latitude': latitude,
      'height': height
    }


# ## toThree
# 
# Reformats latitude longitude and height

# In[27]:


def toThree(v):
    return {
        'latitude':v['latitude'], 'longitude':v['longitude'], 'height':v['height']
    }


# ## addOrbit
# 
# This function takes as input: 
# * A station and its TLE elements
# 
# Then for each minute of its orbit:
# 1. Calculates the position of the satellite with the `_getPositionFromTle` function
# 2. Calculates the azimuth of the satellite relative to a location of interest on the ground with the `calcAzimuth` function (found in azimuth_calc.py)

# In[28]:


def addOrbit(station):
    revsPerDay = station['satrec']['no'] * ixpdotp;
    intervalMinutes = 1;
    minutes = station['orbitMinutes'] or 1440 / revsPerDay;
    initialDate = datetime.utcnow();
    i = 0
    while i <(minutes +1):
        dateVal = (time.time() + i * 60000);
        intervalMinutes += 1
        pos = _getPositionFromTle(station, dateVal);
        calcAzimuth(pos['latitude'], pos['longitude'], pos['height'], i, station)
        i += 1
        if (not pos): continue;


# ### Run the code block below to get the whole system running!

# In[29]:


# loadLteFileStations('http://www.celestrak.com/NORAD/elements/active.txt')


# In[30]:


# satDf.to_csv('satDf.csv')


# In[ ]:




