# FoGS
Requirements:
- numpy
- math
- datetime
- requests
- string
- re
- time
- pandas
- pyorbital.orbital

The folder `Orbit calc` has a script titled `sats_over_gs` that takes as input a file with TLE elements and produces information about each satellite station. For the purposes of our project, we are interested in determining when satellites pass over a ground station at a high enough elevation to record a pass. 

All functions needed for this calculation are activated by running the function `_addTleFileStations` with the appropriate file path to the `celestrak.txt` file in the `parseTleFile` function. This function will update the satDf dataframe with minutes of satellites overhead a point of interest. 

Also included in the datasets folders are:
- `datasets/target_satellites` are some datasets with information about which S-Band and X-Band satellites and ground stations are currently in operation. 
- `datasets/celestrak_txt_files` are some datasets that can be used as an alternative path to urls for the pyorbital script `tlefile.py`. We have uploaded an alternative version of this script and recommend that you install pyorbital then replace this file with the proper pathname on your local computer. This is much faster than calling the url every time
