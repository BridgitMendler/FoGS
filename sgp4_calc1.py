#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from azimuth_calc import *
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


# In[2]:


def dspace(options):
    irez = options['irez'],
    d2201 = options['d2201'],
    d2211 = options['d2211'],
    d3210 = options['d3210'],
    d3222 = options['d3222'],
    d4410 = options['d4410'],
    d4422 = options['d4422'],
    d5220 = options['d5220'],
    d5232 = options['d5232'],
    d5421 = options['d5421'],
    d5433 = options['d5433'],
    dedt = options['dedt'],
    del1 = options['del1'],
    del2 = options['del2'],
    del3 = options['del3'],
    didt = options['didt'],
    dmdt = options['dmdt'],
    dnodt = options['dnodt'],
    domdt = options['domdt'],
    argpo = options['argpo'],
    argpdot = options['argpdot'],
    t = options['t'],
    tc = options['tc'],
    gsto = options['gsto'],
    xfact = options['xfact'],
    xlamo = options['xlamo'],
    no = options['no'];
    atime = options['atime'],
    em = options['em'],
    argpm = options['argpm'],
    inclm = options['inclm'],
    xli = options['xli'],
    mm = options['mm'],
    xni = options['xni'],
    nodem = options['nodem'],
    nm = options['nm'];
    fasx2 = 0.13130908;
    fasx4 = 2.8843198;
    fasx6 = 0.37448087;
    g22 = 5.7686396;
    g32 = 0.95240898;
    g44 = 1.8014998;
    g52 = 1.0508330;
    g54 = 4.4108898;
    rptim = 4.37526908801129966e-3;
    #     // equates to 7.29211514668855e-5 rad/sec

    stepp = 720.0;
    stepn = -720.0;
    step2 = 259200.0;

    dndt = 0.0;
    ft = 0.0;
    #     //  ----------- calculate deep space resonance effects -----------

    theta = (gsto[0] + tc[0] * rptim) % twoPi;
    if type(em) is tuple:
        em = em[0]
    if type(em) is tuple:
        em = em[0]
    if type(em) is tuple:
        em = em[0]
    em = em
    inclm = inclm[0][0][0]
    argpm = argpm[0]
    nodem = nodem[0]
    mm = mm[0]
    em += dedt[0] * t[0];
    inclm += didt[0] * t[0];
    argpm += domdt[0] * t[0];
    nodem += dnodt[0] * t[0];
    mm += dmdt[0] * t[0];
    #     // sgp4fix for negative inclinations
    #     // the following if statement should be commented out
    #     // if (inclm < 0.0)
    #     // {
    #     //   inclm = -inclm;
    #     //   argpm = argpm - pi;
    #     //   nodem = nodem + pi;
    #     // }

    #     /* - update resonances ': numerical (euler-maclaurin) integration - */

    #     /* ------------------------- epoch restart ----------------------  */
    #     //   sgp4fix for propagator problems
    #     //   the following integration works for negative time steps and periods
    #     //   the specific changes are unknown because the original code was so convoluted
    #     // sgp4fix take out atime = 0.0 and fix for faster operation

    if (irez != 0):
#       //  sgp4fix streamline check
        if type(atime) is tuple:
            atime = atime[0]
        if type(atime) is tuple:
            atime = atime[0]
#         print('atime', atime, 't', t, 'argpm', argpm, 'nodem', nodem, 'mm', mm)
        if (atime == 0.0 or t[0] * atime <= 0.0 or abs(t[0]) < abs(atime)):
            atime = 0.0;
            xni = no;
            xli = xlamo;
#         // sgp4fix move check outside loop


        if (t[0] > 0.0):
            delt = stepp;
        else:
            delt = stepn;

        iretn = 381;
#         // added for do loop
        if type(xli) is tuple:
                xli = xli[0]
        while (iretn == 381):

#         //  ------------------- dot terms calculated -------------
#         //  ----------- near - synchronous resonance terms -------
            if (irez != 2):
                if type(xli) is tuple:
                    xli = xli[0]
                if type(del1) is tuple:
                    del1 = del1[0]
                if type(del2) is tuple:
                    del2 = del3[0]
                if type(del3) is tuple:
                    del3 = del3[0]
                if type(xli) is tuple:
                    xli = xli[0]
                if type(del1) is tuple:
                    del1 = del1[0]
                if type(del2) is tuple:
                    del2 = del3[0]
                if type(del3) is tuple:
                    del3 = del3[0]
                if type(xfact) is tuple:
                    xfact = xfact[0]
                if type(xfact) is tuple:
                    xfact = xfact[0]
                xndt = del1 * math.sin(xli - fasx2) + del2 * math.sin(2.0 * (xli - fasx4)) + del3 * math.sin(3.0 * (xli - fasx6));
                xldot = xni + xfact;
                xnddt = del1 * math.cos(xli - fasx2) + 2.0 * del2 * math.cos(2.0 * (xli - fasx4)) + 3.0 * del3 * math.cos(3.0 * (xli - fasx6));
                xnddt *= xldot;
            else:
                #           // --------- near - half-day resonance terms --------
                xomi = argpo + argpdot * atime;
#                 print('xomi', xomi, 'argpo', argpo, 'argpdot', argpdot, 'atime', atime, 'x2omi', x2omi)
                x2omi = xomi + xomi;
                x2li = xli + xli;
                xndt = d2201[0][0] * math.sin(x2omi + xli - g22) + d2211[0][0] * math.sin(xli - g22) + d3210[0][0] * math.sin(xomi + xli - g32) + d3222 * math.sin(-xomi + xli - g32) + d4410 * math.sin(x2omi + x2li - g44) + d4422 * math.sin(x2li - g44) + d5220 * math.sin(xomi + xli - g52) + d5232 * math.sin(-xomi + xli - g52) + d5421 * math.sin(xomi + x2li - g54) + d5433 * math.sin(-xomi + x2li - g54);
                xldot = xni + xfact[0];
                xnddt = d2201[0][0] * math.cos(x2omi + xli - g22) + d2211[0][0] * math.cos(xli - g22) + d3210[0][0] * math.cos(xomi + xli - g32) + d3222 * math.cos(-xomi + xli - g32) + d5220 * math.cos(xomi + xli - g52) + d5232 * math.cos(-xomi + xli - g52) + 2.0 * d4410 * math.cos(x2omi + x2li - g44) + d4422 * math.cos(x2li - g44) + d5421 * math.cos(xomi + x2li - g54) + d5433 * math.cos(-xomi + x2li - g54);
                xnddt *= xldot;

#           //  ----------------------- integrator -------------------
#         //  sgp4fix move end checks to end of routine


            if (abs(t[0] - atime) >= stepp):
                iretn = 381;
            else:
                ft = t[0] - atime;
                iretn = 0;


            if (iretn == 381):
                if type(xli) is tuple:
                    xli = xli[0]
                xli += xldot * delt + xndt * step2;
                xni += xndt * delt + xnddt * step2;
                atime += delt;


        nm = xni + xndt * ft + xnddt * ft * ft * 0.5;
        xl = xli + xldot * ft + xndt * ft * ft * 0.5;

        if (irez != 1):
            mm = xl - 2.0 * nodem + 2.0 * theta;
            dndt = nm - no;
        else:
            mm = xl - nodem - argpm + theta;
            dndt = nm - no;

        nm = no + dndt;

    return {
        'atime': atime,
        'em': em,
        'argpm': argpm,
        'inclm': inclm,
        'xli': xli,
        'mm': mm,
        'xni': xni,
        'nodem': nodem,
        'dndt': dndt,
        'nm': nm
    };



# In[3]:


def sgp4(satrec, tsince):

#     /* ------------------ set mathematical constants --------------- */
#     // sgp4fix divisor for divide by zero check on inclination
#     // the old check used 1.0 + cos(pi-1.0e-9), but then compared it to
#     // 1.5 e-12, so the threshold was changed to 1.5e-12 for consistency

    temp4 = 1.5e-12;
    #     // --------------------- clear sgp4 error flag -----------------

    satrec['t'] = tsince;
    satrec['error'] = 0;
    #     //  ------- update for secular gravity and atmospheric drag -----

    xmdf = satrec['mo'][0] + satrec['mdot'] * satrec['t'];
#     print('HERE WE ARE TESTING!', type(satrec['argpo'][0]),satrec['argpdot'], satrec['t'])
    if type(satrec['argpo'][0]) is tuple:
#         print('yup we are a tuple!!',('else!!', type(satrec['nodeo']), type(satrec['nodedot']), type(satrec['t'])),)
        argpdf = satrec['argpo'][0][0] + satrec['argpdot'] * satrec['t'];
        nodedf = satrec['nodeo'][0] + satrec['nodedot'] * satrec['t'];
    else:
#         print('else!!', type(satrec['nodeo']), type(satrec['nodedot']), type(satrec['t']))
        argpdf = satrec['argpo'][0] + satrec['argpdot'] * satrec['t'];
        nodedf = satrec['nodeo'] + satrec['nodedot'] * satrec['t'];
    argpm = argpdf;
    mm = xmdf;
    t2 = satrec['t'] * satrec['t'];
    nodem = nodedf + satrec['nodecf'] * t2;
    tempa = 1.0 - satrec['cc1'] * satrec['t'];
#     print('satrec bstar', satrec['bstar'], 'satrec cc4', satrec['cc4'])
    tempe = float(satrec['bstar'][0]) * satrec['cc4'] * satrec['t'];
    templ = satrec['t2cof'] * t2;

    if (satrec['isimp'] != 1):
#         print('satrec not isimp')
        delomg = satrec['omgcof'] * satrec['t'];
        #                                  //  sgp4fix use mutliply for speed instead of pow

        delmtemp = 1.0 + satrec['eta'] * math.cos(xmdf);
        delm = satrec['xmcof'] * (delmtemp * delmtemp * delmtemp - satrec['delmo']);
        temp = delomg + delm;
        mm = xmdf + temp;
        argpm = argpdf - temp;
        t3 = t2 * satrec['t'];
        t4 = t3 * satrec['t'];
        tempa = tempa - satrec['d2'] * t2 - satrec['d3'] * t3 - satrec['d4'] * t4;
        tempe += float(satrec['bstar'][0]) * satrec['cc5'] * (math.sin(mm) - satrec['sinmao']);
        templ = templ + satrec['t3cof'] * t3 + t4 * (satrec['t4cof'] + satrec['t'] * satrec['t5cof']);

#     print('done with satrec not isimp')
    nm = satrec['no'];
    em = satrec['ecco'];
    inclm = satrec['inclo'];

    if (satrec['method'] == 'd'):
#         print('satrec method is equal to d in this scenario')
        tc = satrec['t'];
        dspaceOptions = {
        'irez': satrec['irez'],
        'd2201': satrec['d2201'],
        'd2211': satrec['d2211'],
        'd3210': satrec['d3210'],
        'd3222': satrec['d3222'],
        'd4410': satrec['d4410'],
        'd4422': satrec['d4422'],
        'd5220': satrec['d5220'],
        'd5232': satrec['d5232'],
        'd5421': satrec['d5421'],
        'd5433': satrec['d5433'],
        'dedt': satrec['dedt'],
        'del1': satrec['del1'],
        'del2': satrec['del2'],
        'del3': satrec['del3'],
        'didt': satrec['didt'],
        'dmdt': satrec['dmdt'],
        'dnodt': satrec['dnodt'],
        'domdt': satrec['domdt'],
        'argpo': satrec['argpo'],
        'argpdot': satrec['argpdot'],
        't': satrec['t'],
        'tc': tc,
        'gsto': satrec['gsto'],
        'xfact': satrec['xfact'],
        'xlamo': satrec['xlamo'],
        'no': satrec['no'],
        'atime': satrec['atime'],
        'em': em,
        'argpm': argpm,
        'inclm': inclm,
        'xli': satrec['xli'],
        'mm': mm,
        'xni': satrec['xni'],
        'nodem': nodem,
        'nm': nm
        };
#         print('we are just before dspace dspaceOptions')
        dspaceResult = dspace(dspaceOptions);
        em = dspaceResult['em'];
        argpm = dspaceResult['argpm'];
        inclm = dspaceResult['inclm'];
        mm = dspaceResult['mm'];
        nodem = dspaceResult['nodem'];
        nm = dspaceResult['nm'];

    if (nm <= 0.0):
#         print('nm <= 0')
    #       // printf("// error nm %f\n", nm);
        satrec['error'] = 2;
#                       // sgp4fix add return

        return [false, false];

#     print('below the <=')
    am = math.pow(xke / nm, x2o3) * tempa * tempa;
    nm = xke / math.pow(am, 1.5);
#     print(tempe, em)
#     print('THIS IS ABOUT THE EM OKAY!!!', type(em))
    if type(em) is tuple:
        em = em[0]
    em -= float(tempe);
#                       // fix tolerance for error recognition
#     // sgp4fix am is fixed from the previous nm check

    if (em >= 1.0 or em < -0.001):
#       // || (am < 0.95)
#       // printf("// error em %f\n", em);
        satrec['error'] = 1;
        #                       // sgp4fix to return if there is an error in eccentricity

        return [false, false];
#                 //  sgp4fix fix tolerance to avoid a divide by zero


    if (em < 1.0e-6):
        em = 1.0e-6;


    mm += satrec['no'] * templ;
    xlm = mm + argpm + nodem;
    nodem %= twoPi;
    argpm %= twoPi;
    xlm %= twoPi;
    mm = (xlm - argpm - nodem) % twoPi;
    #                  // ----------------- compute extra mean quantities -------------
    if type(inclm) is tuple:
        sinim = math.sin(inclm[0]);
        cosim = math.cos(inclm[0]);
    else:
        sinim = math.sin(inclm);
        cosim = math.cos(inclm);
    #                  // -------------------- add lunar-solar periodics --------------

    ep = em;
    xincp = inclm;
    argpp = argpm;
    nodep = nodem;
    mp = mm;
    sinip = sinim;
    cosip = cosim;

    if (satrec['method'] == 'd'):
#         print('satrec method == d')
        dpperParameters = {
        'inclo': satrec['inclo'],
        'init': 'n',
        'ep': ep,
        'inclp': xincp,
        'nodep': nodep,
        'argpp': argpp,
        'mp': mp,
        'opsmode': satrec['operationmode']
        }
        dpperResult = dpper(satrec, dpperParameters);
        ep = dpperResult['ep'];
        nodep = dpperResult['nodep'];
        argpp = dpperResult['argpp'];
        mp = dpperResult['mp'];
        xincp = dpperResult['inclp'];

        if (xincp[0] < 0.0):
            xincp = xincp[0]
            xincp = -xincp;
            nodep += pi;
            argpp -= pi;

        if type(ep) is tuple:
            ep = ep[0]
        if (ep < 0.0 or ep > 1.0):
    #         //  printf("// error ep %f\n", ep);
            satrec['error'] = 3;
    #             //  sgp4fix add return

            return [false, false];


#                       //  -------------------- long period periodics ------------------


    if (satrec['method'] == 'd'):
#         print('for some reason now satrec method == d')
        if type(xincp) is tuple:
            xincp = xincp[0]
        sinip = math.sin(xincp);
        cosip = math.cos(xincp);
        satrec['aycof'] = -0.5 * j3oj2 * sinip;
#         //  sgp4fix for divide by zero for xincp = 180 deg

        if (abs(cosip + 1.0) > 1.5e-12):
            satrec['xlcof'] = -0.25 * j3oj2 * sinip * (3.0 + 5.0 * cosip) / (1.0 + cosip);
        else:
            satrec['xlcof'] = -0.25 * j3oj2 * sinip * (3.0 + 5.0 * cosip) / temp4;

    if type(ep) is tuple:
        ep = ep[0]
    if type(argpp) is tuple:
        argpp = argpp[0]
    axnl = ep * math.cos(argpp);
    temp = 1.0 / (am * (1.0 - ep * ep));
    aynl = ep * math.sin(argpp) + temp * satrec['aycof'];
    if type(nodep) is tuple:
        nodep = nodep[0]
    xl = mp + argpp + nodep + temp * satrec['xlcof'] * axnl;
    #         // --------------------- solve kepler's equation ---------------

    u = (xl - nodep) % twoPi;
    eo1 = u;
    tem5 = 9999.9;
    ktr = 1;
    #         //    sgp4fix for kepler iteration
    #     //    the following iteration needs better limits on corrections
    #

    while (abs(tem5) >= 1.0e-12 and ktr <= 10):
#         print('in the while loop')
        sineo1 = math.sin(eo1);
        coseo1 = math.cos(eo1);
        tem5 = 1.0 - coseo1 * axnl - sineo1 * aynl;
        tem5 = (u - aynl * coseo1 + axnl * sineo1 - eo1) / tem5;

        if (abs(tem5) >= 0.95):
            if (tem5 > 0.0):
                tem5 = 0.95;
            else:
                tem5 = -0.95;


        eo1 += tem5;
        ktr += 1;
#     } //  ------------- short period preliminary quantities -----------


    ecose = axnl * coseo1 + aynl * sineo1;
    esine = axnl * sineo1 - aynl * coseo1;
    el2 = axnl * axnl + aynl * aynl;
    pl = am * (1.0 - el2);

    if (pl < 0.0):
#       //  printf("// error pl %f\n", pl);
        satrec['error'] = 4;
        #         //  sgp4fix add return

        return [false, false];


    rl = am * (1.0 - ecose);
    rdotl = math.sqrt(am) * esine / rl;
    rvdotl = math.sqrt(pl) / rl;
    betal = math.sqrt(1.0 - el2);
    temp = esine / (1.0 + betal);
    sinu = am / rl * (sineo1 - aynl - axnl * temp);
    cosu = am / rl * (coseo1 - axnl + aynl * temp);
    su = math.atan2(sinu, cosu);
    sin2u = (cosu + cosu) * sinu;
    cos2u = 1.0 - 2.0 * sinu * sinu;
    temp = 1.0 / pl;
    temp1 = 0.5 * j2 * temp;
    temp2 = temp1 * temp;
#         // -------------- update for short period periodics ------------

    if (satrec['method'] == 'd'):
        cosisq = cosip * cosip;
        satrec['con41'] = 3.0 * cosisq - 1.0;
        satrec['x1mth2'] = 1.0 - cosisq;
        satrec['x7thm1'] = 7.0 * cosisq - 1.0;

    mrt = rl * (1.0 - 1.5 * temp2 * betal * satrec['con41']) + 0.5 * temp1 * satrec['x1mth2'] * cos2u;
#     // sgp4fix for decaying satellites

    if (mrt < 1.0):
#       // printf("// decay condition %11.6f \n",mrt);
        satrec['error'] = 6;
        return {
            'position': false,
            'velocity': false
        };


    su -= 0.25 * temp2 * satrec['x7thm1'] * sin2u;
    xnode = nodep + 1.5 * temp2 * cosip * sin2u;
    if type(xincp) is tuple:
        xincp = xincp[0]
    xinc = xincp + 1.5 * temp2 * cosip * sinip * cos2u;
    mvt = rdotl - nm * temp1 * satrec['x1mth2'] * sin2u / xke;
    rvdot = rvdotl + nm * temp1 * (satrec['x1mth2'] * cos2u + 1.5 * satrec['con41']) / xke;
    #                                            // --------------------- orientation vectors -------------------

    sinsu = math.sin(su);
    cossu = math.cos(su);
    snod = math.sin(xnode);
    cnod = math.cos(xnode);
    sini = math.sin(xinc);
    cosi = math.cos(xinc);
    xmx = -snod * cosi;
    xmy = cnod * cosi;
    ux = xmx * sinsu + cnod * cossu;
    uy = xmy * sinsu + snod * cossu;
    uz = sini * sinsu;
    vx = xmx * cossu - cnod * sinsu;
    vy = xmy * cossu - snod * sinsu;
    vz = sini * cossu;
    #                                            // --------- position and velocity (in km and km/sec) ----------

    r = {
        'x': mrt * ux * earthRadius,
        'y': mrt * uy * earthRadius,
        'z': mrt * uz * earthRadius
    };
    v = {
        'x': (mvt * ux + rvdot * vx) * vkmpersec,
        'y': (mvt * uy + rvdot * vy) * vkmpersec,
        'z': (mvt * uz + rvdot * vz) * vkmpersec
    };
#     print('done with another sgp4')
    return {
        'position': r,
        'velocity': v
    };
#     /* eslint-enable no-param-reassign */


# In[4]:



def dsinit(options):
    cosim = options['cosim'],
    argpo = options['argpo'],
    s1 = options['s1'],
    s2 = options['s2'],
    s3 = options['s3'],
    s4 = options['s4'],
    s5 = options['s5'],
    sinim = options['sinim'],
    ss1 = options['ss1'],
    ss2 = options['ss2'],
    ss3 = options['ss3'],
    ss4 = options['ss4'],
    ss5 = options['ss5'],
    sz1 = options['sz1'],
    sz3 = options['sz3'],
    sz11 = options['sz11'],
    sz13 = options['sz13'],
    sz21 = options['sz21'],
    sz23 = options['sz23'],
    sz31 = options['sz31'],
    sz33 = options['sz33'],
    t = options['t'],
    tc = options['tc'],
    gsto = options['gsto'],
    mo = options['mo'],
    mdot = options['mdot'],
    no = options['no'],
    nodeo = options['nodeo'],
    nodedot = options['nodedot'],
    xpidot = options['xpidot'],
    z1 = options['z1'],
    z3 = options['z3'],
    z11 = options['z11'],
    z13 = options['z13'],
    z21 = options['z21'],
    z23 = options['z23'],
    z31 = options['z31'],
    z33 = options['z33'],
    ecco = options['ecco'],
    eccsq = options['eccsq'];
    emsq = options['emsq'],
    em = options['em'],
    argpm = options['argpm'],
    inclm = options['inclm'],
    mm = options['mm'],
    nm = options['nm'],
    nodem = options['nodem'],
    irez = options['irez'],
    atime = options['atime'],
    d2201 = options['d2201'],
    d2211 = options['d2211'],
    d3210 = options['d3210'],
    d3222 = options['d3222'],
    d4410 = options['d4410'],
    d4422 = options['d4422'],
    d5220 = options['d5220'],
    d5232 = options['d5232'],
    d5421 = options['d5421'],
    d5433 = options['d5433'],
    dedt = options['dedt'],
    didt = options['didt'],
    dmdt = options['dmdt'],
    dnodt = options['dnodt'],
    domdt = options['domdt'],
    del1 = options['del1'],
    del2 = options['del2'],
    del3 = options['del3'],
    xfact = options['xfact'],
    xlamo = options['xlamo'],
    xli = options['xli'],
    xni = options['xni'];

    q22 = 1.7891679e-6;
    q31 = 2.1460748e-6;
    q33 = 2.2123015e-7;
    root22 = 1.7891679e-6;
    root44 = 7.3636953e-9;
    root54 = 2.1765803e-9;
    rptim = 4.37526908801129966e-3;
#     // equates to 7.29211514668855e-5 rad/sec

    root32 = 3.7393792e-7;
    root52 = 1.1428639e-7;
    znl = 1.5835218e-4;
    zns = 1.19459e-5;
#     // -------------------- deep space initialization ------------

    irez = 0;

    if (nm[0] < 0.0052359877 and nm[0] > 0.0034906585):
        irez = 1;

    if type(nm) is tuple:
        nm = nm[0]
    if type(nm) is tuple:
        nm = nm[0]
    if type(nm) is tuple:
        nm = nm[0]
    if type(em) is tuple:
        em = em[0]
    if type(em) is tuple:
        em = em[0]
    if type(em) is tuple:
        em = em[0]
#     print('nm', nm, 'em', em)
    if (nm >= 8.26e-3 and nm <= 9.24e-3 and em >= 0.5):
        irez = 2;

# // ------------------------ do solar terms -------------------


    ses = ss1[0] * zns * ss5[0];
    sis = ss2[0] * zns * (sz11[0] + sz13[0]);
#     print('sz1', sz1, 'sz3', sz3)
    sls = -zns * ss3[0] * (sz1[0] + sz3[0] - 14.0 - 6.0 * emsq[0]);
    sghs = ss4[0] * zns * (sz31[0] + sz33[0] - 6.0);
    shs = -zns * ss2[0] * (sz21[0] + sz23[0]);
    #     // sgp4fix for 180 deg incl

    if (inclm[0][0] < 5.2359877e-2 or inclm[0][0] > pi - 5.2359877e-2):
        shs = 0.0;


    if (sinim != 0.0):
        shs /= sinim[0];

#     print('sghs', sghs, 'cosim', cosim, 'shs', shs)
    sgs = sghs - cosim[0] * shs;
# // ------------------------- do lunar terms ------------------

#     print('z11', z11, 'z13', z13, 'sis', sis)
    dedt = ses + s1[0] * znl * s5[0];
    didt = sis + s2[0] * znl * (z11[0] + z13[0]);
    dmdt = sls - znl * s3[0] * (z1[0] + z3[0] - 14.0 - 6.0 * emsq[0]);
    sghl = s4[0] * znl * (z31[0] + z33[0] - 6.0);
    shll = -znl * s2[0] * (z21[0] + z23[0]);
#     // sgp4fix for 180 deg incl

    if (inclm[0][0] < 5.2359877e-2 or inclm[0][0] > pi - 5.2359877e-2):
        shll = 0.0;


    domdt = sgs + sghl;
    dnodt = shs;

#     print('shll', shll, 'cosim', cosim)
    if (sinim != 0.0):
        domdt -= cosim[0] / sinim[0] * shll;
        dnodt += shll / sinim[0];

# // ----------- calculate deep space resonance effects --------


    dndt = 0.0;

    theta = (gsto[0] + tc[0] * rptim) % twoPi;
    if type(em) is tuple:
        em = em[0]
    if type(em) is tuple:
        em = em[0]
    if type(em) is tuple:
        em = em[0]
    em = em
    em += dedt * t[0];
    inclm = inclm[0][0]
    inclm += didt * t[0];
    argpm = argpm[0]
    nodem = nodem[0]
    mm= mm[0]
    argpm += domdt * t[0];
    nodem += dnodt * t[0];
    mm += dmdt * t[0];
#     // sgp4fix for negative inclinations
#   // the following if statement should be commented out
#   // if (inclm < 0.0)
#   // {
#   //   inclm  = -inclm;
#   //   argpm  = argpm - pi;
#   //   nodem = nodem + pi;
#   // }
#   // -------------- initialize the resonance terms -------------

    if (irez != 0):
        if type(nm) is tuple:
            nm = nm[0]
#         print('nm', nm, 'xke', xke, 'x2o3', x2o3)
        aonv = math.pow(nm / xke, x2o3);
#       // ---------- geopotential resonance for 12 hour orbits ------

        if (irez == 2):
            cosisq = cosim[0] * cosim[0];
            emo = em;
            em = ecco;
            emsqo = emsq;
            if type(emsq) is tuple:
                emsq = emsq[0]
            if type(emsq) is tuple:
                emsq = emsq[0]
            if type(em) is tuple:
                em = em[0]
            if type(emsq) is tuple:
                emsq = emsq[0]
            if type(emsq) is tuple:
                emsq = emsq[0]
            if type(em) is tuple:
                em = em[0]
            if type(em) is tuple:
                em = em[0]
            emsq = eccsq;
            if type(emsq) is tuple:
                emsq = emsq[0]
#             print('em', em, 'emsq', emsq)
            eoc = em * emsq;
            g201 = -0.306 - (em - 0.64) * 0.440;

            if (em <= 0.65):
                g211 = 3.616 - 13.2470 * em + 16.2900 * emsq;
                g310 = -19.302 + 117.3900 * em - 228.4190 * emsq + 156.5910 * eoc;
                g322 = -18.9068 + 109.7927 * em - 214.6334 * emsq + 146.5816 * eoc;
                g410 = -41.122 + 242.6940 * em - 471.0940 * emsq + 313.9530 * eoc;
                g422 = -146.407 + 841.8800 * em - 1629.014 * emsq + 1083.4350 * eoc;
                g520 = -532.114 + 3017.977 * em - 5740.032 * emsq + 3708.2760 * eoc;
            else:
                g211 = -72.099 + 331.819 * em - 508.738 * emsq + 266.724 * eoc;
                g310 = -346.844 + 1582.851 * em - 2415.925 * emsq + 1246.113 * eoc;
                g322 = -342.585 + 1554.908 * em - 2366.899 * emsq + 1215.972 * eoc;
                g410 = -1052.797 + 4758.686 * em - 7193.992 * emsq + 3651.957 * eoc;
                g422 = -3581.690 + 16178.110 * em - 24462.770 * emsq + 12422.520 * eoc;

                if (em > 0.715):
                    g520 = -5149.66 + 29936.92 * em - 54087.36 * emsq + 31324.56 * eoc;
                else:
                    g520 = 1464.74 - 4664.75 * em + 3763.64 * emsq;


            if (em < 0.7):
                g533 = -919.22770 + 4988.6100 * em - 9064.7700 * emsq + 5542.21 * eoc;
                g521 = -822.71072 + 4568.6173 * em - 8491.4146 * emsq + 5337.524 * eoc;
                g532 = -853.66600 + 4690.2500 * em - 8624.7700 * emsq + 5341.4 * eoc;
            else:
                g533 = -37995.780 + 161616.52 * em - 229838.20 * emsq + 109377.94 * eoc;
                g521 = -51752.104 + 218913.95 * em - 309468.16 * emsq + 146349.42 * eoc;
                g532 = -40023.880 + 170470.89 * em - 242699.48 * emsq + 115605.82 * eoc;


            sini2 = sinim[0] * sinim[0];
            f220 = 0.75 * (1.0 + 2.0 * cosim[0] + cosisq);
            f221 = 1.5 * sini2;
            f321 = 1.875 * sinim[0] * (1.0 - 2.0 * cosim[0] - 3.0 * cosisq);
            f322 = -1.875 * sinim[0] * (1.0 + 2.0 * cosim[0] - 3.0 * cosisq);
            f441 = 35.0 * sini2 * f220;
            f442 = 39.3750 * sini2 * sini2;
#             if type(sinim) is tuple:
#                 sinim = sinim[0]
            f522 = 9.84375 * sinim[0] * (sini2 * (1.0 - 2.0 * cosim[0] - 5.0 * cosisq) + 0.33333333 * (-2.0 + 4.0 * cosim[0] + 6.0 * cosisq));
            f523 = sinim[0] * (4.92187512 * sini2 * (-2.0 - 4.0 * cosim[0] + 10.0 * cosisq) + 6.56250012 * (1.0 + 2.0 * cosim[0] - 3.0 * cosisq));
            f542 = 29.53125 * sinim[0] * (2.0 - 8.0 * cosim[0] + cosisq * (-12.0 + 8.0 * cosim[0] + 10.0 * cosisq));
            f543 = 29.53125 * sinim[0] * (-2.0 - 8.0 * cosim[0] + cosisq * (12.0 + 8.0 * cosim[0] - 10.0 * cosisq));
            xno2 = nm * nm;
            ainv2 = aonv * aonv;
            temp1 = 3.0 * xno2 * ainv2;
            temp = temp1 * root22;
            d2201 = temp * f220 * g201;
            d2211 = temp * f221 * g211;
            temp1 *= aonv;
            temp = temp1 * root32;
            d3210 = temp * f321 * g310;
            d3222 = temp * f322 * g322;
            temp1 *= aonv;
            temp = 2.0 * temp1 * root44;
            d4410 = temp * f441 * g410;
            d4422 = temp * f442 * g422;
            temp1 *= aonv;
            temp = temp1 * root52;
            d5220 = temp * f522 * g520;
            d5232 = temp * f523 * g532;
            temp = 2.0 * temp1 * root54;
            d5421 = temp * f542 * g521;
            d5433 = temp * f543 * g533;
#             print(dmdt, nodedot, dnodt, rptim)
            xlamo = (mo[0][0] + nodeo[0][0] + nodeo[0][0] - (theta + theta)) % twoPi;
            xfact = mdot[0] + dmdt + 2.0 * (nodedot[0] + dnodt - rptim) - no[0];
            em = emo;
            emsq = emsqo;

#     //  ---------------- synchronous resonance terms --------------


        if (irez == 1):
            g200 = 1.0 + emsq[0] * (-2.5 + 0.8125 * emsq[0]);
            g310 = 1.0 + 2.0 * emsq[0];
            g300 = 1.0 + emsq[0] * (-6.0 + 6.60937 * emsq[0]);
            f220 = 0.75 * (1.0 + cosim[0]) * (1.0 + cosim[0]);
            f311 = 0.9375 * sinim[0] * sinim[0] * (1.0 + 3.0 * cosim[0]) - 0.75 * (1.0 + cosim[0]);
            f330 = 1.0 + cosim[0];
            f330 *= 1.875 * f330 * f330;
#             print('mdot', mdot, 'xpidot', xpidot, 'dmdt', dmdt, 'domdt', domdt, 'dnodt', dnodt, 'no',no, 'rptim', rptim)
            del1 = 3.0 * nm * nm * aonv * aonv;
            del2 = 2.0 * del1 * f220 * g200 * q22;
            del3 = 3.0 * del1 * f330 * g300 * q33 * aonv;
            del1 = del1 * f311 * g310 * q31 * aonv;
            xlamo = (mo[0][0] + nodeo[0][0] + argpo[0][0][0] - theta) % twoPi;
            xfact = mdot[0] + xpidot[0] + dmdt + domdt + dnodt - (no[0] + rptim);

#     //  ------------ for sgp4, initialize the integrator ----------


        xli = xlamo;
        xni = no;
        atime = 0.0;
        nm = no[0] + dndt;


    return {
        'em': em,
        'argpm': argpm,
        'inclm': inclm,
        'mm': mm,
        'nm': nm,
        'nodem': nodem,
        'irez': irez,
        'atime': atime,
        'd2201': d2201,
        'd2211': d2211,
        'd3210': d3210,
        'd3222': d3222,
        'd4410': d4410,
        'd4422': d4422,
        'd5220': d5220,
        'd5232': d5232,
        'd5421': d5421,
        'd5433': d5433,
        'dedt': dedt,
        'didt': didt,
        'dmdt': dmdt,
        'dndt': dndt,
        'dnodt': dnodt,
        'domdt': domdt,
        'del1': del1,
        'del2': del2,
        'del3': del3,
        'xfact': xfact,
        'xlamo': xlamo,
        'xli': xli,
        'xni': xni
    }


# In[5]:


def dpper(satrec, options):
    e3 = satrec['e3'],
    ee2 = satrec['ee2'],
    peo = satrec['peo'],
    pgho = satrec['pgho'],
    pho = satrec['pho'],
    pinco = satrec['pinco'],
    plo = satrec['plo'],
    se2 = satrec['se2'],
    se3 = satrec['se3'],
    sgh2 = satrec['sgh2'],
    sgh3 = satrec['sgh3'],
    sgh4 = satrec['sgh4'],
    sh2 = satrec['sh2'],
    sh3 = satrec['sh3'],
    si2 = satrec['si2'],
    si3 = satrec['si3'],
    sl2 = satrec['sl2'],
    sl3 = satrec['sl3'],
    sl4 = satrec['sl4'],
    t = satrec['t'],
    xgh2 = satrec['xgh2'],
    xgh3 = satrec['xgh3'],
    xgh4 = satrec['xgh4'],
    xh2 = satrec['xh2'],
    xh3 = satrec['xh3'],
    xi2 = satrec['xi2'],
    xi3 = satrec['xi3'],
    xl2 = satrec['xl2'],
    xl3 = satrec['xl3'],
    xl4 = satrec['xl4'],
    zmol = satrec['zmol'],
    zmos = satrec['zmos'];
    init = options['init'],
    opsmode = options['opsmode'];
    ep = options['ep'],
    inclp = options['inclp'],
    nodep = options['nodep'],
    argpp = options['argpp'],
    mp = options['mp'];
    #         // Copy satellite attributes into local iables for convenience
    #     // and symmetry in writing formulae.


    zns = 1.19459e-5;
    zes = 0.01675;
    znl = 1.5835218e-4;
    zel = 0.05490;
    #         //  --------------- calculate time ying periodics -----------


    zm = zmos + zns * t[0];
    #     // be sure that the initial call has time set to zero

    if (init == 'y'):
        zm = zmos;

    zf = zm + 2.0 * zes * math.sin(zm);
    sinzf = math.sin(zf);
    f2 = 0.5 * sinzf * sinzf - 0.25;
    f3 = -0.5 * sinzf * math.cos(zf);
#     print('zmol', zmol, 'znl', znl, 'se3', sl3, 'f3', f3)
    ses = se2[0] * f2 + se3[0] * f3;
    sis = si2[0] * f2 + si3[0] * f3;
    sls = sl2[0] * f2 + sl3[0] * f3 + sl4[0] * sinzf;
    sghs = sgh2[0] * f2 + sgh3[0] * f3 + sgh4[0] * sinzf;
    shs = sh2[0] * f2 + sh3[0] * f3;
    zm = zmol[0] + znl * t[0];

    if (init == 'y'):
        zm = zmol;


    zf = zm + 2.0 * zel * math.sin(zm);
    sinzf = math.sin(zf);
    f2 = 0.5 * sinzf * sinzf - 0.25;
    f3 = -0.5 * sinzf * math.cos(zf);
#     print('ee2', ee2, 'e3', e3)
    sel = ee2[0] * f2 + e3[0] * f3;
    sil = xi2[0] * f2 + xi3[0] * f3;
    sll = xl2[0] * f2 + xl3[0] * f3 + xl4[0] * sinzf;
    sghl = xgh2[0] * f2 + xgh3[0] * f3 + xgh4[0] * sinzf;
    shll = xh2[0] * f2 + xh3[0] * f3;
    pe = ses + sel;
    pinc = sis + sil;
    pl = sls + sll;
    pgh = sghs + sghl;
    ph = shs + shll;

    if (init == 'n'):
        pe -= peo;
        pinc -= pinco;
        pl -= plo;
        pgh -= pgho;
        ph -= pho;
        inclp += pinc;
        ep += pe;
        sinip = math.sin(inclp);
        cosip = math.cos(inclp);
#       /* ----------------- apply periodics directly ------------ */
#       // sgp4fix for lyddane choice
#       // strn3 used original inclination - this is technically feasible
#       // gsfc used perturbed inclination - also technically feasible
#       // probably best to readjust the 0.2 limit value and limit discontinuity
#       // 0.2 rad = 11.45916 deg
#       // use next line for original strn3 approach and original inclination
#       // if (inclo >= 0.2)
#       // use next line for gsfc version and perturbed inclination

        if (inclp >= 0.2):
            ph /= sinip;
            pgh -= cosip * ph;
            argpp += pgh;
            nodep += ph;
            mp += pl;
        else:
#         //  ---- apply periodics with lyddane modification ----
            sinop = math.sin(nodep);
            cosop = math.cos(nodep);
            alfdp = sinip * sinop;
            betdp = sinip * cosop;
            dalf = ph * cosop + pinc * cosip * sinop;
            dbet = -ph * sinop + pinc * cosip * cosop;
            alfdp += dalf;
            betdp += dbet;
            nodep %= twoPi;
#         //  sgp4fix for afspc written intrinsic functions
#         //  nodep used without a trigonometric function ahead

            if (nodep < 0.0 and opsmode == 'a'):
                nodep += twoPi;

            xls = mp + argpp + cosip * nodep;
            dls = pl + pgh - pinc * nodep * sinip;
            xls += dls;
            xnoh = nodep;
            nodep = math.atan2(alfdp, betdp);
#             //  sgp4fix for afspc written intrinsic functions
#             //  nodep used without a trigonometric function ahead

            if (nodep < 0.0 and opsmode == 'a'):
                nodep += twoPi;


            if (math.abs(xnoh - nodep) > pi):
                if (nodep < xnoh):
                    nodep += twoPi;
                else:
                    nodep -= twoPi;



            mp += pl;
            argpp = xls - mp - cosip * nodep;


    return {
        'ep': ep,
        'inclp': inclp,
        'nodep': nodep,
        'argpp': argpp,
        'mp': mp
    };



# In[6]:



def dscom(options):
    epoch = options['epoch'],
    ep = options['ep'],
    argpp = options['argpp'],
    tc = options['tc'],
    inclp = options['inclp'],
    nodep = options['nodep'],
    np = options['np'];

    zes = 0.01675;
    zel = 0.05490;
    c1ss = 2.9864797e-6;
    c1l = 4.7968065e-7;
    zsinis = 0.39785416;
    zcosis = 0.91744867;
    zcosgs = 0.1945905;
    zsings = -0.98088458;
    #     //  --------------------- local iables ------------------------

    nm = np;
    em = ep;
    snodm = math.sin(nodep[0]);
    cnodm = math.cos(nodep[0]);
    sinomm = math.sin(argpp[0][0]);
    cosomm = math.cos(argpp[0][0]);
    sinim = math.sin(inclp[0][0]);
    cosim = math.cos(inclp[0][0]);
    emsq = em[0][0] * em[0][0];
    betasq = 1.0 - emsq;
    rtemsq = math.sqrt(betasq);
    #     //  ----------------- initialize lunar solar terms ---------------

    peo = 0.0;
    pinco = 0.0;
    plo = 0.0;
    pgho = 0.0;
    pho = 0.0;
#     print('tc', tc)
    day = epoch[0][0] + 18261.5 + tc[0] / 1440.0;
    xnodce = (4.5236020 - 9.2422029e-4 * day) % twoPi;
    stem = math.sin(xnodce);
    ctem = math.cos(xnodce);
    zcosil = 0.91375164 - 0.03568096 * ctem;
    zsinil = math.sqrt(1.0 - zcosil * zcosil);
    zsinhl = 0.089683511 * stem / zsinil;
    zcoshl = math.sqrt(1.0 - zsinhl * zsinhl);
    gam = 5.8351514 + 0.0019443680 * day;
    zx = 0.39785416 * stem / zsinil;
    zy = zcoshl * ctem + 0.91744867 * zsinhl * stem;
    zx = math.atan2(zx, zy);
    zx += gam - xnodce;
    zcosgl = math.cos(zx);
    zsingl = math.sin(zx);
    #     //  ------------------------- do solar terms ---------------------

    zcosg = zcosgs;
    zsing = zsings;
    zcosi = zcosis;
    zsini = zsinis;
    zcosh = cnodm;
    zsinh = snodm;
    cc = c1ss;
    xnoi = 1.0 / nm;
    lsflg = 0;

    while (lsflg < 2):
        lsflg += 1;
        a1 = zcosg * zcosh + zsing * zcosi * zsinh;
        a3 = -zsing * zcosh + zcosg * zcosi * zsinh;
        a7 = -zcosg * zsinh + zsing * zcosi * zcosh;
        a8 = zsing * zsini;
        a9 = zsing * zsinh + zcosg * zcosi * zcosh;
        a10 = zcosg * zsini;
        a2 = cosim * a7 + sinim * a8;
        a4 = cosim * a9 + sinim * a10;
        a5 = -sinim * a7 + cosim * a8;
        a6 = -sinim * a9 + cosim * a10;
        x1 = a1 * cosomm + a2 * sinomm;
        x2 = a3 * cosomm + a4 * sinomm;
        x3 = -a1 * sinomm + a2 * cosomm;
        x4 = -a3 * sinomm + a4 * cosomm;
        x5 = a5 * sinomm;
        x6 = a6 * sinomm;
        x7 = a5 * cosomm;
        x8 = a6 * cosomm;
        z31 = 12.0 * x1 * x1 - 3.0 * x3 * x3;
        z32 = 24.0 * x1 * x2 - 6.0 * x3 * x4;
        z33 = 12.0 * x2 * x2 - 3.0 * x4 * x4;
        z1 = 3.0 * (a1 * a1 + a2 * a2) + z31 * emsq;
        z2 = 6.0 * (a1 * a3 + a2 * a4) + z32 * emsq;
        z3 = 3.0 * (a3 * a3 + a4 * a4) + z33 * emsq;
        z11 = -6.0 * a1 * a5 + emsq * (-24.0 * x1 * x7 - 6.0 * x3 * x5);
        z12 = -6.0 * (a1 * a6 + a3 * a5) + emsq * (-24.0 * (x2 * x7 + x1 * x8) + -6.0 * (x3 * x6 + x4 * x5));
        z13 = -6.0 * a3 * a6 + emsq * (-24.0 * x2 * x8 - 6.0 * x4 * x6);
        z21 = 6.0 * a2 * a5 + emsq * (24.0 * x1 * x5 - 6.0 * x3 * x7);
        z22 = 6.0 * (a4 * a5 + a2 * a6) + emsq * (24.0 * (x2 * x5 + x1 * x6) - 6.0 * (x4 * x7 + x3 * x8));
        z23 = 6.0 * a4 * a6 + emsq * (24.0 * x2 * x6 - 6.0 * x4 * x8);
        z1 = z1 + z1 + betasq * z31;
        z2 = z2 + z2 + betasq * z32;
        z3 = z3 + z3 + betasq * z33;
        s3 = cc * xnoi;
        s2 = -0.5 * s3 / rtemsq;
        s4 = s3 * rtemsq;
        s1 = -15.0 * em[0][0] * s4;
        s5 = x1 * x3 + x2 * x4;
        s6 = x2 * x3 + x1 * x4;
        s7 = x2 * x4 - x1 * x3;
        #       //  ----------------------- do lunar terms -------------------

        if (lsflg == 1):
            ss1 = s1;
            ss2 = s2;
            ss3 = s3;
            ss4 = s4;
            ss5 = s5;
            ss6 = s6;
            ss7 = s7;
            sz1 = z1;
            sz2 = z2;
            sz3 = z3;
            sz11 = z11;
            sz12 = z12;
            sz13 = z13;
            sz21 = z21;
            sz22 = z22;
            sz23 = z23;
            sz31 = z31;
            sz32 = z32;
            sz33 = z33;
            zcosg = zcosgl;
            zsing = zsingl;
            zcosi = zcosil;
            zsini = zsinil;
            zcosh = zcoshl * cnodm + zsinhl * snodm;
            zsinh = snodm * zcoshl - cnodm * zsinhl;
            cc = c1l;


    zmol = (4.7199672 + (0.22997150 * day - gam)) % twoPi;
    zmos = (6.2565837 + 0.017201977 * day) % twoPi;
    #     //  ------------------------ do solar terms ----------------------

    se2 = 2.0 * ss1 * ss6;
    se3 = 2.0 * ss1 * ss7;
    si2 = 2.0 * ss2 * sz12;
    si3 = 2.0 * ss2 * (sz13 - sz11);
    sl2 = -2.0 * ss3 * sz2;
    sl3 = -2.0 * ss3 * (sz3 - sz1);
    sl4 = -2.0 * ss3 * (-21.0 - 9.0 * emsq) * zes;
    sgh2 = 2.0 * ss4 * sz32;
    sgh3 = 2.0 * ss4 * (sz33 - sz31);
    sgh4 = -18.0 * ss4 * zes;
    sh2 = -2.0 * ss2 * sz22;
    sh3 = -2.0 * ss2 * (sz23 - sz21);
    #     //  ------------------------ do lunar terms ----------------------

    ee2 = 2.0 * s1 * s6;
    e3 = 2.0 * s1 * s7;
    xi2 = 2.0 * s2 * z12;
    xi3 = 2.0 * s2 * (z13 - z11);
    xl2 = -2.0 * s3 * z2;
    xl3 = -2.0 * s3 * (z3 - z1);
    xl4 = -2.0 * s3 * (-21.0 - 9.0 * emsq) * zel;
    xgh2 = 2.0 * s4 * z32;
    xgh3 = 2.0 * s4 * (z33 - z31);
    xgh4 = -18.0 * s4 * zel;
    xh2 = -2.0 * s2 * z22;
    xh3 = -2.0 * s2 * (z23 - z21);

    return {
    'snodm': snodm,
    'cnodm': cnodm,
    'sinim': sinim,
    'cosim': cosim,
    'sinomm': sinomm,
    'cosomm': cosomm,
    'day': day,
    'e3': e3,
    'ee2': ee2,
    'em': em,
    'emsq': emsq,
    'gam': gam,
    'peo': peo,
    'pgho': pgho,
    'pho': pho,
    'pinco': pinco,
    'plo': plo,
    'rtemsq': rtemsq,
    'se2': se2,
    'se3': se3,
    'sgh2': sgh2,
    'sgh3': sgh3,
    'sgh4': sgh4,
    'sh2': sh2,
    'sh3': sh3,
    'si2': si2,
    'si3': si3,
    'sl2': sl2,
    'sl3': sl3,
    'sl4': sl4,
    's1': s1,
    's2': s2,
    's3': s3,
    's4': s4,
    's5': s5,
    's6': s6,
    's7': s7,
    'ss1': ss1,
    'ss2': ss2,
    'ss3': ss3,
    'ss4': ss4,
    'ss5': ss5,
    'ss6': ss6,
    'ss7': ss7,
    'sz1': sz1,
    'sz2': sz2,
    'sz3': sz3,
    'sz11': sz11,
    'sz12': sz12,
    'sz13': sz13,
    'sz21': sz21,
    'sz22': sz22,
    'sz23': sz23,
    'sz31': sz31,
    'sz32': sz32,
    'sz33': sz33,
    'xgh2': xgh2,
    'xgh3': xgh3,
    'xgh4': xgh4,
    'xh2': xh2,
    'xh3': xh3,
    'xi2': xi2,
    'xi3': xi3,
    'xl2': xl2,
    'xl3': xl3,
    'xl4': xl4,
    'nm': nm,
    'z1': z1,
    'z2': z2,
    'z3': z3,
    'z11': z11,
    'z12': z12,
    'z13': z13,
    'z21': z21,
    'z22': z22,
    'z23': z23,
    'z31': z31,
    'z32': z32,
    'z33': z33,
    'zmol': zmol,
    'zmos': zmos
  };


# In[7]:


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


# In[8]:


def twoline2satrec(longstr1, longstr2):
    earthRadius = 6378.137
    mu = 398600.5
    opsmode = 'i';
    xpdotp = 1440.0 / (2.0 * pi);
    xke = 60.0/ math.sqrt(earthRadius * earthRadius/mu)
    tumin = 1.0 / xke
    deg2rad = pi/ 180.0

    year = 0;
    satrec = {};
    satrec['error'] = 0;
    satrec['satnum'] = longstr1[2:7];
    satrec['epochyr'] = int(longstr1[18:20], 10);
    satrec['epochdays'] = float(longstr1[20:32]);
    satrec['ndot'] = float(longstr1[33:43]);
    satrec['nddot'] = float((("."+(longstr1[45:50])+"E")+(longstr1[50:52])).strip(string.ascii_letters))
    satrec['bstar'] = (""+(longstr1[53:54]+".")+(longstr1[54:59] + "E")+(longstr1[59:61]));
    satrec['inclo'] = float(longstr2[8:16]);
    satrec['nodeo'] = float(longstr2[17:25]);
    satrec['ecco'] = float("."+(longstr2[26:33]));
    satrec['argpo'] = float(longstr2[34:42]);
    satrec['mo'] = float(longstr2[43:51]);
    satrec['no'] = float(longstr2[52:63]);
#     // ---- find no, ndot, nddot ----

    satrec['no'] /= xpdotp;
# ---- convert to sgp4 units ----

    satrec['a'] = math.pow(satrec['no'] * tumin, -2.0 / 3.0);
    satrec['ndot'] /= xpdotp * 1440.0;
#     // ? * minperday

    satrec['nddot'] /= xpdotp * 1440.0 * 1440;
#     // ---- find standard orbital elements ----

    satrec['inclo'] *= deg2rad;
    satrec['nodeo'] *= deg2rad;
    satrec['argpo'] *= deg2rad;
    satrec['mo'] *= deg2rad;
    satrec['alta'] = satrec['a'] * (1.0 + satrec['ecco']) - 1.0;
    satrec['altp'] = satrec['a'] * (1.0 - satrec['ecco']) - 1.0;
#     // ----------------------------------------------------------------
#     // find sgp4epoch time of element set
#     // remember that sgp4 uses units of days from 0 jan 1950 (sgp4epoch)
#     // and minutes from the epoch (time)
#     // ----------------------------------------------------------------
#     // ---------------- temp fix for years from 1957-2056 -------------------
#     // --------- correct fix will occur when year is 4-digit in tle ---------

    if (satrec['epochyr'] < 57):
        year = satrec['epochyr'] + 2000;
    else:
        year = satrec['epochyr'] + 1900;
    mdhmsResult = days2mdhms(year, satrec['epochdays']);
    mon = mdhmsResult['mon'],
    day = mdhmsResult['day'],
    hr = mdhmsResult['hr'],
    minute = mdhmsResult['minute'],
    sec = mdhmsResult['sec'];
    satrec['jdsatepoch'] = jday(year, mon[0], day[0], hr[0], minute[0], sec);
#     //  ---------------- initialize the orbit at sgp4epoch -------------------

    sgp4init(satrec, {
      "opsmode": opsmode,
      "satn": satrec['satnum'],
      "epoch": satrec['jdsatepoch'] - 2433281.5,
      "xbstar": satrec['bstar'],
      "xecco": satrec['ecco'],
      "xargpo": satrec['argpo'],
      "xinclo": satrec['inclo'],
      "xmo": satrec['mo'],
      "xno": satrec['no'],
      "xnodeo": satrec['nodeo']
    });
    return satrec;


# In[9]:


def days2mdhms(year, days):
  lmonth = [31, year % 29 if 4 == 0 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  dayofyr = math.floor(days);
  #     //  ----------------- find month and day of month ----------------

  i = 1;
  inttemp = 0;

  while (dayofyr > inttemp + lmonth[i - 1]) and (i < 12):
      inttemp += lmonth[i - 1]
      i += 1;


  mon = i;
  day = dayofyr - inttemp;
  #     //  ----------------- find hours minutes and seconds -------------

  temp = (days - dayofyr) * 24.0
  hr = math.floor(temp)
  temp = (temp - hr) * 60.0
  minute = math.floor(temp)
  sec = (temp - minute) * 60.0
  return {
    'mon': mon,
    'day': day,
    'hr': hr,
    'minute': minute,
    'sec': sec
  }


# In[10]:


def jdayInternal(year, mon, day, hr, minute, sec,msec = 0):
    return 367.0 * year - math.floor(7 * (year + math.floor((mon + 9) / 12.0)) * 0.25) + math.floor(275 * mon / 9.0) + day + 1721013.5 + ((msec / 60000 + sec / 60.0 + minute) / 60.0 + hr) / 24.0     ;

def jday(year, mon, day, hr, minute, sec, msec = 0):
    if isinstance(year, datetime):
        dateYear = year;
        return jdayInternal(dateYear.getUTCFullYear(), dateYear.getUTCMonth() + 1, dateYear.getUTCDate(), dateYear.getUTCHours(), dateYear.getUTCMinutes(), dateYear.getUTCSeconds(), dateYear.getUTCMilliseconds());
    return jdayInternal(year, mon, day, hr, minute, sec, msec);


# In[11]:



def sgp4init(satrec, options):
#   /* eslint-disable no-param-reassign */
    opsmode = options['opsmode'],
    satn = options['satn'],
    epoch = options['epoch'],
    xbstar = options['xbstar'],
    xecco = options['xecco'],
    xargpo = options['xargpo'],
    xinclo = options['xinclo'],
    xmo = options['xmo'],
    xno = options['xno'],
    xnodeo = options['xnodeo'];

#   /* ------------------------ initialization --------------------- */
#   // sgp4fix divisor for divide by zero check on inclination
#   // the old check used 1.0 + math.cos(pi-1.0e-9)'], but then compared it to
#   // 1.5 e-12'], so the threshold was changed to 1.5e-12 for consistency

    temp4 = 1.5e-12;
#     // ----------- set all near earth iables to zero ------------

    satrec['isimp'] = 0;
    satrec['method'] = 'n';
    satrec['aycof'] = 0.0;
    satrec['con41'] = 0.0;
    satrec['cc1'] = 0.0;
    satrec['cc4'] = 0.0;
    satrec['cc5'] = 0.0;
    satrec['d2'] = 0.0;
    satrec['d3'] = 0.0;
    satrec['d4'] = 0.0;
    satrec['delmo'] = 0.0;
    satrec['eta'] = 0.0;
    satrec['argpdot'] = 0.0;
    satrec['omgcof'] = 0.0;
    satrec['sinmao'] = 0.0;
    satrec['t'] = 0.0;
    satrec['t2cof'] = 0.0;
    satrec['t3cof'] = 0.0;
    satrec['t4cof'] = 0.0;
    satrec['t5cof'] = 0.0;
    satrec['x1mth2'] = 0.0;
    satrec['x7thm1'] = 0.0;
    satrec['mdot'] = 0.0;
    satrec['nodedot'] = 0.0;
    satrec['xlcof'] = 0.0;
    satrec['xmcof'] = 0.0;
    satrec['nodecf'] = 0.0;
    #     // ----------- set all deep space iables to zero ------------

    satrec['irez'] = 0;
    satrec['d2201'] = 0.0;
    satrec['d2211'] = 0.0;
    satrec['d3210'] = 0.0;
    satrec['d3222'] = 0.0;
    satrec['d4410'] = 0.0;
    satrec['d4422'] = 0.0;
    satrec['d5220'] = 0.0;
    satrec['d5232'] = 0.0;
    satrec['d5421'] = 0.0;
    satrec['d5433'] = 0.0;
    satrec['dedt'] = 0.0;
    satrec['del1'] = 0.0;
    satrec['del2'] = 0.0;
    satrec['del3'] = 0.0;
    satrec['didt'] = 0.0;
    satrec['dmdt'] = 0.0;
    satrec['dnodt'] = 0.0;
    satrec['domdt'] = 0.0;
    satrec['e3'] = 0.0;
    satrec['ee2'] = 0.0;
    satrec['peo'] = 0.0;
    satrec['pgho'] = 0.0;
    satrec['pho'] = 0.0;
    satrec['pinco'] = 0.0;
    satrec['plo'] = 0.0;
    satrec['se2'] = 0.0;
    satrec['se3'] = 0.0;
    satrec['sgh2'] = 0.0;
    satrec['sgh3'] = 0.0;
    satrec['sgh4'] = 0.0;
    satrec['sh2'] = 0.0;
    satrec['sh3'] = 0.0;
    satrec['si2'] = 0.0;
    satrec['si3'] = 0.0;
    satrec['sl2'] = 0.0;
    satrec['sl3'] = 0.0;
    satrec['sl4'] = 0.0;
    satrec['gsto'] = 0.0;
    satrec['xfact'] = 0.0;
    satrec['xgh2'] = 0.0;
    satrec['xgh3'] = 0.0;
    satrec['xgh4'] = 0.0;
    satrec['xh2'] = 0.0;
    satrec['xh3'] = 0.0;
    satrec['xi2'] = 0.0;
    satrec['xi3'] = 0.0;
    satrec['xl2'] = 0.0;
    satrec['xl3'] = 0.0;
    satrec['xl4'] = 0.0;
    satrec['xlamo'] = 0.0;
    satrec['zmol'] = 0.0;
    satrec['zmos'] = 0.0;
    satrec['atime'] = 0.0;
    satrec['xli'] = 0.0;
    satrec['xni'] = 0.0;
    #     // sgp4fix - note the following iables are also passed directly via satrec['
    #   // it is possible to streamline the sgp4init call by deleting the "x"
    #   // iables'], but the user would need to set the satrec['* values first. we
    #   // include the additional assignments in case twoline2rv is not used.

    satrec['bstar'] = xbstar;
    satrec['ecco'] = xecco;
    satrec['argpo'] = xargpo;
    satrec['inclo'] = xinclo;
    satrec['mo'] = xmo;
    satrec['no'] = xno;
    satrec['nodeo'] = xnodeo;
    #                                                                                                   //  sgp4fix add opsmode

    satrec['operationmode'] = opsmode;
#                                                                                                   // ------------------------ earth constants -----------------------
#   // sgp4fix identify constants and allow alternate values

    ss = 78.0 / earthRadius + 1.0;
#                                                                                                   // sgp4fix use multiply for speed instead of pow

    qzms2ttemp = (120.0 - 78.0) / earthRadius;
    qzms2t = qzms2ttemp * qzms2ttemp * qzms2ttemp * qzms2ttemp;
    satrec['init'] = 'y';
    satrec['t'] = 0.0;
    initlOptions = {
    'satn': satn,
    'ecco': satrec['ecco'],
    'epoch': epoch,
    'inclo': satrec['inclo'],
    'no': satrec['no'],
    'method': satrec['method'],
    'opsmode': satrec['operationmode']
    };
    initlResult = initl(initlOptions);
    ao = initlResult['ao'],
    con42 = initlResult['con42'],
    cosio = initlResult['cosio'],
    cosio2 = initlResult['cosio2'],
    eccsq = initlResult['eccsq'],
    omeosq = initlResult['omeosq'],
    posq = initlResult['posq'],
    rp = initlResult['rp'],
    rteosq = initlResult['rteosq'],
    sinio = initlResult['sinio'];
    satrec['no'] = initlResult['no'];
    satrec['con41'] = initlResult['con41'];
    satrec['gsto'] = initlResult['gsto'];
    satrec['error'] = 0;
    #                                                                                                   // sgp4fix remove this check as it is unnecessary
#   // the mrt check in sgp4 handles decaying satellite cases even if the starting
#   // condition is below the surface of te earth
#   // if (rp < 1.0)
#   // {
#   //   printf("// *** satn%d epoch elts sub-orbital ***\n"'], satn);
#   //   satrec['error'] = 5;
#   // }
    if (omeosq[0] >= 0.0 or satrec['no'] >= 0.0):
        satrec['isimp'] = 0;
        if (rp[0] < 220.0 / earthRadius + 1.0):
            satrec['isimp'] = 1;


        sfour = ss;
        qzms24 = qzms2t;
        perige = (rp[0] - 1.0) * earthRadius;
#       // - for perigees below 156 km, s and qoms2t are altered -

        if (perige < 156.0):
            sfour = perige - 78.0;

            if (perige < 98.0):
                sfour = 20.0;
#         // sgp4fix use multiply for speed instead of pow


            qzms24temp = (120.0 - sfour) / earthRadius;
            qzms24 = qzms24temp * qzms24temp * qzms24temp * qzms24temp;
            sfour = sfour / earthRadius + 1.0;

        pinvsq = 1.0 / posq[0];
        tsi = 1.0 / (ao[0] - sfour);
        satrec['eta'] = ao[0] * satrec['ecco'][0] * tsi;
        etasq = satrec['eta'] * satrec['eta'];
        eeta = satrec['ecco'][0] * satrec['eta'];
        psisq = abs(1.0 - etasq);
        coef = qzms24 * math.pow(tsi, 4.0);
        coef1 = coef / math.pow(psisq, 3.5);
        cc2 = coef1 * satrec['no'] * (ao[0] * (1.0 + 1.5 * etasq + eeta * (4.0 + etasq)) + 0.375 * j2 * tsi / psisq * satrec['con41'] * (8.0 + 3.0 * etasq * (8.0 + etasq)));
        satrec['cc1'] = float(satrec['bstar'][0]) * cc2;
        cc3 = 0.0;

        if (satrec['ecco'][0] > 1.0e-4):
            cc3 = -2.0 * coef * tsi * j3oj2 * satrec['no'] * sinio / satrec['ecco'][0];


        satrec['x1mth2'] = 1.0 - cosio2[0];
        satrec['cc4'] = 2.0 * satrec['no'] * coef1 * ao[0] * omeosq[0] * (satrec['eta'] * (2.0 + 0.5 * etasq) + satrec['ecco'][0] * (0.5 + 2.0 * etasq) - j2 * tsi / (ao[0] * psisq) * (-3.0 * satrec['con41'] * (1.0 - 2.0 * eeta + etasq * (1.5 - 0.5 * eeta)) + 0.75 * satrec['x1mth2'] * (2.0 * etasq - eeta * (1.0 + etasq)) * math.cos(2.0 * satrec['argpo'][0])));
        satrec['cc5'] = 2.0 * coef1 * ao[0] * omeosq[0] * (1.0 + 2.75 * (etasq + eeta) + eeta * etasq);
        cosio4 = cosio2[0] * cosio2[0];
        temp1 = 1.5 * j2 * pinvsq * satrec['no'];
        temp2 = 0.5 * temp1 * j2 * pinvsq;
        temp3 = -0.46875 * j4 * pinvsq * pinvsq * satrec['no'];

        satrec['mdot'] = satrec['no'] + 0.5 * temp1 * rteosq[0] * satrec['con41'] + 0.0625 * temp2 * rteosq[0] * (13.0 - 78.0 * cosio2[0] + 137.0 * cosio4);
        satrec['argpdot'] = -0.5 * temp1 * con42[0] + 0.0625 * temp2 * (7.0 - 114.0 * cosio2[0] + 395.0 * cosio4) + temp3 * (3.0 - 36.0 * cosio2[0] + 49.0 * cosio4);
        xhdot1 = -temp1 * cosio[0];
        satrec['nodedot'] = xhdot1 + (0.5 * temp2 * (4.0 - 19.0 * cosio2[0]) + 2.0 * temp3 * (3.0 - 7.0 * cosio2[0])) * cosio[0];
        xpidot = satrec['argpdot'] + satrec['nodedot'];
        satrec['omgcof'] = float(satrec['bstar'][0]) * cc3 * math.cos(satrec['argpo'][0]);
        satrec['xmcof'] = 0.0;

        if (satrec['ecco'][0] > 1.0e-4):
            satrec['xmcof'] = -x2o3 * coef * float(satrec['bstar'][0]) / eeta;

        satrec['nodecf'] = 3.5 * omeosq[0] * xhdot1 * satrec['cc1'];
        satrec['t2cof'] = 1.5 * satrec['cc1'];
#         // sgp4fix for divide by zero with xinco = 180 deg

        if (abs(cosio[0] + 1.0) > 1.5e-12):
            satrec['xlcof'] = -0.25 * j3oj2 * sinio * (3.0 + 5.0 * cosio[0]) / (1.0 + cosio[0]);
        else:
            satrec['xlcof'] = -0.25 * j3oj2 * sinio * (3.0 + 5.0 * cosio[0]) / temp4;


        satrec['aycof'] = -0.5 * j3oj2 * sinio;
#         // sgp4fix use multiply for speed instead of pow
        delmotemp = 1.0 + satrec['eta'] * math.cos(satrec['mo'][0]);
        satrec['delmo'] = delmotemp * delmotemp * delmotemp;
        satrec['sinmao'] = math.sin(satrec['mo'][0]);
        satrec['x7thm1'] = 7.0 * cosio2[0] - 1.0;
#                         // --------------- deep space initialization -------------
        if (2 * pi / satrec['no'] >= 225.0):
            satrec['method'] = 'd';
            satrec['isimp'] = 1;
            tc = 0.0;
            inclm = satrec['inclo'];
            dscomOptions = {
            'epoch': epoch,
            'ep': satrec['ecco'],
            'argpp': satrec['argpo'],
            'tc': tc,
            'inclp': satrec['inclo'],
            'nodep': satrec['nodeo'],
            'np': satrec['no'],
            'e3': satrec['e3'],
            'ee2': satrec['ee2'],
            'peo': satrec['peo'],
            'pgho': satrec['pgho'],
            'pho': satrec['pho'],
            'pinco': satrec['pinco'],
            'plo': satrec['plo'],
            'se2': satrec['se2'],
            'se3': satrec['se3'],
            'sgh2': satrec['sgh2'],
            'sgh3': satrec['sgh3'],
            'sgh4': satrec['sgh4'],
            'sh2': satrec['sh2'],
            'sh3': satrec['sh3'],
            'si2': satrec['si2'],
            'si3': satrec['si3'],
            'sl2': satrec['sl2'],
            'sl3': satrec['sl3'],
            'sl4': satrec['sl4'],
            'xgh2': satrec['xgh2'],
            'xgh3': satrec['xgh3'],
            'xgh4': satrec['xgh4'],
            'xh2': satrec['xh2'],
            'xh3': satrec['xh3'],
            'xi2': satrec['xi2'],
            'xi3': satrec['xi3'],
            'xl2': satrec['xl2'],
            'xl3': satrec['xl3'],
            'xl4': satrec['xl4'],
            'zmol': satrec['zmol'],
            'zmos': satrec['zmos']
            };
            dscomResult = dscom(dscomOptions);
            satrec['e3'] = dscomResult['e3'];
            satrec['ee2'] = dscomResult['ee2'];
            satrec['peo'] = dscomResult['peo'];
            satrec['pgho'] = dscomResult['pgho'];
            satrec['pho'] = dscomResult['pho'];
            satrec['pinco'] = dscomResult['pinco'];
            satrec['plo'] = dscomResult['plo'];
            satrec['se2'] = dscomResult['se2'];
            satrec['se3'] = dscomResult['se3'];
            satrec['sgh2'] = dscomResult['sgh2'];
            satrec['sgh3'] = dscomResult['sgh3'];
            satrec['sgh4'] = dscomResult['sgh4'];
            satrec['sh2'] = dscomResult['sh2'];
            satrec['sh3'] = dscomResult['sh3'];
            satrec['si2'] = dscomResult['si2'];
            satrec['si3'] = dscomResult['si3'];
            satrec['sl2'] = dscomResult['sl2'];
            satrec['sl3'] = dscomResult['sl3'];
            satrec['sl4'] = dscomResult['sl4'];
            sinim = dscomResult['sinim'];
            cosim = dscomResult['cosim'];
            em = dscomResult['em'];
            emsq = dscomResult['emsq'];
            s1 = dscomResult['s1'];
            s2 = dscomResult['s2'];
            s3 = dscomResult['s3'];
            s4 = dscomResult['s4'];
            s5 = dscomResult['s5'];
            ss1 = dscomResult['ss1'];
            ss2 = dscomResult['ss2'];
            ss3 = dscomResult['ss3'];
            ss4 = dscomResult['ss4'];
            ss5 = dscomResult['ss5'];
            sz1 = dscomResult['sz1'];
            sz3 = dscomResult['sz3'];
            sz11 = dscomResult['sz11'];
            sz13 = dscomResult['sz13'];
            sz21 = dscomResult['sz21'];
            sz23 = dscomResult['sz23'];
            sz31 = dscomResult['sz31'];
            sz33 = dscomResult['sz33'];
            satrec['xgh2'] = dscomResult['xgh2'];
            satrec['xgh3'] = dscomResult['xgh3'];
            satrec['xgh4'] = dscomResult['xgh4'];
            satrec['xh2'] = dscomResult['xh2'];
            satrec['xh3'] = dscomResult['xh3'];
            satrec['xi2'] = dscomResult['xi2'];
            satrec['xi3'] = dscomResult['xi3'];
            satrec['xl2'] = dscomResult['xl2'];
            satrec['xl3'] = dscomResult['xl3'];
            satrec['xl4'] = dscomResult['xl4'];
            satrec['zmol'] = dscomResult['zmol'];
            satrec['zmos'] = dscomResult['zmos'];
            nm = dscomResult['nm'];
            z1 = dscomResult['z1'];
            z3 = dscomResult['z3'];
            z11 = dscomResult['z11'];
            z13 = dscomResult['z13'];
            z21 = dscomResult['z21'];
            z23 = dscomResult['z23'];
            z31 = dscomResult['z31'];
            z33 = dscomResult['z33'];
            dpperOptions = {
                'inclo': inclm,
                'init': satrec['init'],
                'ep': satrec['ecco'],
                'inclp': satrec['inclo'],
                'nodep': satrec['nodeo'],
                'argpp': satrec['argpo'],
                'mp': satrec['mo'],
                'opsmode': satrec['operationmode']
            }
            dpperResult = dpper(satrec, dpperOptions);
            satrec['ecco'] = dpperResult['ep'];
            satrec['inclo'] = dpperResult['inclp'];
            satrec['nodeo'] = dpperResult['nodep'];
            satrec['argpo'] = dpperResult['argpp'];
            satrec['mo'] = dpperResult['mp'];
            argpm = 0.0;
            nodem = 0.0;
            mm = 0.0;
            dsinitOptions = {
            'cosim': cosim,
            'emsq': emsq,
            'argpo': satrec['argpo'],
            's1': s1,
            's2': s2,
            's3': s3,
            's4': s4,
            's5': s5,
            'sinim': sinim,
            'ss1': ss1,
            'ss2': ss2,
            'ss3': ss3,
            'ss4': ss4,
            'ss5': ss5,
            'sz1': sz1,
            'sz3': sz3,
            'sz11': sz11,
            'sz13': sz13,
            'sz21': sz21,
            'sz23': sz23,
            'sz31': sz31,
            'sz33': sz33,
            't': satrec['t'],
            'tc': tc,
            'gsto': satrec['gsto'],
            'mo': satrec['mo'],
            'mdot': satrec['mdot'],
            'no': satrec['no'],
            'nodeo': satrec['nodeo'],
            'nodedot': satrec['nodedot'],
            'xpidot': xpidot,
            'z1': z1,
            'z3': z3,
            'z11': z11,
            'z13': z13,
            'z21': z21,
            'z23': z23,
            'z31': z31,
            'z33': z33,
            'ecco': satrec['ecco'],
            'eccsq': eccsq,
            'em': em,
            'argpm': argpm,
            'inclm': inclm,
            'mm': mm,
            'nm': nm,
            'nodem': nodem,
            'irez': satrec['irez'],
            'atime': satrec['atime'],
            'd2201': satrec['d2201'],
            'd2211': satrec['d2211'],
            'd3210': satrec['d3210'],
            'd3222': satrec['d3222'],
            'd4410': satrec['d4410'],
            'd4422': satrec['d4422'],
            'd5220': satrec['d5220'],
            'd5232': satrec['d5232'],
            'd5421': satrec['d5421'],
            'd5433': satrec['d5433'],
            'dedt': satrec['dedt'],
            'didt': satrec['didt'],
            'dmdt': satrec['dmdt'],
            'dnodt': satrec['dnodt'],
            'domdt': satrec['domdt'],
            'del1': satrec['del1'],
            'del2': satrec['del2'],
            'del3': satrec['del3'],
            'xfact': satrec['xfact'],
            'xlamo': satrec['xlamo'],
            'xli': satrec['xli'],
            'xni': satrec['xni']
            };
            dsinitResult = dsinit(dsinitOptions);
            satrec['irez'] = dsinitResult['irez'];
            satrec['atime'] = dsinitResult['atime'];
            satrec['d2201'] = dsinitResult['d2201'];
            satrec['d2211'] = dsinitResult['d2211'];
            satrec['d3210'] = dsinitResult['d3210'];
            satrec['d3222'] = dsinitResult['d3222'];
            satrec['d4410'] = dsinitResult['d4410'];
            satrec['d4422'] = dsinitResult['d4422'];
            satrec['d5220'] = dsinitResult['d5220'];
            satrec['d5232'] = dsinitResult['d5232'];
            satrec['d5421'] = dsinitResult['d5421'];
            satrec['d5433'] = dsinitResult['d5433'];
            satrec['dedt'] = dsinitResult['dedt'];
            satrec['didt'] = dsinitResult['didt'];
            satrec['dmdt'] = dsinitResult['dmdt'];
            satrec['dnodt'] = dsinitResult['dnodt'];
            satrec['domdt'] = dsinitResult['domdt'];
            satrec['del1'] = dsinitResult['del1'];
            satrec['del2'] = dsinitResult['del2'];
            satrec['del3'] = dsinitResult['del3'];
            satrec['xfact'] = dsinitResult['xfact'];
            satrec['xlamo'] = dsinitResult['xlamo'];
            satrec['xli'] = dsinitResult['xli'];
            satrec['xni'] = dsinitResult['xni'];

#       // ----------- set iables if not deep space -----------


        if (satrec['isimp'] != 1):
#             print('it is satrec[isimp]')
            cc1sq = satrec['cc1'] * satrec['cc1'];
            satrec['d2'] = 4.0 * ao[0] * tsi * cc1sq;
            temp = satrec['d2'] * tsi * satrec['cc1'] / 3.0;
            satrec['d3'] = (17.0 * ao[0] + sfour) * temp;
            satrec['d4'] = 0.5 * temp * ao[0] * tsi * (221.0 * ao[0] + 31.0 * sfour) * satrec['cc1'];
            satrec['t3cof'] = satrec['d2'] + 2.0 * cc1sq;
            satrec['t4cof'] = 0.25 * (3.0 * satrec['d3'] + satrec['cc1'] * (12.0 * satrec['d2'] + 10.0 * cc1sq));
            satrec['t5cof'] = 0.2 * (3.0 * satrec['d4'] + 12.0 * satrec['cc1'] * satrec['d3'] + 6.0 * satrec['d2'] * satrec['d2'] + 15.0 * cc1sq * (2.0 * satrec['d2'] + cc1sq));
#     /* finally propogate to zero epoch to initialize all others. */
#     // sgp4fix take out check to let satellites process until they are actually below earth surface
#     // if(satrec['error'] == 0)


    sgp4(satrec, 0);
    satrec['init'] = 'n';
#   /* eslint-enable no-param-reassign */


# In[12]:



def initl(options):
    ecco = options['ecco'],
    epoch = options['epoch'],
    inclo = options['inclo'],
    opsmode = options['opsmode'];
    no = options['no'];
# // sgp4fix use old way of finding gst
#   // ----------------------- earth constants ---------------------
#   // sgp4fix identify constants and allow alternate values
#   // ------------- calculate auxillary epoch quantities ----------

    eccsq = ecco[0][0] * ecco[0][0];
    omeosq = 1.0 - eccsq;
    rteosq = math.sqrt(omeosq);
    cosio = math.cos(inclo[0][0]);
    cosio2 = cosio * cosio;
#     // ------------------ un-kozai the mean motion -----------------
    ak = math.pow(xke / no[0], x2o3);
    d1 = 0.75 * j2 * (3.0 * cosio2 - 1.0) / (rteosq * omeosq);
    delPrime = d1 / (ak * ak);
    adel = ak * (1.0 - delPrime * delPrime - delPrime * (1.0 / 3.0 + 134.0 * delPrime * delPrime / 81.0));
    delPrime = d1 / (adel * adel);
    no = no[0] / 1.0 + delPrime;
    ao = math.pow(xke / no, x2o3);
    sinio = math.sin(inclo[0][0]);
    po = ao * omeosq;
    con42 = 1.0 - 5.0 * cosio2;
    con41 = -con42 - cosio2 - cosio2;
    ainv = 1.0 / ao;
    posq = po * po;
    rp = ao * (1.0 - ecco[0][0]);
    method = 'n';
#     //  sgp4fix modern approach to finding sidereal time



    if (opsmode == 'a'):
#     //  sgp4fix use old way of finding gst
#     //  count integer number of days from 0 jan 1970
        ts70 = epoch - 7305.0;
        ds70 = math.floor(ts70 + 1.0e-8);
        tfrac = ts70 - ds70;
        #       //  find greenwich location at epoch

        c1 = 1.72027916940703639e-2;
        thgr70 = 1.7321343856509374;
        fk5r = 5.07551419432269442e-15;
        c1p2p = c1 + twoPi;
        gsto = (thgr70 + c1 * ds70 + c1p2p * tfrac + ts70 * ts70 * fk5r) % twoPi;

        if (gsto < 0.0):
            gsto += twoPi;

    else:
        gsto = gstime(epoch[0][0] + 2433281.5);
    return {
        'no': no,
        'method': method,
        'ainv': ainv,
        'ao': ao,
        'con41': con41,
        'con42': con42,
        'cosio': cosio,
        'cosio2': cosio2,
        'eccsq': eccsq,
        'omeosq': omeosq,
        'posq': posq,
        'rp': rp,
        'rteosq': rteosq,
        'sinio': sinio,
        'gsto': gsto
        };


# In[13]:


def gstime(jdut1):
    tut1 = (jdut1 - 2451545.0) / 36525.0;
    temp = -6.2e-6 * tut1 * tut1 * tut1 + 0.093104 * tut1 * tut1 + (876600.0 * 3600 + 8640184.812866) * tut1 + 67310.54841;
#     // # sec

    temp = temp * deg2rad / 240.0 % twoPi;
#     // 360/86400 = 1/240, to deg, to rad
#     //  ------------------------ check quadrants ---------------------
    if (temp < 0.0):
        temp += twoPi;

    return temp;


# In[ ]:





# In[ ]:





# In[ ]:
