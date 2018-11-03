from py_vollib.ref_python.black_scholes import black_scholes, implied_volatility
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

import re

UNDERLYING_TICKER = 'TMXFUT'

OPTION_DICT = {}
VOL_CURVE = {}

def get_opt_dict(sec_price_dict, T):
    ttm = (1.0 - float(T)/450.0) * (1.0/12.0)
    und_price = sec_price_dict[UNDERLYING_TICKER]
    for security in sec_price_dict:
        opt_pattern = re.compile(r'T(\d*)([CP])$')
        opt_groups = opt_pattern.search(security)
        # returns true if security is an option
        if bool(opt_groups):
            (strike, opt_type) = opt_groups.groups()
            strike = int(strike)
            opt_type = str(opt_type).lower()
            price = sec_price_dict[security]

            print security
            try:
                print (price, und_price, strike, ttm, 0, opt_type)
                iv = implied_volatility.implied_volatility(price, und_price, strike, ttm, 0, opt_type)
            except:
                iv = 0


            option_char = { "opt_price": price,
                            "und_price": und_price,
                            "strike": strike,
                            "opt_type": opt_type,
                            "ttm": ttm,
                            "rf_interest": 0,
                            "dividend_rate": 0,
                            "implied_vol": iv }


            print option_char
            OPTION_DICT[security] = option_char

def get_vol_curve(OPTION_DICT, und_price):
    atm_strike = round(und_price)

    for strike in range (80,121):
        call_opt = "T" + str(strike) + "C"
        put_opt = "T" + str(strike) + "P"
        call_iv = OPTION_DICT[call_opt]["implied_vol"]
        put_iv = OPTION_DICT[put_opt]["implied_vol"]
        VOL_CURVE[strike] = (call_iv + put_iv)/2.0


def get_smoothed(VOL_CURVE):
    x = VOL_CURVE.keys()
    y = VOL_CURVE.values()
    coef = np.polyfit(x, y, 3)
    fitted_val = np.polyval(coef, x)
    smoothed_curve = dict(zip(x, fitted_val))
    return smoothed_curve

def get_greeks(SECURITIES):
    delta_dict = {}
    vega_dict = {}
    delta = re.compile(r'T(\d*)([CP])_delta$')
    vega = re.compile(r'T(\d*)([CP])_vega$')
    for security in SECURITIES:
        delta_groups = delta.search(security)
        vega_groups = vega.search(security)

        if bool(delta_groups):
            delta_dict[security] = SECURITIES[security]
        if bool(vega_groups):
            vega_dict[security] = SECURITIES[security]
    return delta_dict, vega_dict

def update_greeks(pos_dict, delta_dict, vega_dict):
    for strike in range (80, 120):
        call_opt = "T" + strike + "C"
        put_opt = "T" + strike + "P"
        pos_dict[strike] * delta_dict[]

def execute_trade(VOL_CURVE, VOL_SPLINE):
    for strike in range (80,120):
        perc_diff = (VOL_CURVE[strike] - VOL_SPLINE[strike])/VOL_CURVE[strike]
        if perc_diff > 0.05:
            print "Sell:", strike
        if perc_diff < - 0.05:
            print "Buy:", strike


import json


T = 1
with open('C:\Users\evani\OneDrive\Desktop\MIT_TC\case_files\Barclays Options\sample_0.json') as f:
    data = json.load(f)

all_sec = data['securities'].keys()

SECURITIES = {}
for sec in all_sec:
    SECURITIES[sec] = data['securities'][sec]['pricepath']['price'][1]


#print SECURITIES



get_opt_dict(SECURITIES, T)
get_vol_curve(OPTION_DICT, SECURITIES["TMXFUT"])
VOL_SPLINE = get_smoothed(VOL_CURVE)



execute_trade(VOL_CURVE, VOL_SPLINE)

bleh = pd.DataFrame(OPTION_DICT)
#bleh.to_csv("opt_dict0.csv")

#print OPTION_DICT
#print VOL_CURVE
#plt.plot(VOL_CURVE.keys(), VOL_CURVE.values())

#print VOL_CURVE
#vol_df = pd.Series(VOL_CURVE)
#vol_df.to_csv("vol_curve.csv")




