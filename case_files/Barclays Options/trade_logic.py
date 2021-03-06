import numpy as np
import pandas as pd
from py_vollib.ref_python.black_scholes import black_scholes, implied_volatility
from py_vollib.black.greeks.analytical import delta, vega
import re

UNDERLYING_TICKER = 'TMXFUT'
TRADING_THRESHOLD = 0.05

OPTION_DICT = {}
CURR_VOL_CURVE = {}
HIST_VOL_CURVE = {}
SMOOTHED_VOL_CURVE = {}
POSITIONS = {}
DELTA_POSITIONS = {}
VEGA_POSITIONS = {}

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

            #print security
            try:
                #print (price, und_price, strike, ttm, 0, opt_type)
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

            OPTION_DICT[security] = option_char
    return OPTION_DICT

def get_vol_curve(OPTION_DICT):
    CURR_VOL_CURVE = {}
    for strike in range (80,121):
        call_opt = "T" + str(strike) + "C"
        put_opt = "T" + str(strike) + "P"
        call_iv = OPTION_DICT[call_opt]["implied_vol"]
        put_iv = OPTION_DICT[put_opt]["implied_vol"]
        CURR_VOL_CURVE[strike] = (call_iv + put_iv)/2.0
    return CURR_VOL_CURVE

def get_smoothed(CURR_VOL_CURVE, HIST_VOL_CURVE):
    result = {}
    for key in (HIST_VOL_CURVE.viewkeys() | CURR_VOL_CURVE.keys()):
        if key in HIST_VOL_CURVE:
            result.setdefault(key, []).append(HIST_VOL_CURVE[key])
        if key in CURR_VOL_CURVE:
            result.setdefault(key, []).append(CURR_VOL_CURVE[key])
    HIST_VOL_CURVE = result
    lookback = min(len(HIST_VOL_CURVE.values()[0]), 8)

    ema_vol_curve = {}
    for strike in HIST_VOL_CURVE.keys():
        data = pd.Series(HIST_VOL_CURVE[strike])
        ema_vol_curve[strike] = list(data.ewm(span=lookback, adjust=False).mean())[-1]

    x = ema_vol_curve.keys()
    y = ema_vol_curve.values()
    coef = np.polyfit(x, y, 3)
    fitted_val = np.polyval(coef, x)

    SMOOTHED_VOL_CURVE = dict(zip(x, fitted_val))
    return SMOOTHED_VOL_CURVE, HIST_VOL_CURVE

def get_greeks(OPTION_DICT):
    delta_dict = {}
    vega_dict = {}
    for security in OPTION_DICT.keys():
        flag = OPTION_DICT[security]['opt_type']
        F = OPTION_DICT[security]['und_price']
        K = OPTION_DICT[security]['strike']
        t = OPTION_DICT[security]['ttm']
        r = 0
        sigma = OPTION_DICT[security]['implied_vol']
        d1 = delta(flag, F, K, t, r, sigma)
        v1 = vega(flag, F, K, t, r, sigma)
        delta_dict[security] = d1
        vega_dict[security] = v1

    '''
    delta = re.compile(r'T(\d*)([CP])_delta$')
    vega = re.compile(r'T(\d*)([CP])_vega$')
    for security in SECURITIES:
        delta_groups = delta.search(security)
        vega_groups = vega.search(security)

        if bool(delta_groups):
            delta_dict[security] = SECURITIES[security]
        if bool(vega_groups):
            vega_dict[security] = SECURITIES[security]
    '''
    return delta_dict, vega_dict

def update_greeks(pos_dict, OPTION_DICT):
    # TODO make sure to take into account futures
    delta_dict, vega_dict = get_greeks(OPTION_DICT)
    greeks_exposure_dict = {}
    #print pos_dict
    #print delta_dict
    for security in pos_dict.keys():
        if security == UNDERLYING_TICKER:
            greeks_exposure_dict[security] = {"delta": pos_dict[security] * 1,
                                              "vega": pos_dict[security] * 0}
        else:
            greeks_exposure_dict[security] = {"delta": pos_dict[security] * delta_dict[security],
                                              "vega": pos_dict[security] * vega_dict[security]}

    return greeks_exposure_dict

def hedge_delta(greeks_exposure_dict):
    total_delta = sum([greeks_exposure_dict[x]['delta'] for x in greeks_exposure_dict.keys()])

    target_und_pos = - total_delta
    return target_und_pos

def check_target_trade(curr_pos, target_pos, OPTION_DICT):
    '''
    target_dict = {k: curr_pos.get(k, 0) + order_pos.get(k, 0) for k in set(curr_pos) | set(order_pos)}
    if UNDERLYING_TICKER not in target_dict.keys():
        target_dict[UNDERLYING_TICKER] = 0
    '''
    target_greeks = update_greeks(target_pos, OPTION_DICT)
    total_delta = sum([target_greeks[x]['delta'] for x in target_greeks.keys()])
    total_vega = sum([target_greeks[x]['vega'] for x in target_greeks.keys()])
    if abs(total_vega) > 2800:
        #puke if breaching vega limit
        target_pos = {k: 0 for k in curr_pos.keys()}
    if abs(total_delta) > 400:
        target_pos[UNDERLYING_TICKER] = hedge_delta(target_greeks)
    return target_pos


def execute_trade(VOL_CURVE, VOL_SPLINE, curr_pos, OPTION_DICT):
    orders = {}
    target_pos = {}
    for strike in range(80, 120):
        perc_diff = (VOL_CURVE[strike] - VOL_SPLINE[strike])/VOL_CURVE[strike]
        if perc_diff > TRADING_THRESHOLD:
            security = "T" + str(strike) + "C"
            qty = 100.0 * perc_diff
            # sell qty
            target_pos[security] = - qty
        elif perc_diff < - TRADING_THRESHOLD:
            security = "T" + str(strike) + "C"
            qty = 100.0 * perc_diff
            target_pos[security] = qty
        else:
            security = "T" + str(strike) + "C"
            target_pos[security] = 0

    adj_target_pos = check_target_trade(curr_pos, target_pos, OPTION_DICT)
    submit_orders = {k: round(adj_target_pos.get(k, 0) - curr_pos.get(k, 0)) for k in set(adj_target_pos) | set(curr_pos)}
    submit_orders = {x: y for x, y in submit_orders.items() if abs(y) > 0.05}
    if len(submit_orders.keys()) > 3:
        cutoff = sorted([abs(x) for x in submit_orders.values()])[-3]
        submit_orders = {x: y for x, y in submit_orders.items() if abs(y) > cutoff}

    return submit_orders
'''

T = 1
with open('C:\Users\evani\OneDrive\Desktop\MIT_TC\case_files\Barclays Options\sample_0.json') as f:
    data = json.load(f)

all_sec = data['securities'].keys()

SECURITIES = {}
for sec in all_sec:
    SECURITIES[sec] = data['securities'][sec]['pricepath']['price'][1]

curr_pos = {}

OPTION_DICT = get_opt_dict(SECURITIES, T)
CURR_VOL_CURVE = get_vol_curve(OPTION_DICT)
SMOOTHED_VOL_CURVE, HIST_VOL_CURVE = get_smoothed(CURR_VOL_CURVE, HIST_VOL_CURVE)
orders = execute_trade(CURR_VOL_CURVE, SMOOTHED_VOL_CURVE, curr_pos, OPTION_DICT)
print orders


'''