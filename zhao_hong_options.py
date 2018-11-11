import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import re
import tradersbot as tt
import sys

t = tt.TradersBot(host=sys.argv[1], id=sys.argv[2], password=sys.argv[3])

# Keeps track of prices
SECURITIES = {}
HIST_VOL_CURVE = {}
TIME = 0
UNDERLYING_TICKER = 'TMXFUT'
TRADING_THRESHOLD = 0.05

# Supporting Black Scholes
def d1(S, K, t, r, sigma):  # see Hull, page 292
    """Calculate the d1 component of the Black-Scholes PDE.
    """
    N = norm.cdf
    sigma_squared = sigma * sigma
    numerator = np.log(S / float(K)) + (r + sigma_squared / 2.) * t
    denominator = sigma * np.sqrt(t)

    if not denominator:
        print ('')
    return numerator / denominator

def d2(S, K, t, r, sigma):  # see Hull, page 292
    """Calculate the d2 component of the Black-Scholes PDE.
    """
    return d1(S, K, t, r, sigma) - sigma * np.sqrt(t)

def black_scholes(flag, S, K, t, r, sigma):
    """Return the Black-Scholes option price implemented in
        python (for reference).
    """
    N = norm.cdf
    e_to_the_minus_rt = np.exp(-r * t)
    D1 = d1(S, K, t, r, sigma)
    D2 = d2(S, K, t, r, sigma)
    if flag == 'c':
        return S * N(D1) - K * e_to_the_minus_rt * N(D2)
    else:
        return - S * N(-D1) + K * e_to_the_minus_rt * N(-D2)

def delta(flag, F, K, t, r, sigma):
    """Returns the Black delta of an option.
    """
    N = norm.cdf

    D1 = d1(F, K, t, r, sigma)

    if flag == 'p':
        return - np.exp(-r * t) * N(-D1)
    else:
        return np.exp(-r * t) * N(D1)

def vega(flag, F, K, t, r, sigma):
    """Returns the Black vega of an option.
    """
    ONE_OVER_SQRT_TWO_PI = 0.3989422804014326779399460599343818684758586311649
    pdf = lambda x: ONE_OVER_SQRT_TWO_PI * np.exp(-.5 * x * x)
    D1 = d1(F, K, t, r, sigma)
    return F * np.exp(-r * t) * pdf(D1) * np.sqrt(t) * 0.01

def implied_volatility(price, S, K, t, r, flag):
    """Calculate the Black-Scholes implied volatility.
    """
    N = norm.cdf
    f = lambda sigma: price - black_scholes(flag, S, K, t, r, sigma)

    return brentq(
        f,
        a=1e-12,
        b=100,
        xtol=1e-15,
        rtol=1e-15,
        maxiter=1000,
        full_output=False
    )

def get_opt_dict(sec_price_dict, T):
    OPTION_DICT = {}
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

            try:
                iv = implied_volatility(float(price),
                                        float(und_price),
                                        float(strike),
                                        float(ttm),
                                        0.0,
                                        opt_type)
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
    global TIME

    for k in CURR_VOL_CURVE.keys():
        if k in HIST_VOL_CURVE.keys():
            HIST_VOL_CURVE[k].append(CURR_VOL_CURVE[k])
        else:
            HIST_VOL_CURVE[k] = [CURR_VOL_CURVE[k]]

    lookback = min(max(TIME - 1, 1), 8)

    ema_vol_curve = {}

    for strike in HIST_VOL_CURVE.keys():
        data = pd.Series(HIST_VOL_CURVE[strike])
        ema_vol_curve[strike] = list(data.ewm(span=lookback, adjust=False).mean())[-1]

    x = list(ema_vol_curve.keys())
    y = list(ema_vol_curve.values())

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

    return delta_dict, vega_dict

def update_greeks(pos_dict, OPTION_DICT):
    delta_dict, vega_dict = get_greeks(OPTION_DICT)
    greeks_exposure_dict = {}
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
    if abs(total_delta) > 600:
        target_pos[UNDERLYING_TICKER] = hedge_delta(target_greeks)
    else:
        target_pos[UNDERLYING_TICKER] = 0
    return target_pos


def execute_trade(VOL_CURVE, VOL_SPLINE, curr_pos, OPTION_DICT):
    target_pos = {}
    for strike in range(80, 120):
        perc_diff = (VOL_CURVE[strike] - VOL_SPLINE[strike])/VOL_CURVE[strike]
        if perc_diff > TRADING_THRESHOLD:
            security = "T" + str(strike) + "C"
            qty = 100.0 * perc_diff
            target_pos[security] = - qty
            '''
            security = "T" + str(strike) + "P"
            qty = 50.0 * perc_diff
            # sell qty
            target_pos[security] = - qty
            '''
        elif perc_diff < - TRADING_THRESHOLD:
            security = "T" + str(strike) + "C"
            qty = 100.0 * perc_diff
            target_pos[security] = qty
            '''
            security = "T" + str(strike) + "P"
            qty = 50.0 * perc_diff
            target_pos[security] = qty
            '''
        else:
            security = "T" + str(strike) + "C"
            target_pos[security] = 0
            '''
            security = "T" + str(strike) + "P"
            target_pos[security] = 0
            '''

    adj_target_pos = check_target_trade(curr_pos, target_pos, OPTION_DICT)
    submit_orders = {k: round(adj_target_pos.get(k, 0) - curr_pos.get(k, 0)) for k in set(adj_target_pos) | set(curr_pos)}
    submit_orders = {x: y for x, y in submit_orders.items() if abs(y) > 0.05}

    if len(submit_orders.keys()) > 80:
        cutoff = sorted([abs(x) for x in submit_orders.values()])[-80]
        submit_orders = {x: y for x, y in submit_orders.items() if abs(y) > cutoff}

    return submit_orders

# Initializes the prices
def ack_register_method(msg, order):
    global SECURITIES
    security_dict = msg['case_meta']['securities']
    for security in security_dict.keys():
        if not(security_dict[security]['tradeable']):
            continue
        SECURITIES[security] = security_dict[security]['starting_price']

# Updates latest price
def market_update_method(msg, order):
    global SECURITIES
    global TIME
    TIME = msg['elapsed_time']
    SECURITIES[msg['market_state']['ticker']] = msg['market_state']['last_price']

def get_order(curr_pos):
    global SECURITIES
    global TIME
    global HIST_VOL_CURVE
    OPTION_DICT = get_opt_dict(SECURITIES, TIME)
    CURR_VOL_CURVE = get_vol_curve(OPTION_DICT)
    SMOOTHED_VOL_CURVE, HIST_VOL_CURVE = get_smoothed(CURR_VOL_CURVE, HIST_VOL_CURVE)
    orders = execute_trade(CURR_VOL_CURVE, SMOOTHED_VOL_CURVE, curr_pos, OPTION_DICT)
    return orders

def cancel_all_orders(order, open_orders):
    if open_orders is not None:
        for k, v in open_orders.items():
            order.addCancel(v['ticker'], int(k))

def trader_update_method(msg, order):
    global SECURITIES
    print(TIME)
    positions = msg['trader_state']['positions']
    print("positions:", positions)
    open_orders = msg['trader_state']['open_orders']
    orders = get_order(positions)
    print("orders", orders)

    #checking message limits
    if (len(open_orders.items()) + len(orders.keys())) < 85:
        cancel_all_orders(order, open_orders)
        for security in orders.keys():
            quant = int(orders[security])
            if quant > 0:
                order.addBuy(security, quantity=quant, price=SECURITIES[security])
            elif quant < 0:
                order.addSell(security, quantity=abs(quant), price=SECURITIES[security])
    elif (len(open_orders.items()) + len(orders.keys())) >= 85 and len(orders.keys()) <= 80:
        for security in orders.keys():
            quant = int(orders[security])
            if quant > 0:
                order.addBuy(security, quantity=quant, price=SECURITIES[security])
            elif quant < 0:
                order.addSell(security, quantity=abs(quant), price=SECURITIES[security])

t.onAckRegister = ack_register_method
t.onMarketUpdate = market_update_method
t.onTraderUpdate = trader_update_method
t.run()