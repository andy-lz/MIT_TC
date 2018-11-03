import py_vollib
import re

UNDERLYING_TICKER = 'TMXFUT'

OPTION_DICT = {}
VOL_CURVE = {}

def get_opt_dict(sec_price_dict):
    und_price = sec_price_dict[UNDERLYING_TICKER]
    for security in sec_price_dict:
        opt_pattern = re.compile(r'T(\d*)([CP])')
        opt_groups = opt_pattern.search(security)
        # returns true if security is an option
        if not bool(opt_groups):
            continue
        (strike, opt_type) = opt_groups.groups()
        price = sec_price_dict[security]

        iv = py_vollib.implied_volatility(price, und_price, strike, ttm, 0, 0, opt_type)

        option_char = { "opt_price": price,
                        "und_price": und_price,
                        "strike": strike,
                        "opt_type": opt_type,
                        "ttm": ttm,
                        "rf_interest": 0,
                        "dividend_rate": 0,
                        "implied_vol": iv }

        OPTION_DICT[security] = option_char

def get_vol_curve(OPTION_DICT, und_price):
    base = 5
    atm_strike = return int(base * round(float(und_price)/base))
    for security in OPTION_DICT.keys():
        opt_type = OPTION_DICT[security]["opt_type"]
        opt_strike = OPTION_DICT[security]["strike"]
        opt_iv = OPTION_DICT[security]["implied_vol"]
        #USE OOTM puts and calls to construct vol curve
        if opt_type == "C" and opt_strike >= atm_strike:
            VOL_CURVE[str(opt_strike)] = opt_iv
        elif opt_type == "P" and opt_strike < atm_strike:
            VOL_CURVE[str(opt_strike)] = opt_iv













