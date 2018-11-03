#import py_vollib
import re

UNDERLYING_TICKER = 'TMXFUT'

OPTION_DICT = {}
IMPLIED_VOL = {}
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

def get_vol_curve(OPTION_DICT):











