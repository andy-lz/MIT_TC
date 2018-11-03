from tradersbot import *
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import math

#Initialize variables: positions, expectations, future customer orders, etc
C = 1.0/25000
position_limit = 5000
min_order_size = 1000
case_length = 450
last_price = 200.0
fulfill_ticks = 8
news_ticks = 15
fee = 0.0001

cash = 0
time = 0
last_news_time = 0
position_lit = 0
position_dark = 0
edge_req = 0.0

bids_px = None
asks_px = None
bids_sz = None
asks_sz = None
wap = 0.0
best_bid = 0.0
best_ask = 0.0
last_price_lit = 0.0
last_price_dark = 0.0
dark_edge = 0.0
open_orders = None
news_sz = 0.0
news_ABC = ''
news_px = 0.0

news_P0 = 0.0
P0_est = 200.0
P0_conf = 0.0

# indexed: 0 = uninformed, 1 = half-informed, 2 = full-informed
expectations = np.array([0, .5*C, C])
A_expect = np.array([1/3.0, 1/3.0, 1/3.0])
B_expect = np.array([1/3.0, 1/3.0, 1/3.0])
C_expect = np.array([1/3.0, 1/3.0, 1/3.0])
A_total = 3
B_total = 3
C_total = 3
news_updated = False


def register(msg, TradersOrder):
    #Set case information
    case_length = msg['case_meta']['case_length']
    position_limit = msg['case_meta']['underlyings']['TRDRS']['limit']
    time = msg['elapsed_time']
    cash = msg['trader_state']['cash']['USD']
    position_lit = msg['trader_state']['positions']['TRDRS.LIT']
    position_dark = msg['trader_state']['positions']['TRDRS.DARK']
    P0_est = last_price_lit = msg['case_meta']['securities']['TRDRS.LIT']['starting_price']
    last_price_dark =msg['case_meta']['securities']['TRDRS.DARK']['starting_price']

def update_market(msg, TradersOrder):
    global last_price_lit, news_updated, last_news_time
    # Update market information
    time = msg['elapsed_time']
    market_state = msg['market_state']
    if market_state['ticker'] == 'TRDRS.LIT':
        last_price_lit = market_state['last_price']
        bid_dict = market_state['bids']
        ask_dict = market_state['asks']
        bid_book = np.array((list(bid_dict), list(bid_dict.values())))
        ask_book = np.array((list(ask_dict), list(ask_dict.values())))
        bid_sort_order = bid_book[0].argsort()[::-1]
        ask_sort_order = ask_book[0].argsort()
        bids_px = bid_book[0][bid_sort_order][:7]
        bids_sz = bid_book[1][bid_sort_order][:7]
        asks_px = ask_book[0][bid_sort_order][:7]
        asks_sz = ask_book[1][ask_sort_order][:7]
        best_bid = (bids_px[0], bids_sz[0])
        best_ask = (asks_px[0], asks_sz[0])
        wap = get_wap(bids_px, bids_sz, asks_px, asks_sz)
        if (time-last_news_time) >= 10 and news_updated is False:
            update_probs()
            news_updated = True
    else:
        last_price_dark = market_state['last_price']
        dark_edge = last_price_lit - last_price_dark ##TODO idk where this plays in


def update_trader(msg, TradersOrder):
    time = msg['elapsed_time']
    trader_state = msg['trader_state']

    cash = trader_state['cash']['USD']
    position_lit = trader_state['positions']['TRDRS.LIT']
    position_dark = trader_state['positions']['TRDRS.DARK']
    open_orders = trader_state['open_orders']


def update_trade(msg, TradersOrder):
    # Update trade information
    print(msg)
    pass


def update_order(msg, TradersOrder):
    # Update order information
    print(msg)
    pass


def update_news(msg, TradersOrder):
    global P0_est
    # Update news information
    time = last_news_time = msg['news']['time']
    news_sz = int(msg['news']['body'])
    news_ABC = msg['news']['source'][0]
    news_px = P0_est
    cancel_all_orders(TradersOrder)
    news_calc()
    process(TradersOrder)
    #halfway through the case, start trading on dark pool
    if time * 2 > case_length:
        process_dark()


def cancel_all_orders(order):
    for k, v in open_orders.items():
        order.addCancel(v['security'],k) #todo fix params

def process_dark():
    pass


def process(order):
    # Calculate required edge
    global position_dark, position_lit, news_P0, C
    est_size = news_P0*2/C
    informed_shift(est_size)
    pass


def informed_shift(sz):
    global position_dark, position_lit, news_P0, C, P0_conf
    net_pos = position_lit + position_dark

    # pos_slippage = (position_limit - net_pos)*C
    # neg_slippage = -1 * (position_limit + net_pos)*C
    pass


def news_calc():
    global A_expect, B_expect, C_expect
    if news_ABC == 'A':
        probs = A_expect
    elif news_ABC == 'B':
        probs = B_expect
    else:
        probs = C_expect
    news_P0 = mean_discrete(expectations, probs, news_sz)
    P0_conf = sd_discrete(expectations, probs, news_sz)


def update_probs():
    global A_expect, B_expect, C_expect, news_px, P0_est, expectations, A_total, B_total, C_total
    min_diff = chg_px = (P0_est - news_px) / news_sz
    idx = 0
    for e,i in enumerate(expectations):
        if chg_px - i < min_diff:
            min_diff = chg_px - i
            idx = e
    if news_ABC == 'A':
        probs = A_expect
        total = A_total
    elif news_ABC == 'B':
        probs = B_expect
        total = B_total
    else:
        probs = C_expect
        total = C_total
    probs = probs * total
    probs[idx] += 1
    total += 1
    probs = probs / total


def get_wap(bid_px_arr, bid_sz_arr, ask_px_arr, ask_sz_arr):
    bid_sz_sum = bid_sz_arr.sum()
    ask_sz_sum = ask_sz_arr.sum()

    bid_px_wap = bid_px_arr.dot(bid_sz_arr) / bid_sz_sum
    ask_px_wap = ask_px_arr.dot(ask_sz_arr) / ask_sz_sum
    return ((bid_px_wap * ask_sz_sum) + (ask_px_wap * bid_sz_sum)) / (bid_sz_sum + ask_sz_sum)


def mean_discrete(arr_expect, arr_prob, multiplier):
    return np.average(arr_expect, arr_prob) * multiplier


def sd_discrete(arr_expect, arr_prob, multiplier):
    mu = mean_discrete(arr_expect, arr_prob, 1)
    var = np.average((arr_expect - mu)**2, weights=arr_prob)
    return math.sqrt(var)*abs(multiplier)


t = TradersBot('127.0.0.1', 'trader0', 'trader0')

t.onAckRegister = register
t.onMarketUpdate = update_market
t.onTraderUpdate = update_trader
t.onTrade = update_trade
t.onAckModifyOrders = update_order
t.onNews = update_news

t.run()
