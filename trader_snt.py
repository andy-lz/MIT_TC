from tradersbot import *
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import math
import logging
import asyncio as async

# Initialize variables: positions, expectations, future customer orders, etc
C = 1.0/25000
position_limit = 5000.0
max_order_size = 1000
min_order_size = 0
case_length = 450
last_price = 200.0
fulfill_ticks = 8
news_ticks = 15
fee = 0.0001
epsilon = 0.0005
securities = ['TRDRS.LIT', 'TRDRS.DARK']

cash = 0.0
pnl = 0.0
time = 0
last_news_time = 0
position_lit = 0
position_dark = 0
edge_req = 0.0

bids_px = None
asks_px = None
bids_sz = None
asks_sz = None
best_bid = 0.0
best_ask = 0.0
last_price_lit = 0.0
last_price_dark = 0.0
dark_edge = 0.0
open_orders = None
news_sz = 0.0
news_ABC = ''
news_px = 0.0
unfulfilled_sz = 0.0

news_P0 = 0.0
P0_est = 200.0
P0_conf = 0.0

# indexed: 0 = uninformed, 1 = half-informed, 2 = full-informed
expectations = np.array([0, .5*C, C])
A_expect = np.array([1/3.0, 1/3.0, 1/3.0])
B_expect = np.array([1/3.0, 1/3.0, 1/3.0])
C_expect = np.array([1/3.0, 1/3.0, 1/3.0])
dark_orders = np.array([0.0, 0.0, 0.0])
A_total = 3
B_total = 3
C_total = 3
news_updated = False


def register(msg, TradersOrder):
    # Set case information
    global case_length, position_limit, time, cash, position_lit, position_dark, P0_est, last_price_lit, last_price_dark
    case_length = msg['case_meta']['case_length']
    position_limit = msg['case_meta']['underlyings']['TRDRS']['limit']
    time = msg['elapsed_time']
    cash = msg['trader_state']['cash']['USD']
    position_lit = msg['trader_state']['positions']['TRDRS.LIT']
    position_dark = msg['trader_state']['positions']['TRDRS.DARK']
    P0_est = last_price_lit = msg['case_meta']['securities']['TRDRS.LIT']['starting_price']
    last_price_dark = msg['case_meta']['securities']['TRDRS.DARK']['starting_price']
    log_out('R')


def update_market(msg, TradersOrder):
    global last_price_lit, last_price_dark, news_updated, last_news_time, time, asks_px, asks_sz, bids_px, bids_sz, best_ask, best_bid, dark_edge, P0_est
    # Update market information
    time = msg['elapsed_time']
    market_state = msg['market_state']
    if market_state['ticker'] == 'TRDRS.LIT':
        last_price_lit = market_state['last_price']
        bid_dict = market_state['bids']
        ask_dict = market_state['asks']
        if len(bid_dict) > 0:
            num_bids = min(len(bid_dict), 7)
            bid_book = np.array((list(bid_dict), list(bid_dict.values())))
            bid_sort_order = bid_book[0].argsort()[::-1]
            bids_px = bid_book[0][bid_sort_order][:num_bids].astype(float)
            bids_sz = bid_book[1][bid_sort_order][:num_bids].astype(float)
            best_bid = (bids_px[0], bids_sz[0])
        else:
            bids_px = bids_sz = best_bid = None
        if len(ask_dict) > 0:
            num_asks = min(len(ask_dict), 7)
            ask_book = np.array((list(ask_dict), list(ask_dict.values())))
            ask_sort_order = ask_book[0].argsort()
            asks_px = ask_book[0][ask_sort_order][:7].astype(float)
            asks_sz = ask_book[1][ask_sort_order][:7].astype(float)
            best_ask = (asks_px[0], asks_sz[0])
        else:
            asks_px = asks_sz = best_ask = None
        if bids_px is not None and asks_px is not None:
            P0_est = get_wap(bids_px, bids_sz, asks_px, asks_sz)
    else:
        last_price_dark = market_state['last_price']
        dark_edge = last_price_lit - last_price_dark ##TODO idk where this plays in

    log_out('M')


def update_trader(msg, TradersOrder):
    global time, cash, position_lit, position_dark, open_orders
    # time = msg['time']
    trader_state = msg['trader_state']

    cash = trader_state['cash']['USD']
    position_lit = trader_state['positions']['TRDRS.LIT']
    position_dark = trader_state['positions']['TRDRS.DARK']
    net_pos = position_lit + position_dark
    open_orders = trader_state['open_orders']
    # clear out positions near end of case
    if case_length - time < 8 and net_pos != 0:
        if abs(net_pos) > max_order_size:
            amt = max_order_size
        else:
            amt = abs(net_pos)
        if net_pos < 0:
            buy_up(amt, TradersOrder)
        else:
            sell_off(amt, TradersOrder)
    log_out('T')


def update_trade(msg, TradersOrder):
    print(msg)
    global unfulfilled_sz, securities, P0_est, news_P0, position_lit
    isBuy = msg['trades'][0]['buy']
    # manually update lit position
    for i in msg['trades']:
        if i['buy']:
            position_lit += i['quantity']
        else:
            position_lit -= i['quantity']

    if unfulfilled_sz > 0 and msg['trades'][0]['ticker'] == securities[0]:
        if (isBuy and best_ask[0] < P0_est + news_P0) or (not isBuy and best_ask[0] > P0_est + news_P0):
            process_lit(TradersOrder)


def update_order(msg, TradersOrder):
    # Update order information
    print(msg)
    pass


def update_news(msg, TradersOrder):
    print(msg)
    global P0_est, time, last_news_time, news_sz, news_ABC, news_px, news_updated, unfulfilled_sz
    # Update news information
    if news_ABC != '':
        update_probs()
    news_updated = False
    time = last_news_time = msg['news']['time']
    unfulfilled_sz = 0
    news_sz = int(msg['news']['body'])
    if msg['news']['headline'].upper().find("SELL") > -1:
        news_sz = -1*abs(news_sz)
    news_ABC = msg['news']['source'][0]
    news_px = P0_est
    cancel_all_orders(TradersOrder)
    news_calc()
    process_lit(TradersOrder)
    # halfway through the case, start trading on dark pool
    if time * 2 > case_length:
        process_dark()
    log_out('N')


def cancel_all_orders(order):
    global open_orders
    if open_orders is not None:
        for k, v in open_orders.items():
            order.addCancel(v['security'], k)  # todo fix params


def process_dark():
    pass


def process_lit(order):
    # Calculate required edge
    global position_dark, position_lit, news_P0, P0_conf, C, P0_est, fee, best_bid, best_ask, news_px, unfulfilled_sz
    net_pos = position_dark + position_lit
    news_P0_adjust = news_P0 - (P0_est - news_px)
    if news_P0 > 0:
        est_size = max((news_P0_adjust - (P0_conf + best_ask[0] - P0_est)/10 - fee) / 10 / C, 0)
    elif news_P0 < 0:
        est_size = min((news_P0_adjust + (P0_conf - best_bid[0] + P0_est)/10 + fee) / 10 / C, 0)
    else:
        return
    if unfulfilled_sz == 0:
        unfulfilled_sz = abs(est_size)
    print("est", est_size)
    print("remaining", unfulfilled_sz)
    final_sz = informed_shift(est_size)
    print("final", final_sz)
    unfulfilled_sz -= final_sz
    if final_sz > 0:
        buy_up(final_sz, order)
    else:
        sell_off(final_sz, order)


def buy_up(sz, order):
    order.addBuy(securities[0], int(sz))


def sell_off(sz, order):
    order.addSell(securities[0], int(abs(sz)))


def informed_shift(sz):
    # shifts sizing based on existing positions
    global position_dark, position_lit, position_limit, min_order_size, max_order_size
    net_pos = position_lit + position_dark
    sz_final = (position_limit - abs(net_pos))/position_limit * sz
    if abs(sz_final) < min_order_size:
        sz_final = 0
    elif abs(sz_final) > max_order_size:
        if sz_final > 0:
            sz_final = max_order_size
        else:
            sz_final = max_order_size * -1
    elif sz_final > 0:
        if sz_final > position_limit - net_pos:
            if position_limit - net_pos > min_order_size:
                sz_final = min(position_limit - net_pos, max_order_size)
            else:
                sz_final = 0
    else:
        if abs(sz_final) > net_pos + position_limit:
            if position_limit + net_pos > min_order_size:
                sz_final = max(-1 * position_limit - net_pos, -1 * max_order_size)
            else:
                sz_final = 0

    return sz_final


def news_calc():
    global A_expect, B_expect, C_expect, news_P0, P0_conf
    if news_ABC == 'A':
        probs = A_expect
    elif news_ABC == 'B':
        probs = B_expect
    else:
        probs = C_expect
    news_P0 = mean_discrete(expectations, probs, news_sz)
    P0_conf = sd_discrete(expectations, probs, news_sz)


def update_probs():
    global A_expect, B_expect, C_expect, news_px, P0_est, news_P0, expectations, A_total, B_total, C_total, C, news_sz
    chg_px = (P0_est - news_px)
    min_diff = abs(chg_px)
    idx = 0
    for e, i in enumerate(expectations):
        if abs(chg_px - (i * news_sz)) < min_diff:
            min_diff = abs(chg_px - i)
            idx = e
    log_out('P', chg_px, idx, min_diff)
    if news_ABC == 'A':
        A_expect *= A_total
        A_expect[idx] += 1
        A_total += 1
        A_expect /= A_total
    elif news_ABC == 'B':
        B_expect *= B_total
        B_expect[idx] += 1
        B_total += 1
        B_expect /= B_total
    else:
        C_expect *= C_total
        C_expect[idx] += 1
        C_total += 1
        C_expect /= C_total


def get_wap(bid_px_arr, bid_sz_arr, ask_px_arr, ask_sz_arr):
    bid_sz_sum = bid_sz_arr.sum()
    ask_sz_sum = ask_sz_arr.sum()

    bid_px_wap = np.dot(bid_px_arr, bid_sz_arr) / bid_sz_sum
    ask_px_wap = np.dot(ask_px_arr, ask_sz_arr) / ask_sz_sum
    return ((bid_px_wap * ask_sz_sum) + (ask_px_wap * bid_sz_sum)) / (bid_sz_sum + ask_sz_sum)


def mean_discrete(arr_expect, arr_prob, multiplier):
    return np.average(arr_expect * arr_prob) * multiplier


def sd_discrete(arr_expect, arr_prob, multiplier):
    mu = np.average(arr_expect * arr_prob)
    var = np.average((arr_expect - mu)**2 * arr_prob)
    return math.sqrt(var)*abs(multiplier)


def log_out(type, *args, **kwargs):
    global case_length, position_limit, time, cash, pnl, position_lit, position_dark, last_price_lit, last_price_dark, P0_est
    global asks_px, asks_sz, best_ask, bids_px, bids_sz, best_bid, open_orders
    global last_news_time, news_updated, news_ABC, news_px, news_P0, P0_conf
    global A_expect, B_expect, C_expect, A_total, B_total, C_total

    if type == 'R':
        logging.log(msg="Register: case_length={}, position_limit={}, time={}, cash={}, positions={},{}. last_prices={},{}, P0_est={}"
                    .format(case_length, position_limit, time, cash, position_lit, position_dark, last_price_lit,
                            last_price_dark, P0_est), level=30)
    elif type == 'M':
        logging.log(msg="Market: time={}, net_pos = {}, pnl={}, last_prices={},{}, asks = {},{},{},  bids={},{},{}, wap={}"
                    .format(time, position_lit + position_dark, pnl, last_price_lit, last_price_dark,
                            asks_px, asks_sz, best_ask, bids_px, bids_sz, best_bid, P0_est), level=30)
    elif type == 'N':
        logging.log(msg="News: time={}, last_news_time={}, news_updated={}, news_px={}, news_sz={}, news_ABC={}, news_P0={}, P0_conf={}"
                    .format(time, last_news_time, news_updated, news_px, news_sz, news_ABC, news_P0, P0_conf), level=30)
        logging.log(msg="Probs: A={},{}; B={},{}; C={},{}"
                    .format(A_expect, A_total, B_expect, B_total, C_expect, C_total), level=30)
        logging.log(msg="Positions: Lit={}, Dark={}".format(position_lit, position_dark), level=30)

    elif type == 'P':
        logging.log(msg="Updating Probabilities: current_P0={}; old_P0={}; news_P0={}; max_move = {}; chg_px={}; idx ={}; min_diff={}"
                    .format(P0_est, news_px, news_P0, expectations[2]*news_sz, args[0], args[1], args[2]), level=30)

    elif type == 'T':
        logging.log(msg="Trader: net_pos={}, cash={}, pnl ={}, open_orders={}"
                    .format(position_lit + position_dark, cash, pnl, open_orders), level=30)
        logging.log(msg="Positions: Lit={}, Dark={}".format(position_lit, position_dark), level=30)


t = TradersBot('127.0.0.1', 'trader0', 'trader0')

t.onAckRegister = register
t.onMarketUpdate = update_market
t.onTraderUpdate = update_trader
t.onTrade = update_trade
t.onAckModifyOrders = update_order
t.onNews = update_news

t.run()
