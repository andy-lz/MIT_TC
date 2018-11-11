from tradersbot import *
# from datetime import datetime, timedelta
import numpy as np
# import pandas as pd
import math
import logging
import sys

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
multiply_constant = 0.8  # for determining final_sz

cash = 0.0
time = 0
last_news_time = 0
position_lit = 0
position_dark = 0

bids_px = None
asks_px = None
bids_sz = None
asks_sz = None
best_bid = None
best_ask = None
start_price = 0.0
last_price_lit = 0.0
last_price_dark = 0.0
open_orders = None
news_ABC = ''  # Who sent in the news?
news_sz = 0.0  # How much did they want
news_px = 0.0  # Lit price at time of news
unfulfilled_sz = 0.0
final_sz = 0.0

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


def register(msg, TradersOrder):
    # Set case information
    global case_length, position_limit, time, cash, position_lit, position_dark
    global P0_est, last_price_lit, last_price_dark, start_price
    case_length = msg['case_meta']['case_length']
    position_limit = msg['case_meta']['underlyings']['TRDRS.LIT']['limit']
    time = msg['elapsed_time']
    cash = msg['trader_state']['cash']['USD']
    position_lit = msg['trader_state']['positions']['TRDRS.LIT']
    position_dark = msg['trader_state']['positions']['TRDRS.DARK']
    P0_est = last_price_lit = start_price = msg['case_meta']['securities']['TRDRS.LIT']['starting_price']
    last_price_dark = msg['case_meta']['securities']['TRDRS.DARK']['starting_price']
    log_out('R')


def update_market(msg, TradersOrder):
    global last_price_lit, last_price_dark, last_news_time, time
    global asks_px, asks_sz, bids_px, bids_sz, best_ask, best_bid, P0_est
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
            asks_px = ask_book[0][ask_sort_order][:num_asks].astype(float)
            asks_sz = ask_book[1][ask_sort_order][:num_asks].astype(float)
            best_ask = (asks_px[0], asks_sz[0])
        else:
            asks_px = asks_sz = best_ask = None
        if bids_px is not None and asks_px is not None:
            P0_est = get_wap(bids_px, bids_sz, asks_px, asks_sz)
    else:
        last_price_dark = market_state['last_price']


def update_trader(msg, TradersOrder):
    global time, cash, position_lit, position_dark, open_orders, final_sz
    # time = msg['time']
    trader_state = msg['trader_state']

    cash = trader_state['cash']['USD']
    position_lit = trader_state['positions']['TRDRS.LIT']
    position_dark = trader_state['positions']['TRDRS.DARK']
    net_pos = position_lit + position_dark
    open_orders = trader_state['open_orders']
    unfulfilled_sz = final_sz - net_pos
    if case_length - time < 8 and net_pos != 0:
        clear_pos(TradersOrder, net_pos)
    elif unfulfilled_sz != 0 and (9 > time - last_news_time > 2 or 15 > time - last_news_time > 10):
        process_lit(TradersOrder)
    # clear out positions near end of case

    # log_out('T')

def clear_pos(TradersOrder, net_pos):
    cancel_all_orders(TradersOrder)
    global P0_est, securities, max_order_size, asks_sz, bids_sz
    if net_pos < 0:
        amt = min(abs(net_pos), max_order_size, asks_sz.sum())
        TradersOrder.addBuy(securities[0], amt, 1.25 * P0_est)
    else:
        amt = min(abs(net_pos), max_order_size, bids_sz.sum())
        TradersOrder.addSell(securities[0], amt, 0.75 * P0_est)


def update_trade(msg, TradersOrder):
    # print(msg)
    global unfulfilled_sz, securities, position_lit, position_dark, max_order_size, position_limit, start_price, open_orders
    global asks_sz, bids_sz
    for i in msg['trades']:
        if i['ticker'] == securities[1]:
            if i['buy']:
                position_dark -= i['quantity']
                unfulfilled_sz -= i['quantity']
            else:
                position_dark += i['quantity']
                unfulfilled_sz += i['quantity']
    if case_length - time < 8:
        net_pos = position_lit + position_dark
        if net_pos < 0:
            amt = min(abs(net_pos), max_order_size, asks_sz.sum())
            if best_ask[0] < 1.25 * P0_est:
                TradersOrder.addBuy(securities[0], amt)
        else:
            amt = min(abs(net_pos), max_order_size, bids_sz.sum())
            if best_bid[0] > 0.75 * P0_est:
                TradersOrder.addSell(securities[0], amt)
    elif unfulfilled_sz != 0:  # and msg['trades'][0]['ticker'] == securities[0]:
        process_lit(TradersOrder)
    elif 3<time - last_news_time < 12 and case_length - time > 8:
        if not open_orders:
            if asks_sz.sum() >= bids_sz.sum():
                TradersOrder.addBuy(securities[0], 
                                    min(max_order_size, int(position_limit - (position_lit+position_dark))), 
                                    0.25 * start_price)
            else:
                TradersOrder.addSell(securities[0], 
                                     min(max_order_size, int(position_limit + (position_lit+position_dark))), 
                                     2 * start_price)                


def update_order(msg, TradersOrder):
    # Update order information
    #print("ORDUP -- ", msg)
    pass


def update_news(msg, TradersOrder):
    global P0_est, time, last_news_time, news_sz, news_ABC, news_px, unfulfilled_sz
    # Update news information
    cancel_all_orders(TradersOrder)
    if news_ABC != '':
        update_probs()
    time = last_news_time = msg['news']['time']
    unfulfilled_sz = 0
    news_sz = int(msg['news']['body'])
    if msg['news']['headline'].upper().find("SELL") > -1:
        news_sz = -1*abs(news_sz)
    news_ABC = msg['news']['source'][0]
    news_px = P0_est
    news_calc()
    process_lit(TradersOrder)
    process_dark(TradersOrder)


def cancel_all_orders(order):
    global open_orders
    if open_orders:
        print("open orders", open_orders)
        for k, v in open_orders.items():
            order.addCancel(v['ticker'], v['order_id'])
    open_orders = None

def process_dark(order):
    global time, case_length, position_dark, position_lit, position_limit, securities, C, news_sz
    global news_P0, fee, P0_conf, start_price
    net_pos = position_lit + position_dark
    if (position_limit * 0.2) < (position_limit - abs(net_pos)):
        if (net_pos < 0 and news_sz >= 0) or (net_pos > 0 and news_sz <= 0):
            return
    if time * 2 < case_length:
        if news_sz <= 0:
            #weight_avg = (news_px - fee + C*news_sz - P0_conf ) * (1- (2*time/case_length)) + \
            #             (news_P0 - fee - P0_conf)*(2*time/case_length)
            order.addBuy(securities[1], int(max(1000, position_limit*0.7 - net_pos)), news_P0 - fee - (P0_conf*3))
        else:
            #weight_avg = (news_px + fee + C*news_sz + P0_conf) * (1- (2*time/case_length)) + \
            #            (news_P0 + fee + P0_conf)*(2*time/case_length)
            order.addSell(securities[1], int(max(1000, position_limit*0.7 + net_pos)), news_P0 + fee + (P0_conf*3))
    else:
        if case_length - time >20: 
            if news_sz <= 0:
                order.addBuy(securities[1], int(max(1000, position_limit*0.7 - net_pos)), news_P0 - fee - P0_conf)
            # order.addSell(securities[1], 1000, news_P0*1000)
            else:
                order.addSell(securities[1], int(max(1000, position_limit*0.7 + net_pos)), news_P0 + fee + P0_conf)
            # order.addBuy(securities[1], 1000, 0)


def process_lit(order):
    # Calculate required edge
    global position_dark, position_lit, news_P0, P0_conf, C, P0_est, fee, best_bid, best_ask, unfulfilled_sz, final_sz
    net_pos = position_dark + position_lit
    '''
    if news_P0 > 0:
        est_size = max((news_P0_adjust - (P0_conf + best_ask[0] - P0_est)/10 - fee) / 10 / C, 0)
    elif news_P0 < 0:
        est_size = min((news_P0_adjust + (P0_conf - best_bid[0] + P0_est)/10 + fee) / 10 / C, 0)
    else:
        return
    '''
    final_sz = informed_shift()
    unfulfilled_sz = final_sz - net_pos
    # print("final", final_sz)
    # print("unfulfilled", unfulfilled_sz)
    order_size = min(abs(unfulfilled_sz), max_order_size)

    if unfulfilled_sz > 0:
        buy_up(min(unfulfilled_sz, max_order_size), order)
    elif unfulfilled_sz < 0:
        sell_off(min(-1 * unfulfilled_sz, max_order_size), order)

def buy_up(sz, order):
    global unfulfilled_sz, position_lit, securities, start_price, best_ask, C, news_px, news_sz
    filled_sz = int(min(sz, asks_sz.sum() - 500))
    unfulfilled_sz -= filled_sz
    position_lit += filled_sz
    if case_length - time > 8:
        order.addBuy(securities[0], filled_sz, news_px + C*news_sz)
        order.addSell(securities[0], int(filled_sz), 1.01 * P0_est)
        order.addSell(securities[0], int(filled_sz), 1.02 * P0_est)
        order.addSell(securities[0], int(filled_sz), 1.03 * P0_est)
        order.addSell(securities[0], int(filled_sz), 1.04 * P0_est)
        order.addSell(securities[0], int(filled_sz), 1.1 * P0_est)
        order.addSell(securities[0], filled_sz, 2.5 * P0_est)

def sell_off(sz, order):
    global unfulfilled_sz, position_lit, securities, start_price, best_bid, C, news_px, news_sz
    filled_sz = int(min(abs(sz), bids_sz.sum()-500))
    unfulfilled_sz += filled_sz
    position_lit -= filled_sz
    if case_length - time > 8:
        order.addSell(securities[0], filled_sz, news_px - C*news_sz)
        order.addBuy(securities[0], int(filled_sz), 0.99 * P0_est)
        order.addBuy(securities[0], int(filled_sz), 0.98 * P0_est)
        order.addBuy(securities[0], int(filled_sz), 0.97 * P0_est)
        order.addBuy(securities[0], int(filled_sz), 0.96 * P0_est)
        order.addBuy(securities[0], int(filled_sz), 0.95 * P0_est)
        order.addBuy(securities[0], int(filled_sz), 0.94 * P0_est)
        order.addBuy(securities[0], filled_sz, 0.8 * P0_est)

def buy_up_temp(sz, order):
    global unfulfilled_sz, position_lit, securities, start_price, best_ask
    if best_ask[0] < news_px + C*news_sz:
        filled_sz = int(min(sz, asks_sz.sum()-500))
        unfulfilled_sz -= filled_sz
        position_lit += filled_sz
        order.addBuy(securities[0], filled_sz)
        # order.addSell(securities[0], filled_sz, 2 * start_price)


def sell_off_temp(sz, order):
    global unfulfilled_sz, position_lit, securities, start_price, best_bid
    if best_bid[0] > news_px - C*news_sz:
        filled_sz = int(min(abs(sz), bids_sz.sum()-500))
        unfulfilled_sz += filled_sz
        position_lit -= filled_sz
        order.addSell(securities[0], filled_sz)
        # order.addBuy(securities[0], filled_sz, 0.5 * start_price)


def informed_shift():
    # builds size based on confidence interval, fees, best_ask or best_bid
    global news_P0, fee, P0_conf, position_limit, cash, P0_est, multiply_constant
    net_P0 = news_P0
    if news_P0 > P0_est:
        if best_ask is not None:
            market_spread = best_ask[0] - P0_est
            net_P0 -= market_spread
        if net_P0 <= P0_est:
            return 0
    else:
        if best_bid is not None:
            market_spread = P0_est - best_bid[0]
            net_P0 += market_spread
        if net_P0 >= P0_est:
            return 0
    chg_P0 = net_P0 - P0_est
    abs_P0 = abs(chg_P0)
    edge = abs_P0 - fee - P0_conf
    if edge < 0:
        final_sz = 0
    else:
        final_sz = int(max(0, multiply_constant * edge * position_limit / abs_P0))
    # adjust based on buy or sell
    if chg_P0 < 0:
        final_sz *= -1
    return final_sz

    '''
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
    '''
 
 
def news_calc():
    global A_expect, B_expect, C_expect, news_P0, P0_conf, dark_orders, expectations, news_sz
    if news_ABC == 'A':
        dark_orders[0] += news_sz
    elif news_ABC == 'B':
        dark_orders[1] += news_sz
    else:
        dark_orders[2] += news_sz
    news_P0 = start_price + (mean_discrete(expectations, A_expect, dark_orders[0]) +
                             mean_discrete(expectations, B_expect, dark_orders[1]) +
                             mean_discrete(expectations, C_expect, dark_orders[2])) / 3
    P0_conf = (sd_discrete(expectations, A_expect, dark_orders[0]) +
               sd_discrete(expectations, B_expect, dark_orders[1]) +
               sd_discrete(expectations, C_expect, dark_orders[2])) / 3


def update_probs():
    global A_expect, B_expect, C_expect, A_total, B_total, C_total
    global news_px, P0_est, news_P0, expectations, C, news_sz
    chg_px = (P0_est - news_px)
    min_diff = abs(chg_px)
    idx = 0
    for e, i in enumerate(expectations):
        if abs(chg_px - (i * news_sz)) < min_diff:
            min_diff = abs(chg_px - i)
            idx = e
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
 
 
def log_out(type_notif, *args, **kwargs):
    global case_length, position_limit, time, cash, position_lit, position_dark, last_price_lit, last_price_dark, P0_est
    global asks_px, asks_sz, best_ask, bids_px, bids_sz, best_bid, open_orders
    global last_news_time, news_ABC, news_px, news_P0, P0_conf
    global A_expect, B_expect, C_expect, A_total, B_total, C_total
 
    if type_notif == 'R':
        logging.log(msg="Register: case_length={}, position_limit={}, time={}, cash={}, positions={},{}. last_prices={},{}, P0_est={}"
                    .format(case_length, position_limit, time, cash, position_lit, position_dark, last_price_lit,
                            last_price_dark, P0_est), level=30)
    elif type_notif == 'M':
        return
        logging.log(msg="Market: time={}, net_pos = {}, last_prices={},{}, asks = {},{},{},  bids={},{},{}, wap={}"
                    .format(time, position_lit + position_dark, last_price_lit, last_price_dark,
                            asks_px, asks_sz, best_ask, bids_px, bids_sz, best_bid, P0_est), level=30)
    elif type_notif == 'N':
        logging.log(msg="News: time={}, last_news_time={}, news_px={}, news_sz={}, news_ABC={}, news_P0={}, P0_conf={}"
                    .format(time, last_news_time, news_px, news_sz, news_ABC, news_P0, P0_conf), level=30)
        logging.log(msg="Probs: A={},{}; B={},{}; C={},{}"
                    .format(A_expect, A_total, B_expect, B_total, C_expect, C_total), level=30)
        logging.log(msg="Positions: Lit={}, Dark={}".format(position_lit, position_dark), level=30)
 
    elif type_notif == 'P':
        logging.log(msg="Updating Probabilities: current_P0={}; old_P0={}; news_P0={}; max_move = {}; chg_px={}; idx ={}; min_diff={}"
                    .format(P0_est, news_px, news_P0, expectations[2]*news_sz, args[0], args[1], args[2]), level=30)
 
    elif type_notif == 'T':
        logging.log(msg="Trader: net_pos={}, cash={}, open_orders={}"
                    .format(position_lit + position_dark, cash, open_orders), level=30)
        logging.log(msg="Positions: Lit={}, Dark={}, Net Value={}".format(position_lit, position_dark, (position_lit + position_dark)*P0_est + cash), level=30)
 
 
t = TradersBot(host=sys.argv[1], id=sys.argv[2], password=sys.argv[3])
 
t.onAckRegister = register
t.onMarketUpdate = update_market
t.onTraderUpdate = update_trader
t.onTrade = update_trade
t.onAckModifyOrders = update_order
t.onNews = update_news
 
t.run()
