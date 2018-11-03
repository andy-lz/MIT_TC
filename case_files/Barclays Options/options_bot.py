import tradersbot as tt
import random
import numpy as np

t = tt.TradersBot(host='127.0.0.1', id='trader0', password='trader0')

# Keeps track of prices
SECURITIES = {}

def get_wap(bid_px_arr, bid_sz_arr, ask_px_arr, ask_sz_arr):
    bid_sz_sum = bid_sz_arr.sum()
    ask_sz_sum = ask_sz_arr.sum()

    bid_px_wap = bid_px_arr.dot(bid_sz_arr) / bid_sz_sum
    ask_px_wap = ask_px_arr.dot(ask_sz_arr) / ask_sz_sum
    return ((bid_px_wap * ask_sz_sum) + (ask_px_wap * bid_sz_sum)) / (bid_sz_sum + ask_sz_sum)


# return the vwap of the two best bids/offers
def get_sec_price(msg):
	market_state = msg['market_state']
	bid_dict = market_state['bids']
	ask_dict = market_state['asks']
	bid_book = np.array((list(bid_dict), list(bid_dict.values())))
	ask_book = np.array((list(ask_dict), list(ask_dict.values())))
	bid_sort_order = bid_book[0].argsort()[::-1]
	ask_sort_order = ask_book[0].argsort()
	bids_px = bid_book[0][bid_sort_order][:2]
	bids_sz = bid_book[1][bid_sort_order][:2]
	asks_px = ask_book[0][bid_sort_order][:2]
	asks_sz = ask_book[1][ask_sort_order][:2]

	bid_sz_sum = bids_sz.sum()
	ask_sz_sum = asks_sz.sum()

	bid_px_wap = bids_px.dot(bids_sz) / bid_sz_sum
	ask_px_wap = asks_px.dot(asks_sz) / ask_sz_sum
	wap = ((bid_px_wap * ask_sz_sum) + (ask_px_wap * bid_sz_sum)) / (bid_sz_sum + ask_sz_sum)

	return wap


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
	print msg
	#time = msg['elapsed_time']
	#SECURITIES[msg['market_state']['ticker']] = msg['market_state']['last_price']
	SECURITIES[msg['market_state']['ticker']] = get_sec_price(msg)


# Buys or sells in a random quantity every time it gets an update
# You do not need to buy/sell here
def trader_update_method(msg, order):
	global SECURITIES
	#print msg
	positions = msg['trader_state']['positions']
	for security in positions.keys():
		if random.random() < 0.5:
			quant = 10*random.randint(1, 10)
			#order.addBuy(security, quantity=quant,price=SECURITIES[security])
		else:
			quant = 10*random.randint(1, 10)
			#order.addSell(security, quantity=quant,price=SECURITIES[security])

###############################################
#### You can add more of these if you want ####
###############################################

t.onAckRegister = ack_register_method
t.onMarketUpdate = market_update_method
t.onTraderUpdate = trader_update_method
#t.onTrade = trade_method
#t.onAckModifyOrders = ack_modify_orders_method
#t.onNews = news_method
t.run()