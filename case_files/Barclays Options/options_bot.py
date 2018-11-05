import tradersbot as tt
import trade_logic

t = tt.TradersBot(host='127.0.0.1', id='trader0', password='trader0')

# Keeps track of prices
SECURITIES = {}
HIST_VOL_CURVE = {}
TIME = 0

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
	# TODO make this take into account best bids/offers instead
	SECURITIES[msg['market_state']['ticker']] = msg['market_state']['last_price']

def get_order(curr_pos):
	global SECURITIES
	global TIME
	global HIST_VOL_CURVE
	OPTION_DICT = trade_logic.get_opt_dict(SECURITIES, TIME)
	CURR_VOL_CURVE = trade_logic.get_vol_curve(OPTION_DICT)
	SMOOTHED_VOL_CURVE, HIST_VOL_CURVE = trade_logic.get_smoothed(CURR_VOL_CURVE, HIST_VOL_CURVE)
	orders = trade_logic.execute_trade(CURR_VOL_CURVE, SMOOTHED_VOL_CURVE, curr_pos, OPTION_DICT)

	return orders

# Buys or sells in a random quantity every time it gets an update
# Buys or sells in a random quantity every time it gets an update
# You do not need to buy/sell here
def trader_update_method(msg, order):
	global SECURITIES
	positions = msg['trader_state']['positions']
	orders = get_order(positions)

	for security in orders.keys():
		quant = orders[security]
		print quant
		if quant > 0:
			quant = 1
			print ("BUY", security, quant, SECURITIES[security])
			order.addBuy(security, quantity=quant, price=SECURITIES[security])
		elif quant < 0:
			quant = 1
			print ("SELL", security, abs(quant), SECURITIES[security])
			order.addSell(security, quantity=abs(quant), price=SECURITIES[security])


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