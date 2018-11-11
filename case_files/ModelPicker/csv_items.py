CHANGERS = ("SQRT", "LOG", "SQR", "INV")
FUNSTUFF = ("STDDEV", "ARMA")
OPERATIONS = ("+", "-", "/", "*", "%")
FACTORS = ("VIX", "COPP", "3M_R", "US_TRY", "BIG_IX", "SMALL_IX", "SENTI", "TEMP", "RAIN", "OIL", "1")
TICKERS = ("industry", "market_cap", "pb", "returns")
MAX_DELAY = 4

def build_init(signals):
	# build list of factors/transformed factors
	# init w/ time delays
	for factor in FACTORS:
		if factor == "1":
			signals.append(factor)
		else:
			for delay in range(MAX_DELAY + 1):
				if delay == 0:
					pass
					#delay_str = "t"
				else: 
					delay_str = "t-{}".format(str(delay))
					str1 = factor + "_{}".format("{"+ delay_str + "}")
					signals.append(str1)
					print(str1)
	for factor in TICKERS:
		for delay in range(MAX_DELAY + 1):
			if delay != 0:
				delay_str = "t-{}".format(str(delay))
				str1 = factor + "_{}".format("{"+ delay_str + "}")
				signals.append(str1)
				print(str1)

	# add basic ops
	#build_ops(signals)

	# add changers
	temp_list = []
	for factor in signals:
		for chg in CHANGERS:
			str1 = chg + "({})".format(factor)
			temp_list.append(str1)
	for str1 in temp_list:
		signals.append(str1)
		print(str1)


def build_ops(signals):
	# do operations
	temp_list = []
	for e, factor1 in enumerate(signals):
		for i, factor2 in enumerate(signals):
			if factor1 != factor2:
				for op in OPERATIONS:
					if (not((op == "+" or op == "*") and (e > i))):
						str1 = "(" + factor1 + op + factor2 + ")"
						temp_list.append(str1)
	for str1 in temp_list:
		signals.append(str1)
		print(str1)


def build_fun(signals):
	# do funstuff
	temp_list = []
	for factor in signals:
		for fun in FUNSTUFF:
			str1 = fun + "({})".format(factor)
			temp_list.append(str1)
	for str1 in temp_list:
		signals.append(str1)
		print(str1)		


if __name__ == "__main__":
	signals = []
	build_init(signals)
	build_fun(signals)