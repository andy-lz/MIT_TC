# MIT_TC

Clarification of Cases (summary of Piazza posts):
- Will there be click trading?
    Answer: No. 
- Dropbox Link?
    Answer: No. Only need Google Drive Link.
- onAckRegister: msg['case_meta']['case_length']?
    Answer: this gives case length in seconds (from the moment the case starts). 'end_time' parameter isn't supported.
- (@11) message limits per second?
    Answer: Total action limit per second is 90 (orders, cancels). Callbacks intended to be every second, not 0.5 (at this point). Otherwise connection is closed.
- Matching engine documentation?
    Answer: Matching engine matched price-time priority. Latency mainly network latency. Every player given AWS EC2 instances. Server located on AWS in same region. Half time trip approx <1 ms.
- Quantity in AckModifyOrders?
    Answer: represents quantity remaining on order after executing. Quantity of 0 indicates that order was fulfilled completely against existing liquidity. 
- Default path to connect to Mangocore server? 
    Answer: localhost:10914. For the bot, 
                t = tt.TradersBot(host='127.0.0.1', id='trader0', password='trader0')
            should work with the default settings.
- Cash allocation for options case?
    Answer: run the test case to find out. 
- No option tickers?
    Answer: Make sure to specify the correct case file when running the mangocore binary, otherwise mangocore will run the default case which has just the AAPL, IBM, and IDX tickers. 
- Underlying for options? TMxFUT? 
    Answer: Yes, TMxFUT.
- S in vega calculation?
    Answer: the price of the forward contract.
- Time to maturity?
    Answer: every 7.5 min represents 1 month of trading.
- Can trade against other competitors on TRDRS.LIT?
    Answer: Yes.
- Websocket connection closing after hitting delta limits?
    Answer: lower position size.
- OnTrade notification?
    Answer: You get market updates (see onMarketUpdate) which contain the whole order book and last transaction price every 1 second. onTrade only provides updates for your trades.
- Closing out positions in Algo S&T?
    Answer: When closing out, the position evaluated is the position of the underlying: +1200 shares in the dark security and -1200 shares in the lit security will be 0 position. The price will be the final price in the lit market.
- How do you manage TRDRS.DARK position, if you can't move that position to the lit market?
    Answer: The position to be mindful for is the underlying TRDRS, which is the sum of the positions in the lit and dark markets (there is no position limit on TRDRS.LIT or TRDRS.DARK). Thus, you can manage your TRDRS position by trading both TRDRS.LIT and TRDRS.DARK.
- Conflicting Vega limit?
    Answer: Vega Limit is actually 9000.


$ mangocore --help
Usage of mangocore:
  -case string
    	case file to load
  -identity string
    	identity file to load
  -logf string
    	log file format
  -mprofile
    	enable cpu profiling
  -port string
    	port to use (default ":10914")
  -profile
    	enable cpu profiling
  -speedup float
    	how many times faster mangocore should run (default 1)
  -start int
    	automatically start in given seconds
  -test
    	testing mode (default true)
      
Sample setup: 
./mangocore-osx-amd64.x -case /path/to/casefile

Algo S&T timings:
- Orders filled 8 ticks after news update.
- Time between consecutive news updates is 15 ticks. 

Dark json file:
"TRDRS.DARK", "buy": false, "price": 0, "time": 355, "duration": 2, "quantity": 60000
- This line means, in the darkpool, the sell order with the quantity 60,000 shares arrive at the time 355 and keep taking the best buying offer among all LO sent by us until the selling volume reaches at 60,000 shares.

