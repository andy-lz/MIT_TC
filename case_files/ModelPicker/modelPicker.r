#library(glmulti)
library(nlme)
library(MASS)
library(glmnet)

factorSignalsdf <- read.csv("/home/andrew/case3/factor_signals.csv")

tickerSignalsdf <- read.csv("/home/andrew/case3/ticker_signals_arma.csv")
tickerSignalsdf <- subset( tickerSignalsdf, select=-c(delete))

tickerInfodf <- read.csv("/home/andrew/case3/ticker_data.csv")
#tickerInfodf <- read.csv("subset.csv")
tickerInfodf <- subset( tickerInfodf, select=c("returns", "ticker", "timestep","pb","market_cap"))

#factors column names
#VIX,COPP,3M_R,US_TRY,BIG_IX,SMALL_IX,SENTI,TEMP,RAIN,OIL,timestep

#ticker column names
#index,industry,market_cap,pb,returns,ticker,timestep

#industries
# TECH	AGRICULTURE FINANCE	CONSUMER	OTHER

shift <- function(x, n){
	  c(x[-(seq(n))], rep(NA, n))
}


print("Starting to fit models")

for (tickerNumber in 2:2){
	print(tickerNumber)

	singleTickerInfo <- subset(tickerInfodf, ticker == toString(tickerNumber))
	singleTickerInfo <- subset(singleTickerInfo, select=-c(ticker))
	

	singleTickerSignal <- subset(tickerSignalsdf, ticker == toString(tickerNumber))
	singleTickerSignal <- subset(singleTickerSignal, select=-c(ticker))

	
	merged <- merge(singleTickerInfo, factorSignalsdf, by= c("timestep"))
	merged <- merge(merged, singleTickerSignal)

	
	#merged$returns <- shift(merged$returns, 1)

	#multiplied returns by 1000 because they are small, must divide later
	#merged$returns <- merged$returns * 1000

	#did this so that I can cross validate
	merged <- subset(merged, as.integer(timestep) < 1050)
	
	merged <- subset(merged, select = -c(timestep, index))

	merged <- na.omit(merged)
	factors <- subset(merged, select=-c(returns))
	factors <- data.matrix(factors)



	f = file()
	sink(file=f) ## silence upcoming output using anonymous file connection
	


	model <- glmnet(factors, merged$returns, dfmax = 10)
	#model <- glmulti("returns", xr=signalNames, data=merged, level=1, maxsize = 15)
	
	
	sink() # undo silencing
	close(f)
	len <-length(coef(model)[1,])
	#print(coef(model)[,len])
	
	filename <- paste("coefs/",toString(tickerNumber), ".csv", sep="")
	write.csv(coef(model)[,len], file = filename)
}


