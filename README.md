# spred
Stock prediction via blind internet trust

##	Web scraper
* Go
*	media
  *	twitter
  *	reddit
  *	articles/article frequency?
*	Target market(s)
  *	crypto
  *	penny stocks
  *	large stock exchanges
  *	volatility in crypto/penny stocks might make prediction harder but also more fun
    *	penny cryptos/ERC-20 tokens for maximum fun
*	Could run a scraper on my core 2 duo server 24/7
  *	how much data do we need?
  *	how much time does it take to get enough data?
##	Prediction Model
*	Python3? (Gorgonia is a deep learning package for Go if we wanted to go 100% in this direction)
*	type of model
*	figure out how to train/what data to use with historical data/social media dumps?
  *	how big of a time window best predicts
*	what features should be looked at?
  *	general mood
  *	mood per stock
  *	volume of data
*	training
  *	I have an RTX 2070, an R9 380X, and might be able to get ahold of a GTX 970 or two
##	Transactions
*	Go
*	figure out what platform we want to trade on
  *	depends on target market(s)
*	look into APIs
  *	robinhood, binance, ...
