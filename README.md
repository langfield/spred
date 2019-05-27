# spred
Stock prediction via blind internet trust

##	Web scraper
* Go
    * [Go-Colly](http://go-colly.org) might be a useful library
*	media
    *	twitter
    *	reddit
    *	articles/article frequency?
*	Target market(s)
    *	crypto
    *	~~penny stocks~~
    *	~~large stock exchanges~~
    *	volatility in crypto/penny stocks might make prediction harder but also more fun
        * penny cryptos/ERC-20 tokens for maximum fun
*	Could run a scraper on my core 2 duo server 24/7
    *	how much data do we need?
    *	how much time does it take to get enough data?
##	Prediction Model
*	Python3. ~~(Gorgonia is a deep learning package for Go if we wanted to go 100% in this direction)~~
*	type of model
*	figure out how to train/what data to use with historical data/social media dumps?
    *	how big of a time window best predicts
         * 50-Day simple moving average (SMA) and 200-Day SMA to find [golden cross](https://www.investopedia.com/terms/g/goldencross.asp) and [death cross](https://www.investopedia.com/terms/d/deathcross.asp) patterns. Could a similar comparison be used with media data?
*	what features should be looked at?
    *	general mood
    *	mood per stock
    *	volume of data
    * [Market Sentiment](https://www.investopedia.com/terms/m/marketsentiment.asp)
*	training
    *	I have an RTX 2070, an R9 380X, and might be able to get ahold of a GTX 970 or two
##	Transactions
*	Go
*	figure out what platform we want to trade on
    *	depends on target market(s)
*	look into APIs
    * robinhood, binance, ...

## Additional Notes

*   I think it would be unwise to `reinvent the wheel' as it were as far as using Golang over Python3 for model development. 
*   Your note about the time window size is perhaps one of the most interesting problems in all of this. I've actually thought quite a bit about this intermittently over the years. We should talk more next week. 
*   As far as training, I have access to research servers at no cost. We have several machines with dual 1080s, but to be honest I don't think the training will necessitate them. 
*   I think crypto is most interesting because of relatively low commissions and higher probabilities of transparent and well-designed APIs. Kraken in particular is an exchange I think we should consider carefully. There are many other fair candidates as well. 
*   The data question is also very interesting. There are several sources of minute and tick granularity data which are open source. The problem is that they flush what's available to download every 48 hours or so. We'd have to write a simple little script to continuously download and stockpile fine-granularity training data, depending on our proposed time-window solution. 
*   My proposal for a first step is to pick a security-flavor as well as an exchange and learn how to make simple, automated API calls. 
