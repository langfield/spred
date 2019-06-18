package main

import (
	"fmt"
	"github.com/gocolly/colly"
	"time"
    "flag"
    "strconv"
)

type item struct {
	text string
    likes int
    retweets int
    replies int
    datetime string
}

func main() {
    // set up command line arguments
    hashtagPtr := flag.String("h", "bitcoin", "hashtag being scraped")
    tweetPagePtr := flag.Int("c", 1, "number of pages to scrape")
    flag.Parse()

	tweets := []item{}
	var minTweetID string
	// Instantiate default collector
	c := colly.NewCollector(
		// Visit only domains: twitter.com
		colly.AllowedDomains("twitter.com"),
		colly.Async(true),
	)

	// On every p element which has .content attribute call callback
	// This class is unique to the div that holds all information about a tweet
	c.OnHTML(".content", func(e *colly.HTMLElement) {
		temp := item{}
		temp.text = e.ChildText("p")
        tweetData := e.ChildAttrs("span[class=\"ProfileTweet-actionCount\"]", "data-tweet-stat-count")
        temp.replies, _ = strconv.Atoi(tweetData[0])
        temp.retweets, _ = strconv.Atoi(tweetData[1])
        temp.likes, _ = strconv.Atoi(tweetData[2])
        temp.datetime = e.ChildAttr("span[data-long-form=\"true\"]", "data-time")

        tweets = append(tweets, temp)
		fmt.Println(temp.text)
        fmt.Println(tweetData)
        fmt.Println(temp.datetime)
	})

	c.OnHTML(".stream-container", func(e *colly.HTMLElement) {
		minTweetID = e.Attr("data-min-position")
		minTweetID = minTweetID[:len(minTweetID)-1]
	})

	// Set max Parallelism and introduce a Random Delay
	c.Limit(&colly.LimitRule{
		Parallelism: 2,
		RandomDelay: 5 * time.Second,
	})

	// Crawl tweets
	btc := "https://twitter.com/search?f=tweets&vertical=news&q=" +
            *hashtagPtr +
            "&src=typd&lang=en"
	c.Visit(btc)
	c.Wait()

	for i := 0; i < *tweetPagePtr - 1; i++ {
		btc := "https://twitter.com/search?f=tweets&vertical=news&q=" +
                *hashtagPtr + "&src=typd&include_available_features=1&" +
                "include_entities=1&max_position=" + minTweetID +
                "&reset_error_state=false"
		c.Visit(btc)
		c.Wait()
	}

	// print results
	fmt.Printf("Number of tweets: %d\n", len(tweets))
}
