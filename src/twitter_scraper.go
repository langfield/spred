package main

import (
	"fmt"
	"time"
	"github.com/gocolly/colly"
)

type item struct {
	text  string
}

func main() {
	tweets := []item{}
	var minTweetID string
    // Instantiate default collector
	c := colly.NewCollector(
		// Visit only domains: old.reddit.com
		colly.AllowedDomains("twitter.com"),
		colly.Async(true),
	)

	// On every p element which has .content attribute call callback
	// This class is unique to the div that holds all information about a tweet
	c.OnHTML(".content", func(e *colly.HTMLElement) {
        temp := item{}
		temp.text = e.ChildText("p[data-aria-label-part=\"0\"]")
		tweets = append(tweets, temp)
        //fmt.Println(temp.text)
        //fmt.Println()
	})

    c.OnHTML(".stream-container", func(e *colly.HTMLElement) {
        minTweetID = e.Attr("data-min-position")
        minTweetID = minTweetID[:len(minTweetID) - 1]
        fmt.Printf("Min tweet ID: %s\n", minTweetID)
    })

	// Set max Parallelism and introduce a Random Delay
	c.Limit(&colly.LimitRule{
		Parallelism: 2,
		RandomDelay: 5 * time.Second,
	})

	// Before making a request print "Visiting ..."
	c.OnRequest(func(r *colly.Request) {
		fmt.Println("Visiting", r.URL.String())
	})

	// Crawl tweets
	btc := "https://twitter.com/search?f=tweets&vertical=news&q=bitcoin&src=typd&lang=en"
    //btc := "https://twitter.com/search?f=tweets&vertical=news&q=bitcoin&src=typd&include_available_features=1&include_entities=1&max_position=thGAVUV0VFVBaAgLfZyIyxzx8WgoC35d2bsc8fEjUAFQAlAAA%3D&reset_error_state=false"
    //btc:= "https://twitter.com/search?f=tweets&vertical=news&q=bitcoin&src=typd&include_available_features=1&include_entities=1&max_position=1139241875259482112&reset_error_state=false"
    c.Visit(btc)
    c.Wait()

    for i := 0; i < 1; i++ {
        btc := "https://twitter.com/search?f=tweets&vertical=news&q=bitcoin&src=typd&include_available_features=1&include_entities=1&max_position=" + minTweetID + "&reset_error_state=false"
        c.Visit(btc)
        c.Wait()
    }

    // print results
    fmt.Printf("Number of tweets: %d\n", len(tweets))
}
