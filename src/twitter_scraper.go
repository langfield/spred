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
        fmt.Println(temp.text)
        fmt.Println()
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
    c.Visit(btc)

    c.Wait()

    // print results
    fmt.Printf("Number of tweets: %d\n", len(tweets))
}
