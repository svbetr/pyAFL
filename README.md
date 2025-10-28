# AFL lifeline data

I've scraped data from afltables.com to get data on matches and player stats

## Things to improve

- I'm definitely losing data on all players who don't have a unique name in AFL history (e.g. Charlie Cameron). My code searches for the name but for duplicated players it takes the first player with that name who played- so, for example, we have players from 1897 in there. Definitely need to at least fix Charlie Cameron (he's in ~16k legs!)
