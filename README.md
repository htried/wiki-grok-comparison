# What did Elon change? A comprehensive analysis of Grokipedia

This repository contains the code for "What did Elon change? A comprehensive analysis of Grokipedia".

It is structured as follows:
- The `graphics/` directory contains the graphics that were used in the paper
  - `graphics/controversial/` contains graphics for the controversial topics subset of articles
  - `graphics/mps/` contains graphics for the elected officials subset of articles
  - `graphics/overall/` contains graphics for overall comparisons of the two corpora
  - `graphics/sample/` contains graphics pertaining to the 30k article subset that we retrieved article topic and quality class metadata for
  - `graphics/t100/` contains graphics pertaining to the top 100 most cited sources on Wikipedia and Grokipedia
- The `notebooks/` directory contains the python code that was used to scrape, clean, and analyze the Wikipedia and Grokipedia corpora. The notebook names are somewhat descriptive of their contents, and markdown comments have been added throughout.
- The `results/` directory contains intermediate waypoints and final datasets for various features in the Wikipeia and Grokipedia datasets
  - `results/controversial/` contains a list of controversial article titles that are in Grokipedia
  - `results/mps/` contains a csv of elected officials in the US and UK and cited domains on Grokipedia
  - `results/overall/` contains lists of Grokipedia articles with and without CC-licensure, and pickled indices for fast large file access on the full corpora. It also contains the following subfolders
    - `results/overall/cite_sublist/`: CSVs of citations to fringe sites and Wikimedia Foundation-owned sites
    - `results/overall/domains/`: CSVs of citation counts to various sites, the composition of reliability scores, and full article title --> stemmed citation domain maps for Grokipedia and and Wikipedia
    - `results/overall/similarity/`: Parquet files of pairwise similarity statistics, top 1 embedding similarity for each chunk (unhydrated), and a structural heading length comparison file
    - `results/overall/usernames/`: List of links to Grok convos that are cited on Grokipedia, and the most cited usernames for a variety of social media sites on both Grokipedia and Wikipedia
  - `results/sample/` contains predicted topic and article quality classes as json objects from the WMF API
  - `results/t100/` contains a list of the top 100 most cited domains on Grokipedia and Wikipedia, along with their overall citation share
- The `scripts/` directory contains scripts for parallelized scraping of Grokipedia on GCP, as well as parallelized embedding of the corpora on a local GPU cluster. Both subdirectories also contain their own READMEs.
  - `scripts/boomhauer/` contains scripts for building a chunked corpus, embedding it across multiple GPUs, and calculating per chunk similarity (on GCP)
  - `scripts/gcp/` contains scripts for scraping Grokipedia in a parallelized way using transient GCP VMs
- The `supplemental_data/` directory contains various third-party data sources that we use to supplement our analysis:
  - `supplemental_data/domain_lists/` contains a list of academic journal domains from OpenAlex (not used presently)
  - `supplemental_data/news_reliability/` contains Lin et al.'s 2023 domain quality rankings
  - `supplemental_data/pageview_data/` contains CSVs of geolocated Wikipedia pageview data for the month prior to Grokipedia's release
  - `supplemental_data/perennial_sources_enwiki/` contains the raw text of the perennial sources table on enwiki at the time of this project, as well as a CSV of the full table
  - `supplemental_data/wikidata_queries/` contains the queries and results that were used to get elected officials from Wikidata at the time of this project