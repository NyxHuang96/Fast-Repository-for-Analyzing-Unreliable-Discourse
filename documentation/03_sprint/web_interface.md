---
title: "Plan for the Interface"
author: "Tianhao Cao"
date: "2026-03-02"
---

## Overview
This is a web interface for our spam detection model.

## Features
### Top Tool Bar
- **Search**: Returns to the home page, where users can search for the full corpus text based on input keywords.
- **Corpora**: Returns to the corpora page, the content will be the corpora information.
- **Statistics**: Returns to the statistics page, the content will be the statistics of the corpora, such as the number of documents, the number of words, etc.
- **About**: Returns to the about page, the content will be the information of our project and the team members.
- **Login**: Allows users to register and log in.

### Search Page
![img preview](../../img/web/web_interface.png)

- **Search Bar**: Users can enter keywords to search for relevant documents.
- **Search annotated data only**: Users can check this box to search for annotated data only.
- **Corpora**: Users can switch the result displays between Chinese and English corpora.
- **Query Builder**: Users can build complex queries using the query builder.
- **Recent Searches**: Users can view their recent searches.
- **Search Options**: Users can select search options.
    - **Corpus Selection**: Users can select the corpus to search in.
    - **Metadata Filters (Genre, Date, Author)**: Users can filter the search results by metadata.
    - **Concordance View**: Users can view the search results in concordance view.
    - **Frequency List**: Users can view the frequency list of the search results.

**Feature Explanation**
The query builder is the main feature of the search page, where it returns the searched results in the middle of the page so that the user can easily view the results. Aside from search-related features, we also have other features such as corpora information, statistics, and about page, these pages provides a more thothrough understanding of the corpora to users. In the mean time, since we have two different corpora, Chinese and English, we need to have a way to distinguish between them, so we have a corpora selection feature that allows users to switch between the two corpora.

**Web Server Deployment**
 - Render: https://render.com/
    - A Platform as a Service provider that allows users to deploy web applications and services, based on the scale of our project, Render provides free tier for us to deploy our web application with FastAPI and Python Backend.
 - AWS: https://aws.amazon.com/
    - A cloud computing platform that provides a wide range of services, based on the scale of our project, AWS provides free tier for us to deploy our web application with FastAPI and Python Backend.
    - A good practice for us to learn how to deploy web applications and services in the cloud.

**Data Storage**
 - Amazon S3 and Local Woosh Indexing: Our dataset is not that large, so we can store the dataset in Amazon S3 and use local Woosh Indexing for the search functionality. Amazon provides 5 GB of free storage for us to use.

**Future Improvements**
 - Docker Image: We can use Docker Image to package our web application and its dependencies, so that it can be deployed in any environment.
 - IAA Match Integration: Users can annotate existing documents online and compare their annotations with the annotations of other users, return the IAA alpha scores to users.
