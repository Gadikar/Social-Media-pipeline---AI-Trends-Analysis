# data_noobs Project 1 Implementation

## Prerequisites 
Docker

Postgres


## Postgres setup
Create a database with name crawler_db

Use the db scripts from /reddit and /4chan and run the scripts

## Building the 4chan and reddit images

```angular2html
docker build -f Dockerfile.4chan -t 4chan .
docker build -f Dockerfile.reddit -t reddit .
```

## Running the application

Starting the application:
```angular2html
docker compose up --build
```
Stopping the application:
```angular2html
docker compose down
```

  




