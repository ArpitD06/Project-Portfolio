use olympics_data;
CREATE TABLE OLYMPICS_HISTORY (
id INT,
name VARCHAR(300),
sex  VARCHAR(300),
team VARCHAR(300),
noc VARCHAR(300),
games VARCHAR(300),
year INT,
season VARCHAR(300),
city VARCHAR(300),
sport VARCHAR(300),
event VARCHAR(300),
medal VARCHAR(300)
);
CREATE TABLE OLYMPICS_HISTORY_NOC_REGIONS (
noc VARCHAR(100),
region VARCHAR(100),
notes VARCHAR(500)
);
select * from OLYMPICS_HISTORY;
select * from OLYMPICS_HISTORY_NOC_REGIONS;

DROP TABLE IF EXISTS olympics_history;

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/noc_regions.csv'
INTO TABLE olympics_history_noc_regions
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/athlete_events.csv'
INTO TABLE olympics_history
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


select * from olympics_history;
select * from olympics_history_noc_regions;

-- 1. How many olympics games have been held?
SELECT COUNT(DISTINCT games) AS total_olympic_games
FROM olympics_history;

-- 2. List down all Olympics games held so far.
SELECT DISTINCT games
FROM olympics_history
ORDER BY games;

-- 3. Mention the total no of nations who participated in each olympics game?
select games , count(distinct noc) as total_nations
from olympics_history
group by games
order by games;

-- 4. Which year saw the highest and lowest no of countries participating in olympics?
-- Highest no. of participating Countries.
select year, count(distinct noc) as total_nations
from olympics_history
group by year
order by total_nations desc limit 1;
-- Lowest no. of participating Countries.
select year, count(distinct noc) as total_nations
from olympics_history
group by year
order by total_nations asc limit 1;

-- 5. Which nation has participated in all of the olympic games?
select noc
from olympics_history
group by noc
having count(distinct games) = (select count(distinct games) from olympics_history);

-- 6. Identify the sport which was played in all summer olympics.
SELECT sport
FROM olympics_history
WHERE season = 'Summer'
GROUP BY sport
HAVING COUNT(DISTINCT games) = (SELECT COUNT(DISTINCT games) 
FROM olympics_history WHERE season = 'Summer');

-- 7. Which Sports were just played only once in the olympics?
SELECT sport
FROM olympics_history
GROUP BY sport
HAVING COUNT(DISTINCT year) = 1;

-- 8. Fetch the top 5 athletes who have won the most gold medals.
SELECT name, team, COUNT(*) AS total_gold_medals
FROM olympics_history
WHERE medal = 'Gold'
GROUP BY name, team
ORDER BY total_gold_medals DESC
LIMIT 5;


-- 9. Fetch the top 5 athletes who have won the most medals (gold/silver/bronze).
SELECT name, team, count(*) AS total_medals
FROM olympics_history
WHERE medal IN ('Gold', 'Silver', 'Bronze')
GROUP BY name, team
ORDER BY total_medals DESC
LIMIT 5;

-- 10. Fetch the top 5 most successful countries in olympics. Success is defined by no of medals won.
SELECT noc AS country, COUNT(*) AS total_medals
FROM olympics_history
WHERE medal IN ('Gold', 'Silver', 'Bronze')
GROUP BY noc
ORDER BY total_medals DESC
LIMIT 5;

-- 11. List down total gold, silver and broze medals won by each country.
SELECT 
    noc AS country,
    COUNT(CASE WHEN medal = 'Gold' THEN 1 END) AS gold_medals,
    COUNT(CASE WHEN medal = 'Silver' THEN 1 END) AS silver_medals,
    COUNT(CASE WHEN medal = 'Bronze' THEN 1 END) AS bronze_medals
FROM olympics_history
GROUP BY noc
ORDER BY country;

-- 12. List down total gold, silver and broze medals won by each country corresponding to each olympic games.
SELECT 
    noc AS country,
    games,
    COUNT(CASE WHEN medal = 'Gold' THEN 1 END) AS gold_medals,
    COUNT(CASE WHEN medal = 'Silver' THEN 1 END) AS silver_medals,
    COUNT(CASE WHEN medal = 'Bronze' THEN 1 END) AS bronze_medals
FROM olympics_history
GROUP BY noc, games
ORDER BY games, country;

-- 13. Identify which country won the most gold, most silver and most bronze medals in each olympic games.
WITH medal_counts AS (
    SELECT 
        noc AS country,
        games,
        COUNT(CASE WHEN medal = 'Gold' THEN 1 END) AS gold_medals,
        COUNT(CASE WHEN medal = 'Silver' THEN 1 END) AS silver_medals,
        COUNT(CASE WHEN medal = 'Bronze' THEN 1 END) AS bronze_medals
    FROM olympics_history
    GROUP BY noc, games
),
max_medals AS (
    SELECT 
        games,
        country,
        gold_medals,
        silver_medals,
        bronze_medals,
        RANK() OVER (PARTITION BY games ORDER BY gold_medals DESC) AS gold_rank,
        RANK() OVER (PARTITION BY games ORDER BY silver_medals DESC) AS silver_rank,
        RANK() OVER (PARTITION BY games ORDER BY bronze_medals DESC) AS bronze_rank
    FROM medal_counts
)

SELECT 
    games,
    MAX(CASE WHEN gold_rank = 1 THEN country END) AS most_gold_country,
    MAX(CASE WHEN silver_rank = 1 THEN country END) AS most_silver_country,
    MAX(CASE WHEN bronze_rank = 1 THEN country END) AS most_bronze_country
FROM max_medals
GROUP BY games
ORDER BY games;

-- 14. In which Sport/event, India has won highest medals.
SELECT 
    sport,
    event,
    COUNT(*) AS total_medals
FROM olympics_history
WHERE noc = 'IND'  
GROUP BY sport, event
ORDER BY total_medals DESC
LIMIT 1;

-- 15. Break down all olympic games where india won medal for Hockey and how many medals in each olympic games.
SELECT 
    year,
    games,
    COUNT(*) AS total_medals
FROM olympics_history
WHERE noc = 'IND'  -- Assuming 'IND' is the NOC code for India
  AND sport = 'Hockey'
GROUP BY year, games
ORDER BY year;

-- 16. Fetch the total no of sports played in each olympic games.
SELECT 
    games,
    COUNT(DISTINCT sport) AS total_sports
FROM olympics_history
GROUP BY games
ORDER BY games;

-- 17. Which countries have never won gold medal but have won silver/bronze medals?
SELECT noc AS country
FROM olympics_history
GROUP BY noc
HAVING 
    SUM(CASE WHEN medal = 'Gold' THEN 1 ELSE 0 END) = 0 
    AND 
    SUM(CASE WHEN medal IN ('Silver', 'Bronze') THEN 1 ELSE 0 END) > 0;  



