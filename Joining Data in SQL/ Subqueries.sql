-- Subquery inside where
select *
from populations
where life_expectancy > 1.15*(select avg(life_expectancy)
From populations
Where year =2015)
and year = 2015

select name,country_code, urbanarea_pop
From cities
where name in (select capital from countries)
-- Where city name in the field of capital cities
ORDER BY urbanarea_pop DESC;


-- Subquery inside select
/*
SELECT countries.name AS country, COUNT(*) AS cities_num
  FROM cities
    INNER JOIN countries
    ON countries.code = cities.country_code
GROUP BY country
ORDER BY cities_num DESC, country
LIMIT 9;
*/

SELECT countries.name AS country,
  (SELECT count(*)
   FROM cities
   WHERE countries.code = cities.country_code) AS cities_num
FROM countries
ORDER BY cities_num desc, country
LIMIT 9;


-- Subquery inside from
-- Select fields
select local_name, lang_num
  -- From countries
  from countries,
  	-- Subquery (alias as subquery)
  	(SELECT code, COUNT(*) AS lang_num
  	 FROM languages
  	 GROUP BY code) AS lang
  -- Where codes match
  where lang.code = countries.code
-- Order by descending number of languages
order by lang.lang_num desc


-- Advanced subquery
-- Select fields
select local_name, lang_num
  -- From countries
  from countries,
  	-- Subquery (alias as subquery)
  	(SELECT code, COUNT(*) AS lang_num
  	 FROM languages
  	 GROUP BY code) AS lang
  -- Where codes match
  where lang.code = countries.code
-- Order by descending number of languages
order by lang.lang_num desc

-- Select the maximum inflation rate as max_inf
SELECT MAX(inflation_rate) AS max_inf
  -- Subquery using FROM (alias as subquery)
  FROM (
      SELECT name, continent, inflation_rate
      FROM countries
      INNER JOIN economies
      USING (code)
      WHERE year = 2015) AS subquery
-- Group by continent
GROUP BY continent;


-- Select fields
SELECT name, continent, inflation_rate
  -- From countries
  FROM countries
	-- Join to economies
	INNER JOIN economies
	-- Match on code
	using (code)
  -- Where year is 2015
  WHERE year = 2015
    -- And inflation rate in subquery (alias as subquery)
    and inflation_rate in (
        SELECT MAX(inflation_rate) AS max_inf
        FROM (
             SELECT name, continent, inflation_rate
             FROM countries
             INNER JOIN economies
             using (code)
             WHERE year = 2015) AS subquery
      -- Group by continent
        GROUP BY continent);


-- Subquery Challenge

-- Select fields
SELECT code, inflation_rate, unemployment_rate
  -- From economies
  FROM economies
  -- Where year is 2015 and code is not in
  WHERE year = 2015 AND code not in
  	-- Subquery
  	(SELECT code
  	 FROM countries
  	 WHERE (gov_form = 'Constitutional Monarchy' OR gov_form LIKE '%Republic%'))
-- Order by inflation rate
ORDER BY inflation_rate;

-- Final Challenge 1
-- Select fields
SELECT DISTINCT c.name, total_investment, imports
  -- From table (with alias)
  FROM countries AS c
    -- Join with table (with alias)
    LEFT JOIN economies as e
      -- Match on code
      ON (c.code = e.code
      -- and code in Subquery
        AND c.code IN (
          SELECT code
          FROM languages AS lg
          WHERE official = 'true'
        ) )
  -- Where region and year are correct
  WHERE region = 'Central America' AND year = 2015
-- Order by field
ORDER BY name;


-- Final Challenge 2

-- Select fields
SELECT region, continent, avg(fertility_rate) AS avg_fert_rate
  -- From left table
  FROM populations AS p
    -- Join to right table
    INNER JOIN countries AS c
      -- Match on join condition
      ON p.country_code = c.code
  -- Where specific records matching some condition
  WHERE year = 2015
-- Group appropriately
GROUP BY region, continent
-- Order appropriately
ORDER BY avg_fert_rate;

-- Final Challenge 3

SELECT name, country_code, city_proper_pop, metroarea_pop,  
      -- Calculate city_perc
      city_proper_pop/metroarea_pop  * 100 AS city_perc
  -- From appropriate table
  FROM cities
  -- Where 
  WHERE name IN
    -- Subquery
    (SELECT capital
     FROM countries
     WHERE ( continent = 'Europe'
        OR continent LIKE '%America%'))
       AND metroarea_pop IS not null
-- Order appropriately
ORDER BY city_perc desc
-- Limit amount
limit 1