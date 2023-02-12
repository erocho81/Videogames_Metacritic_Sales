--- SENTENCIAS SQL PARA DATASETS DE VIDEOJUEGOS OBTENIDOS DE KAGGLE
--dbo.Metacritic incluye las puntuaciones de videojuegos desde 1995 a  2020-2021 aprox. Incluye tanto las puntuaciones oficiales (prensa),
-- como las puntuaciones de usuarios, su fecha de salida y consola/plataforma.

--dbo.Metacritic includes videogames ratings from 1995 to 2020-2021 aprox. It includes official reviews (press), user reviews, date of release
--and platform
-----
--dbo.vgsales incluye las ventas de videojuegos desde 1980 a 2015 en millones de unidades (incluye datos de 2016, 2017, 2020 pero no actualizados).
--Incluye campos como consola/plataforma, año, genero, empresas publishers, ventas por NA, EU, JP, Otras y globales.

--dbo.vgsales includes videogames sales from 1980 to 2015 in million units (includes some info from 2016,2017,2020 but now completely updated).
--We can find the fields platform, year of release, genre, publishers, global sales and by region.

--- Primero vamos a hacer una selección de los 10 mejores juegos según Metacritic, con las puntuaciones oficiales de prensa:
--- Let's show first a selection of the top 10 games according to the Metacritic scores, which is feeded from official press ratings:

SELECT TOP 10
	name,
	meta_score

FROM dbo.Metacritic

ORDER BY meta_score DESC



---Ahora la misma selección de los 10 mejores pero según los usuarios de Metacritic:
--- Now let's check the same selection, but in this case according to the user ratings:

SELECT 
	top 10
	name,
	user_review

FROM dbo.Metacritic

ORDER BY user_review DESC



--- Vamos a crear una nueva columna "release_year" a partir de release_date de Metacritic para facilitar el código en posteriores sentencias:
--- We are going to create a new column "release_year" from release_date of the Metacritic database to be able to make simpler code in the following queries: 

ALTER TABLE dbo.Metacritic ADD release_year AS RIGHT (release_date, 4) PERSISTED



---Revisamos el estado de la bbdd metacritic con la nueva columna:
---We check now if the column has been created:

SELECT TOP 100 
		name,
		platform,
		release_date,
		release_year,
		summary,
		meta_score,
		user_review

FROM dbo.Metacritic



-- Vamos a utilizar un COUNT para revisar la cantidad de juegos lanzados por año:
-- Here we are using a COUNT to check the total quantity of games release by year:

SELECT 
	release_year,
	COUNT (name) AS Num_games_year

FROM dbo.Metacritic

GROUP BY release_year

ORDER BY release_year ASC



-- Ahora el valor medio de puntuaciones por año según puntuaciones oficiales y de usuarios:
-- Here is the average rating values for the meta_score and user_review per year:

SELECT 
	release_year,
	ROUND (AVG(meta_score),2) AS Average_Meta_Score,
	ROUND (AVG(user_review),2) AS Average_User

FROM dbo.Metacritic

GROUP BY release_year

ORDER BY release_year ASC



--- Vamos a revisar la información de la bbdd vgsales:
--- Let's check some information for the vgsales db:
exec sp_help 'dbo.vgsales'



--- Revisaremos las ventas globales por año en vgsales:
---Now we are checking the global sales per year in the vgsales db:

SELECT 
	year,
	ROUND (SUM (Global_Sales),2) AS total_global_sales

FROM dbo.vgsales

WHERE year is not null

GROUP BY Year

order by year ASC


--- Ahora vamos a focalizarnos en las ventas para los juegos con nombre que incluyan "Zelda" en vgsales ---
--- We are going to focus on the sales of games with a name that includes the word "Zelda" for the vgsales db---

SELECT
	Year,
	Name

FROM dbo.vgsales

WHERE name like '%zelda%'

ORDER BY Year


--- Revisamos ahora los juegos con títulos que incluyen "Zelda" en dbo.Metacritic ---
--- Let's check the Zelda titles for the Metacritic db:

SELECT

	release_year,
	name

FROM dbo.Metacritic

WHERE name like '%zelda%'

ORDER BY release_year


---Y ahora revisamos la información de ventas y puntuaciones para Zelda, entre otros datos, usando un join entre Metacritic y vgsales:
--- We will show the sales and ratings for the Zelda titles, using a Join between Metacritic and vgsales:

SELECT
	vgsales.Year,
	vgsales.Name,
	vgsales.Platform,
	Metacritic.meta_score,
	vgsales.Global_Sales

FROM dbo.vgsales 

LEFT JOIN dbo.Metacritic
ON Metacritic.name = vgsales.Name

WHERE vgsales.Name like '%zelda%'

ORDER BY vgsales.Year


--- Ahora vamos a ver la media de ventas y puntuación por región:
--- Now let's check the sales and ratings per region:

SELECT 
	vgsales.Year,
	ROUND (AVG(Metacritic.meta_score),2) AS Average_Meta_Score,
	ROUND (AVG(Metacritic.user_review),2) AS Average_User,
	ROUND (SUM(vgsales.Global_Sales),2) AS World_Sales,
	ROUND (SUM(vgsales.NA_Sales),2) AS Nort_Am_Sales,
	ROUND (SUM(vgsales.EU_Sales),2) AS Europe_Sales,
	ROUND (SUM(vgsales.JP_Sales),2) AS Japan_Sales

FROM dbo.vgsales 

LEFT JOIN dbo.Metacritic
ON Metacritic.release_year = vgsales.Year

WHERE vgsales.Year is not null

GROUP BY  vgsales.Year

ORDER BY vgsales.Year



--- Aquí las ventas por plataformas (consolas):
--- Here we can see the sales per platform:

SELECT
	Platform,		
	ROUND (SUM(vgsales.Global_Sales),2) AS Total_Sales_Platform

FROM dbo.vgsales

WHERE vgsales.Platform is not null

GROUP BY  Platform

ORDER BY Total_Sales_Platform DESC



--- Ahora las ventas totales por Publisher:
--- Here we can find the total sales per Publisher:

SELECT TOP 25
	Publisher,		
	ROUND (SUM(vgsales.Global_Sales),2) AS Total_Sales_Publisher

FROM dbo.vgsales

WHERE Publisher is not null

GROUP BY Publisher

ORDER BY Total_Sales_Publisher DESC



---¿Cuales son los 3 juegos con más ventas por consola:
---Which are the top 3 selling games per platform:


WITH cte AS

	(SELECT
		Platform,
		Name,
		Global_Sales,
		RANK() OVER (PARTITION BY Platform ORDER BY Global_Sales DESC) AS vg_rk

	FROM dbo.vgsales

	WHERE Platform is not null)

SELECT
	vg_rk,
	Platform,
	Name, 
	Global_Sales

FROM cte

WHERE vg_rk <=3

ORDER BY Platform


---Estos son los Top 3 juegos por publisher que han tenido al menos 5M en ventas:
---These are the Top 3 games per publisher with at least 5M in sales:

WITH cte2 AS
	(SELECT
		Publisher,
		Name,
		Year,
		Platform,
		Global_Sales,
		RANK() OVER (PARTITION BY Publisher ORDER BY Global_Sales DESC) AS pub_rk

	FROM dbo.vgsales

	WHERE Publisher is not null
	and Global_Sales >5)

SELECT
	*

FROM cte2

WHERE pub_rk <=3

ORDER BY Publisher



--- En este caso, Top 3 Juegos puntuación oficial por género (se repiten si hay empates).
--- Now we are checking the top 3 games with highest ratings per Genre.

WITH cte5 AS
	(SELECT
		vgsales.Genre,
		vgsales.Name,
		vgsales.Year,
		vgsales.Platform,
		Metacritic.meta_score,
		RANK() OVER (PARTITION BY Genre ORDER BY meta_score DESC) AS Gnr_rnk
	 
	FROM dbo.vgsales

	INNER JOIN dbo.Metacritic
	ON Metacritic.name = vgsales.Name

	WHERE vgsales.Name is not null)

SELECT
	*

FROM cte5

WHERE Gnr_rnk <=3

ORDER BY Genre, Year



--- Y ahora puntuaciones oficiales por Genero desde el año 2000:
--- Now, the Metacritic scores per Genre since year 2000:

SELECT 
	vgsales.Genre,
	vgsales.Year,
	ROUND (AVG (Metacritic.meta_score),2) AS AVG_GEN_MTC

FROM dbo.vgsales 

INNER JOIN dbo.Metacritic
ON Metacritic.name = vgsales.Name

WHERE vgsales.Year>= '2000'

GROUP BY vgsales.Genre, vgsales.Year

ORDER BY vgsales.Genre,vgsales.Year


---A continuación usaremos CASE WHEN para agrupar los sistemas por los principales fabricantes de consolas (Nintendo, Sega, Sony,..)
--para ver las ventas de juegos por consola/platform y por zona:
---Here we are using CASE WHEN to group all platforms with their manufacturers (Nintendo, Sega, Sony,...) to check sales per platform and region:


WITH cte3 AS 

	(SELECT
		vgsales.Global_Sales,
		vgsales.NA_Sales,
		vgsales.EU_Sales,
		vgsales.JP_Sales,

		CASE 
			WHEN Platform ='NES'
			OR Platform ='SNES'
			OR Platform ='GB'
			OR Platform ='GBA'
			OR Platform ='DS'
			OR Platform ='Wii'
			OR Platform ='3DS'
			OR Platform ='N64'
			OR Platform ='GC'
			OR Platform ='WiiU'
		THEN 'NINTENDO'

		WHEN Platform ='GG'
			OR Platform ='SCD'
			OR Platform ='DC'
			OR Platform ='GEN'
			OR Platform ='SAT'
		THEN 'SEGA'

		WHEN Platform ='PS'
			OR Platform ='PS2'
			OR Platform ='PS3'
			OR Platform ='PS4'
			OR Platform ='PSP'
			OR Platform ='PSV'
		THEN 'SONY'

		WHEN Platform = 'X360'
			OR Platform ='XB'
			OR Platform ='XOne'
		THEN 'MICROSOFT'

		ELSE 'OTHERS'

		END AS System_Games

		FROM dbo.vgsales)

SELECT
	System_Games,
	ROUND (SUM(Global_Sales),2) AS World_Sales,
	ROUND (SUM(NA_Sales),2) AS Nort_Am_Sales,
	ROUND (SUM(EU_Sales),2) AS Europe_Sales,
	ROUND (SUM(JP_Sales),2) AS Japan_Sale

FROM cte3

GROUP BY System_Games

ORDER BY System_Games


--- Ejemplo de vista para las tabla Metacritic:
--- Example of a Metacritic db view:

CREATE VIEW Metacritic_View AS
	SELECT 
		name,
		platform,
		release_year,
		meta_score,
		user_review
	FROM dbo.Metacritic

	WHERE meta_score>=90

SELECT * FROM Metacritic_View

DROP VIEW Metacritic_View



---Juegos con diferencias mayores entre puntuaciones meta_score y user_review, donde mostramos mayor puntuación otorgada por meta_score que de users:
--- Here are the games with higher discrepancies between meta_score and user_review, showing the cases where meta_score is higher than user reviews:

SELECT 
	top 25
	name,
	platform,
	release_year,
	meta_score,
	user_review*10 AS user_review_adapted,
	MAX(meta_score -(user_review*10)) AS Diff_scores

FROM dbo.Metacritic

WHERE meta_score is not null
and user_review is not null

GROUP BY name, meta_score, user_review,platform,release_year

ORDER BY Diff_scores DESC

---Diferencias mayores entre puntuaciones meta_score y user_review, mostrando en este caso mayores puntuaciones de usuarios que de meta_score:
--- Here are the games with higher discrepancies between meta_score and user_review, showing the cases where user reviews is higher than user meta_score:

SELECT 
	top 25
	name,
	platform,
	release_year,
	meta_score,
	user_review*10 AS user_review_adapted,

MAX(meta_score -(user_review*10)) AS Diff_scores

FROM dbo.Metacritic

WHERE meta_score is not null
and user_review is not null

GROUP BY name, meta_score, user_review,platform, release_year
ORDER BY Diff_scores ASC




