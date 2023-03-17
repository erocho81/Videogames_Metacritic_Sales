#!/usr/bin/env python
# coding: utf-8

# In[1]:


#This is the data study for 2 datasets. One includes the metacritic information about user and official ratings until 2016
# and the other one includes the sales of videogames.

#Este es un estudio de datos para 2 datasets. Uno incluye información sobre las puntuaciones de usuarios y
# puntuaciones oficiales hasta 2016 aprox y el otro incluye ventas de videojuegos.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Importacion de los datasets
#Dataset import

metacritic = pd.read_excel ('E:/Análisis de datos/DATASETS/Videojuegos/all_games_python.xlsx')
                     
vgsales = pd.read_excel ('E:/Análisis de datos/DATASETS/Videojuegos/vgsales.xlsx')


# In[2]:


#Revisamos uno de los datasets
#We check one of the datasets
metacritic.head()


# In[3]:


#Revisamos uno de los datasets
#We check one of the datasets
vgsales.head()


# In[4]:


#Let's check if there are Na in some of the columns
#Queremos revisar si hay NA's en alguna de las columnas que utilizaremos
metacritic.user_review.isna().any()


# In[5]:


vgsales.Year.isna().any()


# In[6]:


#Revisamos las columnas de uno de los datasets
#We check the columns for one of the datasets
metacritic.columns


# In[7]:


#Revisamos las columnas del otro dataset.
#We also check the columns of the next dataset.
vgsales.columns


# In[8]:


#Unimos los datasets
#We join the datasets.

sales_rtg= metacritic.join (vgsales, lsuffix='name', rsuffix='Name')

#No conseguiremos todos los datos que nos gustaría, por que el Join está creado con el campo Name
# y  algunos de los títulos no van a coincidir exactamente, pero seguiremos teniendo una gran cantidad de información.

#We are not going to get a match for all the rows, since the join is done with the fields Name
# and some of them may have mismatches, however we will have a large chunk of information.


# In[9]:


sales_rtg.head()


# In[10]:


#Tendremos columnas duplicadas
#After the join we will find some column duplications
sales_rtg.columns


# In[11]:


#Quitaremos Name, Platform, release_date
#We will drop Name, Platform, release_date
sales_rtg = sales_rtg.drop(columns=['Name', 'Platform', 'release_date', 'Rank'])


# In[12]:


#Revisamos
#Let's check the new df
sales_rtg.head()


# In[13]:


#Let's check data type for the columns:
#Este es el tipo de datos para las columnas:
sales_rtg.dtypes


# In[14]:


#OJO ESTE PASO NO ES NECESARIO Y CAMBIA VALORES DE ANÁLISIS!!

#The empty values or NA are preventing us from changing data types. user_review should be int instead of float.
#First we will replace NA with 0

#sales_rtg.user_review = sales_rtg.user_review.fillna(0)

#Check later if 0 should be changed to NA again to avoid incorrect calculations


# In[15]:


#OJO ESTE PASO NO ES NECESARIO Y CAMBIA VALORES DE ANÁLISIS!!

#Now we will be able to change the column data type

#sales_rtg.user_review= sales_rtg.user_review.astype('int')


# In[16]:


#Necesitamos cambiar el tipo de dato para Year, para hacerlo primero cambiaremos los NA's por 0.
#We need to change the data type for the Year, first we will replace Na's with 0's.
sales_rtg.Year = sales_rtg.Year.fillna(0)


# In[17]:


#Ahora sí que podemos cambiar el formato
#Here we change the data type for Year:

sales_rtg.Year= sales_rtg.Year.astype('int')


# In[18]:


#Year ya aparece como int:
#Year appears as int now:
sales_rtg.head()


# In[19]:


#El tipo de datos para Year es correcto. Sin embargo no podemos cambiar los 0's a NA (o dejar vacíos los campos), 
#ya que NA es float, lo que cambiaría el tipo de dato de nuevo.

#The types are correct now. Unfortunately we can't convert the Year 0 values to NA again, because NA is float, 
#and this would change data type to float again.

sales_rtg.dtypes


# In[20]:


#Vamos a crear una columna calculada con la diferencia entre meta_score y user_reviews, 
#como hicimos en el código SQL y PowerBI

#We are going to create a calculated column with the difference between meta_score
#and user_review as we did in the SQL code and the PowerBi file

sales_rtg["reviews_diff"] = sales_rtg["meta_score"] - sales_rtg["user_review"]


# In[21]:


#También vamos a revisar los 5 juegos que están más sobrevalorados según los usuarios de metacritic,
#ordenando por el campo de diferencia de ratings de mayor a menor.
#Vemos que los resultados son similares a los que obtuvimos en SQL y Power BI

#We are also going to check the 5 top overrated games according to metacritic user reviews, by sorting
#by the new created field.
#The results are similar to the ones obtained with SQL and Power Bi.

df1 = sales_rtg.sort_values('reviews_diff',ascending = False).groupby('name').head(5)
print (df1)


# In[22]:


#Ahora haremos lo mismo pero con los juegos más infravalorados según los ratings de los usuarios.

#We will do the same but sorting the top underrated games according to user_ratings.

df2 = sales_rtg.sort_values('reviews_diff',ascending = True).groupby('name').head(5)
print (df2)


# In[23]:


#Let's check the empty values:

#Revisamos valores vacíos:

total = sales_rtg.isnull().sum().sort_values(ascending=False)
percent = (sales_rtg.isnull().sum()/sales_rtg.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data[missing_data['Total'] > 0]


# In[24]:


# Con el código siguiente quitaríamos valores según porcentaje de missing del 15%, pero en este caso el código no actuará,
#al estar los porcentajes debajo de ese umbral. Por el momento no realizaremos cambio en valores vacíos.

#With the following code we would remove values with a missing % above 15%. In this case the code will have no effect,
#since the %'s for all columns are below this threshold. We will not change empty values for now.

sales_rtg = sales_rtg[missing_data[missing_data['Percent'] < 0.15].index]
sales_rtg


# In[25]:


#Tenemos que volver a cambiar los 0 de Year por NAN porque sino nos contará los años 0, 
#ya que si creamos visualizaciones aparecerían valores incorrectos contados como 0, afectando a los gráficos:

#However for the following visualizations we will change back the Year 0's to Na's, 
#to avoid a count of the 0 values that would mess the graphs:

sales_rtg.Year.replace(0, np.nan, inplace=True)


# In[26]:


#Ahora realizamos las visualizaciones para los valores numéricos del df.
#Here we show the visualizations for the numerical values of the df.

sales_rtg.hist(bins=50, figsize=(30,20))


# In[27]:


#Vamos a revisar las visualizaciones de solo algunas de las columnas:

#We are going to check the visualizations for only some columns:

sales_rtg.hist(bins=50, figsize=(30,20), column=["user_review", "meta_score", "reviews_diff"])


# In[28]:


#Vamos a visualizar los valores medios de "user_review" y "meta_score" por año.

#Let's plot the average "user_review" y "meta_score" per year.

sales_rtg.groupby('Year')["user_review", "meta_score"].mean().plot()


# In[29]:


#Esta visualización muestra las ventas totales globales y por región:

#This is a visualization of the total sales per region & global:

sales_rtg.groupby('Year')["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"].sum().plot()


# In[30]:


#Podemos revisar las medias, min, max, count de los valores numéricos del df.
#We can also check avg, min, max, std, count for the numerical values of the df.

sales_rtg.describe()


# In[31]:


#Este es un análisis bivariante para user_review y meta_score.

#This is a bivariant visualization for user_review and meta_score.

var = 'meta_score'
data = pd.concat([sales_rtg['user_review'], sales_rtg['meta_score']], axis=1)
data.plot.scatter(x='meta_score', y='user_review')


# In[32]:


#Cantidad de juegos según genero
sales_rtg["Genre"].value_counts().plot(kind='bar')


# In[33]:


#Cantidad de juegos según platform
sales_rtg["platform"].value_counts().plot(kind='bar')


# In[34]:


#Realizamos un análisis multivariante para las variables numéricas. 
#Vemos que hay una relación clara entre el meta_score y las ventas a nivel global.

#We create a multivariant visualization.
#We can find a high correlation between meta_score and global sales.

#Multivariante sin normalizar:
corrmat = sales_rtg.corr(method='spearman')
f, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corrmat, ax=ax, cmap="YlGnBu", linewidths=0.1)


# In[35]:


#Podemos agrupar las variables más relacionadas para ver este detalle más claramente.

#We can group the most related columns to see this more clearly.

corrmat = sales_rtg.corr(method='spearman')
cg = sns.clustermap(corrmat, cmap="YlGnBu", linewidths=0.1);
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
cg


# In[36]:


#Bivariante entre meta_score y Genre con boxplot. 
#Vemos valores más bajos para Puzzle, Adventure y Strategy y valores más altos para Platform y Shooter.
#Precisamente Platform fue el género más popular hasta mediados de los 90 y
#Shooter el más popular a partir de finales de los 90.

#The following is a bivariant boxplot visualization between meta_score and Genre. 
#We see lower reviews for Puzzle, Adventure and Strategy but higher values for Platform and Shooter games. 
#Platform was the most popular Genre until mid 90's and Shooter were very popular from the end of the 90's.

var = 'Genre'
data = pd.concat([sales_rtg['meta_score'], sales_rtg[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="meta_score", data=data)
plt.xticks(rotation=90);


# In[37]:


#Multivariant visualization between all the numeric variables.

#Relación multivariante de las variables cuantitativas.

sns.set()
cols = ['meta_score', 'user_review', 'Global_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales']
sns.pairplot(sales_rtg[cols], height = 2.5)
plt.show();


# In[38]:


sales_rtg.head()


# In[39]:


#Drop de filas con na, para evitar problemas a la hora de normalizar y aplicar modelos:

sales_rtg_nona = sales_rtg.dropna()

#Using mean instead of dropping na's would get worse results for a linear regression:

#sales_rtg_nona = sales_rtg.fillna(sales_rtg.mean())


# In[40]:


sales_rtg_nona.head()


# In[41]:


#LINEAR REGRESSION

#Entrenamiento y Test.
#let's study the relation between sales and meta_score

from sklearn.model_selection import train_test_split

X = sales_rtg_nona.meta_score.values #This is the meta_score column

Y = sales_rtg_nona.Global_Sales.values #This is the global_sales column


X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.2, random_state = 0) 



# In[42]:


#We use reshape to avoid the following error for lear regression: Expected 2D array, got 1D array instead
X_train= X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)


# In[43]:


#Fitting Simple Linear Regression Model to the training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression(normalize= True)
regressor = regressor.fit(X_train, Y_train)


# In[44]:


#The coefficient of determination, denoted as 𝑅², tells you which amount of variation in 𝑦 can be explained
#by the dependence on 𝐱, using the particular regression model. 
#A larger 𝑅² indicates a better fit and means that the model can better explain the variation
#of the output with different inputs.

#The value 𝑅² = 1 corresponds to SSR = 0. That’s the perfect fit, since the values
#of predicted and actual responses fit completely to each other.

#In this case the value is only 0.23 which is low for the model

r_sq = regressor.score(X_train, Y_train)
print(f"coefficient of determination: {r_sq}")


# In[45]:


print(f"intercept: {regressor.intercept_}")


# In[46]:


print(f"slope: {regressor.coef_}")


# In[47]:


#The value of 𝑏₀ is approx -6.159
#This illustrates that the model predicts the response -6.159 when X is zero.
#The value 𝑏₁ = 0.09058 means that the predicted response rises by 0.09058 when X is increased by one.


# In[48]:


#Predicted Response

predicted = regressor.predict(X_test)

print(f"predicted response:\n{predicted}")


# In[49]:


#Visualization of training results

plt.scatter(X_train , Y_train, color = 'red')
plt.plot(X_train , regressor.predict(X_train), color ='blue')


# In[50]:


#Visualization of the test results

plt.scatter(X_test , Y_test, color = 'red')
plt.plot(X_test , regressor.predict(X_test), color ='blue')


# In[51]:


#We can use this fitted model to calculate the outputs based on new inputs:

x_new = np.arange(5).reshape((-1, 1))
x_new


# In[52]:


#LOGISTIC REGRESSION
#Vamos a crear un subset categorico.

#Let's create a categorical subset.

sales_rtg_categorical = sales_rtg_nona[['Genre', 'Publisher', 'platform']]
sales_rtg_categorical 


# In[53]:


#Creamos nuevas columnas categóricas binarizadas:

#We create the new binary columns:

import pandas as pd
sales_rtg_categorical = pd.concat([pd.get_dummies(sales_rtg_categorical[col], prefix=col) for col in sales_rtg_categorical], axis=1)


# In[54]:


#We concat some of the numeric columns to the new categorical ones.

#Unimos varias variables numéricas con las columnas categóricas nuevas.

df_categ = pd.concat([sales_rtg_nona[['meta_score', 'user_review','Global_Sales' ]], sales_rtg_categorical], axis=1)
df_categ.head()


# In[55]:


sales_rtg_categorical.head()


# In[56]:


df_categ.columns


# In[57]:


#Changing columns names to remove spaces so they are easier to manage.
#Cambiamos los normbres de las columnas para que sean más fáciles de utilizar.

df_categ.columns = df_categ.columns.str.replace(' ', '_')


# In[58]:


df_categ.columns


# In[59]:


#LOGISTIC REGRESSION

#Vamos a crear un nuevo par de test/train
#We are gpoing to create a new test/train pair

from sklearn.model_selection import train_test_split

X2 = df_categ.drop('platform__Wii_U', 1)
y2 = df_categ.platform__Wii_U

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X2, y2, test_size=0.30, random_state=42)


# In[60]:


y_test_2


# In[61]:


#We use reshape to avoid the following error for lear regression: Expected 2D array, got 1D array instead
#X_train_2 = X_train_2.reshape(-1, 1)
#X_test_2 = X_test_2.reshape(-1, 1)


# In[62]:


#Aplicamos el modelo
#We apply the model

from sklearn import linear_model, datasets

logreg = linear_model.LogisticRegression(max_iter=600, solver='lbfgs')
model = logreg.fit(X_train_2, y_train_2)
model


# In[63]:


#We check the predicted data
#Revisamos la información predicha

predicted_2 = model.predict(X_test_2)
predicted_2


# In[64]:


#Creamos una matriz de confusión para revisar cuantos datos han sido correctamente clasificados.
#We create a confusion matrix to check how many data have been correctly/incorrectly classified.

#In this case the false data has been correctly classified, but the true data has been incorrectly classified.

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

#predicted_2 = np.round(predicted_2)
matrix2 = confusion_matrix(y_test_2, predicted_2)
sns.heatmap(matrix2, annot=True, fmt="d", cmap='Blues', square=True)
plt.xlabel("predicción")
plt.ylabel("real")
plt


# In[65]:


#Esta es la accuracy que obtenemos del modelo.
#This is the accuracy obtained:

from sklearn.metrics import accuracy_score

accuracy_score(y_test_2, predicted_2)


# In[67]:


#We can check more detailed information about the trained model:

from sklearn.metrics import classification_report

report = classification_report(y_test_2, predicted_2)
print(report)


# In[68]:


# DECISION TREES

# Load libraries
# Cargamos las librerías
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


# In[69]:


#Split dataset in features and target variable. Meta_score will be the variable that we want to predict.

#Dividimos de nuevo el dataset en features y target. Meta_score será la variable que queremos predecir

feature_cols = ['user_review', 'Global_Sales', 'Genre_Action','Genre_Platform','platform__Xbox_360','platform__Wii','platform__PlayStation_3']
X3 = df_categ[feature_cols] # Features
y3 = df_categ.meta_score # Target variable


# In[70]:


X3


# In[71]:


# Split dataset into training set and test set
#Creamos de nuevo test/entrenamiento

X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X3, y3, test_size=0.3, random_state=1) # 70% training and 30% test


# In[72]:



# Creation of Decision Tree classifer object
# Creamos el objeto con el clasificador del arbol.

clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
# Entrenamos el arbol
clf = clf.fit(X_train_3,y_train_3)

#Predict the response for test dataset
#Predecimos la respuesta para el dataset test

y_pred_3 = clf.predict(X_test_3)


# In[73]:


# Model Accuracy, 82%
# La precisión del modelo es del 82%

print("Accuracy:",metrics.accuracy_score(y_test_3, y_pred_3))


# In[ ]:





# In[74]:


#KNN Model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#Se asignan 2 variables a X y una a y:
#We assign 2 variables to X and one to y:
X4 = df_categ[['Global_Sales','user_review']].values
y4 = df_categ['meta_score'].values

#realizamos de nuevo la división test/training
#we split the data into test/training again:

X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(X4, y4, random_state=0)
scaler = MinMaxScaler()
X_train_4 = scaler.fit_transform(X_train_4)
X_test_4 = scaler.transform(X_test_4)


# In[75]:


#Definimos k como 3 ya que da un poco mejor accuracy
#We have assigned k=3 since it seems to provide a slightly better accuracy:

n_neighbors = 3

knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train_4, y_train_4)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train_4, y_train_4)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test_4, y_test_4)))


# In[76]:


#Precisión del modelo:

pred = knn.predict(X_test_4)
print(confusion_matrix(y_test_4, pred))
print(classification_report(y_test_4, pred))


# In[79]:


#Hacemos fit knn de X4, y4, no de test/training
#We fit X4, y4, not training/test

clf2= knn.fit(X4, y4)


# In[80]:


#Con esto podemos obtener una predicción. Como vemos obtenemos un valor alto para meta_score con Global_Sales también altas.
#El valor es más alto que user_reviews, que acostumbra a estar por debajo de meta_score.

#With this we can try to make a prediction.We get a high meta_score with high Global_Sales.
#The meta_score is higher than user_reviews, which matches the data trends we have been checking.

print(clf2.predict([[25, 78]]))

