import numpy as np # linear algebrarehelp
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import os
from dask import dataframe as dd
from sqlalchemy import create_engine
import pymysql
from matplotlib import pyplot as plt

def readRawDataFiles(dirName, fileName):
    print("reading file: ", fileName)
    firstRow = True
    nanCount = 0
    rowCnt = 0
    CHUNK_SIZE = 100000
    pd.set_option('max_columns', None)
    pd.set_option('max_rows', None)
    for chunk in pd.read_csv(os.path.join(dirName, fileName), chunksize=CHUNK_SIZE):
        if ('o' in chunk.columns):
            nanCount = nanCount + chunk['o'].isna().sum()
            chunk.dropna(subset=['o'],inplace=True)

        if (firstRow == True):
            rawDF = pd.DataFrame().reindex_like(chunk)
            firstRow = False

        rawDF = pd.concat([rawDF, chunk])
        print("New DF size: ", rawDF.shape)
        if (rawDF.shape[0] > 1000000):
            return rawDF

    return rawDF

#    newDF = playsDF.merge(trackingDF,on=["gameId","playId"],how="left")
#    print("newDF shape: " + str(newDF.shape))

def countRowsRawData(dirName, fileName):
    CHUNK_SIZE = 10000
    rowCnt = 0
    pd.set_option('max_columns', None)
    pd.set_option('max_rows', None)
    for chunk in pd.read_csv(os.path.join(dirName, fileName), chunksize=CHUNK_SIZE):
        rowCnt = rowCnt + chunk.shape[0]
        if (rowCnt % CHUNK_SIZE == 0):
            print ("row count: ", rowCnt)
    return rowCnt

def cleanData(trackingDF):
    import numpy as np

    # Drop non-numeric columns
    numeric_cols = trackingDF.select_dtypes(exclude='number')
    trackingDF.drop(numeric_cols, axis=1, inplace=True)
    print(trackingDF.shape)
    print("dropped columns")

    # Clean the Data
    for (columnName, columnData) in trackingDF.iteritems():
        # print('Column Name : ', columnName)
        if (trackingDF[columnName].isnull().sum() > 0):
            print(columnName, ' NaN Count : ', trackingDF[columnName].isnull().sum())
            trackingDF[columnName].fillna(0, inplace=True)
            print(columnName, ' NaN Count : ', trackingDF[columnName].isnull().sum())
        if (np.isinf(trackingDF[columnName]).sum() > 0):
            print('Column inf Count : ', np.isinf(trackingDF[columnName]).sum())

    print("got result column")
    trackingDF.drop(columns=["dis"], inplace=True)
    return trackingDF

def doMLStuff(trackingDF):
    print("ready to fit")
    from sklearn.ensemble import RandomForestRegressor

    for (columnName, columnData) in trackingDF.iteritems():
        print('Column Name : ', columnName)

    # define the model
    model = RandomForestRegressor(n_jobs=-1)

    # fit the model
    model.fit(trackingDF, y)
    print("done with fit!")

    # get importance
    importance = model.feature_importances_

    # summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))

    # plot feature importance
    # pyplot.bar([x for x in range(len(importance))], importance)
    # pyplot.show()
    return

def visualizeTrackingData(trackingData):
    trackingData.groupby('event').plot.bar()
    trackingData['event'].value_counts().sort_index().plot(kind='bar', rot=0, ylabel='count')
    pd.value_counts(trackingData['event']).plot.bar()
    plt.show()
#    ['product'].nunique()
#    .plot.bar()
    return

# Create SQLAlchemy engine to connect to MySQL Database
sqlEngine = create_engine("mysql+pymysql://root:skitaos1@/neural-nfl-data")
dbConn    = sqlEngine.connect()

# Read Seperate CSV files into DataFrames
#scoutingDataDF = pd.read_csv(os.path.join(dirname, 'PFFScoutingData.csv'))
#trackingDF = pd.concat([tracking2018DF,tracking2019DF,tracking2020DF])
#gamesDF = pd.read_csv(os.path.join(dirname, 'games.csv'))

dirName = "/Users/stevefurlong/dev/ml-nfl-kaggle/nfl-big-data-bowl-2022"
#print("players.csv row cnt: " , countRowsRawData(dirName, 'players.csv'))
#print("tracking2018.csv row cnt: " , countRowsRawData(dirName, 'tracking2018.csv'))
#print("tracking2019.csv row cnt: " , countRowsRawData(dirName, 'tracking2019.csv'))
#print("tracking2020.csv row cnt: " , countRowsRawData(dirName, 'tracking2020.csv'))

df = readRawDataFiles(dirName, 'tracking2018.csv')
print(df.shape)
#df.to_sql(name='tracking2018', con=dbConn, if_exists='replace', index=False, chunksize=100000)
#visualizeTrackingData(df)

#df = readRawDataFiles(dirName, 'tracking2019.csv')
#print(df.shape)
#df = readRawDataFiles(dirName, 'tracking2020.csv')
#print(df.shape)
df = readRawDataFiles(dirName, 'plays.csv')
print(df.shape)
df['specialTeamsPlayType'].value_counts().sort_index().plot(kind = 'bar', rot = 50, ylabel = 'count')
df['specialTeamsResult'].value_counts().sort_index().plot(kind = 'bar', rot = 75, ylabel = 'count')
plt.show()

df = readRawDataFiles(dirName, 'games.csv')
print(df.shape)
df.to_sql(name='games', con=dbConn, if_exists='replace', index=False)

df = readRawDataFiles(dirName, 'players.csv')
print(df.shape)
df.to_sql(name='players', con=dbConn, if_exists='replace', index=False)

#df.to_csv("./result.csv")

#readRawDataFiles()
#trackingDF = pd.read_csv('/Users/stevefurlong/dev/ml-nfl-kaggle/result.csv')
#cleanData()
#doMLStuff()






