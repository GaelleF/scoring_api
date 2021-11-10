from flask_sqlalchemy import SQLAlchemy
from .views import app
import logging as lg
from numpy import genfromtxt
from time import time
from datetime import datetime
from sqlalchemy import Integer, Float, Date, String
#from sqlalchemy.ext.declarative import declarative_base
#from flask_sqlalchemy import create_engine
#from flask_sqlalchemy.orm import sessionmaker

# Create database connection object
db = SQLAlchemy(app)
"""
def Load_Data(file_name):
    data = genfromtxt(file_name, delimiter=',', skip_header=1, converters={0: lambda s: str(s)})
    return data.tolist()

Base = db

class data_application(db): #'Price_History'
    #Tell SQLAlchemy what the table name is and if there's any table-specific arguments it should know about
    __tablename__ = 'test'
    __table_args__ = {'sqlite_autoincrement': True}
    #tell SQLAlchemy the name of column and its attributes:
    id = db.Column(Integer, primary_key=True, nullable=False) 
    #date = Column(Date)
    TARGET = db.Column(Float) # not in test!
    NAME_CONTRACT_TYPE = db.Column(Float)
    CODE_GENDER = db.Column(String)
    FLAG_OWN_CAR = db.Column(Float)
    FLAG_OWN_REALTY = db.Column(Float)
    CNT_CHILDREN = db.Column(Float)
    AMT_INCOME_TOTAL = db.Column(Float)
    AMT_CREDIT = db.Column(Float)

def import_csv(name_db, type_db='test'): #csv_test.db' application_test
  #Create the database
  engine = create_engine('sqlite:///'+name_db+'.db')
  Base.metadata.create_all(engine)
  #Create the session
  session = sessionmaker()
  session.configure(bind=engine)
  s = session()
  start_inc = 1
  if type_db == 'test':
    start_inc= 0
  try:
    file_name = "scoringapp/data/"+name_db+".csv" #sample CSV file used:  http://www.google.com/finance/historical?q=NYSE%3AT&ei=W4ikVam8LYWjmAGjhoHACw&output=csv
    data = Load_Data(file_name) 

    for i in data:
      record = data_application(**{
        id : i[0],
         #TARGET = i[1] # not in test!
        'NAME_CONTRACT_TYPE' : i[start_inc+1],
        'CODE_GENDER' : i[start_inc+2],
        'FLAG_OWN_CAR' : i[start_inc+3],
        'FLAG_OWN_REALTY' : i[start_inc+4],
        'CNT_CHILDREN' : i[start_inc+5],
        'AMT_INCOME_TOTAL' : i[start_inc+6],
        'AMT_CREDIT' : i[start_inc+7],
        }, name=type_db)
      s.add(record) #Add all the records

      s.commit() #Attempt to commit all the records
  except:
        s.rollback() #Rollback the changes on error
  finally:
    s.close() #Close the connection
    print("Time elapsed: " + str(time() - t) + " s.") #0.091s
"""

# load data application for train and test db
def init_db():
   lg.warning('Database initialized!')