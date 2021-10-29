from flask_sqlalchemy import SQLAlchemy
from .views import app
import logging as lg

# Create database connection object
db = SQLAlchemy(app)

def init_db():
   lg.warning('Database initialized!')