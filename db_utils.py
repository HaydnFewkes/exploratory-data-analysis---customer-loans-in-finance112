import yaml
from sqlalchemy import create_engine
import pandas as pd

with open('credentials.yaml', 'r') as file:
    creds = yaml.safe_load(file)

print(creds)

class RDSDatabaseConnector:
    def __init__(self, creds):
        DATABASE_TYPE = 'postgresql'
        DBAPI = 'psycopg2'
        HOST = creds['RDS_HOST']
        USER = creds['RDS_USER']
        PASSWORD = creds['RDS_PASSWORD']
        DATABASE = creds['RDS_DATABASE']
        PORT = creds['RDS_PORT']
        engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")
