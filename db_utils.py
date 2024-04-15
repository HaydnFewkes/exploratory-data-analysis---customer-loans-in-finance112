import yaml
from sqlalchemy import create_engine, inspect
import pandas as pd

with open('credentials.yaml', 'r') as file:
    creds = yaml.safe_load(file)

print(creds)

class RDSDatabaseConnector:
    def __init__(self, creds):
        self.creds = creds

    def RDSConnection(self):
        self.DATABASE_TYPE = 'postgresql'
        self.DBAPI = 'psycopg2'
        self.HOST = self.creds['RDS_HOST']
        self.USER = self.creds['RDS_USER']
        self.PASSWORD = self.creds['RDS_PASSWORD']
        self.DATABASE = self.creds['RDS_DATABASE']
        self.PORT = self.creds['RDS_PORT']
        self.engine = create_engine(f"{self.DATABASE_TYPE}+{self.DBAPI}://{self.USER}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.DATABASE}")

    def RDSExtract(self):
        connection = self.engine.connect()
        inspector = inspect(self.engine)
        table_names = inspector.get_table_names()
        self.loan_payments = pd.read_sql_table(f"{table_names[0]}", self.engine)
        connection.close()

    def RDSSaveToCSV(self):
        self.loan_payments.to_csv('loan_payments.csv', index=False)

a = RDSDatabaseConnector(creds)
a.RDSConnection()
a.RDSExtract()
a.RDSSaveToCSV()