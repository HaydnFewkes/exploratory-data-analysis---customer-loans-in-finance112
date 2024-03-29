import yaml

with open('credentials.yaml', 'r') as file:
    creds = yaml.safe_load(file)


class RDSDatabaseConnector:
    def __init__(self, creds):
        self.creds = creds