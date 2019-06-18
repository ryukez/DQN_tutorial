import os
from dotenv import load_dotenv

load_dotenv('.env')
isLocal = (os.environ.get('ENV') == 'local')
