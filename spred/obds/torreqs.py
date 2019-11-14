import time
from torrequest import TorRequest

with TorRequest() as tr:
    response = tr.get('http://ipecho.net/plain')
    print(response.text)  # not your IP address
    
print("Sleeping.")
time.sleep(1)

with TorRequest() as tr:
    response = tr.get('http://ipecho.net/plain')
    print(response.text)  # another IP address, not yours
