from stem import Signal
from stem.control import Controller

with Controller.from_port(port = 9051) as controller:
    controller.authenticate(password='16:872860B76453A77D60CA2BB8C1A7042072093276A3D701AD684053EC4C')
    print("Success!")
    controller.signal(Signal.NEWNYM)
    print("New Tor connection processed")
