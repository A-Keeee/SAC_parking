import sys
sys.path.append("/home/ake/1-500/HOPE_ge/src")
from env.parking_map_normal import ParkingMapNormal
m = ParkingMapNormal()
m.reset()
print(m.start.get_pos())
