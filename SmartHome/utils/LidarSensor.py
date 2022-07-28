import time
import threading
from smbus import SMBus
class LidarSensor:
    
    # LIDAR I2c Paramaters
    I2C_speed = 400000
    I2C_address = "0x70"
    
    distance = 0.0
    
    def start(self,t):
        self.startTime = t
        
    def getDistance(self):
        return self.distance
    
    def main(self):
        
        i2cbus = SMBus(1)
        
        while True:
            try:
                self.distance = i2cbus.read_block_data(self.I2C_address)
                print(self.distance)
                time.sleep(.20) 
            except Exception as e:
                print(e)
    
    def __init__(self):
        thread = threading.Thread(target=self.main, daemon=True)
        thread.start()