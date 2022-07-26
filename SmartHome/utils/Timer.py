import time
import threading

class Timer:
    startTime = 0.0
    duration = 0
    isExpired = False
    elp = 0.0
    def start(self,t):
        self.startTime = t
        
    def checkExpired(self):
        elapsedTime  = time.time() - self.startTime
        self.elp = elapsedTime
        if elapsedTime < self.duration:
            self.isExpired = False
        else:
            self.isExpired = True
    
    def main(self):
        while True:
            self.checkExpired()
            time.sleep(.05)
    
    def __init__(self,dur):
        self.duration = dur
        thread = threading.Thread(target=self.main, daemon=True)
        thread.start()