class PIController:
    def __init__(self, KP, KI):
        self.KP = KP
        self.KI = KI
        self.error = 0.0
        self.set_point = 0.0
        self.integral = 0.0
        self.throttle = 0.0

    def set_desired(self, desired):
        self.set_point = desired

    def updated(self, measurement):
        self.error = self.set_point - measurement
        self.integral += self.error
        self.throttle = self.error * self.KP + self.integral * self.KI
        return self.throttle
