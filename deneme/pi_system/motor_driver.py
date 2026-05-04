class MotorDriver:
    # BCM pin numbers. Change these to match your wiring.
    LEFT_IN1 = 17
    LEFT_IN2 = 27
    LEFT_PWM = 18
    RIGHT_IN1 = 22
    RIGHT_IN2 = 23
    RIGHT_PWM = 13

    PWM_FREQ = 1000

    def __init__(self):
        self.left_in1 = None
        self.left_in2 = None
        self.left_pwm = None
        self.right_in1 = None
        self.right_in2 = None
        self.right_pwm = None
        self.ready = False

    def setup(self):
        try:
            from gpiozero import DigitalOutputDevice, PWMOutputDevice
        except ImportError:
            print(
                "[WARN] gpiozero kutuphanesi bulunamadi. MotorDriver pasif modda. "
                "Raspberry Pi uzerinde 'sudo apt install python3-gpiozero' gerekli."
            )
            self.ready = False
            return

        self.left_in1 = DigitalOutputDevice(self.LEFT_IN1, initial_value=False)
        self.left_in2 = DigitalOutputDevice(self.LEFT_IN2, initial_value=False)
        self.left_pwm = PWMOutputDevice(self.LEFT_PWM, frequency=self.PWM_FREQ, initial_value=0)

        self.right_in1 = DigitalOutputDevice(self.RIGHT_IN1, initial_value=False)
        self.right_in2 = DigitalOutputDevice(self.RIGHT_IN2, initial_value=False)
        self.right_pwm = PWMOutputDevice(self.RIGHT_PWM, frequency=self.PWM_FREQ, initial_value=0)

        self.ready = True
        self.stop()

    def forward(self, speed):
        speed = self._clamp_speed(speed)
        self.set_motor(speed, speed)

    def backward(self, speed):
        speed = self._clamp_speed(speed)
        self.set_motor(-speed, -speed)

    def left(self, speed):
        speed = self._clamp_speed(speed)
        self.set_motor(speed * 0.45, speed)

    def right(self, speed):
        speed = self._clamp_speed(speed)
        self.set_motor(speed, speed * 0.45)

    def stop(self):
        if not self.ready:
            return

        self._stop_side(self.left_in1, self.left_in2, self.left_pwm)
        self._stop_side(self.right_in1, self.right_in2, self.right_pwm)

    def cleanup(self):
        self.stop()

        for device in (
            self.left_in1,
            self.left_in2,
            self.left_pwm,
            self.right_in1,
            self.right_in2,
            self.right_pwm,
        ):
            if device is not None:
                device.close()

        self.ready = False

    def set_motor(self, left_speed, right_speed):
        if not self.ready:
            return

        self._set_side(self.left_in1, self.left_in2, self.left_pwm, left_speed)
        self._set_side(self.right_in1, self.right_in2, self.right_pwm, right_speed)

    def _set_side(self, in1, in2, pwm, speed):
        speed = self._clamp_signed_speed(speed)
        duty = abs(speed) / 100.0

        if speed > 0:
            in1.on()
            in2.off()
            pwm.value = duty
        elif speed < 0:
            in1.off()
            in2.on()
            pwm.value = duty
        else:
            self._stop_side(in1, in2, pwm)

    def _stop_side(self, in1, in2, pwm):
        in1.off()
        in2.off()
        pwm.value = 0

    def _clamp_speed(self, speed):
        return max(0.0, min(100.0, float(speed)))

    def _clamp_signed_speed(self, speed):
        return max(-100.0, min(100.0, float(speed)))
