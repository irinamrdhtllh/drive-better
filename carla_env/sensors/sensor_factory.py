from sensors.sensors import (
    CameraRGB,
    CameraDepth,
    Lidar,
    Radar,
    Gnss,
    Imu,
    LaneInvasion,
    Collision,
)


class SensorFactory(object):
    @staticmethod
    def spawn(name, attributes, interface, parent):
        attributes = attributes.copy()
        sensor_type = attributes.get("type", "")

        if sensor_type == "sensor.camera.rgb":
            sensor = CameraRGB(name, attributes, interface, parent)
        elif sensor_type == "sensor.camera.depth":
            sensor = CameraDepth(name, attributes, interface, parent)
        elif sensor_type == "sensor.lidar.ray_cast":
            sensor = Lidar(name, attributes, interface, parent)
        elif sensor_type == "sensor.other.radar":
            sensor = Radar(name, attributes, interface, parent)
        elif sensor_type == "sensor.other.gnss":
            sensor = Gnss(name, attributes, interface, parent)
        elif sensor_type == "sensor.other.imu":
            sensor = Imu(name, attributes, interface, parent)
        elif sensor_type == "sensor.other.lane_invasion":
            sensor = LaneInvasion(name, attributes, interface, parent)
        elif sensor_type == "sensor.other.collision":
            sensor = Collision(name, attributes, interface, parent)
        else:
            raise RuntimeError(f"Sensor of type {sensor_type} is not supported.")

        return sensor
