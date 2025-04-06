import copy
import math
import cv2

import carla
import numpy as np


class BaseSensor(object):
    def __init__(self, name, attributes, interface, parent):
        self.name = name
        self.attributes = attributes
        self.interface = interface
        self.parent = parent

        self.interface.register(self.name, self)

    def is_event_sensor(self):
        return False

    def parse(self):
        raise NotImplementedError

    def update_sensor(self, data, frame):
        if not self.is_event_sensor():
            self.interface._data_buffers.put((self.name, frame, self.parse(data)))
        else:
            self.interface._event_data_buffers.put((self.name, frame, self.parse(data)))

    def callback(self, data):
        frame = data.frame
        self.update_sensor(data, frame)

    def destroy(self):
        raise NotImplementedError


class CarlaSensor(BaseSensor):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

        world = self.parent.get_world()

        sensor_type = self.attributes.pop("type", "")
        transform = self.attributes.pop("transform", "0, 0, 0, 0, 0, 0")
        if isinstance(transform, str):
            transform = [float(x) for x in transform.split(", ")]
        assert len(transform) == 6

        blueprint = world.get_blueprint_library().find(sensor_type)
        blueprint.set_attribute("role_name", name)
        for key, value in attributes.items():
            blueprint.set_attribute(str(key), str(value))

        transform = carla.Transform(
            carla.Location(transform[0], transform[1], transform[2]),
            carla.Rotation(transform[4], transform[5], transform[3]),
        )
        self.sensor = world.spawn_actor(
            blueprint,
            transform,
            attach_to=self.parent,
            attachment_type=carla.AttachmentType.Rigid,
        )
        self.sensor.listen(self.callback)

    def destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()
            self.sensor = None


class BaseCamera(CarlaSensor):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

    def parse(self, sensor_data):
        # sensor_data: [fov, height, width, raw_data]
        array = np.frombuffer(sensor_data.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (sensor_data.height, sensor_data.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array


class CameraRGB(BaseCamera):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)


class CameraDepth(BaseCamera):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)


class Lidar(CarlaSensor):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

    def parse(self, sensor_data):
        # sensor_data: [x, y, z, intensity]
        points = np.frombuffer(sensor_data.raw_data, dtype=np.dtype("f4"))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        return points


class Radar(CarlaSensor):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

    def parse(self, sensor_data):
        # sensor_data: [depth, azimuth, altitute, velocity]
        points = np.frombuffer(sensor_data.raw_data, dtype=np.dtype("f4"))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        points = np.flip(points, 1)
        return points


class Gnss(CarlaSensor):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

    def parse(self, sensor_data):
        # sensor_data: [latitude, longitude, altitude]
        return np.array(
            [sensor_data.latitude, sensor_data.longitude, sensor_data.altitude],
            dtype=np.float64,
        )


class Imu(CarlaSensor):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

    def parse(self, sensor_data):
        # sensor_data: [accelerometer, gyroscope, compass]
        return np.array(
            [
                sensor_data.accelerometer.x,
                sensor_data.accelerometer.y,
                sensor_data.accelerometer.z,
                sensor_data.gyroscope.x,
                sensor_data.gyroscope.y,
                sensor_data.gyroscope.z,
                sensor_data.compass,
            ],
            dtype=np.float64,
        )


class LaneInvasion(CarlaSensor):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

    def is_event_sensor(self):
        return True

    def parse(self, sensor_data):
        # sensor_data: [transform, lane marking]
        return [sensor_data.transform, sensor_data.crossed_lane_markings]


class Collision(CarlaSensor):
    def __init__(self, name, attributes, interface, parent):
        self._last_event_frame = 0
        super().__init__(name, attributes, interface, parent)

    def callback(self, data):
        if self._last_event_frame != data.frame:
            self._last_event_frame = data.frame
            self.update_sensor(data, data.frame)

    def is_event_sensor(self):
        return True

    def parse(self, sensor_data):
        # sensor_data: [other actor, impulse]
        impulse = sensor_data.normal_impulse
        impulse_value = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        return [sensor_data.other_actor, impulse_value]
