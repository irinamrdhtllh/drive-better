import os
import random
import subprocess
import time
import carla
import cv2

from helper import is_used
from config import read_config
from sensors.sensor_factory import SensorFactory
from sensors.sensor_interface import SensorInterface


class InitEnv:
    def __init__(self, config):
        self.carla_config = config["carla"]
        self.exp_config = config["experiment"]

        self.client = None
        self.world = None
        self.map = None
        self.traffic_manager = None

        self.hero = None
        self.sensor_interface = SensorInterface()
        self.spawn_point = None
        self.goal_point = None

        self.init_server()
        self.connect_client()

    def init_server(self):
        self.server_port = random.randint(15000, 32000)

        time.sleep(random.uniform(0, 1))

        uses_server_port = is_used(self.server_port)
        uses_stream_port = is_used(self.server_port + 1)
        while uses_server_port and uses_stream_port:
            if uses_server_port:
                print(f"Is using the server port: {self.server_port}")
            if uses_stream_port:
                print(f"Is using the streaming port: {self.server_port + 1}")
            self.server_port += 2
            uses_server_port = is_used(self.server_port)
            uses_stream_port = is_used(self.server_port + 1)

        if self.carla_config["show_display"]:
            server_command = [
                "D:/Documents/CARLA_0.9.15/CarlaUE4.exe",
                "-windowed",
                "-ResX={}".format(self.carla_config["res_x"]),
                "-ResY={}".format(self.carla_config["res_y"]),
                "-quality-level={}".format(self.carla_config["quality_level"]),
            ]
        else:
            server_command = [
                "D:/Documents/CARLA_0.9.15/CarlaUE4.exe",
                "-RenderOffScreen",
            ]

        server_command += [
            "--carla-rpc-port={}".format(self.server_port),
        ]

        server_command_text = " ".join(map(str, server_command))
        print(server_command_text)
        server_process = subprocess.Popen(
            server_command,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def connect_client(self):
        max_retries = self.carla_config["retries_on_error"]
        for i in range(max_retries):
            try:
                self.client = carla.Client(self.carla_config["host"], self.server_port)
                self.client.set_timeout(self.carla_config["timeout"])
                self.world = self.client.get_world()

                settings = self.world.get_settings()
                settings.no_rendering_mode = not self.carla_config["enable_rendering"]
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = self.carla_config["timestep"]
                self.world.apply_settings(settings)
                self.world.tick()

                return
            except Exception as e:
                print(
                    f"Waiting for server to be ready: {e}. Attempt {i + 1}/{max_retries}."
                )
                time.sleep(5)

        raise Exception(
            "Cannot connect to server. Try increasing 'timeout' or 'retries_on_error' at the carla configuration."
        )

    def setup_experiment(self):
        self.world = self.client.get_world()
        self.map = self.world.get_map()

        weather = getattr(carla.WeatherParameters, self.exp_config["weather"])
        self.world.set_weather(weather)

        self.tm_port = 8000
        while is_used(self.tm_port):
            print(
                f"Traffic manager's port {self.tm_port} is already being used. Checking the next one."
            )
            self.tm_port += 1
        print(f"Traffic manager connected to port {self.tm_port}.")

        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        self.traffic_manager.set_hybrid_physics_mode(self.exp_config["tm_hybrid_mode"])

        seed = self.exp_config["seed"]
        if seed is not None:
            self.traffic_manager.set_random_device_seed(seed)

    def reset_hero(self):
        if self.hero is not None:
            self.hero.destroy()
            self.hero = None
            self.sensor_interface.destroy()
            cv2.destroyAllWindows()

        self.world.tick()

        hero_model = "".join(self.exp_config["hero_model"])
        hero_blueprint = self.world.get_blueprint_library().find(hero_model)
        hero_blueprint.set_attribute("role_name", "hero")

        spawn_points = self.map.get_spawn_points()
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()

        self.hero = self.world.spawn_actor(hero_blueprint, spawn_point)
        if self.hero is None:
            raise AssertionError(
                f"Error spawning her {hero_blueprint} at point {spawn_point}."
            )

        self.world.tick()

        hero_sensors = self.exp_config["sensors"]

        if self.hero is not None:
            print("Hero spawned.")
            for name, attributes in hero_sensors.items():
                sensor = SensorFactory.spawn(
                    name, attributes, self.sensor_interface, self.hero
                )

        sensor_data = self.sensor_interface.get_data()
        image = sensor_data["camera"][1]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Camera Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def spectator_camera_view(self): ...

    def get_sensor_data(self): ...

    def control_hero(self): ...

    def generate_traffic(self): ...

    def tick(self): ...

    def destroy(self): ...


if __name__ == "__main__":
    config = read_config()
    env = InitEnv(config)
    env.setup_experiment()
    env.reset_hero()
