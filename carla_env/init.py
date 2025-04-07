import os
import logging
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

        self.vehicles_list = []
        self.walkers_list = []
        self.all_walkers_id = []

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
                settings.synchronous_mode = True
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

    def set_spectator_view(self):
        transform = self.hero.get_transform()

        # The camera position
        view_x = transform.location.x - 8 * transform.get_forward_vector().x
        view_y = transform.location.y - 5 * transform.get_forward_vector().y
        view_z = transform.location.z + 3

        # The camera orientation
        view_roll = transform.rotation.roll
        view_yaw = transform.rotation.yaw
        view_pitch = transform.rotation.pitch

        self.spectator = self.world.get_spectator()
        self.spectator.set_transform(
            carla.Transform(
                carla.Location(x=view_x, y=view_y, z=view_z),
                carla.Rotation(pitch=view_pitch, yaw=view_yaw, roll=view_roll),
            )
        )

    def get_sensor_data(self):
        sensor_data = self.sensor_interface.get_data()
        return sensor_data

    def control_hero(self): ...

    def generate_traffic(self):
        vehicle_blueprints = self.world.get_blueprint_library().filter("vehicle.*")
        num_vehicles = self.exp_config["num_vehicles"]

        walker_blueprints = self.world.get_blueprint_library().filter(
            "walker.pedestrian.*"
        )
        num_walkers = self.exp_config["num_walkers"]

        spawn_points = self.map.get_spawn_points()
        num_spawn_points = len(spawn_points)

        if num_vehicles <= num_spawn_points:
            random.shuffle(spawn_points)
        else:
            print(
                f"Requested {num_vehicles} vehicles, but could only find {num_spawn_points} spawn points."
            )
            num_vehicles = num_spawn_points

        # Spawn all vehicles
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= num_vehicles:
                break

            vehicle = random.choice(vehicle_blueprints)
            if vehicle.has_attribute("color"):
                color = random.choice(vehicle.get_attribute("color").recommended_values)
                vehicle.set_attribute("color", color)
            if vehicle.has_attribute("driver_id"):
                driver_id = random.choice(
                    vehicle.get_attribute("driver_id").recommended_values
                )
                vehicle.set_attribute("driver_id", driver_id)

            batch.append(
                carla.command.SpawnActor(vehicle, transform).then(
                    carla.command.SetAutopilot(
                        carla.command.FutureActor, True, self.traffic_manager.get_port()
                    )
                )
            )

        for response in self.client.apply_batch_sync(batch, True):
            if response.error:
                logging.error(response.error)
            else:
                self.vehicles_list.append(response.actor_id)

        # Spawn all walkers
        walkers_running = 0.2
        walkers_crossing = 0.2
        spawn_points = []
        for i in range(num_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                spawn_points.append(spawn_point)

        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker = random.choice(walker_blueprints)
            if walker.has_attribute("is_invicible"):
                walker.set_attribute("is_invicible", "false")
            if walker.has_attribute("speed"):
                if random.random() > walkers_running:
                    walker_speed.append(
                        walker.get_attribute("speed").recommended_values[1]
                    )
                else:
                    walker_speed.append(
                        walker.get_attribute("speed").recommended_values[2]
                    )
            else:
                walker_speed.append(0.0)
            batch.append(carla.command.SpawnActor(walker, spawn_point))

        walker_speed_2 = []
        for i, response in enumerate(self.client.apply_batch_sync(batch, True)):
            if response.error:
                logging.error(response.error)
            else:
                self.walkers_list.append({"id": response.actor_id})
                walker_speed_2.append(walker_speed[i])
        walker_speed = walker_speed_2

        # Add AI controller to walkers
        batch = []
        walker_controller = self.world.get_blueprint_library().find(
            "controller.ai.walker"
        )
        for i in range(len(self.walkers_list)):
            batch.append(
                carla.command.SpawnActor(
                    walker_controller, carla.Transform(), self.walkers_list[i]["id"]
                )
            )

        for i, response in enumerate(self.client.apply_batch_sync(batch, True)):
            if response.error:
                logging.error(response.error)
            else:
                self.walkers_list[i]["ctrl"] = response.actor_id

        for i in range(len(self.walkers_list)):
            self.all_walkers_id.append(self.walkers_list[i]["ctrl"])
            self.all_walkers_id.append(self.walkers_list[i]["id"])
        self.all_walkers = self.world.get_actors(self.all_walkers_id)

        self.world.tick()

        self.world.set_pedestrians_cross_factor(walkers_crossing)

        for i in range(0, len(self.all_walkers), 2):
            self.all_walkers[i].start()
            self.all_walkers[i].go_to_location(
                self.world.get_random_location_from_navigation()
            )
            self.all_walkers[i].set_max_speed(float(walker_speed[int(i / 2)]))

        print(
            f"spawned {len(self.vehicles_list)} vehicles and {len(self.walkers_list)} walkers."
        )

    def tick(self, control): ...

    def destroy(self): ...


if __name__ == "__main__":
    config = read_config()
    env = InitEnv(config)
    env.setup_experiment()
    env.reset_hero()
    env.generate_traffic()
    env.set_spectator_view()

    try:
        while True:
            env.world.tick()
            time.sleep(0.02)
    except KeyboardInterrupt:
        settings = env.world.get_settings()
        settings.synchronous_mode = False
        settings.no_rendering_mode = False
        settings.fixed_delta_seconds = None
        env.world.apply_settings(settings)

        # Destroy all vehicles
        print(f"Destroying {len(env.vehicles_list)} vehicles")
        env.client.apply_batch(
            [carla.command.DestroyActor(x) for x in env.vehicles_list]
        )

        # Stop walkers' controller
        for i in range(0, len(env.all_walkers), 2):
            env.all_walkers[i].stop()

        # Destroy all walkers
        print(f"Destroying {len(env.walkers_list)} walkers")
        env.client.apply_batch(
            [carla.command.DestroyActor(x) for x in env.all_walkers_id]
        )

        time.sleep(0.5)
