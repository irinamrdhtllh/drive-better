import os
import random
import subprocess
import time
import carla

from helper import is_used
from config import read_config


class InitEnv:
    def __init__(self, config):
        self.config = config

        self.client = None
        self.world = None
        self.map = None
        self.traffic_manager = None

        self.hero = None
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

        if self.config["show_display"]:
            server_command = [
                "D:/Documents/CARLA_0.9.15/CarlaUE4.exe",
                "-windowed",
                "-ResX={}".format(self.config["res_x"]),
                "-ResY={}".format(self.config["res_y"]),
                "-quality-level={}".format(self.config["quality_level"]),
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
        max_retries = self.config["retries_on_error"]
        for i in range(max_retries):
            try:
                self.client = carla.Client(self.config["host"], self.server_port)
                self.client.set_timeout(self.config["timeout"])
                self.world = self.client.get_world()

                settings = self.world.get_settings()
                settings.no_rendering_mode = not self.config["enable_rendering"]
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = self.config["timestep"]
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


if __name__ == "__main__":
    config = read_config()
    env = InitEnv(config)
