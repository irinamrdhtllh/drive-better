import psutil


def is_used(port: bool):
    return port in [c.laddr.port for c in psutil.net_connections()]
