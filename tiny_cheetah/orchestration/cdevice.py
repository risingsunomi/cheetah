from __future__ import annotations

from tiny_cheetah.models.shard import Shard
from tiny_cheetah.orchestration.device_info import collect_host_info


class CDevice:
    def __init__(
        self,
        peer_client_id: str,
        ip_address: str,
        port: int,
        tg_device: str = "CPU",
        get_self_info: bool = True
    ) -> None:
        self.peer_client_id = peer_client_id
        self.ip_address = ip_address
        self.port = port
        self.tg_device = tg_device
        self.cpu_model = ""
        self.cpu_proc_speed = ""
        self.cpu_cores = 0
        self.cpu_ram = ""
        self.gpu_model = ""
        self.gpu_vram = ""
        self.gpu_flops = 0.0
        self.ping_ms = 0.0
        
        if get_self_info:
            host_info = collect_host_info()
            devices = host_info.get("devices", [])
            cpu_info = next((d for d in devices if d.get("kind") == "CPU"), {})
            gpu_info = next((d for d in devices if d.get("kind") == "GPU"), {})
            self.cpu_model = str(cpu_info.get("name", ""))
            self.cpu_proc_speed = str(cpu_info.get("speed", ""))
            self.cpu_cores = int(cpu_info.get("cores", 0) or 0)
            self.cpu_ram = str(cpu_info.get("ram_gb", ""))
            self.gpu_model = str(gpu_info.get("name", ""))
            self.gpu_vram = str(gpu_info.get("vram_gb", gpu_info.get("ram_gb", "")))
            self.gpu_flops = float(gpu_info.get("flops", 0.0) or 0.0)

    def as_dict(self) -> dict:
        return {
            "peer_client_id": self.peer_client_id,
            "ip_address": self.ip_address,
            "port": self.port,
            "tg_device": self.tg_device,
            "cpu_model": self.cpu_model,
            "cpu_proc_speed": self.cpu_proc_speed,
            "cpu_cores": self.cpu_cores,
            "cpu_ram": self.cpu_ram,
            "gpu_model": self.gpu_model,
            "gpu_vram": self.gpu_vram,
            "gpu_flops": self.gpu_flops,
        }
