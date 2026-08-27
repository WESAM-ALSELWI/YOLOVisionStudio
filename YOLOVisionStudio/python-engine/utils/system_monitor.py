import psutil
from typing import Dict, Any

class SystemMonitor:
    @staticmethod
    def get_cpu_percent() -> float:
        return psutil.cpu_percent(interval=0.1)

    @staticmethod
    def get_ram_stats() -> Dict[str, Any]:
        vm = psutil.virtual_memory()
        return {
            "used_gb": round(vm.used / (1024**3), 2),
            "total_gb": round(vm.total / (1024**3), 2),
            "percent": vm.percent
        }

    @staticmethod
    def get_disk_stats() -> Dict[str, Any]:
        disk = psutil.disk_usage('/')
        return {
            "used_gb": round(disk.used / (1024**3), 2),
            "total_gb": round(disk.total / (1024**3), 2),
            "percent": disk.percent
        }

    @staticmethod
    def get_gpu_stats() -> Dict[str, Any]:
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                name = pynvml.nvmlDeviceGetName(handle)
                # handle bytestring return in newer/older versions
                if isinstance(name, bytes):
                    name = name.decode('utf-8')

                return {
                    "name": name,
                    "memory_used_mb": info.used // (1024**2),
                    "memory_total_mb": info.total // (1024**2),
                    "utilization_percent": util.gpu,
                    "temperature": temp
                }
        except Exception:
            pass
            
        # Fallback if no GPU or pynvml fails
        return {
            "name": "N/A (CPU Only / No NVML)",
            "memory_used_mb": 0,
            "memory_total_mb": 0,
            "utilization_percent": 0,
            "temperature": 0
        }

    @staticmethod
    def get_all_stats() -> Dict[str, Any]:
        ram = SystemMonitor.get_ram_stats()
        disk = SystemMonitor.get_disk_stats()
        gpu = SystemMonitor.get_gpu_stats()
        
        return {
            "cpu_percent": SystemMonitor.get_cpu_percent(),
            "ram_used_gb": ram["used_gb"],
            "ram_total_gb": ram["total_gb"],
            "disk_used_gb": disk["used_gb"],
            "disk_total_gb": disk["total_gb"],
            "gpu_name": gpu["name"],
            "gpu_memory_used_mb": gpu["memory_used_mb"],
            "gpu_memory_total_mb": gpu["memory_total_mb"],
            "gpu_utilization_percent": gpu["utilization_percent"],
            "gpu_temperature": gpu["temperature"]
        }
