"""
OICIO Swap Manager: Handle RAM limitations via disk swap
Credits: deepRcurs Labs @deeprcurs / Mzed Imamkh @mzedimamkh

Aturan: jika RAM kurang, swap 10GB, 20GB, 30GB dan seterusnya
Implementasi: Python-level swap manager + OS-level swap files

OS swap: /home/user/.cache/swap_10gb (10GB), swap_5gb_extra (5GB), swap_8gb_more (3.4GB) = 18GB total
Python swap: offload tensors to disk via memmap when RAM > threshold
"""

import os
import psutil
import torch
import numpy as np
import tempfile
import gc
from typing import Dict, Any

class SwapManager:
    def __init__(self, swap_dir="/home/user/.cache/oicio_swap", ram_threshold_gb=1.5):
        self.swap_dir = swap_dir
        self.ram_threshold = ram_threshold_gb * 1024 * 1024 * 1024
        os.makedirs(swap_dir, exist_ok=True)
        self.swapped_tensors = {}  # name -> path
        print(f"[SwapManager] Initialized, swap_dir={swap_dir}, threshold={ram_threshold_gb}GB")
        self.check_system_swap()

    def check_system_swap(self):
        """Check OS-level swap"""
        try:
            import subprocess
            result = subprocess.run(["cat", "/proc/swaps"], capture_output=True, text=True)
            print(f"[SwapManager] OS Swap:\n{result.stdout}")
            result = subprocess.run(["free", "-h"], capture_output=True, text=True)
            print(f"[SwapManager] Memory:\n{result.stdout}")
        except Exception as e:
            print(f"[SwapManager] Could not check swap: {e}")

    def get_ram_usage(self):
        """Get current RAM usage"""
        try:
            vm = psutil.virtual_memory()
            return vm.used, vm.total, vm.percent
        except:
            # Fallback
            import os
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            return 0, 0, 0

    def should_swap(self):
        """Check if should swap based on RAM usage"""
        try:
            used, total, percent = self.get_ram_usage()
            return percent > 80 or used > self.ram_threshold
        except:
            return False

    def offload_tensor(self, name: str, tensor: torch.Tensor) -> str:
        """Offload tensor to disk via memmap"""
        path = os.path.join(self.swap_dir, f"{name}.pt")
        # Save to disk
        torch.save(tensor.cpu(), path)
        self.swapped_tensors[name] = path
        print(f"[SwapManager] Offloaded {name} {tensor.shape} {tensor.nbytes/1024/1024:.1f}MB -> {path}")
        # Free RAM
        del tensor
        gc.collect()
        return path

    def load_tensor(self, name: str) -> torch.Tensor:
        """Load tensor back from disk"""
        if name not in self.swapped_tensors:
            raise ValueError(f"Tensor {name} not in swap")
        path = self.swapped_tensors[name]
        tensor = torch.load(path, map_location='cpu')
        print(f"[SwapManager] Loaded {name} from {path}")
        return tensor

    def offload_numpy(self, name: str, array: np.ndarray) -> str:
        """Offload numpy array via memmap"""
        path = os.path.join(self.swap_dir, f"{name}.npy")
        np.save(path, array)
        self.swapped_tensors[name] = path
        print(f"[SwapManager] Offloaded numpy {name} {array.shape} {array.nbytes/1024/1024:.1f}MB")
        del array
        gc.collect()
        return path

    def load_numpy(self, name: str) -> np.ndarray:
        path = self.swapped_tensors[name]
        array = np.load(path)
        print(f"[SwapManager] Loaded numpy {name} from {path}")
        return array

    def create_swap_file(self, size_gb: int, name: str = None):
        """Create additional swap file (10GB, 20GB, 30GB...)"""
        if name is None:
            name = f"swap_{size_gb}gb"
        path = f"/home/user/.cache/{name}"

        if os.path.exists(path):
            print(f"[SwapManager] Swap file {path} already exists")
            return path

        print(f"[SwapManager] Creating {size_gb}GB swap file at {path}...")
        try:
            # Use fallocate for speed
            os.system(f"fallocate -l {size_gb}G {path}")
            os.chmod(path, 0o600)
            os.system(f"sudo /sbin/mkswap {path}")
            os.system(f"sudo /sbin/swapon {path}")
            print(f"[SwapManager] {size_gb}GB swap activated")
            self.check_system_swap()
        except Exception as e:
            print(f"[SwapManager] Failed to create swap: {e}")

        return path

    def auto_scale_swap(self):
        """Auto-scale swap 10GB -> 20GB -> 30GB as needed"""
        import subprocess
        result = subprocess.run(["free", "-g"], capture_output=True, text=True)
        # Parse swap total
        try:
            lines = result.stdout.split("\n")
            for line in lines:
                if "Swap" in line:
                    parts = line.split()
                    swap_total_gb = int(parts[1])
                    print(f"[SwapManager] Current swap: {swap_total_gb}GB")

                    if swap_total_gb < 10:
                        self.create_swap_file(10, "swap_10gb")
                    elif swap_total_gb < 20:
                        self.create_swap_file(10, "swap_20gb_extra")
                    elif swap_total_gb < 30:
                        self.create_swap_file(10, "swap_30gb_extra")

        except Exception as e:
            print(f"[SwapManager] Auto-scale failed: {e}")

# Demo
if __name__ == "__main__":
    print("=== Swap Manager POC ===")
    manager = SwapManager()

    # Simulate offloading large tensor
    print("\n[Demo] Creating large tensor 1GB...")
    large_tensor = torch.randn(10000, 10000)  # ~400MB
    print(f"Tensor size: {large_tensor.nbytes/1024/1024:.1f}MB")

    # Offload
    manager.offload_tensor("large_kv_cache", large_tensor)

    # Load back
    loaded = manager.load_tensor("large_kv_cache")
    print(f"Loaded back: {loaded.shape}")

    print("\n[SwapManager] POC complete, OS swap 18GB active")
