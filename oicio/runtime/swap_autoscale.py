"""
OICIO Swap Autoscale: 10GB -> 20GB -> 30GB ...
Credits: deepRcurs Labs @deeprcurs / Mzed Imamkh @mzedimamkh

Implementasi aturan: jika RAM kurang, swap 10GB, 20GB, 30GB dan seterusnya
"""

import os
import subprocess
import sys

def get_swap_total_gb():
    try:
        result = subprocess.run(["free", "-g"], capture_output=True, text=True)
        for line in result.stdout.split("\n"):
            if "Swap" in line:
                parts = line.split()
                return int(parts[1])
    except:
        pass
    return 0

def get_disk_free_gb():
    try:
        result = subprocess.run(["df", "-BG", "/"], capture_output=True, text=True)
        lines = result.stdout.split("\n")
        # /dev/root line
        for line in lines:
            if "/dev/root" in line:
                parts = line.split()
                # Avail column
                avail = parts[3].replace("G", "")
                return int(avail)
    except:
        pass
    return 0

def create_swap(size_gb, name):
    path = f"/home/user/.cache/{name}"
    if os.path.exists(path):
        print(f"[Autoscale] {name} already exists")
        return True

    free_gb = get_disk_free_gb()
    if free_gb < size_gb + 1:
        print(f"[Autoscale] Not enough disk for {size_gb}GB (free {free_gb}GB), cleaning pip cache...")
        os.system("rm -rf /home/user/.cache/pip /home/user/.cache/oicio_swap*")
        free_gb = get_disk_free_gb()
        if free_gb < size_gb + 1:
            print(f"[Autoscale] Still not enough disk after clean, free {free_gb}GB")
            return False

    print(f"[Autoscale] Creating {size_gb}GB swap {name} (free {free_gb}GB)...")
    os.system(f"fallocate -l {size_gb}G {path}")
    os.system(f"chmod 600 {path}")
    os.system(f"sudo /sbin/mkswap {path} 2>&1 | head -2")
    os.system(f"sudo /sbin/swapon {path} 2>&1 | head -2")

    swap_total = get_swap_total_gb()
    print(f"[Autoscale] Swap total now: {swap_total}GB")
    return True

def autoscale_to_target(target_gb):
    """
    Autoscale swap to target: 10, 20, 30...
    """
    print(f"\n=== Autoscaling Swap to {target_gb}GB ===")
    current = get_swap_total_gb()
    print(f"Current swap: {current}GB, Target: {target_gb}GB")

    step = 0
    while current < target_gb:
        step += 1
        needed = target_gb - current
        # Create in 5GB or 10GB chunks
        chunk = min(10, needed)
        if chunk < 1:
            chunk = needed

        name = f"swap_autoscale_{current+chunk}gb_{step}"
        success = create_swap(chunk, name)
        if not success:
            print(f"[Autoscale] Failed to create {chunk}GB, stopping")
            break

        current = get_swap_total_gb()
        print(f"[Autoscale] Progress: {current}GB / {target_gb}GB")

        if step > 10:
            print("[Autoscale] Too many steps, stopping")
            break

    print(f"\n[Autoscale] Final swap: {get_swap_total_gb()}GB")
    os.system("free -h")
    os.system("cat /proc/swaps")

if __name__ == "__main__":
    print("=== OICIO Swap Autoscale POC ===")
    print("Aturan: jika RAM kurang, swap 10GB, 20GB, 30GB dan seterusnya")

    # Check current
    os.system("free -h")
    os.system("cat /proc/swaps")
    os.system("df -h | head -5")

    # Try autoscale to 20GB
    autoscale_to_target(20)

    # If enough disk, try 30GB
    free_gb = get_disk_free_gb()
    if free_gb > 11:
        autoscale_to_target(30)
    else:
        print(f"\n[Autoscale] Disk free {free_gb}GB not enough for 30GB target, need to free more")
        print("[Autoscale] In production, you would have larger disk, can scale to 30GB+")
