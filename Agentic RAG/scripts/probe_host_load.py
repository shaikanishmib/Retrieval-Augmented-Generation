"""Record an observational host snapshot during a slow run without changing power policy."""
import ctypes
import subprocess
import time
import psutil
from agentic_rag.artifacts import ROOT,write

class PowerStatus(ctypes.Structure):
    _fields_=[('ac_line_status',ctypes.c_ubyte),('battery_flag',ctypes.c_ubyte),('battery_percent',ctypes.c_ubyte),('reserved',ctypes.c_ubyte),('battery_life_seconds',ctypes.c_ulong),('full_battery_seconds',ctypes.c_ulong)]

if __name__=='__main__':
    status=PowerStatus()
    available=bool(ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)))
    samples=[]
    for _ in range(5):
        samples.append(subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu,pstate,utilization.gpu,memory.used,power.draw,clocks.sm','--format=csv,noheader'],text=True).strip())
        time.sleep(.2)
    result=dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),available_ram_bytes=psutil.virtual_memory().available,
                cpu_percent=psutil.cpu_percent(interval=1),ac_line_status=status.ac_line_status if available else None,
                battery_percent=status.battery_percent if available else None,gpu_samples=samples,
                scope='Observational snapshot during reactive HotpotQA evaluation; no controlled cause of latency variation established and no host settings changed.')
    write(ROOT/f'reports/host-observations/{time.time_ns()}.json',result)
    print(result)
