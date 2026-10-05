"""Read actual accelerator FDs and locality; never initialize a device here."""
import os
from pathlib import Path
import re


def cpu_set(text):
    values = set()
    for part in text.strip().split(','):
        if not part:
            continue
        ends = part.split('-')
        values.update(range(int(ends[0]), int(ends[-1])+1))
    return values


def verify_device():
    devices = {}
    for fd in Path('/proc/self/fd').iterdir():
        try:
            target = fd.readlink()
            if target.parent == Path('/dev/accel') and re.fullmatch(r'accel\d+', target.name):
                pci = (Path('/sys/class/accel')/target.name/'device').resolve()
                devices[target.name] = dict(bus_id=pci.name,
                    numa_node=int((pci/'numa_node').read_text()),
                    local_cpus=sorted(cpu_set((pci/'local_cpulist').read_text())))
        except FileNotFoundError:
            continue
    expected = os.getenv('GAUDI_KERNELS_BUS_ID')
    if expected:
        assert devices and {d['bus_id'] for d in devices.values()} == {expected}, (
            'Actual accelerator differs from assigned PCI device', expected, devices)
        actual = next(iter(devices.values()))
        assert os.sched_getaffinity(0) <= set(actual['local_cpus']), (
            'Probe CPU affinity is not local to its actual accelerator', actual)
    return dict(module_id=os.getenv('HLS_MODULE_ID'), visible_modules=os.getenv('HABANA_VISIBLE_MODULES'),
                expected_bus_id=expected, device_fds=devices,
                cpu_affinity=sorted(os.sched_getaffinity(0)))
