"""Read-only GPU telemetry gate; never kill or wait through another workload."""

import pytest

from mjlab_microduck.gpu_idle_gate import wait_idle


def probes(utilizations,*,state="inactive",pids="",temperature=45,memory=12):
    clock=[0.];values=iter(utilizations);reads=[]
    def read(*args):
        reads.append(args)
        if args[0]=="systemctl":return state
        if "--query-compute-apps=pid" in args:return pids
        return f"{next(values)}, {temperature}, {memory}"
    def sleep(seconds):clock[0]+=seconds
    return read,sleep,lambda:clock[0],reads


def test_requires_two_idle_samples_and_retains_transient_utilization():
    read,sleep,now,reads=probes([30,0,20,0,0])
    r=wait_idle(reader=read,sleep=sleep,now=now)
    assert [s["utilization_percent"] for s in r["samples"]]==[30,0,20,0,0]
    assert r["samples"][-1]["elapsed_s"]==4
    assert len(reads)==20


@pytest.mark.parametrize("kw",[dict(state="active"),dict(pids="1234"),dict(temperature=80),dict(memory=100)])
def test_busy_or_unsafe_never_waits(kw):
    read,sleep,now,reads=probes([0],**kw)
    with pytest.raises(ValueError,match="occupied/unsafe GPU"):wait_idle(reader=read,sleep=sleep,now=now)
    assert now()==0


def test_stale_utilization_is_bounded_and_never_launches_anything():
    read,sleep,now,reads=probes([30]*20)
    with pytest.raises(ValueError):wait_idle(reader=read,sleep=sleep,now=now,timeout=3)
    assert now()<=3
    assert all(a[0] in ("systemctl","nvidia-smi") for a in reads)


def test_default_subprocess_timeout_never_exceeds_budget(monkeypatch):
    from mjlab_microduck import gpu_idle_gate as gate
    read,sleep,now,_=probes([0,0]);timeouts=[]
    def check(args,**kwargs):
        timeouts.append(kwargs["timeout"]);return read(*args)
    monkeypatch.setattr(gate.subprocess,"check_output",check)
    gate.wait_idle(sleep=sleep,now=now,timeout=3)
    assert len(timeouts)==8 and all(0<x<=2 for x in timeouts)
