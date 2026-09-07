"""Bounded telemetry settling; never wait through a competing compute workload."""

import json
import subprocess
import time

from mjlab_microduck.first_attempt_smoke import require

SERVICES=("recomo-ai-mission-vllm.service","recomo-ai-mission-subject-model-worker.service")


def wait_idle(*, reader=None, sleep=time.sleep, now=time.monotonic, timeout=10):
    require(type(timeout) in (int,float) and 2<=timeout<=10,"bounded idle-settling window")
    start=now();samples=[];consecutive=0
    def probe(*args):
        remaining=timeout-(now()-start)
        require(remaining>0,"idle probe deadline")
        if reader is not None:return reader(*args)
        return subprocess.check_output(args,text=True,timeout=min(2,remaining)).strip()
    for _ in range(11):
        states={s:probe("systemctl","show",s,"-p","ActiveState","--value") for s in SERVICES}
        pids=probe("nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits")
        raw=probe("nvidia-smi","--query-gpu=utilization.gpu,temperature.gpu,memory.used","--format=csv,noheader,nounits")
        utilization,temperature,memory=map(int,raw.split(","))
        sample=dict(elapsed_s=now()-start,services=states,compute_pids=pids,
                    utilization_percent=utilization,temperature_c=temperature,memory_mib=memory)
        samples.append(sample)
        require(all(s=="inactive" for s in states.values()) and not pids
                and 0<=utilization<=100 and 0<=temperature<80 and 0<=memory<100,
                "occupied/unsafe GPU, no wait: "+json.dumps(sample))
        consecutive=consecutive+1 if utilization==0 else 0
        if consecutive==2:
            require(now()-start<=timeout,"idle probe exceeded deadline")
            return dict(protocol="two-idle-samples-v1",samples=samples,timeout_s=timeout)
        if now()-start+1>timeout:break
        sleep(1)
    raise ValueError("idle telemetry did not settle: "+json.dumps(samples))
