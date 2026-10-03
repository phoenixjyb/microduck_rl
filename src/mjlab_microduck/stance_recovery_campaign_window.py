"""Renewed October 3 authorization; the expired 08:00 window stays immutable."""
import math
import time

from mjlab_microduck.first_attempt_smoke import require

PROTOCOL = 'microduck-simulation-window-20261003-2000-v1'
START = 1790988600  # 2026-10-03 08:50 Asia/Shanghai
CUTOFF = 1791028800  # 2026-10-03 20:00 Asia/Shanghai


def check(*, reserve_seconds=0, now=None):
    current = time.time() if now is None else now
    require(type(current) in (int, float) and math.isfinite(current), 'finite renewed-window clock')
    require(type(reserve_seconds) in (int, float) and math.isfinite(reserve_seconds)
            and reserve_seconds >= 0, 'finite nonnegative complete-work reserve')
    require(START <= current and current + reserve_seconds < CUTOFF,
            'renewed Duck work and complete closeout before fixed 20:00 Shanghai cutoff')


def declaration():
    return dict(protocol=PROTOCOL, start_unix=START, cutoff_unix=CUTOFF,
                timezone='Asia/Shanghai', no_physical_motion=True,
                preserve_expired_protocols=True, restore_protected_services=False)
