"""
restart_runner.py — Stops the experiment runner of THIS project (and only its own
worker processes) and relaunches it detached. Finished stages are cached, so nothing
that has been written to disk is recomputed.

  python revision/restart_runner.py [--wait-for N wustl latent searches] [--plan main] [--workers 10]
"""

import argparse
import glob
import os
import subprocess
import sys
import time

import psutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEEP_PLAN = None


def own_runners():
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (p.info["name"] or "").lower()
            args = p.info["cmdline"] or []
        except Exception:
            continue
        # only Python interpreters whose script argument is this project's run_cv.py
        if name.startswith("python") and any(a.replace("\\", "/").endswith("revision/run_cv.py") for a in args[1:3]) \
                and p.pid != os.getpid() and not (KEEP_PLAN and KEEP_PLAN in args):
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-glob", default=None, help="wait until this glob matches --wait-count files")
    ap.add_argument("--wait-count", type=int, default=0)
    ap.add_argument("--plan", default="main")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--log", default="runner.log")
    ap.add_argument("--keep-plan", default=None, help="do not stop runners started with this --plan value")
    args = ap.parse_args()
    global KEEP_PLAN
    KEEP_PLAN = args.keep_plan
    if args.wait_glob:
        while len(glob.glob(os.path.join(ROOT, args.wait_glob))) < args.wait_count:
            time.sleep(15)
    victims = []
    for r in own_runners():
        try:
            victims += r.children(recursive=True) + [r]
        except psutil.NoSuchProcess:
            pass
    for v in victims:
        try:
            v.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(victims, timeout=20)
    print(f"stopped {len(victims)} processes", flush=True)
    log = open(os.path.join(ROOT, "results", "revision", "_logs", args.log), "w")
    flags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    subprocess.Popen([sys.executable, os.path.join(ROOT, "revision", "run_cv.py"), "--plan", args.plan,
                      "--workers", str(args.workers)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, creationflags=flags, close_fds=True)
    print("runner relaunched", time.strftime("%H:%M:%S"), flush=True)


if __name__ == "__main__":
    main()
