#!/usr/bin/env python3
"""Full 89-task terminal-bench eval via sbatch job arrays on FAIR cluster."""
import argparse, json, logging, os, shlex, subprocess, time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

TASK_IDS = [
    "adaptive-rejection-sampler", "bn-fit-modify", "break-filter-js-from-html",
    "build-cython-ext", "build-pmars", "build-pov-ray", "caffe-cifar-10",
    "cancel-async-tasks", "chess-best-move", "circuit-fibsqrt",
    "cobol-modernization", "code-from-image", "compile-compcert",
    "configure-git-webserver", "constraints-scheduling", "count-dataset-tokens",
    "crack-7z-hash", "custom-memory-heap-crash", "db-wal-recovery",
    "distribution-search", "dna-assembly", "dna-insert", "extract-elf",
    "extract-moves-from-video", "feal-differential-cryptanalysis",
    "feal-linear-cryptanalysis", "filter-js-from-html",
    "financial-document-processor", "fix-code-vulnerability", "fix-git",
    "fix-ocaml-gc", "gcode-to-text", "git-leak-recovery", "git-multibranch",
    "gpt2-codegolf", "headless-terminal", "hf-model-inference",
    "install-windows-3.11", "kv-store-grpc", "large-scale-text-editing",
    "largest-eigenval", "llm-inference-batching-scheduler",
    "log-summary-date-ranges", "mailman", "make-doom-for-mips",
    "make-mips-interpreter", "mcmc-sampling-stan", "merge-diff-arc-agi-task",
    "model-extraction-relu-logits", "modernize-scientific-stack",
    "mteb-leaderboard", "mteb-retrieve", "multi-source-data-merger",
    "nginx-request-logging", "openssl-selfsigned-cert", "overfull-hbox",
    "password-recovery", "path-tracing", "path-tracing-reverse",
    "polyglot-c-py", "polyglot-rust-c", "portfolio-optimization",
    "protein-assembly", "prove-plus-comm", "pypi-server",
    "pytorch-model-cli", "pytorch-model-recovery", "qemu-alpine-ssh",
    "qemu-startup", "query-optimize", "raman-fitting", "regex-chess",
    "regex-log", "reshard-c4-data", "rstan-to-pystan", "sam-cell-seg",
    "sanitize-git-repo", "schemelike-metacircular-eval", "sparql-university",
    "sqlite-db-truncate", "sqlite-with-gcov", "torch-pipeline-parallelism",
    "torch-tensor-parallelism", "train-fasttext", "tune-mjcf",
    "video-processing", "vulnerable-secret", "winning-avg-corewars",
    "write-compressor",
]
MODEL = "bedrock/us.anthropic.claude-opus-4-6-v1"
DATASET = "terminal-bench@2.0"

def _build_env():
    env = dict(os.environ)
    sd = Path(__file__).resolve().parent
    tr = sd.parent
    pp = [str(tr), str(tr / "agent")]
    ex = env.get("PYTHONPATH", "")
    if ex: pp.append(ex)
    env["PYTHONPATH"] = ":".join(pp)
    return env

def run_eval(nc):
    env = _build_env()
    ar = env.get("AGENT_ROOT_DIR", "")
    sld = env.get("SRUN_LD_LIBRARY_PATH", "")
    if ar:
        rdh = os.path.join(ar, "harbor_array")
        rdl = "/root/harbor_array"
    else:
        rdh = f"/checkpoint/ram-h100-2/tianhaowu/harbor_full_{int(time.time())}"
        rdl = rdh
    os.makedirs(f"{rdl}/logs", exist_ok=True)
    os.makedirs(f"{rdl}/jobs", exist_ok=True)
    rd = rdh
    with open(f"{rdl}/task_list.txt", "w") as f:
        for t in TASK_IDS: f.write(t + "\n")
    tlh = f"{rd}/task_list.txt"
    pp = env.get("PYTHONPATH", "")
    if ar:
        pp = pp.replace("/root", ar)
        ai = "agent:AgentHarness"
        hpp = f"{ar}/agent:{ar}:{pp}"
    else:
        sd = Path(__file__).resolve().parent
        tr = sd.parent
        ai = "agent:AgentHarness"
        hpp = f"{tr}/agent:{tr}:{pp}"
    script_path = f"{rdl}/array_task.sh"
    script_path_host = f"{rd}/array_task.sh"
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f'TASK=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "{tlh}")\n')
        f.write('if [[ -z "$TASK" ]]; then echo "ERROR: No task"; exit 1; fi\n')
        f.write('echo "=== Task $SLURM_ARRAY_TASK_ID: $TASK on $(hostname) ==="\n')
        f.write("set -a; source ~/.mobius/config.env 2>/dev/null || true; set +a\n")
        f.write(f"export PYTHONPATH={shlex.quote(hpp)}\n")
        f.write(f'JOB_DIR="{rd}/jobs/task_${{SLURM_ARRAY_TASK_ID}}"\n')
        f.write('mkdir -p "$JOB_DIR"\n')
        f.write(f"harbor run -d {shlex.quote(DATASET)} "
                f"--agent-import-path {shlex.quote(ai)} "
                f"-m {shlex.quote(MODEL)} "
                '--environment-import-path "mobius.harbor_apptainer:ApptainerEnvironment" '
                '-o "$JOB_DIR" '
                '--job-name "task_${SLURM_ARRAY_TASK_ID}_${TASK}" '
                "-n 1 --n-concurrent 1 "
                '-t "$TASK" -q 2>&1\n')
        f.write('echo "Exit: $?"\n')
    os.chmod(script_path, 0o755)
    nt = len(TASK_IDS)
    aspec = f"0-{nt-1}%{nc}"
    se = dict(os.environ); se["LD_LIBRARY_PATH"] = sld
    log.info("sbatch array: %d tasks, concurrency=%d, model=%s", nt, nc, MODEL)
    r = subprocess.run(
        ["sbatch", "--parsable", f"--array={aspec}", "--cpus-per-task=4", "--mem=8G",
         "--time=01:00:00", "--exclusive", "--partition=cpu", "--account=ram",
         "--qos=cpu_lowest", f"--output={rd}/logs/task_%a.log",
         f"--error={rd}/logs/task_%a.err", f"--chdir={rd}",
         f"--wrap=bash {script_path_host}"],
        capture_output=True, text=True, env=se)
    if r.returncode != 0:
        log.error("sbatch failed (rc=%d): %s", r.returncode, r.stderr.strip() or r.stdout.strip())
        return {t: 0.0 for t in TASK_IDS}
    jid = r.stdout.strip().split(";")[0]
    log.info("Submitted job array %s", jid)
    while True:
        sq = subprocess.run(["squeue", "-j", jid, "-h"], capture_output=True, text=True, env=se)
        if not sq.stdout.strip(): break
        done = sum(1 for i in range(nt) if list(Path(f"{rdl}/jobs/task_{i}").glob("**/result.json")))
        log.info("[array] %d/%d complete", done, nt)
        time.sleep(30)
    log.info("All jobs done for %s", jid)
    rewards = {}
    for i, tid in enumerate(TASK_IDS):
        jd = Path(f"{rdl}/jobs/task_{i}")
        rfs = list(jd.glob("**/result.json"))
        if not rfs: rewards[tid] = 0.0; continue
        ok = False
        for rf in rfs:
            try:
                d = json.loads(rf.read_text())
                if not d.get("task_name"): continue
                vr = d.get("verifier_result") or {}
                rw = vr.get("rewards") or {}
                reward = float(rw.get("reward", 0))
                rewards[tid] = reward
                log.info("  %s: %s (%.2f)", tid, "PASS" if reward > 0.5 else "FAIL", reward)
                ok = True; break
            except (json.JSONDecodeError, OSError) as e:
                log.warning("Parse error %s: %s", rf, e)
        if not ok: rewards[tid] = 0.0
    return rewards

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--concurrency", type=int, default=89)
    a = p.parse_args()
    log.info("Evaluating %d tasks, concurrency=%d", len(TASK_IDS), a.concurrency)
    rewards = run_eval(a.concurrency)
    correct = sum(1 for r in rewards.values() if r > 0.5)
    total = len(TASK_IDS)
    mean = sum(rewards.values()) / total if total > 0 else 0.0
    print("---")
    print(f"mean_pass_rate:   {mean:.3f}")
    print(f"correct:          {correct}")
    print(f"total:            {total}")

if __name__ == "__main__":
    main()
