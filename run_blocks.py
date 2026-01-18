import argparse
import os
import subprocess
import json
from datetime import datetime

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ================== 你在这里直接写多个命令块 ==================
# 写法1：每块是一个 dict（推荐：有 name，日志不乱）
# BLOCKS = [
#     {
#         "name": "output_vil_iD_all",
#         "body": r"""
# python main_all2.py --dataset iDigits --batch-size 24  --num_tasks 5 --versatile_inc  --seed 42 --output_dir /root/VIL_main13/output5/output_vil_iD_all  --IC --thre 0.0 --beta 0.01 --use_cast_loss --k 2 --use_sce --sce_sensitivity 0.80
#
# """.strip()
#     },
#     {
#         "name": "output_cil_iD_all",
#         "body": r"""
# python main_all2.py --dataset iDigits --batch-size 24  --num_tasks 5   --seed 42 --output_dir /root/VIL_main13/output5/output_cil_iD_all  --IC --thre 0.0 --beta 0.01 --use_cast_loss --k 2 --use_sce --sce_sensitivity 0.80
#
# """.strip()
#     },
#     {
#         "name": "output_dil_iD_all",
#         "body": r"""
# python main_all2.py --dataset iDigits --batch-size 24  --num_tasks 4   --seed 42 --output_dir /root/VIL_main13/output5/output_dil_iD_all  --IC --thre 0.0 --beta 0.01 --use_cast_loss --k 2 --use_sce --sce_sensitivity 0.80 --domain_inc
#
# """.strip()
#     },
# ]

# BLOCKS = [
#     {
#         "name": "output_vil_Do_all",
#         "body": r"""
# python main_all.py --dataset DomainNet --batch-size 24  --num_tasks 5 --versatile_inc  --seed 42 --output_dir /root/VIL_main13/output5/output_vil_Do_all  --IC --thre 0.0 --beta 0.01 --use_cast_loss --k 2 --use_sce --sce_sensitivity 0.60
#
# """.strip()
#     },
#     {
#         "name": "output_cil_Do_all",
#         "body": r"""
# python main_all.py --dataset DomainNet --batch-size 24  --num_tasks 5   --seed 42 --output_dir /root/VIL_main13/output5/output_cil_Do_all  --IC --thre 0.0 --beta 0.01 --use_cast_loss --k 2 --use_sce --sce_sensitivity 0.60
#
# """.strip()
#     },
#     {
#         "name": "output_dil_Do_all",
#         "body": r"""
# python main_all.py --dataset DomainNet --batch-size 24  --num_tasks 6   --seed 42 --output_dir /root/VIL_main13/output5/output_dil_Do_all  --IC --thre 0.0 --beta 0.01 --use_cast_loss --k 2 --use_sce --sce_sensitivity 0.60 --domain_inc
#
# """.strip()
#     },
# ]

BLOCKS = [
    {
        "name": "output_vil_CO_all",
        "body": r"""
python main_all2.py --dataset CORe50 --batch-size 24  --num_tasks 5 --versatile_inc  --seed 42 --output_dir /root/VIL_main13/output5/output_vil_CO_all  --IC --thre 0.0 --beta 0.01 --use_cast_loss --k 2 --use_sce --sce_sensitivity 1.20

""".strip()
    },
    {
        "name": "output_cil_CO_all",
        "body": r"""
python main_all2.py --dataset CORe50 --batch-size 24  --num_tasks 5   --seed 42 --output_dir /root/VIL_main13/output5/output_cil_CO_all  --IC --thre 0.0 --beta 0.01 --use_cast_loss --k 2 --use_sce --sce_sensitivity 1.20

""".strip()
    },
    {
        "name": "output_dil_CO_all",
        "body": r"""
python main_all2.py --dataset CORe50 --batch-size 24  --num_tasks 1   --seed 42 --output_dir /root/VIL_main13/output5/output_dil_CO_all  --IC --thre 0.0 --beta 0.01 --use_cast_loss --k 2 --use_sce --sce_sensitivity 1.20 --versatile_inc --random_inc --n_tasks 8

""".strip()
    },
]


def load_state(path: str):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"done": {}, "history": []}

def save_state(path: str, state: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def normalize_blocks(blocks):
    out = []
    for i, b in enumerate(blocks):
        if isinstance(b, str):
            out.append({"name": f"block_{i:03d}", "body": b.strip()})
        elif isinstance(b, dict):
            name = b.get("name") or f"block_{i:03d}"
            body = (b.get("body") or "").strip()
            out.append({"name": name, "body": body})
        else:
            raise TypeError(f"Unsupported block type at index {i}: {type(b)}")
    return out

def run_block(name: str, body: str, log_dir: str, stop_on_fail: bool):
    print(f"\n[{now()}] >>> START {name}")

    # 保证能找到 main_all2.py + 尽量实时输出
    full_body = f"""

export PYTHONUNBUFFERED=1
{body}
""".strip()

    cmd = ["bash", "-lc", full_body]

    try:
        # 关键：不重定向 stdout/stderr => 终端会像 sweep 一样滚动输出
        subprocess.run(cmd, check=True)
        print(f"[{now()}] <<< OK  {name}")
        return True, 0
    except subprocess.CalledProcessError as e:
        print(f"[{now()}] !!! FAIL {name} rc={e.returncode}")
        if stop_on_fail:
            raise SystemExit(e.returncode)
        return False, e.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default="logs", help="log dir")
    ap.add_argument("--state", default="run_state.json", help="resume state file")
    ap.add_argument("--continue_on_fail", action="store_true", help="continue after failures")
    ap.add_argument("--rerun_done", action="store_true", help="rerun blocks already done")
    args = ap.parse_args()

    blocks = normalize_blocks(BLOCKS)
    state = load_state(args.state)

    stop_on_fail = not args.continue_on_fail

    print(f"Total blocks: {len(blocks)}")
    for blk in blocks:
        name, body = blk["name"], blk["body"]

        if not body:
            print(f"[{now()}] SKIP (empty): {name}")
            continue

        if (not args.rerun_done) and state["done"].get(name) is True:
            print(f"[{now()}] SKIP (already done): {name}")
            continue

        ok, rc = run_block(name, body, args.log_dir, stop_on_fail=stop_on_fail)
        state["done"][name] = ok
        state["history"].append({"time": now(), "name": name, "ok": ok, "rc": rc})
        save_state(args.state, state)

    print(f"\n[{now()}] ALL DONE")

if __name__ == "__main__":
    main()
