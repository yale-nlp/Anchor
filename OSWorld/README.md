# OSWorld (Anchor)
This is the OSWorld environment **as used in this repo** for generating new trajectories via **branch expansion**.

If you’re looking for the full benchmark docs/leaderboard/etc, see the upstream OSWorld project. This README focuses only on the setup needed to run our pipeline.

## ✅ What we do here
- Start from existing OSWorld trajectories
- Choose “branch points” (intermediate steps)
- Use an LLM to **propose new follow-up tasks** from the current UI state
- Use an executor agent (Claude or Qwen3-VL) to **complete the new task**
- Save the resulting **new trajectories** to disk

The main script is `generate_branch_trajectories.py`.

## 💾 Install (Python)
From `OSWorld/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 🖥️ Environment provider setup (our use cases)
OSWorld runs against a VM provider. In this repo we primarily use **AWS** (recommended for parallelism), and optionally **VMware** for local development on macOS.

### AWS (recommended)
- You need working AWS credentials in your environment (prefer `AWS_PROFILE`; avoid long-lived `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` in `.env`).
- You must set the following **required** networking variables (the AWS VM manager will error if these are missing):
  - `AWS_REGION` (or pass `--region`)
  - `AWS_SUBNET_ID`
  - `AWS_SECURITY_GROUP_ID`
- Optional (but recommended for debugging): `AWS_KEY_NAME` (EC2 keypair name)
- Choose region + screen size (defaults in the script are `us-east-1` and 1920×1080).

Example flags you’ll commonly use:
- `--provider_name aws`
- `--region us-east-1`
- `--screen_width 1920 --screen_height 1080`
- `--aws_profile <profile>` (optional)

Notes:
- The security group must allow OSWorld ports (VNC, backend, Chrome control). See `desktop_env/providers/aws/AWS_GUIDELINE.md` for the exact inbound/outbound rules we use.
- **Region env var name matters**: this code requires `AWS_REGION` to be set (setting only `AWS_DEFAULT_REGION` is not enough).
- **AMI selection is hardcoded**: the VM allocator chooses the AMI from `desktop_env/providers/aws/manager.py` (`IMAGE_ID_MAP`) based on region + screen size. If your region/screen size isn’t in the map, you must update it.
- **Instance TTL auto-termination** (recommended): instances are scheduled for termination via EventBridge Scheduler by default. If your AWS role can’t create IAM roles/policies, set `AWS_SCHEDULER_ROLE_ARN` explicitly or disable auto-create (see below).

### VMware (local dev on macOS)
- Install **VMware Fusion** on Apple Silicon (or Workstation Pro elsewhere).
- Ensure `vmrun` is available on your PATH.

Example flags:
- `--provider_name vmware`
- `--path_to_vm /path/to/Ubuntu.vmx`

## 🔐 API keys (`.env`)
Create `OSWorld/.env` (the scripts call `load_dotenv()` automatically).

```bash
# AWS (required for --provider_name aws)
AWS_REGION="us-east-1"
AWS_SUBNET_ID="subnet-xxxxxxxxxxxxxxxxx"
AWS_SECURITY_GROUP_ID="sg-xxxxxxxxxxxxxxxxx"
AWS_KEY_NAME="oskey"   # optional, but recommended for SSH/debugging

# AWS (optional)
AWS_INSTANCE_TYPE="t3.medium"          # default is in code; set if you want different hardware
ENABLE_TTL="true"                      # default: true
DEFAULT_TTL_MINUTES="180"              # default: 180
AWS_SCHEDULER_ROLE_ARN=""              # if set, used for EventBridge Scheduler
AWS_SCHEDULER_ROLE_NAME="osworld-scheduler-ec2-terminate"  # used to derive ARN if ARN not set
AWS_AUTO_CREATE_SCHEDULER_ROLE="true"  # default: true (may require IAM create-role permissions)

# Task generation (Azure OpenAI)
AZURE_OPENAI_API_KEY="..."
AZURE_OPENAI_ENDPOINT="https://<your-resource-name>.openai.azure.com/"
AZURE_OPENAI_API_VERSION="2024-xx-xx"

# Executor (choose one)
ANTHROPIC_API_KEY="..."      # for --executor_agent claude
DASHSCOPE_API_KEY="..."      # for --executor_agent qwen3vl (DashScope)
```

Azure OpenAI note:
- In Azure OpenAI, the `model="..."` field is typically your **deployment name**. This code calls Azure with `model="gpt-5.1-chat"`.
- Either create an Azure deployment named `gpt-5.1-chat`, or edit `generate_branch_trajectories.py` and change the `model=` values to your deployment name(s).

Security note:
- Keep `.env` out of git (don’t commit credentials), and rotate keys if they ever leak.

## 🧾 Generate new trajectories (branch expansion)
### Inputs
- **Task configs**: `evaluation_examples/examples/<domain>/<task_id>.json`
- **Branch points**: `branch_point_selected/*.json`
  - A *branch point* is the step you want to **branch from** (i.e. an intermediate point along an existing trajectory where we want to generate new follow-up tasks).
  - We include an example branch-point file in `branch_point_selected/` you can copy/modify.
- **Existing seed trajectories**: a directory tree under `--trajectory_base_dir` containing `traj.jsonl` (+ screenshots alongside it is best)
  - Easy option: download seed trajectories from the Hugging Face dataset [xlangai/ubuntu_osworld_verified_trajs](https://huggingface.co/datasets/xlangai/ubuntu_osworld_verified_trajs).

Each branch-point JSON looks like:

```json
{
  "task_id": "<uuid>",
  "app_type": "<domain>",
  "model": "Claude-4.5",
  "analysis": {
    "branch_points": [
      { "after_step": 4, "num_tasks": 5, "reason": "...", "new_task_examples": ["..."] }
    ]
  }
}
```

### Run
From `OSWorld/`:

```bash
python generate_branch_trajectories.py \
  --branch_analysis_dir branch_point_selected \
  --trajectory_base_dir /path/to/your/trajectory_data_root \
  --output_dir branch_generated \
  --provider_name aws \
  --region us-east-1 \
  --os_type Ubuntu \
  --screen_width 1920 \
  --screen_height 1080 \
  --max_continuation_steps 30 \
  --executor_agent claude
```

Helpful flags:
- **Select tasks**: `--task_ids <id1,id2,...>`
- **Parallelize**: `--num_envs 4`
- **Debug**: `--limit_tasks 1 --limit_branches_per_task 1 --max_continuation_steps 5`

### Outputs
Written under `--output_dir` (default `branch_generated/`):
- Per generated trajectory: `metadata.json`, `trajectory.jsonl`, and `screenshots/`
- Run artifacts: `generation_summary.json`, `proposed_tasks.json` (global dedupe), plus logs under `logs/`

### Troubleshooting
- **Trajectory not found**: adjust `--trajectory_base_dir` and ensure your `traj.jsonl` paths include the `task_id`.
- **Task config not found**: ensure `app_type` matches a folder under `evaluation_examples/examples/`.
- **Replay state mismatch**: the script writes a mismatch record (images + metadata) into `--output_dir` for debugging.