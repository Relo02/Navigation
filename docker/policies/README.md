# `policies/` — model assets (mounted to `/workspace/policies`)

Drop exported policy model files here. This directory is bind-mounted into the
`amo_policy` container at `/workspace/policies` and surfaced via the
`AMO_POLICY_PATH` env var.

Note: the AMO RL gait itself is loaded by RoboJuDo from its own asset tree
(`RoboJuDo/assets/models/g1/amo/amo_jit.pt` + adapter + norm stats), not from a
single file here — `amo/amo_inference.py` drives `robojudo.policy.AMOPolicy`.
Use this folder for any additional/alternative exported models you want the
container to see.
