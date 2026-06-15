from mmengine.config import read_base
from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

# Re-imports chat_rocketkv inside the worker subprocess so RocketKVChatBot is
# registered there too (the launcher already registers it in the driver).
custom_imports = dict(imports=['chat_rocketkv'], allow_failed_imports=False)

with read_base():
    from opencompass.configs.datasets.text2json.text2json import (
        text2json_datasets,
    )
    from opencompass.configs.datasets.needlebench_v2.needlebench_v2_128k.needlebench_v2_multi_retrieval_128k import (  # noqa: E501
        needlebench_en_datasets,
    )

datasets = needlebench_en_datasets + text2json_datasets

models = [
    dict(
        type='RocketKVChatBot',
        path='meta-llama/Llama-3.1-8B-Instruct',
        token_budget=2048,
        max_seq_len=128 * 1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        abbr='rocket;llama;token_budget=2048',
    ),
    dict(
        type='RocketKVChatBot',
        path='Qwen/Qwen3-4B-Instruct-2507',
        token_budget=0.015625,
        max_seq_len=128 * 1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        abbr='rocket;qwen3-4b;token_budget=0.015625',
    ),
    # Qwen3-MoE (e.g. Qwen3-30B-A3B): same wrapper, just give it more GPUs.
    # dict(
    #     type='RocketKVChatBot',
    #     path='Qwen/Qwen3-30B-A3B-Instruct-2507',
    #     token_budget=0.015625,
    #     max_seq_len=128 * 1024,
    #     batch_size=1,
    #     run_cfg=dict(num_gpus=2),
    #     abbr='rocket;qwen3-moe-30b;token_budget=0.015625',
    # ),
]

infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=1,
        keep_keys=['custom_imports'],
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=1,
        retry=2,
        task=dict(type=OpenICLInferTask),
    ),
)
