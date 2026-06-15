#!/usr/bin/env python
"""Launcher that runs OpenCompass with the RocketKV model registered.

Usage (from the RocketKV folder):

    HF_HOME=/mnt/LLM CUDA_VISIBLE_DEVICES=0 \
        python opencompass_run.py run_config.py -w outputs

Running this file puts /workspace/RocketKV on sys.path automatically (it is the
script's directory), so the driver can ``import chat_rocketkv`` via the config's
``custom_imports``. The worker subprocess OpenCompass spawns is, however, a
direct ``python .../opencompass/tasks/openicl_infer.py`` run whose sys.path does
NOT include this folder -- so we export it via PYTHONPATH, which the subprocess
inherits, letting its ``custom_imports`` find chat_rocketkv too.
"""

import os

os.environ['PYTHONPATH'] = os.pathsep.join(
    p for p in (os.path.dirname(os.path.abspath(__file__)),
                os.environ.get('PYTHONPATH', '')) if p)

if __name__ == '__main__':
    from opencompass.cli.main import main
    main()
