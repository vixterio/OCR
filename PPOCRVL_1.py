"""PaddleOCR-VL — original script, fixed to run on this Mac.

Was: vl_rec_backend="vllm-server" against http://127.0.0.1:8080/v1.
vLLM has no macOS/Metal build, so that server could never exist here and every
run died with "Exception from the 'vlm' worker: Connection error."

Now: the same client/server split, but the server is mlx-vlm serving the model
4-bit quantised on the M2 GPU. Start it first (see start_server.sh), then:

    ./safe_run.sh PPOCRVL_1.py demo.png
"""
import os
import sys

import _compat

from paddleocr import PaddleOCRVL

_compat.apply()  # Python 3.9 asyncio fix; no-op on 3.10+

pipeline = PaddleOCRVL(
    pipeline_version="v1",
    vl_rec_backend="mlx-vlm-server",
    vl_rec_server_url="http://127.0.0.1:8080/v1",
    vl_rec_api_model_name="mlx-community/PaddleOCR-VL-4bit",
    vl_rec_max_concurrency=1,
    use_queues=False,
)

output = pipeline.predict(sys.argv[1] if len(sys.argv) > 1 else "demo.png")

os.makedirs("output", exist_ok=True)
for res in output:
    res.print()
    res.save_to_json(save_path="output")
    res.save_to_markdown(save_path="output")
