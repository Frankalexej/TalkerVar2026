"""Execute a notebook with this Python's kernel, saving outputs after each cell.

Uses jupyter_client, already present in the available Anaconda environments.
Example: python src/execute_notebook.py src/train_vowels.ipynb
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from jupyter_client import KernelManager


def execute(path: Path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="talkervar-kernel-") as directory:
        manager = KernelManager(connection_file=str(Path(directory) / "kernel.json"))
        # Use the current Anaconda environment even if the global kernelspec differs.
        manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        kernel_environment = dict(os.environ)
        kernel_environment["IPYTHONDIR"] = str(Path(directory) / "ipython")
        manager.start_kernel(cwd=str(path.parent), env=kernel_environment)
        client = manager.blocking_client()
        client.start_channels()
        try:
            client.wait_for_ready(timeout=60)
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                cell["outputs"] = []
                print(f"Executing cell {index + 1}/{len(notebook['cells'])}", flush=True)

                def capture(message):
                    kind, content = message["msg_type"], message["content"]
                    if kind == "stream":
                        cell["outputs"].append({"output_type": kind, "name": content["name"], "text": content["text"]})
                        print(content["text"], end="", flush=True)
                    elif kind in ("display_data", "execute_result"):
                        output = {"output_type": kind, "data": content["data"], "metadata": content["metadata"]}
                        if kind == "execute_result":
                            output["execution_count"] = content["execution_count"]
                        cell["outputs"].append(output)
                    elif kind == "error":
                        cell["outputs"].append({"output_type": kind, **{k: content[k] for k in ("ename", "evalue", "traceback")}})

                reply = client.execute_interactive("".join(cell["source"]), timeout=7200, output_hook=capture)
                cell["execution_count"] = reply["content"]["execution_count"]
                path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
                if reply["content"]["status"] != "ok":
                    raise RuntimeError(f"Notebook cell {index + 1} failed: {reply['content']}")
        finally:
            client.stop_channels()
            manager.shutdown_kernel(now=True)
    print(f"Executed notebook saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    execute(parser.parse_args().notebook.resolve())
