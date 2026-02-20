"""
dpu_runner.py

"""
import numpy as np
import vart
import xir
import time


class DpuRunner:
    """
    Wraps VART Runner for YOLOv3 inference on DPU.
    """

    def __init__(self, xmodel_path):

        print(f"Loading xmodel: {xmodel_path}")
        graph = xir.Graph.deserialize(xmodel_path)
        subgraphs = graph.get_root_subgraph().toposort_child_subgraph()

        dpu_sg = None
        for sg in subgraphs:
            if sg.has_attr("device") and sg.get_attr("device") == "DPU":
                dpu_sg = sg
                break

        if dpu_sg is None:
            raise RuntimeError("No DPU subgraph found in xmodel")

        self._runner = vart.Runner.create_runner(dpu_sg, "run")

        # Input tensor info — preserves fix_point scaling
        in_tensors = self._runner.get_input_tensors()
        self._in_dims = tuple(in_tensors[0].dims)
        self._in_fixpos = in_tensors[0].get_attr("fix_point")
        self._in_scale = 2.0 ** self._in_fixpos
        _, self.model_h, self.model_w, _ = self._in_dims

        # Output tensor info — preserves fix_point scaling
        out_tensors = self._runner.get_output_tensors()
        self._out_info = []
        for t in out_tensors:
            self._out_info.append({
                "dims": tuple(t.dims),
                "scale": 2.0 ** t.get_attr("fix_point"),
            })
            print(f"  Output: {tuple(t.dims)} fix={t.get_attr('fix_point')}")

        print(f"  Input: {self.model_w}x{self.model_h} fix={self._in_fixpos}")
        print(f"  DPU runner ready.")

    @property
    def num_outputs(self):
        return len(self._out_info)

    @property
    def output_shapes(self):
        return [info["dims"] for info in self._out_info]

    def preprocess(self, frame):

        import cv2
        img = cv2.resize(frame, (self.model_w, self.model_h))
        img = img.astype(np.float32) * (self._in_scale / 255.0)
        img = np.clip(img, -128, 127).astype(np.int8)
        return img.reshape(self._in_dims)

    def run(self, frame):

        input_data = self.preprocess(frame)

        # Allocate fresh output buffers each call (avoids segfault)
        out_bufs = [np.zeros(info["dims"], dtype=np.int8)
                    for info in self._out_info]

        # DPU execution — preserves async pattern
        job_id = self._runner.execute_async([input_data], out_bufs)
        self._runner.wait(job_id)

        # Dequantize: INT8 → float32
        float_outputs = []
        for i, buf in enumerate(out_bufs):
            float_out = buf.astype(np.float32) / self._out_info[i]["scale"]
            float_outputs.append(float_out)

        return float_outputs

    def __del__(self):
        if hasattr(self, '_runner'):
            del self._runner