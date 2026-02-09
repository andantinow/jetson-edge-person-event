#!/usr/bin/env python3
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401


class TRTInferYOLO:
    """
    TensorRT 10.3 runner (explicit tensors) for YOLOv8 ONNX-exported engine.

    Engine I/O (your engine):
      input : images   (1,3,640,640) float32
      output: output0  (1,84,8400)   float32
    """
    def __init__(self, engine_path: str, input_name="images", output_name="output0"):
        self.engine_path = engine_path
        self.input_name = input_name
        self.output_name = output_name

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            plan = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(plan)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize engine")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        self.stream = cuda.Stream()
        self._alloc()

    def _alloc(self):
        in_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        out_shape = tuple(self.engine.get_tensor_shape(self.output_name))

        self.in_shape = in_shape
        self.out_shape = out_shape

        in_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
        out_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))

        self.in_host = np.empty(np.prod(in_shape), dtype=in_dtype).reshape(in_shape)
        self.out_host = np.empty(np.prod(out_shape), dtype=out_dtype).reshape(out_shape)

        self.in_dev = cuda.mem_alloc(self.in_host.nbytes)
        self.out_dev = cuda.mem_alloc(self.out_host.nbytes)

        # Explicit tensor addresses (TRT10)
        self.context.set_tensor_address(self.input_name, int(self.in_dev))
        self.context.set_tensor_address(self.output_name, int(self.out_dev))

    def infer(self, inp: np.ndarray):
        """
        inp: (1,3,640,640) float32 contiguous
        returns: output0 copy, (t_h2d_ms, t_infer_d2h_ms)
        """
        if inp.shape != self.in_shape:
            raise ValueError(f"bad input shape: {inp.shape} expected {self.in_shape}")

        if inp.dtype != self.in_host.dtype:
            inp = np.ascontiguousarray(inp, dtype=self.in_host.dtype)
        if not inp.flags["C_CONTIGUOUS"]:
            inp = np.ascontiguousarray(inp)

        t0 = cuda.Event()
        t1 = cuda.Event()
        t2 = cuda.Event()

        t0.record(self.stream)
        cuda.memcpy_htod_async(self.in_dev, inp, self.stream)
        t1.record(self.stream)

        ok = self.context.execute_async_v3(stream_handle=int(self.stream.handle))
        if not ok:
            raise RuntimeError("execute_async_v3 failed")

        cuda.memcpy_dtoh_async(self.out_host, self.out_dev, self.stream)
        t2.record(self.stream)
        self.stream.synchronize()

        # Event time is in ms
        t_h2d = t0.time_till(t1)
        t_infer_d2h = t1.time_till(t2)

        return self.out_host.copy(), float(t_h2d), float(t_infer_d2h)

