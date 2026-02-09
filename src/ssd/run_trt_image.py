import os, argparse
import numpy as np
from PIL import Image

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def trt_dtype_to_np(dtype: trt.DataType):
    if dtype == trt.float32: return np.float32
    if dtype == trt.float16: return np.float16
    if dtype == trt.int32:   return np.int32
    if dtype == trt.int8:    return np.int8
    if dtype == trt.bool:    return np.bool_
    raise ValueError(f"Unsupported TRT dtype: {dtype}")

def vol(shape):
    v = 1
    for d in shape:
        v *= int(d)
    return int(v)

def load_engine(path):
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("Failed to deserialize engine")
    return engine

def preprocess_pil_to_nchw(img: Image.Image, mode: str):
    """
    Returns float32 NCHW shape (1,3,300,300)
    mode:
      - tf:   (x - 127.5) / 127.5        -> [-1,1]  (Mobilenet 계열 흔함)
      - raw01: x / 255.0                 -> [0,1]
      - meanstd: (x-mean)/std (ImageNet)
    """
    img = img.convert("RGB").resize((300, 300), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32)  # HWC, 0..255
    if mode == "tf":
        x = (x - 127.5) / 127.5
    elif mode == "raw01":
        x = x / 255.0
    elif mode == "meanstd":
        x = x / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
    else:
        raise ValueError(f"unknown mode: {mode}")

    x = np.transpose(x, (2, 0, 1))          # CHW
    x = np.expand_dims(x, axis=0)           # NCHW
    return x.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--mode", default="tf", choices=["tf", "raw01", "meanstd"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    engine = load_engine(args.engine)
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create execution context")

    # I/O tensor names (your engine: 1 input + 2 outputs)
    io_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]

    # set input shape
    for name in io_names:
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            context.set_input_shape(name, (1, 3, 300, 300))

    # allocate buffers
    hbuf = {}
    dbuf = {}
    bindings = {}

    for name in io_names:
        shape = tuple(context.get_tensor_shape(name))
        dtype = engine.get_tensor_dtype(name)
        np_dtype = trt_dtype_to_np(dtype)

        hbuf[name] = np.empty(vol(shape), dtype=np_dtype)
        dbuf[name] = cuda.mem_alloc(hbuf[name].nbytes)
        bindings[name] = int(dbuf[name])

    # preprocess image -> input
    img = Image.open(args.image)
    x = preprocess_pil_to_nchw(img, args.mode)  # float32 NCHW

    # fill input host buffer (convert to engine dtype)
    for name in io_names:
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            in_dtype = trt_dtype_to_np(engine.get_tensor_dtype(name))
            hbuf[name] = x.astype(in_dtype).ravel()
            break

    stream = cuda.Stream()

    # H2D
    for name in io_names:
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            cuda.memcpy_htod_async(dbuf[name], hbuf[name], stream)

    # execute
    for name in io_names:
        context.set_tensor_address(name, bindings[name])

    ok = context.execute_async_v3(stream_handle=stream.handle)
    if not ok:
        raise RuntimeError("TensorRT execution failed")

    # D2H
    for name in io_names:
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            cuda.memcpy_dtoh_async(hbuf[name], dbuf[name], stream)

    stream.synchronize()

    # reshape + save outputs
    for name in io_names:
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            shape = tuple(context.get_tensor_shape(name))
            arr = np.array(hbuf[name]).reshape(shape)

            # 파일명 매핑: 너가 이미 쓰는 이름으로 고정
            if name == "Squeeze:0":
                out_path = os.path.join(args.out_dir, "Squeeze_0.npy")
            elif name == "concat_1:0":
                out_path = os.path.join(args.out_dir, "concat_1_0.npy")
            else:
                safe = name.replace("/", "_").replace(":", "_")
                out_path = os.path.join(args.out_dir, f"{safe}.npy")

            np.save(out_path, arr)
            print(f"[save] {name} -> {out_path} dtype={arr.dtype} shape={arr.shape} min={arr.min():.6g} max={arr.max():.6g} mean={arr.mean():.6g}")

if __name__ == "__main__":
    main()

