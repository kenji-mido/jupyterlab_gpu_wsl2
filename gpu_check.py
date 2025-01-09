try:
    import cv2
except ImportError:
    cv2 = None

try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import tensorrt as trt
except ImportError:
    trt = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    cuda = None

def check_opencv_gpu():
    opencv_info = ["============ OpenCV ============"]
    if cv2:
        if cv2.getBuildInformation().find('CUDA') != -1:
            opencv_info.append("GPU (CUDA) is supported")
            try:
                # 利用可能なCUDAデバイスの数を取得
                num_cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
                opencv_info.append(f" - CUDA Enabled Device Count: {num_cuda_devices}")

                for i in range(num_cuda_devices):
                    # DeviceInfo クラスを使用して詳細情報を取得
                    device_info = cv2.cuda.DeviceInfo(i)
                    major_version = device_info.majorVersion()  # メジャーバージョン
                    minor_version = device_info.minorVersion()  # マイナーバージョン
                    total_memory = device_info.totalMemory()  # メモリサイズ

                    opencv_info.append(f"   - GPU {i}")
                    opencv_info.append(f"     - CUDA Capability: {major_version}.{minor_version}")
                    opencv_info.append(f"     - Total Memory: {total_memory / 1024**3:.2f} GB")
            except Exception as e:
                opencv_info.append(f"Error retrieving CUDA device info: {str(e)}")
        else:
            opencv_info.append("GPU (CUDA) is not supported")
    else:
        opencv_info.append("OpenCV is not installed")
    return opencv_info



def check_pytorch_gpu():
    pytorch_info = ["============ PyTorch ============"]
    if torch:
        if torch.cuda.is_available():
            pytorch_info.append("GPU is supported")
            pytorch_info.append(f" - CUDA Version: {torch.version.cuda}")
            pytorch_info.append(f" - GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                pytorch_info.append(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
                pytorch_info.append(f"     - Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
                pytorch_info.append(f"     - CUDA Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
        else:
            pytorch_info.append("GPU is not supported")
    else:
        pytorch_info.append("PyTorch is not installed")
    return pytorch_info


def check_tensorflow_gpu():
    tensorflow_info = ["============ TensorFlow ============"]
    if tf:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            tensorflow_info.append("GPU is supported")
            for i, gpu in enumerate(gpus):
                gpu_info = tf.config.experimental.get_device_details(gpu)
                tensorflow_info.append(f" - GPU {i}: {gpu_info['device_name']}")
                # 'memory_limit' キーが存在するかチェック
                if 'memory_limit' in gpu_info:
                    tensorflow_info.append(f"   - Memory Limit: {gpu_info['memory_limit'] / 1024**3:.2f} GB")
                else:
                    tensorflow_info.append("   - Memory Limit information is not available")
        else:
            tensorflow_info.append("GPU is not supported")
    else:
        tensorflow_info.append("TensorFlow is not installed")
    return tensorflow_info


def check_tensorrt_gpu():
    tensorrt_info = ["============ TensorRT ============"]
    if trt:
        trt_version = trt.__version__
        tensorrt_info.append(f"TensorRT is installed (Version: {trt_version})")
        tensorrt_info.append("TensorRT does not provide detailed GPU information directly, but uses CUDA cores.")
    else:
        tensorrt_info.append("TensorRT is not installed")
    return tensorrt_info


def check_onnx_gpu():
    onnx_info = ["============ ONNX Runtime ============"]
    if ort:
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            onnx_info.append("GPU is supported")
            onnx_info.append("ONNX Runtime does not provide detailed GPU information directly.")
        else:
            onnx_info.append("GPU is not supported")
    else:
        onnx_info.append("ONNX Runtime is not installed")
    return onnx_info


def check_pycuda_gpu():
    pycuda_info = ["============ PyCUDA ============"]
    if cuda:
        try:
            cuda.init()
            device_count = cuda.Device.count()
            pycuda_info.append(f"GPU Count: {device_count}")
            for i in range(device_count):
                dev = cuda.Device(i)
                pycuda_info.append(f" - GPU {i}: {dev.name()}")
                pycuda_info.append(f"   - Total Memory: {dev.total_memory() / 1024**3:.2f} GB")
                attrs = dev.get_attributes()
                pycuda_info.append(f"   - CUDA Capability: {attrs[cuda.device_attribute.COMPUTE_CAPABILITY_MAJOR]}.{attrs[cuda.device_attribute.COMPUTE_CAPABILITY_MINOR]}")
        except cuda.Error as e:
            pycuda_info.append(f"PyCUDA Error: {str(e)}")
    else:
        pycuda_info.append("PyCUDA is not installed")
    return pycuda_info


def main():
    # 各フレームワークの情報を取得
    opencv_info = check_opencv_gpu()
    pytorch_info = check_pytorch_gpu()
    tensorflow_info = check_tensorflow_gpu()
    tensorrt_info = check_tensorrt_gpu()
    onnx_info = check_onnx_gpu()
    pycuda_info = check_pycuda_gpu()

    # すべての情報をまとめて出力
    all_info = opencv_info + pytorch_info + tensorflow_info + tensorrt_info + onnx_info + pycuda_info
    for line in all_info:
        print(line)


if __name__ == "__main__":
    main()
