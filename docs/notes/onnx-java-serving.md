# ONNX Conversion & Java Serving (CPU & GPU)

High-performance inference on CPU and GPU using **ONNX Runtime (ORT)** in a **Java** environment.


### Why ONNX + Java?
- **JVM Ecosystem**: Integrate directly with existing Java/Scala backend services (Spring Boot, Flink, Spark).
- **Lower Latency**: Avoid HTTP overhead of calling an external model server (TensorFlow Serving / TorchServe) by running in-process (JNI).
- **CPU optimization**: ONNX Runtime is highly optimized for CPU inference (AVX2/AVX512).

---

## 1. Model Conversion (Python)

Convert PyTorch or TensorFlow models to `.onnx` format.

### PyTorch to ONNX

```python
import torch
import torch.nn as nn

# 1. Load trained model
model = MyModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# 2. Define dummy input (shape must match model input)
dummy_input = torch.randn(1, 3, 224, 224)

# 3. Export
torch.onnx.export(
    model, 
    dummy_input, 
    "model.onnx",
    input_names=["input"], 
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=14
)
```

### TensorFlow to ONNX

Use `tf2onnx`:

```bash
pip install tf2onnx
python -m tf2onnx.convert --saved-model ./tf_model --output model.onnx --opset 14
```

---

## 2. Java Inference (ONNX Runtime)

Use the `onnxruntime` Java API to load and query the model.

### Dependency (Maven)
```xml
<!-- Optimized Native Binaries for CPU -->
<dependency>
    <groupId>com.microsoft.onnxruntime</groupId>
    <artifactId>onnxruntime</artifactId>
    <version>1.17.1</version>
</dependency>
```

### Inference Code

```java
import ai.onnxruntime.*;
import java.nio.FloatBuffer;
import java.util.Collections;

public class ModelService {
    private OrtEnvironment env;
    private OrtSession session;

    public ModelService(String modelPath) throws OrtException {
        this.env = OrtEnvironment.getEnvironment();
        // Optimization: Enable graph optimizations
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        opts.setIntraOpNumThreads(4); // Tune based on CPU cores
        
        this.session = env.createSession(modelPath, opts);
    }

    public float[] predict(float[] inputData, long[] shape) throws OrtException {
        // 1. Create Tensor from Java array
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), shape);

        // 2. Run Inference
        OrtSession.Result result = session.run(Collections.singletonMap("input", inputTensor));

        // 3. Extract Output
        float[][] output = (float[][]) result.get(0).getValue();
        result.close(); // Important: Close native resource to prevent leaks (or use try-with-resources)

        return output[0];
    }
}
```

---

## 3. Java Inference (GPU / CUDA)

To run on NVIDIA GPUs, switch dependencies and configure the session options.

### Dependency (Maven)

Replace `onnxruntime` with `onnxruntime_gpu`:

```xml
<dependency>
    <groupId>com.microsoft.onnxruntime</groupId>
    <artifactId>onnxruntime_gpu</artifactId>
    <version>1.17.1</version>
</dependency>
```

### CUDA Configuration Code

```java
import ai.onnxruntime.providers.OrtCUDAProviderOptions;

// ... inside constructor ...

OrtSession.SessionOptions opts = new OrtSession.SessionOptions();

// 1. Configure CUDA Provider
OrtCUDAProviderOptions cudaOpts = new OrtCUDAProviderOptions(0); // GPU Device ID 0
opts.addCUDA(cudaOpts);

// 2. Create Session (will load model onto GPU)
this.session = env.createSession(modelPath, opts);
```

**Note**: Ensure the host machine has compatible NVIDIA Drivers and CUDA Toolkit installed (matching the ORT version).


---

## 4. Performance Tuning (CPU)

| Setting | Recommendation |
| :--- | :--- |
| **`IntraOpNumThreads`** | Set to number of physical cores available to the request. Don't oversubscribe. |
| **`InterOpNumThreads`** | Keep low (1) unless running parallel subgraphs (rare). |
| **Execution Mode** | `SEQUENTIAL` is usually faster for simple models; `PARALLEL` increases latency but throughput for complex graphs. |
| **Memory Mapping** | Use `sessionOptions.addSessionConfigEntry("session.load_model_format", "ORT")` for faster loading if using optimized format. |

### Memory Management
**Critical**: ONNX Runtime uses off-heap memory (C++). 
- Always close `OrtSession.Result` and `OnnxTensor` objects explicitly or use `try-with-resources`. 
- Garbage Collection (GC) won't reclaim native memory fast enough, leading to OOM.

---

## 5. Architecture Diagram

```mermaid
graph TD
    subgraph Python[Training Phase (Python)]
        TF[TensorFlow / PyTorch] -->|Export| ONNX[model.onnx]
    end

    subgraph Java[Serving Phase (JVM)]
        App[Spring Boot / Flink] 
        Runtime[ONNX Runtime (C++)]
        
        ONNX -->|Load| Runtime
        App -- JNI --> Runtime
        Runtime -- Result --> App
    end
    
    style Runtime fill:#f9f,stroke:#333
```
