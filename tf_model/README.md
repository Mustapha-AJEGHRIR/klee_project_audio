# Speed result on the Raspberry Pi 3B+ 1 Gb RAM
This result comes from the execution of `_speed_test_script.py` on the Raspberry Pi 3B+ 1 Gb RAM.
```yaml
**********
---ONNX---
**********
        Activations : [-1.0243063  0.4067883]
        Duration : 5.51ms
**********
-TF-lite--
**********
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
        Activations : [-1.0269492  0.5141687]
        Duration : 5.17ms
```
In my personal computer, only on cpu (lenovo 5i pro 16" with i7-11370H), the results are :

```yaml
**********
---ONNX---
**********
        Activations : [-1.10536    0.3417935]
        Duration : 0.35ms
**********
-TF-lite--
**********
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
        Activations : [-1.0295922   0.49271876]
        Duration : 17.47ms
```

I don't know why tf-lite performs badly on my computer.
Concerning `ONNX`, it is probably using my GPU (MX 450 2GB) since I have previously the gpu version, I'm only interested in tf-lite's performance.