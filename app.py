2026-03-06T17:11:25.381335618Z [inf]  v7.0 Starting on PORT: 8080
2026-03-06T17:11:25.381345563Z [err]  [2026-03-06 17:11:25 +0000] [2] [INFO] Starting gunicorn 21.2.0
2026-03-06T17:11:25.381350818Z [err]  [2026-03-06 17:11:25 +0000] [2] [INFO] Listening at: http://0.0.0.0:8080 (2)
2026-03-06T17:11:25.381355473Z [err]  [2026-03-06 17:11:25 +0000] [2] [INFO] Using worker: gthread
2026-03-06T17:11:25.381360281Z [err]  [2026-03-06 17:11:25 +0000] [3] [INFO] Booting worker with pid: 3
2026-03-06T17:11:25.752982107Z [inf]  Starting Container
2026-03-06T17:11:25.881075149Z [err]  2026-03-06 17:11:25.879841: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2026-03-06T17:11:25.883144021Z [err]  2026-03-06 17:11:25.882413: I external/local_tsl/tsl/cuda/cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
2026-03-06T17:11:25.935522664Z [err]  2026-03-06 17:11:25.924062: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2026-03-06T17:11:25.935533016Z [err]  2026-03-06 17:11:25.924111: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2026-03-06T17:11:25.935544029Z [err]  2026-03-06 17:11:25.925686: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2026-03-06T17:11:25.935549217Z [err]  2026-03-06 17:11:25.933496: I external/local_tsl/tsl/cuda/cudart_stub.cc:31] Could not find cuda drivers on your machine, GPU will not be used.
2026-03-06T17:11:25.935554684Z [err]  2026-03-06 17:11:25.933770: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
2026-03-06T17:11:25.935559985Z [err]  To enable the following instructions: AVX2 AVX512F AVX512_VNNI AVX512_BF16 AVX_VNNI AMX_TILE AMX_INT8 AMX_BF16 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2026-03-06T17:11:27.024334624Z [err]  2026-03-06 17:11:27.019863: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
2026-03-06T17:11:27.699114806Z [err]  INFO:app:================================================================================
2026-03-06T17:11:27.699124956Z [err]  INFO:app:🚀 I-TRANSLATION v5.2 - RAILWAY PRODUCTION
2026-03-06T17:11:27.699142061Z [err]  INFO:app:================================================================================
2026-03-06T17:11:27.699148876Z [err]  INFO:app:
2026-03-06T17:11:27.699157674Z [err]  📦 Starting model download and loading process...
2026-03-06T17:11:27.699165083Z [err]  INFO:app:Total models to load: 4
2026-03-06T17:11:27.699170838Z [err]  INFO:app:
2026-03-06T17:11:27.699177823Z [err]  ============================================================
2026-03-06T17:11:27.699184662Z [err]  INFO:app:[F] Processing Generator F
2026-03-06T17:11:27.699191069Z [err]  INFO:app:============================================================
2026-03-06T17:11:27.799219802Z [err]  INFO:app:📥 Downloading file ID: 1O1hQSOoizPt5fJyVuEfxRpq0LibmaGeM
2026-03-06T17:11:30.683378948Z [err]  ERROR:app:❌ Download error: 'NoneType' object has no attribute 'groups'
2026-03-06T17:11:30.683385569Z [err]  ERROR:app:[F] ❌ Download failed, skipping model load
2026-03-06T17:11:30.683391893Z [err]  INFO:app:
2026-03-06T17:11:30.683398056Z [err]  ============================================================
2026-03-06T17:11:30.683406570Z [err]  INFO:app:[G] Processing Generator G
2026-03-06T17:11:30.683413310Z [err]  INFO:app:============================================================
2026-03-06T17:11:30.683420001Z [err]  INFO:app:📥 Downloading file ID: 1nQnBaEyjQyTp3LJ6DF9tfaXrZxIHkROQ
2026-03-06T17:11:33.778352978Z [err]  ERROR:app:❌ Download error: 'NoneType' object has no attribute 'groups'
2026-03-06T17:11:33.778360178Z [err]  ERROR:app:[G] ❌ Download failed, skipping model load
2026-03-06T17:11:33.778368300Z [err]  INFO:app:
2026-03-06T17:11:33.778374230Z [err]  ============================================================
2026-03-06T17:11:33.778380574Z [err]  INFO:app:[I] Processing Generator I
2026-03-06T17:11:33.778386222Z [err]  INFO:app:============================================================
2026-03-06T17:11:33.778392893Z [err]  INFO:app:📥 Downloading file ID: 1QIvFXO0LzDa6IH683OWXkedRAXpcDvk-
2026-03-06T17:11:36.476840070Z [err]  ERROR:app:[I] ❌ Download failed, skipping model load
2026-03-06T17:11:36.476851957Z [err]  INFO:app:
2026-03-06T17:11:36.476859800Z [err]  ============================================================
2026-03-06T17:11:36.476867362Z [err]  INFO:app:[J] Processing Generator J
2026-03-06T17:11:36.476873375Z [err]  INFO:app:============================================================
2026-03-06T17:11:36.476886338Z [err]  INFO:app:📥 Downloading file ID: 1-Quu4cDJhTpH7RDj-HZ-6c4VsQl1mc6j
2026-03-06T17:11:36.476932609Z [err]  ERROR:app:❌ Download error: 'NoneType' object has no attribute 'groups'
2026-03-06T17:11:37.896408404Z [err]  ERROR:app:❌ Download error: 'NoneType' object has no attribute 'groups'
2026-03-06T17:11:37.903479534Z [err]  INFO:app:✓ POST /convert-batch - Convert multiple images
2026-03-06T17:11:37.903492721Z [err]  INFO:app:================================================================================
2026-03-06T17:11:37.903502797Z [err]  INFO:app:✅ APPLICATION READY TO SERVE REQUESTS
2026-03-06T17:11:37.903508620Z [err]  ERROR:app:[J] ❌ Download failed, skipping model load
2026-03-06T17:11:37.903512126Z [err]  INFO:app:================================================================================
2026-03-06T17:11:37.903521355Z [err]  
2026-03-06T17:11:37.903525602Z [err]  INFO:app:
2026-03-06T17:11:37.903584442Z [err]  ================================================================================
2026-03-06T17:11:37.903594695Z [err]  INFO:app:📊 FINAL STATUS: 0/4 models loaded
2026-03-06T17:11:37.903603013Z [err]  INFO:app:================================================================================
2026-03-06T17:11:37.903614696Z [err]  WARNING:app:⚠️  WARNING: Only 0/4 models loaded
2026-03-06T17:11:37.903623217Z [err]  WARNING:app:Available: []
2026-03-06T17:11:37.903631941Z [err]  WARNING:app:Missing: ['f', 'g', 'i', 'j']
2026-03-06T17:11:37.903639679Z [err]  INFO:app:
2026-03-06T17:11:37.903648201Z [err]  ================================================================================
2026-03-06T17:11:37.903657269Z [err]  INFO:app:🌐 API ENDPOINTS REGISTERED
2026-03-06T17:11:37.903665615Z [err]  INFO:app:================================================================================
2026-03-06T17:11:37.903673126Z [err]  INFO:app:✓ GET  /health        - Check system status
2026-03-06T17:11:37.903681501Z [err]  INFO:app:✓ POST /convert       - Convert single image
2026-03-06T17:12:04.863050725Z [inf]  Starting Container
2026-03-06T17:25:53.553005101Z [inf]  Starting Container
2026-03-06T17:30:41.025679124Z [inf]  Starting Container
2026-03-06T17:38:18.414441419Z [inf]  Starting Container
2026-03-06T17:42:57.742302848Z [inf]  Starting Container
2026-03-06T17:44:21.913063758Z [inf]  Starting Container
2026-03-06T17:54:04.931314199Z [inf]  Starting Container
2026-03-06T18:00:40.200311967Z [inf]  Starting Container
