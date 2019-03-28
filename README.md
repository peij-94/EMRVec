# MIMIC-III 电子病历

- 电子病历数据提取
  - 疾病诊断码序列（ diagnose code sequence ）
  - 用药记录序列（ medication sequence ）

- EMRVec模型
  - Recursive AutoEncoder - diagnose code sequence
  - Attention Based Model - medication sequence

- requirement
  - python: 3.5以上
  - tensorflow
  - tqdm
  - pandas
  - numpy
  - psutil
  - joblib
