---
typora-root-url: ./results
---

# FashionCLIP_KG_garments
**Download Files:**

- [Dataset](https://drive.google.com/file/d/1VMoHI8XpuZYAb4Ne3KFRUXIjileguQNv/view?usp=drive_link) 
- [Model](https://drive.google.com/file/d/1oBwG_ArN1m8xy2z0WemZYCj2kUJ1x-o-/view?usp=drive_link) 

**Training Loss Curve (Epoch 100)**

![](/loss_curve.png)


**Training Accuracy for Different Tasks:**
Here is a summary of the training accuracy for each classification task based on the final epoch (Epoch 99):

| **Classification Task**        | **Training Accuracy (%)** |
| ------------------------------ | ------------------------- |
| Gender Classification          | 45.05                     |
| Coarse Category Classification | 4.25                      |
| Fine Category Classification   | 3.47                      |
| Materials Classification       | 10.07                     |
| Features Classification        | 4.19                      |
| Colour Classification          | 3.21                      |
| Sleeve Length Classification   | 34.46                     |
| Closure Classification         | 14.67                     |
| Fabric Type Classification     | 5.38                      |
| Neckline Classification        | 13.54                     |
| Pattern Classification         | 44.26                     |

**mAP and CMC for Each Task:**

| **Classification Task**        | **Top-1 CMC** | **Top-3 CMC** | **Top-5 CMC** | **Top-10 CMC** | **mAP** |
| ------------------------------ | ------------- | ------------- | ------------- | -------------- | ------- |
| Gender Classification          | 1.0           | 1.0           | 1.0           | 1.0            | 1.0     |
| Coarse Category Classification | 1.0           | 1.0           | 1.0           | 1.0            | 1.0     |
| Fine Category Classification   | 0.91          | 0.93          | 0.94          | 1.0            | 0.95    |
| Materials Classification       | 0.87          | 0.88          | 0.92          | 1.0            | 0.92    |
| Features Classification        | 0.56          | 0.68          | 0.71          | 0.86           | 0.73    |
| Colour Classification          | 0.97          | 1.0           | 1.0           | 1.0            | 0.99    |
| Sleeve Length Classification   | 1.0           | 1.0           | 1.0           | 1.0            | 1.0     |
| Closure Classification         | 0.96          | 0.96          | 0.96          | 1.0            | 0.97    |
| Fabric Type Classification     | 0.92          | 0.93          | 0.93          | 1.0            | 0.95    |
| Neckline Classification        | 0.91          | 1.0           | 1.0           | 1.0            | 0.98    |
| Pattern Classification         | 0.85          | 0.85          | 0.86          | 0.95           | 0.91    |

**Testing Accuracy Summary:**

| **Classification Task**        | **Testing Accuracy (%)** |
| ------------------------------ | ------------------------ |
| Gender Classification          | 55.31                    |
| Coarse Category Classification | 0.00                     |
| Fine Category Classification   | 0.63                     |
| Materials Classification       | 18.16                    |
| Features Classification        | 4.06                     |
| Colour Classification          | 7.33                     |
| Sleeve Length Classification   | 17.70                    |
| Closure Classification         | 14.37                    |
| Fabric Type Classification     | 5.00                     |
| Neckline Classification        | 23.79                    |
| Pattern Classification         | 61.17                    |