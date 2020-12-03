# Driver_Monitoring
부주의 운전에 의한 사고를 예방하기 위한 운전자 모니터링 시스템

# Train, Test Data
Kaggle state-farm-distracted-driver-detection </br>
![data_train](https://user-images.githubusercontent.com/34363323/100238377-7d9b3400-2f73-11eb-938e-8d1adb334438.png)
# Model
basic CNN classification model 적용 </br>
<모델 구성> </br>
conv_block : batch_norm, conv2d, maxpooling </br>
feature layer : 64, 128, 256, 512, 512 개의 커널 필터를 가지는 conv_block </br>
softmax layer : 10개의 아웃풋으로 주행자 모니터링 상황 추정

# Result
![test](https://user-images.githubusercontent.com/34363323/100238390-7f64f780-2f73-11eb-8b03-ff99bcc416b8.png)

# Reference
<a href="https://www.kaggle.com/c/state-farm-distracted-driver-detection"> Data </a> </br>
