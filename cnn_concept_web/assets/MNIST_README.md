# MNIST 그림과 측정값 출처

`mnist_*.png`는 실제 MNIST 손글씨에서 선택한 표본과 이 자료의 작은 CNN을 실제로 실행해 만든 그림입니다.
원 데이터: Yann LeCun, Corinna Cortes, Christopher J. C. Burges의 MNIST handwritten digit database.

- 데이터 구성: https://www.tensorflow.org/datasets/catalog/mnist
- 사용한 원본 파일·미러·체크섬의 근거: https://docs.pytorch.org/vision/stable/_modules/torchvision/datasets/mnist.html
- 다운로드 미러: https://ossci-datasets.s3.amazonaws.com/mnist/

`samples`는 실제 test 표본, `pixels`는 한 표본의 실제 밝기, `filters`는 수작업 필터 계산입니다.
`features`는 학습된 첫 층, `learning`은 실제 train/validation 곡선, `prediction`은 한 test 예측, `errors`는 실제 오분류 앞6장입니다.
MNIST 전체 원본은 Git·배포 ZIP에 넣지 않습니다. 패키지에 포함하는 것은 수업에 사용한 그림과 측정 메타데이터입니다.

재현: `python src/prepare_mnist.py`(cnn_concept_web 폴더에서 실행).
기본 seed17, CPU, Adam lr0.003, batch128,6epoch. train6000/val1000/test1000을 클래스별로 균형 있게 선택합니다.
선택 test 인덱스·혼동 행렬·실제 성능은 `mnist_metrics.json`에 있습니다. 공식 test10000장 전체 성능이 아닙니다.
