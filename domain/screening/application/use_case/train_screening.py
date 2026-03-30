from domain.prediction.application.train_prediction import TrainPrediction


class TrainScreening(TrainPrediction):
    """스크리닝 모델 학습 Use Case입니다.

    예측 모델과 동일한 학습 과정을 사용하되,
    평가 시 민감도 우선 임계값을 적용합니다.

    Train_Prediction을 상속받아 CV 및 전체 데이터 학습 과정을 재사용합니다.
    """
    pass