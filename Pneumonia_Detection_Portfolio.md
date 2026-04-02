# 흉부 X-ray 폐렴 진단 — FastAI ResNet50 전이 학습

> **Binary Classification** | Chest X-Ray | FastAI · PyTorch · TensorFlow · Keras  
> **Dataset:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — Kaggle / Mendeley Data (CC BY 4.0)  
> **환경:** Kaggle Notebook (Tesla T4 GPU × 2) | Internet: 일부 STEP에서만 ON

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [STEP 0 — 데이터셋 맥락](#2-step-0--데이터셋-맥락)
3. [STEP 1 — EDA & 환경 확인](#3-step-1--eda--환경-확인)
4. [STEP 2 — 전처리 파이프라인](#4-step-2--전처리-파이프라인)
5. [STEP 3 — Baseline CNN 정의](#5-step-3--baseline-cnn-정의)
6. [STEP 4 — 학습 실행](#6-step-4--학습-실행)
7. [STEP 5 — 모델 평가](#7-step-5--모델-평가)
8. [STEP 6 — 하이퍼파라미터 튜닝](#8-step-6--하이퍼파라미터-튜닝)
9. [STEP 7 — Transfer Learning: FastAI ResNet50](#9-step-7--transfer-learning-fastai-resnet50)
10. [최종 결론 & 향후 과제](#10-최종-결론--향후-과제)

---

## 1. 프로젝트 개요

### 문제 정의

폐렴(Pneumonia)은 세계적으로 소아 사망률 1위 원인 중 하나이며, 조기 진단이 치료 예후를 결정한다. 흉부 X-ray는 1차 진단 수단으로 광범위하게 사용되지만, 판독에는 숙련된 전문의가 필요하다. 본 프로젝트는 딥러닝 모델을 통해 **NORMAL vs PNEUMONIA의 이진 분류**를 자동화하는 파이프라인을 구축하고, 모델 선택 과정의 논리적 흐름을 단계적으로 검증한다.

### 성능 목표 (사전 정의)

| 지표 | 목표값 | 근거 |
|:---|:---:|:---|
| **Test Accuracy** | ≥ 0.90 | 임상 보조 도구로서의 최소 신뢰 수준 |
| **Recall (PNEUMONIA)** | ≥ 0.93 | FN(폐렴 누락)이 FP보다 의료적으로 더 치명적 |

### 실험 흐름 요약

```
Baseline CNN → 아키텍처 튜닝 → Transfer Learning (ResNet50)
   Acc 0.86       개선 실패          Acc 0.91 ✅  Recall 0.93 ✅
```

---

## 2. STEP 0 — 데이터셋 맥락

### 데이터셋 개요

| 항목 | 내용 |
|:---|:---|
| **데이터셋명** | Chest X-Ray Images (Pneumonia) |
| **출처** | Kaggle — Paul Mooney / Mendeley Data |
| **총 이미지** | 5,863장 (JPEG) |
| **클래스** | NORMAL / PNEUMONIA (이진) |
| **공식 분할** | train(5,216) / val(16) / test(624) |
| **라이선스** | CC BY 4.0 |

- **기관:** 중국 광저우 여성아동의료센터 소아 환자 (1–5세) 회고적 코호트
- **촬영 방식:** 루틴 임상 진료 중 전후방(AP) 흉부 X-ray

### 품질 관리 3단계

1. 저품질·판독 불가 스캔 일차 제거
2. 전문의 2인 독립 진단 등급 부여
3. 제3의 전문가에 의한 평가 세트 재검증

### 클래스별 X-ray 소견

| 클래스 | X-ray 소견 |
|:---|:---|
| **NORMAL** | 폐야 청명, 비정상적 혼탁 없음 |
| **PNEUMONIA (세균성)** | 국소적 엽 경화(focal lobar consolidation), 주로 우상엽 |
| **PNEUMONIA (바이러스성)** | 양측 폐 미만성 간질 패턴(diffuse interstitial pattern) |

> **프로젝트 범위:** 세균성·바이러스성 세부 구분 없이 **NORMAL vs PNEUMONIA 이진 분류**만 수행.

---

## 3. STEP 1 — EDA & 환경 확인

**핵심 발견:** Kaggle 기본 제공 val 세트는 **단 16장**으로, 신뢰할 수 있는 검증 세트로 사용 불가.  
→ **설계 결정:** train(5,216) + val(16) = 5,232장을 합산한 뒤, **80/20으로 재분할** (4,185 / 1,047장).

```
클래스 분포 (train 기준):
  NORMAL    : 1,341장 (25.7%)
  PNEUMONIA : 3,875장 (74.3%)
  → 불균형 비율 ≈ 1 : 2.89
```

**불균형 대응 전략:** 손실 함수에 `class_weight` 적용 (NORMAL: 3.91, PNEUMONIA: 1.34)으로 소수 클래스의 학습 기여도를 보정.

---

## 4. STEP 2 — 전처리 파이프라인

### 설계 원칙

- **이중 정규화 방지:** 파이프라인에서 픽셀 정규화를 수행하므로, 모델 내부에 Rescaling 레이어를 중복 삽입하지 않는다.
- **의료 영상 특성 반영:** 흉부 X-ray는 상하 반전이 임상적으로 불가능하므로, Augmentation은 `horizontal flip`만 허용.

### 전처리 순서

```python
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0          # [0, 1] 범위로 스케일
    mean  = tf.math.reduce_mean(image)
    std   = tf.math.reduce_std(image)
    image = (image - mean) / (std + 1e-7)               # 이미지별 z-score 표준화
    return image, label
```

### 파이프라인 결과 체크

| 항목 | 결과 | 상태 | 비고 |
|:---|:---|:---:|:---|
| 이미지 shape | `(64, 300, 300, 3)` | ✅ | 300×300 리사이즈 정상 |
| 픽셀 값 범위 | `[-3.49, 6.03]` | ✅ | z-score 정상 적용 |
| class_weight | NORMAL: 3.91 / PNEUMONIA: 1.34 | ✅ | 불균형 보정 완료 |
| 80/20 재분할 | 4,185 / 1,047장 | ✅ | val 16장 신뢰 불가 문제 해결 |

---

## 5. STEP 3 — Baseline CNN 정의

### 아키텍처

```
Input (300×300×3)
→ [Augmentation Layer]
→ Conv2D(32) + MaxPooling
→ Conv2D(64) + MaxPooling
→ Conv2D(128) + MaxPooling
→ Flatten → Dense(128, ReLU) → Dropout(0.5)
→ Dense(1, Sigmoid)
```

**컴파일:** Adam(lr=5e-4) / Binary Crossentropy / AUC

### 구조적 경고 사항

| 레이어 | 파라미터 수 | 비율 |
|:---|---:|---:|
| Conv Blocks × 3 | ~148K | ~0.7% |
| **첫 번째 Dense (Flatten 직후)** | **~22.4M** | **~99.3%** |

- `37×37×128 = 175,232` 차원이 Flatten 되면서 Dense 레이어에서 파라미터가 폭발.
- **과적합 위험**을 사전에 인식하고, 이후 STEP 6에서 `GlobalAveragePooling2D`로 교체.

---

## 6. STEP 4 — 학습 실행

### 콜백 전략

| 콜백 | 설정 | 목적 |
|:---|:---|:---|
| `EarlyStopping` | patience=5, monitor=val_loss | 과적합 방지 |
| `ModelCheckpoint` | monitor=val_accuracy | 최적 가중치 보존 |
| `ReduceLROnPlateau` | factor=0.5, patience=3 | 지역 최솟값 탈출 |

### 학습 결과

| 지표 | 결과 | 목표 | 상태 |
|:---|:---:|:---:|:---|
| val_accuracy | **0.9742** | ≥ 0.90 | ✅ 초과 달성 |
| val_auc | **0.9973** | — | ✅ |
| 과적합 gap | −0.0074 (val > train) | < 0.05 | ✅ 없음 |
| 최적 epoch | **10 / 30** (EarlyStopping: 15) | — | ✅ |

> **해석:** Epoch 1에서 val_accuracy 0.72로 시작해 Epoch 10에서 0.9742 달성. `class_weight`로 인해 초기 loss가 1.74로 높게 시작한 것은 NORMAL 클래스(weight=3.91)의 페널티가 반영된 정상적 현상.

---

## 7. STEP 5 — 모델 평가

### 테스트 세트 최종 성능 (threshold=0.75)

| 지표 | 결과 | 목표 | 상태 |
|:---|:---:|:---:|:---|
| **Test Accuracy** | 0.8622 | ≥ 0.90 | ⚠️ **미달** |
| **Recall (PNEUMONIA)** | 0.9641 | ≥ 0.93 | ✅ |
| **AUC** | 0.9518 | — | ✅ |

### Confusion Matrix 분석

```
                예측: NORMAL   예측: PNEUMONIA
실제: NORMAL       162 (TN)       72 (FP) ⚠️
실제: PNEUMONIA     14 (FN) ✅   376 (TP)
```

**핵심 문제:** val에서는 0.9742의 높은 정확도를 보였으나, **test에서 Accuracy가 0.8622로 급락**.  
→ NORMAL을 PNEUMONIA로 오진하는 **FP가 72건으로 과다 발생**하여 NORMAL Recall이 0.6923으로 붕괴.

> **원인 가설:** Baseline CNN이 폐렴 판정에 과도하게 민감해지는 "과신뢰(Overconfidence)" 현상. 얕은 CNN의 특징 추출 능력 한계로 인한 예측 확률 보정(Calibration) 부족.

---

## 8. STEP 6 — 하이퍼파라미터 튜닝

Baseline의 Accuracy 미달 원인을 구조적으로 해결하기 위해 두 차례의 튜닝을 시도.

### 변경 사항 요약

| 항목 | Baseline (STEP 5) | Tuned (STEP 6) | 목적 |
|:---|:---|:---|:---|
| Conv Blocks | 3개 (32→64→128) | **4개 (+Conv256)** | 표현력 강화 |
| Pooling | Flatten | **GlobalAveragePooling2D** | 파라미터 폭발 억제 |
| BatchNorm | 없음 | **각 Conv Block 후 추가** | 학습 안정화 |
| Dropout | 0.5 × 1 | **0.6 + 0.4 × 2** | 과적합 억제 |
| L2 정규화 | 없음 | **Dense에 1e-4** | 과적합 억제 |
| Augmentation | Flip + Rotation + Zoom | **+ Brightness + Contrast** | 다양성 확보 |
| RandomFlip | `horizontal_and_vertical` | **`horizontal` 전용** | X-ray 도메인 현실성 반영 |

### 트러블슈팅 기록

- **PNEUMONIA 가중치 스케일다운 시도 (×0.85) → 실패:** FP 억제를 위해 가중치를 낮췄더니 오히려 정상/폐렴 간 학습 불균형이 심화되어 원래 `1/freq` 공식으로 복원.
- **상하 반전(vertical flip) 제거:** 임상적으로 흉부 X-ray가 상하 반전된 채 촬영되는 경우는 존재하지 않으므로, 불필요한 노이즈를 제거.

**결과:** 아키텍처 개선 후에도 Accuracy 목표(≥ 0.90) 미달 → **Transfer Learning으로 전략 전환**.

---

## 9. STEP 7 — Transfer Learning: FastAI ResNet50

### 전략 선택 근거

- **Baseline CNN + 튜닝**으로도 개선 실패 → 얕은 CNN의 구조적 한계로 판단.
- ImageNet으로 사전 학습된 ResNet50의 **깊은 특징 추출기(Feature Extractor)** 를 활용하여 예측 확률 보정 문제를 해결.
- FastAI의 `fine_tune()` API를 사용해 **Frozen → Unfreeze** 2단계 학습 수행.

### 학습 구성

```python
# FastAI DataBlock
dblock = DataBlock(
    blocks    = (ImageBlock, CategoryBlock),
    get_items = get_image_files,
    splitter  = RandomSplitter(valid_pct=0.20, seed=42),
    get_y     = parent_label,
    item_tfms = Resize(300),
    batch_tfms = aug_transforms(flip_vert=False, ...)  # 상하 반전 제외
)

learn = vision_learner(dls, resnet50, metrics=[accuracy, RocAucBinary()])
learn.fine_tune(4)   # Frozen 1 epoch + Unfreeze 4 epoch
```

### 단계별 성능 추이

| 단계 | Epoch | val_acc | Recall | F1-Score |
|:---|:---:|:---:|:---:|:---:|
| **Frozen** | 0 | 0.9168 | 0.9087 | 0.9428 |
| **Fine-tune** | 0 | 0.8681 | 0.8264 | 0.9043 |
| **Fine-tune** | 1 | 0.9111 | 0.8859 | 0.9376 |
| **Fine-tune** | 2 | 0.9264 | 0.9062 | 0.9490 |
| **Fine-tune** | 3 | 0.9331 | 0.9138 | 0.9537 |
| **Fine-tune** | 4 | **0.9350** | **0.9176** | **0.9551** |

> Fine-tune 초기(epoch 0)에 val_acc가 0.8681로 하락하는 것은, 전체 레이어 동시 학습 시작 직후 발생하는 정상적인 현상.

### 테스트 세트 최종 성능 (threshold=0.75)

| 지표 | Baseline CNN | **ResNet50** | 변화 |
|:---|:---:|:---:|:---:|
| **Test Accuracy** | 0.8622 ⚠️ | **0.9119** ✅ | ↑ +0.050 |
| **Test AUC** | 0.9518 | **0.9558** | ↑ +0.004 |
| **Recall (PNEUMONIA)** | 0.9641 | **0.9308** ✅ | ↓ −0.033 |
| **FP (NORMAL 오진)** | 72건 ⚠️ | **28건** | ↓ −44건 |
| **FN (폐렴 누락)** | 14건 | **27건** | ↑ +13건 |

### Confusion Matrix 비교

```
[Baseline CNN]                    [ResNet50]
              예측: N  예측: P               예측: N  예측: P
실제: NORMAL   162    72 ⚠️    실제: NORMAL   206    28
실제: PNEUM.    14   376       실제: PNEUM.    27   363 ✅
```

---

## 10. 최종 결론 & 향후 과제

### 결론

| 구분 | 내용 |
|:---|:---|
| **달성 목표** | Accuracy ≥ 0.90 ✅ (0.9119), Recall ≥ 0.93 ✅ (0.9308) |
| **핵심 개선 요인** | FP 72건 → 28건 (−61%), NORMAL Recall 0.69 → 0.88 |
| **성공 원인** | ResNet50의 사전 학습 특징 추출기가 얕은 CNN의 과신뢰 문제를 구조적으로 해결 |

Baseline CNN은 폐렴 판정에 과도하게 편향되어 정상 환자를 폐렴으로 오진(FP)하는 한계를 보였다. ResNet50은 동일한 임계값(threshold=0.75)에서도 훨씬 안정적으로 보정된(Calibrated) 예측 확률을 출력하며, **FP를 대폭 감소시켜 두 가지 성능 목표를 동시에 충족**했다.

### 실험 과정에서 얻은 주요 인사이트

1. **Validation 신뢰성이 모델 선택에 선행한다.** 공식 val 세트 16장은 통계적으로 무의미하며, train+val 합산 후 재분할이 필수.
2. **도메인 지식이 Augmentation을 결정한다.** 상하 반전(vertical flip)이 X-ray 데이터에 적용되면 오히려 모델이 비현실적 패턴을 학습한다.
3. **클래스 가중치 조정은 조심스럽게.** 의도적 가중치 축소가 오히려 학습 불균형을 심화시킨 역설적 결과를 경험.
4. **얕은 CNN 아키텍처 개선만으로는 한계가 있다.** 구조적 표현력 부족은 하이퍼파라미터 튜닝이 아닌 Transfer Learning으로 해결.

### 향후 과제

- **FN 최소화:** ResNet50에서 폐렴 누락(FN)이 14→27건으로 증가했으며, 임계값(threshold) 세밀한 재탐색 또는 추가 fine-tuning으로 개선 가능.
- **EfficientNet / Vision Transformer 비교 실험:** ResNet50 대비 경량·고성능 모델과의 체계적인 비교.
- **Grad-CAM 시각화:** 모델이 어느 폐 영역에 주목하는지 시각적으로 해석하여 임상적 타당성 검증.

---

<details>
<summary>실험 환경 및 주요 라이브러리</summary>

| 항목 | 내용 |
|:---|:---|
| **환경** | Kaggle Notebook |
| **GPU** | Tesla T4 × 2 (13,757 MB each) |
| **Python** | 3.x |
| **TensorFlow / Keras** | 2.x |
| **FastAI** | 최신 (pip install) |
| **PyTorch** | FastAI 의존 버전 |
| **scikit-learn** | confusion matrix, ROC AUC |
| **Backbone** | ResNet50 (ImageNet pretrained) |
| **Random Seed** | 42 (전 실험 고정) |

</details>
