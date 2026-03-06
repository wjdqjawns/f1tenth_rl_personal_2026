# F1-like RL demo (lap-time oriented)

이 버전은 centerline 추종보다 **랩타임 최소화** 쪽으로 보상을 바꾼 버전이다.

## 핵심 변경점

- centerline은 진행도와 기준 heading 계산용으로만 사용
- 보상은 `forward progress - step penalty - oscillation penalty - reverse penalty - offtrack penalty + lap bonus`
- start/finish gate 교차 기반 lap 판정 추가
- 단순 bicycle dynamics 기반으로 속도, yaw rate, steering 반영
- 평가 시 episode별 궤적과 시계열 그래프를 한 그림에 누적 overlay
- 학습 중 episode reward / success / lap time / progress 그래프 저장

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 트랙 보기

```bash
python plot_track.py
```

## 학습

빠른 확인:

```bash
python train_ppo.py --timesteps 30000
```

조금 더 학습:

```bash
python train_ppo.py --timesteps 200000
```

출력:
- 모델: `models/ppo_f1_track.zip`
- 학습 로그/그래프: `runs/ppo_f1_track/`
  - `training_metrics.csv`
  - `training_summary.png`
  - `monitor.csv`

## 평가 + 그래프 저장

렌더 없이:

```bash
python evaluate.py --model-path models/ppo_f1_track.zip --episodes 5
```

렌더 포함:

```bash
python evaluate.py --model-path models/ppo_f1_track.zip --episodes 5 --render
```

출력:
- `eval_outputs/episode_overlay.png`
- `eval_outputs/episode_metrics.png`

## 관측 / 행동

관측:
- speed
- yaw rate
- normalized lateral position
- heading error
- left/right boundary margin
- lookahead curvature 6개
- previous steer/throttle command

행동:
- steering command
- throttle/brake command

## 해석 포인트

학습이 잘 되면 보통 아래가 같이 좋아져야 한다.

- success rate 상승
- lap time 감소
- mean speed 증가
- trajectory가 코너에서 바깥-안쪽-apex-바깥 형태로 바뀜

accuracy는 분류 문제처럼 직접적인 지표가 아니어서 여기서는 **success rate**로 대신 본다.
