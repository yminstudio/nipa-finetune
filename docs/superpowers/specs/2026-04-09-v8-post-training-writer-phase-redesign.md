# v8 post-training writer phase 재설계

## 배경

`round2` 학습은 `1000/1000` step과 `final-export` 저장까지 완료했지만, 그 뒤 `reload validation` 직전/직후 구간에서 `NCCL collective timeout`으로 실패 처리되었다.

로그와 산출물을 대조한 결과 문제는 `final-export` 자체가 아니라, 분산 프로세스가 살아 있는 상태에서 writer rank만 오래 걸리는 후처리를 수행하고 다른 rank는 barrier에서 대기하도록 만든 종료부 구조에 있었다.

## 목표

- `final-export` 저장은 기존처럼 모든 rank가 참여하는 collective 단계로 유지한다.
- `reload validation`과 `train result report`는 NCCL이 없는 후처리 단계로 분리한다.
- non-writer rank도 writer 결과를 확인할 수 있어야 하지만, 더 이상 분산 barrier 때문에 장시간 묶이지 않아야 한다.

## 설계

### 1. `final export` 이후 분산 종료

- `final export` collective가 성공하면, 후속 writer-only 단계에 들어가기 전에 process group을 명시적으로 종료한다.
- 종료 후에는 `distributed_info`를 finalized 상태로 표시해 `finally` 블록에서 중복 종료가 일어나지 않게 한다.

### 2. 후처리 전용 writer phase 분리

- 새 단계는 `coordinated_post_training_writer_phase`로 분리한다.
- writer rank는 실제 작업을 수행하고 상태 파일을 기록한다.
- non-writer rank는 NCCL barrier 대신 상태 파일이 생길 때까지 폴링한 뒤 결과만 읽는다.
- 이 단계는 `reload validation`, `train result report`에 공통으로 사용한다.

### 3. stale 상태 파일 제거

- resume 재실행에서도 이전 상태 파일을 잘못 읽지 않도록, `final export` 직후 짧은 collective 정리 단계에서 post-training phase 상태 파일을 제거한다.

## 기대 효과

- `final-export` 이후 writer rank만 오래 걸리는 validation/report 때문에 발생하던 NCCL timeout을 제거한다.
- `train_result.json`이 다시 `success` 또는 `validation_failed`로 정상 기록될 수 있게 한다.
- 확보된 `final-export` 산출물에 대한 reload validation을 안전하게 후속 실행할 수 있다.
