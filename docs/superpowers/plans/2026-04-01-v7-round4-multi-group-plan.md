# V7 Round4 Multi Group Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 사용자 제공 표를 기준으로 6개 QA 묶음과 각 11개 질문 변형을 포함한 `round4` 데이터셋을 만들고, 베이스 모델에서 새로 LoRA 학습을 실행한다.

**Architecture:** 기존 `v7` 학습 스크립트는 그대로 재사용하고, 새 dataset/config/output 경로만 `round4`로 분리한다. `round4`는 총 66개 `system + user + assistant` 레코드로 구성하며, 조기 종료 기준은 `loss < 1.0` 및 `mean_token_accuracy >= 0.98`로 둔다.

**Tech Stack:** Python, JSONL, TRL `SFTTrainer`, PEFT LoRA, pytest

---

### Task 1: Round4 기대값 테스트 추가

**Files:**
- Modify: `/home/work/dev_data/fine-tuning/.worktrees/feature-gen-dataset-v7/tests/test_v7_single_qa_training.py`

- [ ] **Step 1: Write the failing test**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Write minimal implementation**
- [ ] **Step 4: Run test to verify it passes**

### Task 2: Round4 dataset/config 생성

**Files:**
- Create: `/home/work/dev_data/fine-tuning/.worktrees/feature-gen-dataset-v7/llm_datasets/seed_v7/seed_v7_single_qa_round4.jsonl`
- Create: `/home/work/dev_data/fine-tuning/.worktrees/feature-gen-dataset-v7/configs/gpt_oss_20b_seed_v7_single_qa_round4.json`
- Modify: `/home/work/dev_data/fine-tuning/.worktrees/feature-gen-dataset-v7/docs/v7_single_qa_training.md`

- [ ] **Step 1: Create round4 files with exact paths**
- [ ] **Step 2: Re-run pytest to verify tests pass**

### Task 3: Round4 base-model training

**Files:**
- Modify: `/home/work/dev_data/fine-tuning/docs/context-log.md`
- Output: `/home/work/dev_data/fine-tuning/.worktrees/feature-gen-dataset-v7/llm_model_lora/gpt-oss-20b-seed-v7-single-qa-round4`

- [ ] **Step 1: Run round4 training from base model**
- [ ] **Step 2: Inspect train result and early-stop event**
- [ ] **Step 3: Report completion and remaining risks**
