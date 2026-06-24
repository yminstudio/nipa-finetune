# V7 Round3 Multi Question Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `round3`에서 같은 답변을 공유하는 11개 질문 변형 데이터셋과 학습 설정을 만들고, 학습 및 직접 검증까지 수행한다.

**Architecture:** 기존 `v7` 단일 Q/A 흐름을 유지하되, `round3` 데이터셋만 다중 레코드로 확장한다. 학습 스크립트는 그대로 재사용하고, 조기 종료 조건과 산출물 경로만 새 round에 맞춰 분리한다.

**Tech Stack:** Python, JSONL, TRL `SFTTrainer`, PEFT LoRA, pytest

---

### Task 1: Round3 테스트 추가

**Files:**
- Modify: `/home/work/dev_data/fine-tuning/.worktrees/feature-gen-dataset-v7/tests/test_v7_single_qa_training.py`

- [ ] **Step 1: Write the failing test**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Write minimal implementation**
- [ ] **Step 4: Run test to verify it passes**

### Task 2: Round3 dataset/config 생성

**Files:**
- Create: `/home/work/dev_data/fine-tuning/.worktrees/feature-gen-dataset-v7/llm_datasets/seed_v7/seed_v7_single_qa_round3.jsonl`
- Create: `/home/work/dev_data/fine-tuning/.worktrees/feature-gen-dataset-v7/configs/gpt_oss_20b_seed_v7_single_qa_round3.json`
- Modify: `/home/work/dev_data/fine-tuning/.worktrees/feature-gen-dataset-v7/docs/v7_single_qa_training.md`

- [ ] **Step 1: Create round3 files with exact paths**
- [ ] **Step 2: Re-run pytest to verify tests pass**

### Task 3: Round3 학습과 검증

**Files:**
- Modify: `/home/work/dev_data/fine-tuning/docs/context-log.md`
- Output: `/home/work/dev_data/fine-tuning/.worktrees/feature-gen-dataset-v7/llm_model_lora/gpt-oss-20b-seed-v7-single-qa-round3`

- [ ] **Step 1: Run round3 training**
- [ ] **Step 2: Inspect training result and early-stop event**
- [ ] **Step 3: Run direct adapter validation**
- [ ] **Step 4: Record context log**
