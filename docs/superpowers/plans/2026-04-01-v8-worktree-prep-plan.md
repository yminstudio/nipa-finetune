# V8 Worktree Prep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `v7` 기반의 새 워크트리 `feature-gen-dataset-v8`를 만들고, 기존 학습 산출물과 로그를 제거한 깨끗한 초기 상태를 준비한다.

**Architecture:** 저장소 루트에서 새 git worktree를 생성한 뒤, `v7`의 스크립트/테스트/설정/문서 패턴은 유지하고 모델 산출물과 smoke 로그만 비운다. 이후 `v8`는 독립된 실험 공간으로 사용한다.

**Tech Stack:** git worktree, Python test tooling, Markdown docs

---

### Task 1: Worktree 생성 안전 확인

**Files:**
- Modify: `/home/work/dev_data/fine-tuning/docs/superpowers/specs/2026-04-01-v8-worktree-prep-design.md`

- [ ] **Step 1: 기존 `.worktrees` 디렉터리와 ignore 상태 확인**
- [ ] **Step 2: 새 브랜치/경로 이름 충돌 여부 확인**
- [ ] **Step 3: worktree 생성 명령 실행**
- [ ] **Step 4: 생성된 경로 진입 및 기본 상태 확인**

### Task 2: V8 초기 정리

**Files:**
- Modify: `/home/work/dev_data/fine-tuning/.worktrees/feature-gen-dataset-v8/docs/context-log.md`
- Modify: `/home/work/dev_data/fine-tuning/.worktrees/feature-gen-dataset-v8/docs/v7_single_qa_training.md`
- Delete: `/home/work/dev_data/fine-tuning/.worktrees/feature-gen-dataset-v8/llm_model_lora/*`
- Delete: `/home/work/dev_data/fine-tuning/.worktrees/feature-gen-dataset-v8/tests/log/*`

- [ ] **Step 1: 제거 대상 경로 확인**
- [ ] **Step 2: 기존 round 산출물과 smoke 로그 제거**
- [ ] **Step 3: `v8` 출발점 문서 문구 보정**
- [ ] **Step 4: 불필요 산출물이 남지 않았는지 확인**

### Task 3: V8 시작 상태 검증

**Files:**
- Test: `/home/work/dev_data/fine-tuning/.worktrees/feature-gen-dataset-v8/tests/test_v7_single_qa_training.py`

- [ ] **Step 1: 기본 파일 구조 확인**
- [ ] **Step 2: 필요한 테스트 또는 최소 검증 명령 실행**
- [ ] **Step 3: 결과를 정리해 사용자에게 보고**
