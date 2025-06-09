# hyu_llm_2025

## 🔀 Git Branch Naming Convention & Workflow

### 🔧 Main Branches
- `main`: 배포 가능한 **최종 버전**만 유지하는 브랜치
- `develop`: 모든 기능 브랜치를 병합하는 **통합 개발 브랜치**

### 🌿 Feature Branches
- 형식: `feature/{이름}/{기능명}`
- 예시: `feature/jaeseong/login-api`, `feature/yujin/front-header`

개발자는 항상 `develop` 브랜치에서 분기하여 자신의 `feature` 브랜치를 만들고, 작업 완료 후 **Pull Request (PR)**를 통해 `develop`에 병합합니다.

### 🐛 Bugfix Branches
- 형식: `bugfix/{이름}/{버그설명}`
- 예시: `bugfix/minju/login-error`

### 📦 Release & Hotfix
- `release/{버전}`: 배포 직전 최종 조정 (ex. `release/v1.0`)
- `hotfix/{이름}/{수정내용}`: `main`에서 긴급 수정할 경우 사용

---

## 👥 팀원별 Prefix 예시
| 이름   | Prefix        |
|--------|---------------|
| 재성   | `jaesung`    |
| 민주   | `minjoo`       |
| 단익   | `danik`       |
| 유진   | `yujin`       |
| 주현   | `joohyun`     |
| 승환   | `seunghwan`   |

각자 작업할 때는 자신의 prefix로 브랜치명을 시작해주세요.

---

## 🔄 PR & Merge Rules
- 모든 기능은 `feature` 브랜치에서 개발하고, `develop`으로 PR을 생성합니다.
- 리뷰어 1명 이상 승인 후 merge합니다.
- `main` 브랜치는 직접 push하지 않고 `release` 혹은 `hotfix`를 통해 merge만 수행합니다.
- 커밋 메시지는 명확히! 예: `feat: 로그인 기능 구현`, `fix: 로그인 오류 수정`

---

## ✅ 브랜치 예시 요약

- `main`  
- `develop`  
- `feature/jaeseong/model-training`  
- `feature/minjoo/data-cleaning`  
- `bugfix/joohyun/null-error`  
- `hotfix/seunghwan/login-crash`  
