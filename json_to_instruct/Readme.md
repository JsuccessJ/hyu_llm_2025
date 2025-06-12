#  데이터 설명

##  perfume_instruct.py
- json 파일명을 입력하면 template 별로 train test를 나누고, train은 하나로 병합, test만 각 템플릿 별로 출력
- 총 5개의 템플릿이 train으로 4개의 템플릿이 test로 출력
- 남자, 여자 각각 9:1로 템플릿 별 train, test를 나눈 상태이며, 합치면 8:2로 되게끔 해놓음
- cmd 창에서 python 3.10 기준
  1) python perfume_instruct.py로 실행
  2) 변환하고자 하는 your_file.json 입력
  3) 변환 완료



## train_template_man.json, train_template_woman.json
- instruction tuning에 사용할 학습 데이터 셋
- 각각 따로 분리해놓은 상태이며, 추후 학습 때 합쳐서 사용할 수 있음

## test_ㅇㅇㅇ.json
- 모델 성능을 평가하기 위한 테스트 셋 파일
  
## failed_items_ㅇㅇㅇ.json
- 어떤 정보가 없어서 템플릿을 만들 수 없는지 따로 추출해놓은 파일
