# V2-2d 도로 표식 유형·배치 증거 감사

이 디렉터리는 2024-10-05 10:20 UTC OSM 원본의 표식 관련 태그를 현재 렌더
도로 way 결속과 대조한다. 목적은 후보 도로를 좁히는 것이며, 후보를 실제 표식
배치로 간주하지 않는다.

## 결과

| 분류 | way | 렌더 구간 | 중심선 길이 | 판정 |
|---|---:|---:|---:|---|
| 기존 명시적 일방통행 차로 경계 | 66 | 4,387 | 48.57 km | V2-2c 적용 유지 |
| `oneway=yes`가 아닌 다차로 중앙선 검토 후보 | 37 | 502 | 5.14 km | 유형·색·범위 미확정 |
| `lanes:forward/backward` 후보 | 5 | 103 | 1.12 km | 방향 분할 후보일 뿐 표식 유형 미확정 |
| 자전거 차로 태그 후보 | 31 | 694 | 7.41 km | 기호·선·폭·측면 미확정 |
| `crossing:markings=yes` 후보 | 3 | 6 | 0.04 km | 존재만 표시하며 형상 미확정 |

원본에는 `divider`, `shoulder`, `turn:lanes*`, `road_marking`, `lane_markings`,
`centre_line`/`center_line` 태그가 없다. 따라서 양방향 중앙선의 색과 실선/점선,
가장자리선, 회전 화살표, 자전거 기호와 횡단 표식 형상은 모두 차단 상태다.
이번 단계는 런타임 형상을 변경하지 않는다.

## 증거 능력과 한계

- 행사 시점 OSM은 way ID, 차로 수와 일부 자전거 시설 후보를 제공하지만 도색의
  정확한 색·형태·종방향 범위를 제공하지 않는다.
- [도로교통법 시행규칙 별표 6](https://law.go.kr/LSW/flDownload.do?bylClsCd=110201&flSeq=144234901)과
  [국토교통부 2024 차선도색 유지·관리 매뉴얼](https://www.molit.go.kr/portal/common/download/DownloadMltm2.jsp?FileName=%EC%B0%A8%EC%84%A0%EB%8F%84%EC%83%89+%EC%9C%A0%EC%A7%80%C2%B7%EA%B4%80%EB%A6%AC+%EB%A7%A4%EB%89%B4%EC%96%BC.pdf&FilePath=portal%2FDextUpload%2F202404%2F20240429_133733_839.pdf)은
  표식 종류와 치수 근거지만 각 OSM way의 행사일 배치를 지정하지 않는다.
- [서울 S-Map 공식 가이드](https://smap.seoul.go.kr/guide/guide.html)는 2024년
  배경지도 선택을 지원한다. 아직 영상 획득일, 재사용 범위, 원본 해상도와
  월드 좌표 등록 잔차를 확인하지 않았으므로 현재 판정에는 픽셀 증거로 쓰지 않았다.
- [서울 자전거도로 통계](https://data.seoul.go.kr/bsp/wgs/dataView/data300View/428.do)는
  시설 유형의 정의와 집계이며 행사일 way별 도색 형상이 아니다.

다음 단계 V2-2e는 S-Map 2024 영상지도에서 획득일과 사용 범위를 보존한 로컬 검증
캡처를 만들고, 각 크롭을 최소 제어점 4개·독립 검사점 2개로 등록해 평면 잔차
1.0m 이하를 통과시킨 뒤 확인된 표식을 OSM way ID에 연결한다.

## 재생성

```powershell
python -m tools.audit_road_marking_evidence
python -m pytest tests/test_road_marking_evidence.py -q
```

- `road_marking_evidence_report.json`: 입력 체크섬, 태그 전수, way별 후보와 독립 게이트
- `road_marking_evidence_map.png`: 후보 그룹의 전체 렌더 도로 공간 분포
