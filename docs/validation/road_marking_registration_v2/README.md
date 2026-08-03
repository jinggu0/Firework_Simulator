# V2-2e S-Map 2024 영상지도 좌표 등록

서울 S-Map의 2024 배경지도를 여의도한강공원 자전거도로 후보 구간에 직접
등록했다. 제공자 픽셀은 재배포 범위가 확인되지 않았으므로 `local_reference/`에만
보존하고 Git에는 좌표 계약, URL, 체크섬과 후보 결속 결과만 기록한다.

## 확보 결과

- 브라우저에서 배경지도 `2024`, 3D 건물 `비활성화`, 고화질 상태를 확인했다.
- 실제 네트워크 응답은 `ortho_drone_25cm_2024` JPEG 타일이며 HTTP 200이었다.
- S-Map 지도 설정 소스가 명시한 EPSG:5186 원점·extent·해상도 배열·256px 타일을
  SHA-256으로 잠갔다.
- zoom 10의 0.25m/px 타일 36장을 로컬에 내려받아 1,536×1,536px,
  384m×384m 북향 모자이크와 world file을 생성했다.
- 크롭 안에는 OSM 자전거 차로 후보 3 way(`384826214`, `474507374`,
  `474507381`)의 렌더 구간 34개가 들어 있다.

로컬 원본 위치:

```text
local_reference/smap_2024_ortho/focus_z10_x634-639_y520-525.png
local_reference/smap_2024_ortho/focus_z10_x634-639_y520-525.pgw
```

## 판정

제공자 타일 격자를 그대로 사용하므로 별도 GCP 피팅 없이 픽셀 좌표가 EPSG:5186에
직접 결속된다. 다만 설정 소스의 `[202410]`과 `20241209` 주석은 2024년 10월
기간과 배포 시점을 시사할 뿐 정확한 촬영일을 증명하지 않는다. 2024-10-05 이전
촬영 여부와 독립 검사점 오차를 확인하기 전에는 보이는 자전거 기호·차선 형태를
행사일 런타임 표식으로 적용하지 않는다.

[S-Map 공식 사용 가이드](https://smap.seoul.go.kr/guide/guide.html)는 연도별
배경지도 선택과 현재 지도 화면의 PNG 저장 기능을 안내한다. 사이트 표기는
`© 서울특별시 모든 권리 보유`이며 원시 타일 재배포 허용 문구를 확인하지 못했다.
따라서 원본과 모자이크는 로컬 검증 전용이다.

## 재생성

사용자가 허용한 로컬 검증 범위에서만 실행한다.

```powershell
python -m tools.acquire_smap_ortho --accept-local-validation
python -m pytest tests/test_smap_ortho_registration.py -q
```

- `smap_2024_registration_report.json`: 타일별 URL·체크섬·EPSG:5186 범위와 게이트
- `smap_2024_candidate_coverage.png`: 제공자 픽셀을 포함하지 않는 후보 구간 지도
