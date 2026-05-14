# kor-minish HTTP API 클라이언트 예시

서버 띄우기:

```bash
uv run uvicorn server.server:app --host 0.0.0.0 --port 8000
```

또는 Docker:

```bash
docker build -t kor-minish .
docker run --rm -p 8000:8000 kor-minish
```

## 엔드포인트

| Method | Path | Body | 반환 |
|---|---|---|---|
| GET | `/health` | — | `{status, model, dim, vocab}` |
| POST | `/encode` | `{texts: string[], normalize?: bool}` | `{dim, embeddings: number[][]}` |
| POST | `/similarity` | `{query: string, docs: string[]}` | `{scores, order}` |

## 클라이언트 예시

### curl

```bash
HOST=http://localhost:8000 bash examples/curl.sh
```

### Node.js (18+)

```bash
HOST=http://localhost:8000 node examples/client.js
```

### Java (17+, single-file mode)

```bash
HOST=http://localhost:8000 java examples/Client.java
```

> Java HttpClient는 기본 HTTP/2이지만 uvicorn은 HTTP/1.1만 지원합니다. 클라이언트 코드는 `HttpClient.Version.HTTP_1_1`을 강제합니다.

### 기타 언어

표준 HTTP POST + JSON으로 어디서든 호출 가능합니다. 한국어가 들어가는 body는 반드시 **UTF-8**로 인코딩하세요.

## 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `KOR_MINISH_MODEL` | `kor-minish-bge-m3-ko` | 로컬 경로 또는 HF repo |
| `KOR_MINISH_BATCH` | `256` | `/encode` 한번에 받을 텍스트 수 상한 |
