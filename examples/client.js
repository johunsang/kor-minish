// kor-minish HTTP API — Node.js 클라이언트 예시
// node 18+ (전역 fetch). 실행: node examples/client.js

const HOST = process.env.HOST || 'http://127.0.0.1:8765';

async function encode(texts, normalize = true) {
  const r = await fetch(`${HOST}/encode`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ texts, normalize }),
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

async function similarity(query, docs) {
  const r = await fetch(`${HOST}/similarity`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, docs }),
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

(async () => {
  const enc = await encode(['안녕하세요', '반갑습니다']);
  console.log('dim:', enc.dim, 'count:', enc.embeddings.length);

  const docs = ['김치찌개 레시피', '주식 매수 타이밍', '된장국 끓이는 법', '자동차 보험'];
  const sim = await similarity('한국 음식 만들기', docs);
  console.log('\nranked:');
  sim.order.forEach((idx, rank) => {
    console.log(`  ${rank + 1}. ${sim.scores[idx].toFixed(3)}  ${docs[idx]}`);
  });
})();
