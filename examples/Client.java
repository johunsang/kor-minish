// kor-minish HTTP API — Java 클라이언트 예시 (Java 17+, JEP 408 단일파일 실행)
// 실행:  java examples/Client.java
//
// JSON은 단순 정규식으로 파싱 — 실서비스에서는 Jackson/Gson 권장.

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.util.Comparator;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Client {
    static final String HOST = System.getenv().getOrDefault("HOST", "http://127.0.0.1:8765");
    static final HttpClient http = HttpClient.newBuilder()
            .version(HttpClient.Version.HTTP_1_1)
            .build();

    static String post(String path, String body) throws Exception {
        HttpRequest req = HttpRequest.newBuilder(URI.create(HOST + path))
                .header("Content-Type", "application/json; charset=utf-8")
                .POST(HttpRequest.BodyPublishers.ofByteArray(body.getBytes(StandardCharsets.UTF_8)))
                .build();
        HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
        if (resp.statusCode() / 100 != 2) {
            throw new RuntimeException("HTTP " + resp.statusCode() + ": " + resp.body());
        }
        return resp.body();
    }

    static String jsonStringArray(List<String> items) {
        return items.stream()
                .map(s -> "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\"")
                .collect(Collectors.joining(",", "[", "]"));
    }

    static double[] parseScores(String json) {
        Matcher m = Pattern.compile("\"scores\"\\s*:\\s*\\[([^\\]]*)\\]").matcher(json);
        if (!m.find()) throw new RuntimeException("scores not found: " + json);
        String[] parts = m.group(1).split(",");
        double[] out = new double[parts.length];
        for (int i = 0; i < parts.length; i++) out[i] = Double.parseDouble(parts[i].trim());
        return out;
    }

    public static void main(String[] args) throws Exception {
        String docsJson = jsonStringArray(List.of(
                "김치찌개 레시피", "주식 매수 타이밍", "된장국 끓이는 법", "자동차 보험"));
        String body = "{\"query\":\"한국 음식 만들기\",\"docs\":" + docsJson + "}";

        String resp = post("/similarity", body);
        double[] scores = parseScores(resp);

        List<String> docs = List.of("김치찌개 레시피", "주식 매수 타이밍", "된장국 끓이는 법", "자동차 보험");
        List<Integer> order = IntStream.range(0, scores.length).boxed()
                .sorted(Comparator.comparingDouble((Integer i) -> -scores[i]))
                .collect(Collectors.toList());

        System.out.println("ranked:");
        for (int rank = 0; rank < order.size(); rank++) {
            int idx = order.get(rank);
            System.out.printf("  %d. %.3f  %s%n", rank + 1, scores[idx], docs.get(idx));
        }
    }
}
