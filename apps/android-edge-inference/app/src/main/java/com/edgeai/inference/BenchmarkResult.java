package com.edgeai.inference;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

class BenchmarkResult {
    final InferenceResult lastInference;
    final long firstRunMs;
    final long minMs;
    final long p50Ms;
    final long p95Ms;
    final long maxMs;

    BenchmarkResult(InferenceResult lastInference, long firstRunMs, long minMs, long p50Ms, long p95Ms, long maxMs) {
        this.lastInference = lastInference;
        this.firstRunMs = firstRunMs;
        this.minMs = minMs;
        this.p50Ms = p50Ms;
        this.p95Ms = p95Ms;
        this.maxMs = maxMs;
    }

    static BenchmarkResult from(InferenceResult firstInference, InferenceResult lastInference, List<Long> warmTimes) {
        List<Long> sorted = new ArrayList<>(warmTimes);
        sorted.sort(Long::compareTo);
        return new BenchmarkResult(
                lastInference,
                firstInference.latencyMs,
                sorted.get(0),
                percentile(sorted, 0.50),
                percentile(sorted, 0.95),
                sorted.get(sorted.size() - 1));
    }

    String format(String formattedInference) {
        return formattedInference
                + String.format(Locale.US,
                "\nFirst run latency: %d ms\nWarm min latency: %d ms\nWarm P50 latency: %d ms\nWarm P95 latency: %d ms\nWarm max latency: %d ms",
                firstRunMs, minMs, p50Ms, p95Ms, maxMs);
    }

    private static long percentile(List<Long> sortedTimes, double percentile) {
        int index = Math.min(sortedTimes.size() - 1,
                Math.max(0, (int) Math.ceil(sortedTimes.size() * percentile) - 1));
        return sortedTimes.get(index);
    }
}
