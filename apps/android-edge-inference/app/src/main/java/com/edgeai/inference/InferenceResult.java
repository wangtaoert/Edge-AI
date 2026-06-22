package com.edgeai.inference;

class InferenceResult {
    final int classIndex;
    final float score;
    final long latencyMs;

    InferenceResult(int classIndex, float score, long latencyMs) {
        this.classIndex = classIndex;
        this.score = score;
        this.latencyMs = latencyMs;
    }
}
