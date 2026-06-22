package com.edgeai.inference;

import android.content.Context;
import android.content.SharedPreferences;

class AppPreferences {
    static final String DEFAULT_MODEL_URL =
            "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.25_224_quantized/1/default/1?lite-format=tflite";
    static final String DEFAULT_LABELS_URL =
            "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt";
    static final String MOBILENET_V1_FULL_INT8_URL =
            "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_1.0_224_quantized/1/default/1?lite-format=tflite";
    static final String MOBILENET_V1_FULL_FLOAT_URL =
            "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_1.0_224/1/metadata/1?lite-format=tflite";

    private static final String PREFS = "edge_ai_inference";
    private static final String KEY_MODEL_URL = "model_url";
    private static final String KEY_LABELS_URL = "labels_url";
    private static final String KEY_THREAD_COUNT = "thread_count";
    private static final String KEY_BENCHMARK_ITERATIONS = "benchmark_iterations";
    private static final String KEY_PREPROCESS_MODE = "preprocess_mode";
    private static final String KEY_RUNTIME_MODE = "runtime_mode";

    private final SharedPreferences preferences;

    AppPreferences(Context context) {
        preferences = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
    }

    String getModelUrl() {
        return preferences.getString(KEY_MODEL_URL, DEFAULT_MODEL_URL);
    }

    String getLabelsUrl() {
        return preferences.getString(KEY_LABELS_URL, DEFAULT_LABELS_URL);
    }

    void saveUrls(String modelUrl, String labelsUrl) {
        preferences.edit()
                .putString(KEY_MODEL_URL, modelUrl)
                .putString(KEY_LABELS_URL, labelsUrl)
                .apply();
    }

    int getThreadCount() {
        return preferences.getInt(KEY_THREAD_COUNT,
                Math.max(1, Runtime.getRuntime().availableProcessors() / 2));
    }

    int getBenchmarkIterations() {
        return preferences.getInt(KEY_BENCHMARK_ITERATIONS, 30);
    }

    void saveRuntimeSettings(int threadCount, int benchmarkIterations) {
        preferences.edit()
                .putInt(KEY_THREAD_COUNT, threadCount)
                .putInt(KEY_BENCHMARK_ITERATIONS, benchmarkIterations)
                .apply();
    }

    PreprocessConfig getPreprocessConfig() {
        return new PreprocessConfig(preferences.getString(KEY_PREPROCESS_MODE, PreprocessConfig.MODE_ZERO_ONE));
    }

    void savePreprocessMode(String mode) {
        preferences.edit()
                .putString(KEY_PREPROCESS_MODE, mode)
                .apply();
    }

    RuntimeConfig getRuntimeConfig() {
        return new RuntimeConfig(preferences.getString(KEY_RUNTIME_MODE, RuntimeConfig.MODE_CPU));
    }

    void saveRuntimeMode(String mode) {
        preferences.edit()
                .putString(KEY_RUNTIME_MODE, mode)
                .apply();
    }
}
