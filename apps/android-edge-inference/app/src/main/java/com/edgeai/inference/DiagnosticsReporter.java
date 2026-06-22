package com.edgeai.inference;

import android.os.Build;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

class DiagnosticsReporter {
    String buildReport(
            ModelRepository modelRepository,
            AppPreferences preferences,
            String modelUrl,
            String labelsUrl,
            String lastResultText,
            File externalFilesDir,
            File errorLogFile,
            String lastError) {
        StringBuilder builder = new StringBuilder();
        builder.append("Edge AI Diagnostics\n");
        builder.append("Generated: ")
                .append(new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(new Date()))
                .append("\n\n");

        builder.append("App\n");
        builder.append("Package: com.edgeai.inference\n");
        builder.append("Version: 0.1.0 (1)\n\n");

        builder.append("Device\n");
        builder.append("Manufacturer: ").append(Build.MANUFACTURER).append("\n");
        builder.append("Model: ").append(Build.MODEL).append("\n");
        builder.append("Android SDK: ").append(Build.VERSION.SDK_INT).append("\n");
        builder.append("Android release: ").append(Build.VERSION.RELEASE).append("\n");
        builder.append("ABI: ").append(Build.SUPPORTED_ABIS.length > 0 ? Build.SUPPORTED_ABIS[0] : "unknown").append("\n\n");

        builder.append("Runtime settings\n");
        builder.append("Runtime: ").append(preferences.getRuntimeConfig().getLabel()).append("\n");
        builder.append("Threads: ").append(preferences.getThreadCount()).append("\n");
        builder.append("Benchmark runs: ").append(preferences.getBenchmarkIterations()).append("\n");
        builder.append("Preprocess: ").append(preferences.getPreprocessConfig().getLabel()).append("\n\n");

        builder.append("Model\n");
        ModelInfo modelInfo = modelRepository.getModelInfo();
        if (modelInfo != null) {
            builder.append(modelInfo.reportSummary(modelRepository.getLabels().size())).append("\n");
        } else {
            builder.append("Model: not loaded\n");
        }
        builder.append("Downloaded model exists: ").append(modelRepository.hasDownloadedModel()).append("\n");
        builder.append("Model URL: ").append(modelUrl).append("\n");
        builder.append("Labels URL: ").append(labelsUrl).append("\n");
        builder.append("External files dir: ")
                .append(externalFilesDir != null ? externalFilesDir.getAbsolutePath() : "unavailable")
                .append("\n\n");

        builder.append("Logs\n");
        builder.append("Error log: ")
                .append(errorLogFile != null ? errorLogFile.getAbsolutePath() : "unavailable")
                .append("\n");
        builder.append("Last error: ").append(lastError).append("\n\n");

        builder.append("Last result\n");
        builder.append(lastResultText == null || lastResultText.trim().isEmpty() ? "No result yet." : lastResultText);
        return builder.toString();
    }
}
