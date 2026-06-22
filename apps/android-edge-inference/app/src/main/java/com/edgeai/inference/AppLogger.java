package com.edgeai.inference;

import android.content.Context;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

class AppLogger {
    private static final String LOG_FILE = "edge_ai_errors.txt";

    private final File logFile;
    private String lastError = "";

    AppLogger(Context context) {
        logFile = new File(context.getExternalFilesDir(null), LOG_FILE);
    }

    void log(String area, Throwable throwable) {
        String timestamp = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(new Date());
        StringWriter stackTrace = new StringWriter();
        throwable.printStackTrace(new PrintWriter(stackTrace));
        lastError = "[" + timestamp + "] " + area + ": " + throwable.getMessage();

        try (FileWriter writer = new FileWriter(logFile, true)) {
            writer.write(lastError);
            writer.write("\n");
            writer.write(stackTrace.toString());
            writer.write("\n");
        } catch (IOException ignored) {
            // Logging must never break the app's user-facing flow.
        }
    }

    String getLastError() {
        return lastError.isEmpty() ? "No logged errors in this session." : lastError;
    }

    File getLogFile() {
        return logFile;
    }
}
