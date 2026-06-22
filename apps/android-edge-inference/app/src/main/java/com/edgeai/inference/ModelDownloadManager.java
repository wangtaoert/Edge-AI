package com.edgeai.inference;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Locale;

class ModelDownloadManager {
    interface ProgressListener {
        void onProgress(String message);
    }

    void downloadToFile(String urlText, File target, String label, ProgressListener listener) throws IOException {
        URL url = new URL(urlText);
        String protocol = url.getProtocol();
        if (!"https".equals(protocol)) {
            throw new IOException("Only https URLs are supported.");
        }

        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.setConnectTimeout(15_000);
        connection.setReadTimeout(30_000);
        connection.setInstanceFollowRedirects(true);

        int responseCode = connection.getResponseCode();
        if (responseCode < 200 || responseCode >= 300) {
            throw new IOException("HTTP " + responseCode + " for " + urlText);
        }

        int contentLength = connection.getContentLength();
        File part = new File(target.getParentFile(), target.getName() + ".part");
        try (InputStream inputStream = connection.getInputStream();
             FileOutputStream outputStream = new FileOutputStream(part)) {
            byte[] buffer = new byte[64 * 1024];
            long totalRead = 0;
            int lastPercent = -1;
            int read;
            while ((read = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, read);
                totalRead += read;
                if (contentLength > 0) {
                    int percent = (int) (totalRead * 100 / contentLength);
                    if (percent >= lastPercent + 5 || percent == 100) {
                        lastPercent = percent;
                        listener.onProgress(String.format(Locale.US, "Downloading %s: %d%%", label, percent));
                    }
                } else {
                    listener.onProgress(String.format(Locale.US,
                            "Downloading %s: %.2f MB",
                            label,
                            totalRead / 1024.0 / 1024.0));
                }
            }
        } finally {
            connection.disconnect();
        }

        if (part.length() == 0) {
            deleteQuietly(part);
            throw new IOException("Downloaded file is empty.");
        }
        replaceFile(part, target);
    }

    static void replaceFile(File source, File target) throws IOException {
        if (target.exists() && !target.delete()) {
            throw new IOException("Failed to replace " + target.getName());
        }
        if (!source.renameTo(target)) {
            throw new IOException("Failed to save " + target.getName());
        }
    }

    static void deleteQuietly(File file) {
        if (file.exists()) {
            file.delete();
        }
    }
}
