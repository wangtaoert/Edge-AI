package com.edgeai.inference;

import android.content.Context;
import android.content.res.AssetFileDescriptor;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

class ModelRepository {
    static final String MODEL_ASSET = "model_int8.tflite";
    static final String LABELS_ASSET = "labels.txt";
    static final String DOWNLOADED_MODEL = "downloaded_model.tflite";
    static final String DOWNLOADED_LABELS = "downloaded_labels.txt";

    private final Context context;
    private Interpreter interpreter;
    private ModelInfo modelInfo;
    private List<String> labels = new ArrayList<>();
    private int threadCount = Math.max(1, Runtime.getRuntime().availableProcessors() / 2);
    private RuntimeConfig runtimeConfig = new RuntimeConfig(RuntimeConfig.MODE_CPU);

    ModelRepository(Context context) {
        this.context = context.getApplicationContext();
    }

    void loadModel() throws IOException {
        close();
        File downloadedModel = downloadedModelFile();
        if (downloadedModel.exists()) {
            try {
                openModelFromFile(downloadedModel, "downloaded file");
                return;
            } catch (IOException | IllegalArgumentException e) {
                ModelDownloadManager.deleteQuietly(downloadedModel);
                ModelDownloadManager.deleteQuietly(downloadedLabelsFile());
                throw new IOException("Downloaded model was invalid and has been removed: " + e.getMessage(), e);
            }
        }
        openModelFromAsset();
    }

    void setThreadCount(int threadCount) {
        this.threadCount = Math.max(1, threadCount);
    }

    void setRuntimeConfig(RuntimeConfig runtimeConfig) {
        this.runtimeConfig = runtimeConfig;
    }

    void installDownloadedModel(File candidateModel, File candidateLabels) throws IOException {
        validateModelFile(candidateModel);
        ModelDownloadManager.replaceFile(candidateModel, downloadedModelFile());
        if (candidateLabels != null) {
            ModelDownloadManager.replaceFile(candidateLabels, downloadedLabelsFile());
        } else {
            ModelDownloadManager.deleteQuietly(downloadedLabelsFile());
        }
    }

    void clearDownloadedModel() {
        ModelDownloadManager.deleteQuietly(downloadedModelFile());
        ModelDownloadManager.deleteQuietly(downloadedLabelsFile());
    }

    boolean hasDownloadedModel() {
        return downloadedModelFile().exists();
    }

    Interpreter getInterpreter() {
        return interpreter;
    }

    ModelInfo getModelInfo() {
        return modelInfo;
    }

    List<String> getLabels() {
        return labels;
    }

    File newCandidateModelFile() {
        return new File(context.getCacheDir(), "candidate_model.tflite");
    }

    File newCandidateLabelsFile() {
        return new File(context.getCacheDir(), "candidate_labels.txt");
    }

    File downloadedModelFile() {
        return new File(context.getFilesDir(), DOWNLOADED_MODEL);
    }

    File downloadedLabelsFile() {
        return new File(context.getFilesDir(), DOWNLOADED_LABELS);
    }

    void close() {
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
        }
    }

    private void openModelFromFile(File modelFile, String source) throws IOException {
        Interpreter newInterpreter = createInterpreter(loadModelFile(modelFile));
        applyInterpreter(newInterpreter, source, modelFile.length(), true);
    }

    private void openModelFromAsset() throws IOException {
        Interpreter newInterpreter = createInterpreter(loadModelFile(MODEL_ASSET));
        applyInterpreter(newInterpreter, "asset fallback", assetLength(MODEL_ASSET), false);
    }

    private Interpreter createInterpreter(MappedByteBuffer modelBuffer) {
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(threadCount);
        options.setUseNNAPI(runtimeConfig.useNnapi());
        return new Interpreter(modelBuffer, options);
    }

    private void validateModelFile(File modelFile) throws IOException {
        Interpreter testInterpreter = null;
        try {
            testInterpreter = createInterpreter(loadModelFile(modelFile));
            Tensor inputTensor = testInterpreter.getInputTensor(0);
            int[] inputShape = inputTensor.shape();
            if (inputShape.length != 4 || inputShape[0] != 1 || inputShape[3] != 3) {
                throw new IllegalArgumentException("Only NHWC image input [1,H,W,3] is supported. Got "
                        + Arrays.toString(inputShape));
            }
        } finally {
            if (testInterpreter != null) {
                testInterpreter.close();
            }
        }
    }

    private void applyInterpreter(Interpreter newInterpreter, String source, long modelSizeBytes, boolean useDownloadedLabels) {
        interpreter = newInterpreter;
        Tensor inputTensor = interpreter.getInputTensor(0);
        Tensor outputTensor = interpreter.getOutputTensor(0);
        modelInfo = new ModelInfo(
                source,
                inputTensor.dataType(),
                inputTensor.shape(),
                outputTensor.dataType(),
                outputTensor.shape(),
                modelSizeBytes);
        labels = loadLabels(useDownloadedLabels);
    }

    private long assetLength(String assetName) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(assetName);
        return fileDescriptor.getDeclaredLength();
    }

    private MappedByteBuffer loadModelFile(String assetName) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(assetName);
        try (FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
             FileChannel fileChannel = inputStream.getChannel()) {
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }
    }

    private MappedByteBuffer loadModelFile(File modelFile) throws IOException {
        try (FileInputStream inputStream = new FileInputStream(modelFile);
             FileChannel fileChannel = inputStream.getChannel()) {
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, modelFile.length());
        }
    }

    private List<String> loadLabels(boolean preferDownloadedLabels) {
        List<String> loaded = new ArrayList<>();
        File downloadedLabels = downloadedLabelsFile();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                preferDownloadedLabels && downloadedLabels.exists()
                        ? new FileInputStream(downloadedLabels)
                        : context.getAssets().open(LABELS_ASSET)))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (!line.trim().isEmpty()) {
                    loaded.add(line.trim());
                }
            }
        } catch (IOException ignored) {
            // Labels are optional; numeric class ids are still useful for benchmark runs.
        }
        return loaded;
    }
}
