package com.edgeai.inference;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.view.Gravity;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class MainActivity extends Activity {
    private static final int PICK_IMAGE_REQUEST = 1001;
    private static final int PAGE_MODEL = 0;
    private static final int PAGE_SETTINGS = 1;
    private static final int PAGE_IMAGE = 2;
    private static final int PAGE_INFERENCE = 3;
    private static final int PAGE_SUPPORT = 4;

    private ModelRepository modelRepository;
    private ModelDownloadManager downloadManager;
    private TfliteImageClassifier classifier;
    private AppPreferences appPreferences;
    private DiagnosticsReporter diagnosticsReporter;
    private AppLogger appLogger;

    private Bitmap selectedBitmap;
    private boolean downloadInProgress;

    private EditText modelUrlInput;
    private EditText labelsUrlInput;
    private EditText threadCountInput;
    private EditText benchmarkIterationsInput;
    private ImageView preview;
    private ProgressBar progressBar;
    private LinearLayout modelCard;
    private LinearLayout settingsCard;
    private LinearLayout imageCard;
    private LinearLayout runCard;
    private LinearLayout supportCard;
    private TextView modelStatus;
    private TextView modelDescription;
    private TextView downloadStatus;
    private TextView imageStatus;
    private TextView result;
    private Button downloadButton;
    private Button defaultModelButton;
    private Button clearModelButton;
    private final List<Button> presetButtons = new ArrayList<>();
    private final List<Button> preprocessButtons = new ArrayList<>();
    private final List<Button> runtimeButtons = new ArrayList<>();
    private final List<Button> navButtons = new ArrayList<>();
    private Button applySettingsButton;
    private Button pickImageButton;
    private Button runOnceButton;
    private Button benchmarkButton;
    private Button copyResultButton;
    private Button shareResultButton;
    private Button saveResultButton;
    private Button aboutButton;
    private Button privacyButton;
    private Button diagnosticsButton;
    private String lastResultText = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        modelRepository = new ModelRepository(this);
        downloadManager = new ModelDownloadManager();
        classifier = new TfliteImageClassifier();
        appPreferences = new AppPreferences(this);
        diagnosticsReporter = new DiagnosticsReporter();
        appLogger = new AppLogger(this);
        modelRepository.setThreadCount(appPreferences.getThreadCount());
        modelRepository.setRuntimeConfig(appPreferences.getRuntimeConfig());

        setContentView(buildUi());
        restoreInputs();
        loadModel();
        updateActions();
    }

    @Override
    protected void onDestroy() {
        modelRepository.close();
        super.onDestroy();
    }

    private LinearLayout buildUi() {
        LinearLayout screen = new LinearLayout(this);
        screen.setOrientation(LinearLayout.VERTICAL);
        screen.setBackgroundColor(Color.rgb(246, 248, 250));

        ScrollView scrollView = new ScrollView(this);
        scrollView.setBackgroundColor(Color.rgb(246, 248, 250));
        screen.addView(scrollView, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                0,
                1f));

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(16), dp(20), dp(16), dp(24));
        scrollView.addView(root);

        TextView title = new TextView(this);
        title.setText("端侧 AI 推理");
        title.setTextSize(26);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        title.setTextColor(Color.rgb(20, 28, 38));
        root.addView(title, matchWrap());

        TextView subtitle = new TextView(this);
        subtitle.setText("管理模型、选择图片、运行推理，并记录端侧性能证据。");
        subtitle.setTextSize(14);
        subtitle.setTextColor(Color.rgb(82, 94, 110));
        subtitle.setPadding(0, dp(4), 0, dp(12));
        root.addView(subtitle, matchWrap());

        modelCard = card();
        root.addView(modelCard, matchWrapWithBottom());
        modelCard.addView(sectionTitle("模型"));

        modelUrlInput = new EditText(this);
        modelUrlInput.setHint("模型 URL（.tflite，HTTPS 直链）");
        modelUrlInput.setSingleLine(true);
        modelCard.addView(modelUrlInput, matchWrap());

        labelsUrlInput = new EditText(this);
        labelsUrlInput.setHint("标签 URL（可选，.txt）");
        labelsUrlInput.setSingleLine(true);
        modelCard.addView(labelsUrlInput, matchWrap());

        TextView presetHint = bodyText();
        presetHint.setText("预设模型库");
        modelCard.addView(presetHint, matchWrap());

        LinearLayout presetRowOne = row();
        Button presetSmall = presetButton("轻量 INT8", AppPreferences.DEFAULT_MODEL_URL,
                "MobileNet V1 0.25 量化模型，体积小，适合快速验证下载、加载和推理链路。");
        presetRowOne.addView(presetSmall, weighted());
        Button presetFull = presetButton("完整 INT8", AppPreferences.MOBILENET_V1_FULL_INT8_URL,
                "MobileNet V1 1.0 量化模型，适合做更完整的 INT8 图片分类性能基线。");
        presetRowOne.addView(presetFull, weighted());
        modelCard.addView(presetRowOne, matchWrap());

        LinearLayout presetRowTwo = row();
        Button presetFloat = presetButton("浮点模型", AppPreferences.MOBILENET_V1_FULL_FLOAT_URL,
                "MobileNet V1 1.0 浮点模型，适合和 INT8 模型对比体积、速度和输出差异。");
        presetRowTwo.addView(presetFloat, weighted());
        modelCard.addView(presetRowTwo, matchWrap());

        modelDescription = bodyText();
        modelDescription.setText("选择一个预设模型，或手动输入模型 URL。建议先使用“轻量 INT8”。");
        modelCard.addView(modelDescription, matchWrap());

        LinearLayout modelActions = row();
        downloadButton = new Button(this);
        downloadButton.setText("下载模型");
        downloadButton.setOnClickListener(v -> downloadModelFromInputs());
        modelActions.addView(downloadButton, weighted());

        defaultModelButton = new Button(this);
        defaultModelButton.setText("默认模型");
        defaultModelButton.setOnClickListener(v -> useDefaultModelUrls());
        modelActions.addView(defaultModelButton, weighted());

        clearModelButton = new Button(this);
        clearModelButton.setText("使用内置");
        clearModelButton.setOnClickListener(v -> clearDownloadedModel());
        modelActions.addView(clearModelButton, weighted());
        modelCard.addView(modelActions, matchWrap());

        progressBar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        progressBar.setIndeterminate(true);
        progressBar.setVisibility(ProgressBar.GONE);
        modelCard.addView(progressBar, matchWrap());

        modelStatus = bodyText();
        modelCard.addView(modelStatus, matchWrap());

        downloadStatus = bodyText();
        downloadStatus.setText("可以下载模型。");
        modelCard.addView(downloadStatus, matchWrap());

        settingsCard = card();
        root.addView(settingsCard, matchWrapWithBottom());
        settingsCard.addView(sectionTitle("设置"));

        LinearLayout settingsRow = row();
        threadCountInput = new EditText(this);
        threadCountInput.setHint("线程数");
        threadCountInput.setSingleLine(true);
        settingsRow.addView(threadCountInput, weighted());

        benchmarkIterationsInput = new EditText(this);
        benchmarkIterationsInput.setHint("测试次数");
        benchmarkIterationsInput.setSingleLine(true);
        settingsRow.addView(benchmarkIterationsInput, weighted());
        settingsCard.addView(settingsRow, matchWrap());

        TextView runtimeHint = bodyText();
        runtimeHint.setText("运行后端");
        settingsCard.addView(runtimeHint, matchWrap());

        LinearLayout runtimeRow = row();
        runtimeRow.addView(runtimeButton("CPU", RuntimeConfig.MODE_CPU), weighted());
        runtimeRow.addView(runtimeButton("NNAPI", RuntimeConfig.MODE_NNAPI), weighted());
        settingsCard.addView(runtimeRow, matchWrap());

        TextView preprocessHint = bodyText();
        preprocessHint.setText("图像预处理");
        settingsCard.addView(preprocessHint, matchWrap());

        LinearLayout preprocessRowOne = row();
        preprocessRowOne.addView(preprocessButton("[0,1]", PreprocessConfig.MODE_ZERO_ONE), weighted());
        preprocessRowOne.addView(preprocessButton("[-1,1]", PreprocessConfig.MODE_MINUS_ONE_ONE), weighted());
        settingsCard.addView(preprocessRowOne, matchWrap());

        LinearLayout preprocessRowTwo = row();
        preprocessRowTwo.addView(preprocessButton("ImageNet", PreprocessConfig.MODE_IMAGENET), weighted());
        settingsCard.addView(preprocessRowTwo, matchWrap());

        applySettingsButton = new Button(this);
        applySettingsButton.setText("应用设置");
        applySettingsButton.setOnClickListener(v -> applyRuntimeSettings());
        settingsCard.addView(applySettingsButton, matchWrap());

        imageCard = card();
        root.addView(imageCard, matchWrapWithBottom());
        imageCard.addView(sectionTitle("图片"));

        preview = new ImageView(this);
        preview.setAdjustViewBounds(true);
        preview.setScaleType(ImageView.ScaleType.CENTER_CROP);
        preview.setBackgroundColor(Color.rgb(226, 232, 240));
        imageCard.addView(preview, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                dp(260)));

        pickImageButton = new Button(this);
        pickImageButton.setText("选择图片");
        pickImageButton.setOnClickListener(v -> pickImage());
        imageCard.addView(pickImageButton, matchWrap());

        imageStatus = bodyText();
        imageStatus.setText("尚未选择图片。");
        imageCard.addView(imageStatus, matchWrap());

        runCard = card();
        root.addView(runCard, matchWrapWithBottom());
        runCard.addView(sectionTitle("推理"));

        LinearLayout runActions = row();
        runOnceButton = new Button(this);
        runOnceButton.setText("单次推理");
        runOnceButton.setOnClickListener(v -> runInferenceOnce());
        runActions.addView(runOnceButton, weighted());

        benchmarkButton = new Button(this);
        benchmarkButton.setOnClickListener(v -> runBenchmark(appPreferences.getBenchmarkIterations()));
        runActions.addView(benchmarkButton, weighted());
        runCard.addView(runActions, matchWrap());

        LinearLayout resultActions = row();
        copyResultButton = new Button(this);
        copyResultButton.setText("复制");
        copyResultButton.setOnClickListener(v -> copyLastResult());
        resultActions.addView(copyResultButton, weighted());

        shareResultButton = new Button(this);
        shareResultButton.setText("分享");
        shareResultButton.setOnClickListener(v -> shareLastResult());
        resultActions.addView(shareResultButton, weighted());

        saveResultButton = new Button(this);
        saveResultButton.setText("保存");
        saveResultButton.setOnClickListener(v -> saveLastResult());
        resultActions.addView(saveResultButton, weighted());
        runCard.addView(resultActions, matchWrap());

        result = bodyText();
        setResultText("请先加载模型并选择图片。");
        runCard.addView(result, matchWrap());

        supportCard = card();
        root.addView(supportCard, matchWrapWithBottom());
        supportCard.addView(sectionTitle("支持"));

        LinearLayout supportRow = row();
        aboutButton = new Button(this);
        aboutButton.setText("关于");
        aboutButton.setOnClickListener(v -> showAbout());
        supportRow.addView(aboutButton, weighted());

        privacyButton = new Button(this);
        privacyButton.setText("隐私");
        privacyButton.setOnClickListener(v -> showPrivacy());
        supportRow.addView(privacyButton, weighted());

        diagnosticsButton = new Button(this);
        diagnosticsButton.setText("诊断");
        diagnosticsButton.setOnClickListener(v -> showDiagnostics());
        supportRow.addView(diagnosticsButton, weighted());
        supportCard.addView(supportRow, matchWrap());

        LinearLayout bottomNav = row();
        bottomNav.setPadding(dp(8), dp(6), dp(8), dp(8));
        bottomNav.setBackgroundColor(Color.WHITE);
        bottomNav.addView(navButton("模型", PAGE_MODEL), weighted());
        bottomNav.addView(navButton("设置", PAGE_SETTINGS), weighted());
        bottomNav.addView(navButton("图片", PAGE_IMAGE), weighted());
        bottomNav.addView(navButton("推理", PAGE_INFERENCE), weighted());
        bottomNav.addView(navButton("支持", PAGE_SUPPORT), weighted());
        screen.addView(bottomNav, matchWrap());

        showPage(PAGE_MODEL);
        return screen;
    }

    private void restoreInputs() {
        modelUrlInput.setText(appPreferences.getModelUrl());
        labelsUrlInput.setText(appPreferences.getLabelsUrl());
        threadCountInput.setText(String.valueOf(appPreferences.getThreadCount()));
        benchmarkIterationsInput.setText(String.valueOf(appPreferences.getBenchmarkIterations()));
        updateBenchmarkButtonText();
    }

    private void useDefaultModelUrls() {
        applyPresetUrls(AppPreferences.DEFAULT_MODEL_URL);
        modelDescription.setText("已恢复默认轻量 INT8 图片分类模型。点击“下载模型”即可使用。");
        downloadStatus.setText("已恢复默认模型 URL。");
    }

    private Button presetButton(String label, String modelUrl, String description) {
        Button button = new Button(this);
        button.setText(label);
        button.setOnClickListener(v -> {
            applyPresetUrls(modelUrl);
            modelDescription.setText(description);
            downloadStatus.setText("已选择：" + label + "。点击“下载模型”开始下载。");
        });
        presetButtons.add(button);
        return button;
    }

    private Button navButton(String label, int page) {
        Button button = new Button(this);
        button.setText(label);
        button.setOnClickListener(v -> showPage(page));
        navButtons.add(button);
        return button;
    }

    private void showPage(int page) {
        if (modelCard == null) {
            return;
        }
        modelCard.setVisibility(page == PAGE_MODEL ? View.VISIBLE : View.GONE);
        settingsCard.setVisibility(page == PAGE_SETTINGS ? View.VISIBLE : View.GONE);
        imageCard.setVisibility(page == PAGE_IMAGE ? View.VISIBLE : View.GONE);
        runCard.setVisibility(page == PAGE_INFERENCE ? View.VISIBLE : View.GONE);
        supportCard.setVisibility(page == PAGE_SUPPORT ? View.VISIBLE : View.GONE);
        for (Button button : navButtons) {
            boolean selected = ("模型".contentEquals(button.getText()) && page == PAGE_MODEL)
                    || ("设置".contentEquals(button.getText()) && page == PAGE_SETTINGS)
                    || ("图片".contentEquals(button.getText()) && page == PAGE_IMAGE)
                    || ("推理".contentEquals(button.getText()) && page == PAGE_INFERENCE)
                    || ("支持".contentEquals(button.getText()) && page == PAGE_SUPPORT);
            button.setTextColor(selected ? Color.WHITE : Color.rgb(20, 28, 38));
            button.setBackgroundColor(selected ? Color.rgb(0, 109, 119) : Color.rgb(232, 237, 243));
        }
    }

    private Button preprocessButton(String label, String mode) {
        Button button = new Button(this);
        button.setText(label);
        button.setOnClickListener(v -> {
            appPreferences.savePreprocessMode(mode);
            setStatus(downloadStatus, StatusKind.INFO,
                    "预处理模式已设置为：" + appPreferences.getPreprocessConfig().getLabel());
            updatePreprocessButtons();
        });
        preprocessButtons.add(button);
        return button;
    }

    private Button runtimeButton(String label, String mode) {
        Button button = new Button(this);
        button.setText(label);
        button.setOnClickListener(v -> {
            appPreferences.saveRuntimeMode(mode);
            modelRepository.setRuntimeConfig(appPreferences.getRuntimeConfig());
            loadModel();
            setStatus(downloadStatus, StatusKind.INFO,
                    "运行后端已设置为：" + appPreferences.getRuntimeConfig().getLabel());
            updateRuntimeButtons();
        });
        runtimeButtons.add(button);
        return button;
    }

    private void applyPresetUrls(String modelUrl) {
        modelUrlInput.setText(modelUrl);
        labelsUrlInput.setText(AppPreferences.DEFAULT_LABELS_URL);
        appPreferences.saveUrls(modelUrl, AppPreferences.DEFAULT_LABELS_URL);
    }

    private void loadModel() {
        try {
            modelRepository.loadModel();
            ModelInfo modelInfo = modelRepository.getModelInfo();
            setStatus(modelStatus, StatusKind.SUCCESS, modelInfo.format(modelRepository.getLabels().size()));
        } catch (IOException | IllegalArgumentException e) {
            appLogger.log("loadModel", e);
            setStatus(modelStatus, StatusKind.ERROR, "没有可用模型。请下载模型，或把 "
                    + ModelRepository.MODEL_ASSET + " 放到 assets 目录。\n" + e.getMessage());
        }
        updateActions();
    }

    private void downloadModelFromInputs() {
        String modelUrl = modelUrlInput.getText().toString().trim();
        String labelsUrl = labelsUrlInput.getText().toString().trim();
        if (modelUrl.isEmpty()) {
            setStatus(downloadStatus, StatusKind.WARNING, "请先输入模型 URL。");
            return;
        }

        appPreferences.saveUrls(modelUrl, labelsUrl);
        setDownloadInProgress(true, "正在准备下载...");

        new Thread(() -> {
            File candidateModel = modelRepository.newCandidateModelFile();
            File candidateLabels = modelRepository.newCandidateLabelsFile();
            try {
                downloadManager.downloadToFile(modelUrl, candidateModel, "model", this::postDownloadStatus);
                boolean hasNewLabels = !labelsUrl.isEmpty();
                if (hasNewLabels) {
                    downloadManager.downloadToFile(labelsUrl, candidateLabels, "labels", this::postDownloadStatus);
                }

                modelRepository.installDownloadedModel(candidateModel, hasNewLabels ? candidateLabels : null);

                runOnUiThread(() -> {
                    loadModel();
                    setDownloadInProgress(false, String.format(Locale.US,
                            "模型已下载并校验通过。大小：%.2f MB",
                            modelRepository.downloadedModelFile().length() / 1024.0 / 1024.0));
                });
            } catch (IOException | IllegalArgumentException e) {
                appLogger.log("downloadModel", e);
                ModelDownloadManager.deleteQuietly(candidateModel);
                ModelDownloadManager.deleteQuietly(candidateLabels);
                runOnUiThread(() -> setDownloadInProgress(false,
                        "下载失败，已保留现有模型。\n" + e.getMessage()));
            }
        }).start();
    }

    private void applyRuntimeSettings() {
        try {
            int threadCount = parsePositiveInt(threadCountInput.getText().toString(), "thread count");
            int benchmarkIterations = parsePositiveInt(benchmarkIterationsInput.getText().toString(), "benchmark runs");
            if (threadCount > 16) {
                throw new IllegalArgumentException("线程数建议不超过 16。");
            }
            if (benchmarkIterations > 500) {
                throw new IllegalArgumentException("测试次数建议不超过 500。");
            }
            appPreferences.saveRuntimeSettings(threadCount, benchmarkIterations);
            modelRepository.setThreadCount(threadCount);
            modelRepository.setRuntimeConfig(appPreferences.getRuntimeConfig());
            loadModel();
            updateBenchmarkButtonText();
            setStatus(downloadStatus, StatusKind.SUCCESS, "设置已应用。模型已用 "
                    + threadCount + " 个线程重新加载。运行后端：" + appPreferences.getRuntimeConfig().getLabel()
                    + "。预处理：" + appPreferences.getPreprocessConfig().getLabel());
        } catch (IllegalArgumentException e) {
            appLogger.log("applyRuntimeSettings", e);
            setStatus(downloadStatus, StatusKind.ERROR, "设置未应用：" + e.getMessage());
        }
    }

    private void postDownloadStatus(String message) {
        runOnUiThread(() -> setStatus(downloadStatus, StatusKind.INFO, message));
    }

    private void setDownloadInProgress(boolean inProgress, String message) {
        downloadInProgress = inProgress;
        setStatus(downloadStatus, inProgress ? StatusKind.INFO : StatusKind.SUCCESS, message);
        progressBar.setVisibility(inProgress ? ProgressBar.VISIBLE : ProgressBar.GONE);
        updateActions();
    }

    private void clearDownloadedModel() {
        modelRepository.clearDownloadedModel();
        setStatus(downloadStatus, StatusKind.INFO, "已清除下载模型。如 assets 中有内置模型，将使用内置模型。");
        loadModel();
    }

    private void pickImage() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("image/*");
        startActivityForResult(Intent.createChooser(intent, "选择图片"), PICK_IMAGE_REQUEST);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == PICK_IMAGE_REQUEST && resultCode == RESULT_OK && data != null) {
            Uri uri = data.getData();
            if (uri == null) {
                return;
            }
            try (InputStream inputStream = getContentResolver().openInputStream(uri)) {
                selectedBitmap = BitmapFactory.decodeStream(inputStream);
                if (selectedBitmap == null) {
                    setStatus(imageStatus, StatusKind.ERROR, "Failed to decode selected image.");
                    return;
                }
                preview.setImageBitmap(selectedBitmap);
                setStatus(imageStatus, StatusKind.SUCCESS, String.format(Locale.US,
                        "Selected image: %d x %d",
                        selectedBitmap.getWidth(),
                        selectedBitmap.getHeight()));
                setResultText("Image ready.");
            } catch (IOException e) {
                appLogger.log("pickImage", e);
                setStatus(imageStatus, StatusKind.ERROR, "Failed to load image: " + e.getMessage());
            }
            updateActions();
        }
    }

    private void runInferenceOnce() {
        if (!readyForInference()) {
            return;
        }
        try {
            InferenceResult inferenceResult = classifier.runInference(
                    modelRepository.getInterpreter(),
                    selectedBitmap,
                    appPreferences.getPreprocessConfig());
            setResultText(formatResult(inferenceResult, "Single run"));
        } catch (IllegalArgumentException e) {
            appLogger.log("runInferenceOnce", e);
            setResultText("Inference failed: " + e.getMessage());
        }
    }

    private void runBenchmark(int iterations) {
        if (!readyForInference()) {
            return;
        }
        try {
            BenchmarkResult benchmarkResult = classifier.benchmark(
                    modelRepository.getInterpreter(),
                    selectedBitmap,
                    iterations,
                    appPreferences.getPreprocessConfig());
            setResultText(benchmarkResult.format(formatResult(benchmarkResult.lastInference, "Benchmark x" + iterations)));
        } catch (IllegalArgumentException e) {
            appLogger.log("runBenchmark", e);
            setResultText("Benchmark failed: " + e.getMessage());
        }
    }

    private boolean readyForInference() {
        if (modelRepository.getInterpreter() == null) {
            setResultText("Model is not loaded.");
            return false;
        }
        if (selectedBitmap == null) {
            setResultText("Pick an image first.");
            return false;
        }
        return true;
    }

    private String formatResult(InferenceResult inferenceResult, String title) {
        List<String> labels = modelRepository.getLabels();
        String label = inferenceResult.classIndex < labels.size()
                ? labels.get(inferenceResult.classIndex)
                : "class_" + inferenceResult.classIndex;
        return String.format(Locale.US,
                "%s\nTop-1: %s\nScore: %.5f\nLatency: %d ms",
                title,
                label,
                inferenceResult.score,
                inferenceResult.latencyMs);
    }

    private void setResultText(String text) {
        lastResultText = text;
        if (result != null) {
            result.setText(text);
        }
        updateActions();
    }

    private void copyLastResult() {
        if (!hasUsefulResult()) {
            Toast.makeText(this, "No result to copy.", Toast.LENGTH_SHORT).show();
            return;
        }
        copyText("Edge AI result", buildExperimentReport());
    }

    private void shareLastResult() {
        if (!hasUsefulResult()) {
            Toast.makeText(this, "No result to share.", Toast.LENGTH_SHORT).show();
            return;
        }
        shareText("Edge AI inference result", buildExperimentReport());
    }

    private void saveLastResult() {
        if (!hasUsefulResult()) {
            Toast.makeText(this, "No result to save.", Toast.LENGTH_SHORT).show();
            return;
        }
        saveText("edge_ai_results.txt", buildExperimentReport());
    }

    private void showAbout() {
        new AlertDialog.Builder(this)
                .setTitle("About Edge AI Inference")
                .setMessage("Version 0.1.0\n\n"
                        + "A local Android Edge AI lab for downloading TFLite models, inspecting tensor shapes, "
                        + "running image classification, and recording benchmark evidence.")
                .setPositiveButton("OK", null)
                .show();
    }

    private void showPrivacy() {
        new AlertDialog.Builder(this)
                .setTitle("Privacy")
                .setMessage("This app performs inference on device.\n\n"
                        + "Network access is used only to download model and labels files from HTTPS URLs you provide or select.\n\n"
                        + "Downloaded models, labels, settings, and saved experiment reports stay in this app's private storage unless you choose to share them.")
                .setPositiveButton("OK", null)
                .show();
    }

    private void showDiagnostics() {
        String report = diagnosticsReporter.buildReport(
                modelRepository,
                appPreferences,
                modelUrlInput.getText().toString().trim(),
                labelsUrlInput.getText().toString().trim(),
                lastResultText,
                getExternalFilesDir(null),
                appLogger.getLogFile(),
                appLogger.getLastError());
        new AlertDialog.Builder(this)
                .setTitle("Diagnostics")
                .setMessage(report)
                .setPositiveButton("Copy", (dialog, which) -> copyText("Edge AI diagnostics", report))
                .setNegativeButton("Share", (dialog, which) -> shareText("Edge AI diagnostics", report))
                .setNeutralButton("Save", (dialog, which) -> saveText("edge_ai_diagnostics.txt", report))
                .show();
    }

    private void copyText(String label, String text) {
        ClipboardManager clipboard = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
        clipboard.setPrimaryClip(ClipData.newPlainText(label, text));
        Toast.makeText(this, "Copied.", Toast.LENGTH_SHORT).show();
    }

    private void shareText(String subject, String text) {
        Intent intent = new Intent(Intent.ACTION_SEND);
        intent.setType("text/plain");
        intent.putExtra(Intent.EXTRA_SUBJECT, subject);
        intent.putExtra(Intent.EXTRA_TEXT, text);
        startActivity(Intent.createChooser(intent, "Share"));
    }

    private void saveText(String fileName, String text) {
        File output = new File(getExternalFilesDir(null), fileName);
        String timestamp = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(new Date());
        try (FileWriter writer = new FileWriter(output, true)) {
            writer.write("[" + timestamp + "]\n");
            writer.write(text);
            writer.write("\n\n");
            Toast.makeText(this, "Saved to " + output.getAbsolutePath(), Toast.LENGTH_LONG).show();
        } catch (IOException e) {
            Toast.makeText(this, "Save failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    private String buildExperimentReport() {
        StringBuilder builder = new StringBuilder();
        builder.append("Edge AI Inference Report\n");
        builder.append("Generated: ")
                .append(new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(new Date()))
                .append("\n\n");

        ModelInfo modelInfo = modelRepository.getModelInfo();
        if (modelInfo != null) {
            builder.append(modelInfo.reportSummary(modelRepository.getLabels().size())).append("\n");
        } else {
            builder.append("Model: not loaded\n");
        }
        builder.append("Model URL: ").append(modelUrlInput.getText().toString().trim()).append("\n");
        builder.append("Labels URL: ").append(labelsUrlInput.getText().toString().trim()).append("\n");
        builder.append("Threads: ").append(appPreferences.getThreadCount()).append("\n");
        builder.append("Runtime: ").append(appPreferences.getRuntimeConfig().getLabel()).append("\n");
        builder.append("Benchmark runs: ").append(appPreferences.getBenchmarkIterations()).append("\n");
        builder.append("Preprocess: ").append(appPreferences.getPreprocessConfig().getLabel()).append("\n");
        if (selectedBitmap != null) {
            builder.append("Image: ")
                    .append(selectedBitmap.getWidth())
                    .append(" x ")
                    .append(selectedBitmap.getHeight())
                    .append("\n");
        }
        builder.append("\nResult:\n").append(lastResultText);
        return builder.toString();
    }

    private boolean hasUsefulResult() {
        return lastResultText != null
                && !lastResultText.trim().isEmpty()
                && !lastResultText.startsWith("Load a model");
    }

    private int parsePositiveInt(String text, String name) {
        try {
            int value = Integer.parseInt(text.trim());
            if (value <= 0) {
                throw new IllegalArgumentException(name + " must be positive.");
            }
            return value;
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(name + " must be a number.");
        }
    }

    private void updateBenchmarkButtonText() {
        if (benchmarkButton != null) {
            benchmarkButton.setText("Benchmark x" + appPreferences.getBenchmarkIterations());
        }
    }

    private void updateActions() {
        boolean hasModel = modelRepository != null && modelRepository.getInterpreter() != null;
        boolean hasImage = selectedBitmap != null;
        if (downloadButton != null) {
            downloadButton.setEnabled(!downloadInProgress);
        }
        if (defaultModelButton != null) {
            defaultModelButton.setEnabled(!downloadInProgress);
        }
        for (Button button : presetButtons) {
            button.setEnabled(!downloadInProgress);
        }
        if (clearModelButton != null) {
            clearModelButton.setEnabled(!downloadInProgress && modelRepository.hasDownloadedModel());
        }
        if (runOnceButton != null) {
            runOnceButton.setEnabled(hasModel && hasImage && !downloadInProgress);
        }
        if (benchmarkButton != null) {
            benchmarkButton.setEnabled(hasModel && hasImage && !downloadInProgress);
        }
        if (pickImageButton != null) {
            pickImageButton.setEnabled(!downloadInProgress);
        }
        boolean hasResult = hasUsefulResult();
        if (copyResultButton != null) {
            copyResultButton.setEnabled(hasResult);
        }
        if (shareResultButton != null) {
            shareResultButton.setEnabled(hasResult);
        }
        if (saveResultButton != null) {
            saveResultButton.setEnabled(hasResult);
        }
        if (applySettingsButton != null) {
            applySettingsButton.setEnabled(!downloadInProgress);
        }
        updatePreprocessButtons();
        updateRuntimeButtons();
    }

    private void updatePreprocessButtons() {
        String currentMode = appPreferences.getPreprocessConfig().getMode();
        for (Button button : preprocessButtons) {
            boolean selected = ("[0,1]".contentEquals(button.getText()) && PreprocessConfig.MODE_ZERO_ONE.equals(currentMode))
                    || ("[-1,1]".contentEquals(button.getText()) && PreprocessConfig.MODE_MINUS_ONE_ONE.equals(currentMode))
                    || ("ImageNet".contentEquals(button.getText()) && PreprocessConfig.MODE_IMAGENET.equals(currentMode));
            button.setEnabled(!downloadInProgress);
            button.setTextColor(selected ? Color.WHITE : Color.rgb(20, 28, 38));
            button.setBackgroundColor(selected ? Color.rgb(0, 109, 119) : Color.rgb(232, 237, 243));
        }
    }

    private void updateRuntimeButtons() {
        String currentMode = appPreferences.getRuntimeConfig().getMode();
        for (Button button : runtimeButtons) {
            boolean selected = ("CPU".contentEquals(button.getText()) && RuntimeConfig.MODE_CPU.equals(currentMode))
                    || ("NNAPI".contentEquals(button.getText()) && RuntimeConfig.MODE_NNAPI.equals(currentMode));
            button.setEnabled(!downloadInProgress);
            button.setTextColor(selected ? Color.WHITE : Color.rgb(20, 28, 38));
            button.setBackgroundColor(selected ? Color.rgb(0, 109, 119) : Color.rgb(232, 237, 243));
        }
    }

    private void setStatus(TextView view, StatusKind kind, String text) {
        view.setText(text);
        view.setTextColor(kind.textColor);
        GradientDrawable background = new GradientDrawable();
        background.setColor(kind.backgroundColor);
        background.setCornerRadius(dp(6));
        background.setStroke(1, kind.borderColor);
        view.setBackground(background);
        view.setPadding(dp(10), dp(8), dp(10), dp(8));
    }

    private LinearLayout card() {
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setPadding(dp(14), dp(14), dp(14), dp(14));
        GradientDrawable background = new GradientDrawable();
        background.setColor(Color.WHITE);
        background.setCornerRadius(dp(8));
        background.setStroke(1, Color.rgb(220, 226, 235));
        layout.setBackground(background);
        return layout;
    }

    private LinearLayout row() {
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.HORIZONTAL);
        layout.setGravity(Gravity.CENTER);
        return layout;
    }

    private TextView sectionTitle(String text) {
        TextView view = new TextView(this);
        view.setText(text);
        view.setTextSize(18);
        view.setTypeface(Typeface.DEFAULT_BOLD);
        view.setTextColor(Color.rgb(20, 28, 38));
        view.setPadding(0, 0, 0, dp(8));
        return view;
    }

    private TextView bodyText() {
        TextView view = new TextView(this);
        view.setTextSize(14);
        view.setTextColor(Color.rgb(61, 73, 92));
        view.setPadding(0, dp(8), 0, 0);
        return view;
    }

    private LinearLayout.LayoutParams matchWrap() {
        return new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
    }

    private LinearLayout.LayoutParams matchWrapWithBottom() {
        LinearLayout.LayoutParams params = matchWrap();
        params.setMargins(0, 0, 0, dp(12));
        return params;
    }

    private LinearLayout.LayoutParams weighted() {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f);
        params.setMargins(dp(2), dp(4), dp(2), dp(4));
        return params;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private enum StatusKind {
        INFO(Color.rgb(45, 83, 112), Color.rgb(232, 244, 252), Color.rgb(176, 212, 235)),
        SUCCESS(Color.rgb(28, 96, 64), Color.rgb(232, 246, 238), Color.rgb(177, 220, 192)),
        WARNING(Color.rgb(116, 84, 27), Color.rgb(255, 247, 225), Color.rgb(233, 199, 128)),
        ERROR(Color.rgb(136, 44, 44), Color.rgb(253, 235, 235), Color.rgb(236, 174, 174));

        final int textColor;
        final int backgroundColor;
        final int borderColor;

        StatusKind(int textColor, int backgroundColor, int borderColor) {
            this.textColor = textColor;
            this.backgroundColor = backgroundColor;
            this.borderColor = borderColor;
        }
    }
}
