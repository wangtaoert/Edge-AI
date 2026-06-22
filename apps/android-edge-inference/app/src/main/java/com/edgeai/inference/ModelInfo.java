package com.edgeai.inference;

import org.tensorflow.lite.DataType;

import java.util.Arrays;
import java.util.Locale;

class ModelInfo {
    final String source;
    final DataType inputType;
    final int[] inputShape;
    final DataType outputType;
    final int[] outputShape;
    final long modelSizeBytes;

    ModelInfo(String source, DataType inputType, int[] inputShape,
              DataType outputType, int[] outputShape, long modelSizeBytes) {
        this.source = source;
        this.inputType = inputType;
        this.inputShape = inputShape;
        this.outputType = outputType;
        this.outputShape = outputShape;
        this.modelSizeBytes = modelSizeBytes;
    }

    String format(int labelCount) {
        return String.format(Locale.US,
                "Source: %s\nModel size: %.2f MB\nInput: %s %s\nOutput: %s %s\nLabels: %d",
                source,
                modelSizeBytes / 1024.0 / 1024.0,
                inputType,
                Arrays.toString(inputShape),
                outputType,
                Arrays.toString(outputShape),
                labelCount);
    }

    String reportSummary(int labelCount) {
        return String.format(Locale.US,
                "Model source: %s\nModel size: %.2f MB\nInput: %s %s\nOutput: %s %s\nLabels: %d",
                source,
                modelSizeBytes / 1024.0 / 1024.0,
                inputType,
                Arrays.toString(inputShape),
                outputType,
                Arrays.toString(outputShape),
                labelCount);
    }
}
