package com.edgeai.inference;

import android.graphics.Bitmap;
import android.os.SystemClock;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.Tensor.QuantizationParams;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

class TfliteImageClassifier {
    InferenceResult runInference(Interpreter interpreter, Bitmap bitmap, PreprocessConfig preprocessConfig) {
        Tensor inputTensor = interpreter.getInputTensor(0);
        Tensor outputTensor = interpreter.getOutputTensor(0);
        int[] inputShape = inputTensor.shape();
        if (inputShape.length != 4 || inputShape[0] != 1 || inputShape[3] != 3) {
            throw new IllegalArgumentException("Only NHWC image input [1,H,W,3] is supported. Got "
                    + Arrays.toString(inputShape));
        }

        ByteBuffer inputBuffer = makeInputBuffer(bitmap, inputTensor, preprocessConfig);
        ByteBuffer outputBuffer = ByteBuffer.allocateDirect(outputTensor.numBytes()).order(ByteOrder.nativeOrder());

        long started = SystemClock.elapsedRealtimeNanos();
        interpreter.run(inputBuffer, outputBuffer);
        long latencyMs = (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000L;

        float[] scores = readOutputScores(outputBuffer, outputTensor);
        int topIndex = argmax(scores);
        return new InferenceResult(topIndex, scores[topIndex], latencyMs);
    }

    BenchmarkResult benchmark(Interpreter interpreter, Bitmap bitmap, int iterations, PreprocessConfig preprocessConfig) {
        InferenceResult first = runInference(interpreter, bitmap, preprocessConfig);
        List<Long> warmTimes = new ArrayList<>();
        InferenceResult last = first;
        for (int i = 0; i < iterations; i++) {
            last = runInference(interpreter, bitmap, preprocessConfig);
            warmTimes.add(last.latencyMs);
        }
        return BenchmarkResult.from(first, last, warmTimes);
    }

    private ByteBuffer makeInputBuffer(Bitmap source, Tensor inputTensor, PreprocessConfig preprocessConfig) {
        int[] shape = inputTensor.shape();
        int height = shape[1];
        int width = shape[2];
        Bitmap resized = Bitmap.createScaledBitmap(source, width, height, true);
        ByteBuffer buffer = ByteBuffer.allocateDirect(inputTensor.numBytes()).order(ByteOrder.nativeOrder());
        int[] pixels = new int[width * height];
        resized.getPixels(pixels, 0, width, 0, 0, width, height);

        DataType type = inputTensor.dataType();
        QuantizationParams quant = inputTensor.quantizationParams();
        for (int pixel : pixels) {
            float r = ((pixel >> 16) & 0xFF) / 255.0f;
            float g = ((pixel >> 8) & 0xFF) / 255.0f;
            float b = (pixel & 0xFF) / 255.0f;
            putValue(buffer, type, quant, preprocessConfig.transform(r, 0));
            putValue(buffer, type, quant, preprocessConfig.transform(g, 1));
            putValue(buffer, type, quant, preprocessConfig.transform(b, 2));
        }
        buffer.rewind();
        return buffer;
    }

    private void putValue(ByteBuffer buffer, DataType type, QuantizationParams quant, float value) {
        if (type == DataType.FLOAT32) {
            buffer.putFloat(value);
        } else if (type == DataType.UINT8) {
            int q = Math.round(value / checkedScale(quant)) + quant.getZeroPoint();
            buffer.put((byte) clamp(q, 0, 255));
        } else if (type == DataType.INT8) {
            int q = Math.round(value / checkedScale(quant)) + quant.getZeroPoint();
            buffer.put((byte) clamp(q, -128, 127));
        } else {
            throw new IllegalArgumentException("Unsupported input type: " + type);
        }
    }

    private float[] readOutputScores(ByteBuffer outputBuffer, Tensor outputTensor) {
        outputBuffer.rewind();
        int count = outputTensor.numElements();
        float[] scores = new float[count];
        DataType type = outputTensor.dataType();
        QuantizationParams quant = outputTensor.quantizationParams();
        for (int i = 0; i < count; i++) {
            if (type == DataType.FLOAT32) {
                scores[i] = outputBuffer.getFloat();
            } else if (type == DataType.UINT8) {
                scores[i] = ((outputBuffer.get() & 0xFF) - quant.getZeroPoint()) * checkedScale(quant);
            } else if (type == DataType.INT8) {
                scores[i] = (outputBuffer.get() - quant.getZeroPoint()) * checkedScale(quant);
            } else {
                throw new IllegalArgumentException("Unsupported output type: " + type);
            }
        }
        return scores;
    }

    private float checkedScale(QuantizationParams quant) {
        if (quant.getScale() == 0.0f) {
            throw new IllegalArgumentException("Quantized tensor scale is 0.");
        }
        return quant.getScale();
    }

    private int argmax(float[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++) {
            if (values[i] > values[best]) {
                best = i;
            }
        }
        return best;
    }

    private int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }
}
