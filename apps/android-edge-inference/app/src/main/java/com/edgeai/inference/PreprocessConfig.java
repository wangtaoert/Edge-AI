package com.edgeai.inference;

class PreprocessConfig {
    static final String MODE_ZERO_ONE = "zero_one";
    static final String MODE_MINUS_ONE_ONE = "minus_one_one";
    static final String MODE_IMAGENET = "imagenet";

    private final String mode;

    PreprocessConfig(String mode) {
        if (!MODE_MINUS_ONE_ONE.equals(mode) && !MODE_IMAGENET.equals(mode)) {
            this.mode = MODE_ZERO_ONE;
        } else {
            this.mode = mode;
        }
    }

    String getMode() {
        return mode;
    }

    String getLabel() {
        switch (mode) {
            case MODE_MINUS_ONE_ONE:
                return "RGB [-1,1]";
            case MODE_IMAGENET:
                return "RGB ImageNet mean/std";
            case MODE_ZERO_ONE:
            default:
                return "RGB [0,1]";
        }
    }

    float transform(float channelValue, int channelIndex) {
        switch (mode) {
            case MODE_MINUS_ONE_ONE:
                return channelValue * 2.0f - 1.0f;
            case MODE_IMAGENET:
                return (channelValue - mean(channelIndex)) / std(channelIndex);
            case MODE_ZERO_ONE:
            default:
                return channelValue;
        }
    }

    private float mean(int channelIndex) {
        if (channelIndex == 0) {
            return 0.485f;
        }
        if (channelIndex == 1) {
            return 0.456f;
        }
        return 0.406f;
    }

    private float std(int channelIndex) {
        if (channelIndex == 0) {
            return 0.229f;
        }
        if (channelIndex == 1) {
            return 0.224f;
        }
        return 0.225f;
    }
}
