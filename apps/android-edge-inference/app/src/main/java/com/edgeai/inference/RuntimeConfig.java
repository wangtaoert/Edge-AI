package com.edgeai.inference;

class RuntimeConfig {
    static final String MODE_CPU = "cpu";
    static final String MODE_NNAPI = "nnapi";

    private final String mode;

    RuntimeConfig(String mode) {
        if (MODE_NNAPI.equals(mode)) {
            this.mode = MODE_NNAPI;
        } else {
            this.mode = MODE_CPU;
        }
    }

    String getMode() {
        return mode;
    }

    String getLabel() {
        if (MODE_NNAPI.equals(mode)) {
            return "NNAPI";
        }
        return "CPU";
    }

    boolean useNnapi() {
        return MODE_NNAPI.equals(mode);
    }
}
