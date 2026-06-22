plugins {
    id("com.android.application")
}

android {
    namespace = "com.edgeai.inference"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.edgeai.inference"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            isShrinkResources = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
}

dependencies {
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
}
