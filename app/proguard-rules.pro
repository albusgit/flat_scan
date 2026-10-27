# R8 rules for FlatScan (release builds, and debug with -Pminify=true).
# The native methods of NativeLib keep their names through the default rules (classes with native methods).
# jni_bridge.cpp calls the progress listener by name: GetMethodID(listener, "onProgress", "(ILjava/lang/String;)V").
-keep interface com.albus.flatscan.ProgressListener { *; }
-keepclassmembers class * implements com.albus.flatscan.ProgressListener {
    void onProgress(int, java.lang.String);
}

# Tesseract4Android (OCR, feature 3): its native code reads and writes these classes' fields and calls
# back into them by name; the AAR ships no rules of its own
-keep class com.googlecode.tesseract.android.** { *; }
-keep class com.googlecode.leptonica.android.** { *; }

# security-crypto (sync credentials, feature 6) pulls in Tink, which references compile-only
# errorprone annotations
-dontwarn com.google.errorprone.annotations.**
