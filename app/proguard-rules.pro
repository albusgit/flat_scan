# R8 rules for FlatScan (release builds, and debug with -Pminify=true).
# The native methods of NativeLib keep their names through the default rules (classes with native methods).
# jni_bridge.cpp calls the progress listener by name: GetMethodID(listener, "onProgress", "(ILjava/lang/String;)V").
-keep interface com.albus.flatscan.ProgressListener { *; }
-keepclassmembers class * implements com.albus.flatscan.ProgressListener {
    void onProgress(int, java.lang.String);
}
