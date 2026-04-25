"""Shared constants for the KubeMedic server implementation."""

from __future__ import annotations


CHALLENGE_NAMESPACE = "challenge"
MAX_STEPS = 30
MAX_SCALE_REPLICAS = 5

PROTECTED_NAMESPACES = {
    "kube-system",
    "default",
    "kube-public",
    "kube-node-lease",
    "monitoring",
}

MUTATING_TOOLS = {
    "kubectl_patch_resources",
    "kubectl_patch_tolerations",
    "kubectl_cordon",
    "kubectl_uncordon",
    "kubectl_delete_pod",
    "kubectl_delete_workload",
}

SCENARIO_DESCRIPTIONS = {
    "KUBE-01": "Node MemoryPressure — BestEffort pods evicted",
    "KUBE-03": "OOMKill CrashLoopBackOff — memory limit too low",
    "KUBE-04": "Unschedulable — memory requests too high",
    "KUBE-05": "Taint/toleration mismatch — pods stuck Pending",
    "KUBE-06": "DiskPressure — log flood DaemonSet",
}

SCENARIO_ROOT_CAUSES = {
    "KUBE-01": "A stress pod exhausts node memory and triggers eviction pressure.",
    "KUBE-03": "payment-svc has a memory limit lower than its actual usage, causing OOMKilled CrashLoopBackOff.",
    "KUBE-04": "ml-inference requests far more memory than it actually uses, so it is unschedulable.",
    "KUBE-05": "gpu-workload targets a tainted node without the required toleration.",
    "KUBE-06": "A log-flood DaemonSet fills node disk and causes DiskPressure evictions.",
}

SUPPORTED_SCENARIOS = {"KUBE-01", "KUBE-03", "KUBE-04", "KUBE-05", "KUBE-06"}

POD_PRIORITY = {
    "api-gw": "critical",
    "auth-svc": "critical",
    "payment-svc": "critical",
    "order-svc": "high",
    "worker-svc": "high",
    "ml-inference": "medium",
    "gpu-workload": "medium",
    "batch-job": "background",
}
