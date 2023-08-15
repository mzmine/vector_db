def create_MilvusIVFSP(metric_type, nprobe):
    search_params = {
        "metric_type": metric_type,
        "nprobe": nprobe
    }
    return search_params