import numpy as np

def analysis_single_file(name):
    file = open(name)
    lines = file.readlines()
    is_metrics = True
    metrics_data = None
    layer_data = None
    s = ''
    if len(lines) == 0:
        return None
    for line in lines:
        s = s + line[:-1]
        if s[-1] == ']':
            # print(is_metrics, s)
            if is_metrics:
                metrics = np.array([list(map(float,s[1:-1].split()))])
                # print(metrics_data)
                if metrics_data is None:
                    metrics_data = metrics
                else:
                    metrics_data = np.concatenate((metrics_data, metrics), axis=0)
                is_metrics = not is_metrics
            else:
                layer_similarity = np.array([list(map(float,s[1:-1].split()))])
                # print(layer_data)
                if layer_data is None:
                    layer_data = layer_similarity
                else:
                    layer_data = np.concatenate((layer_data, layer_similarity), axis=0)
                is_metrics = not is_metrics
            s = ''
    return layer_data, metrics_data