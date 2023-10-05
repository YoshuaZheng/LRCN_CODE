import numpy as np
import pywt

def moving_average_filter(signal):
    window_length = 5
    return np.convolve(signal, np.ones(window_length)/window_length, mode='valid')
    
def normalization(signal):
    normalized_signal = signal-signal.mean()
    normalized_signal = normalized_signal / np.abs(normalized_signal).max()
    return normalized_signal

def segment_signal(signal,weight,indx, segment_size=128):
    # Segment the signal
    if weight is None:
        step_size = segment_size//2
    else:
        step_size = int(weight[int(indx)])
    step_size = int(step_size)
    segments = [signal[i:i+segment_size]-signal[i:i+segment_size].mean() for i in range(0, len(signal)-segment_size + 1, step_size)]
    valid_segments = [segment for segment in segments if len(segment) == segment_size]
    
    # Convert to matrix
    matrix = np.vstack(valid_segments)

    return matrix

def segment_long_intervals(long_interval_signals, segment_size=128):
    segmented_matrices = []

    for signal_portion in long_interval_signals:
        # Segment the signal portion
        step_size = segment_size//2 # 50% overlap
        segments = [signal_portion[i:i+segment_size]-signal_portion[i:i+segment_size].mean() for i in range(0, len(signal_portion)-segment_size + 1, step_size)]
        valid_segments = [segment for segment in segments if len(segment) == segment_size]
        
        # Convert to matrix if there are any valid segments
        if valid_segments:
            matrix = np.vstack(valid_segments)
            segmented_matrices.append(matrix)

    matrix = np.vstack(segmented_matrices)
    
    return matrix

def Anomalous(x, frame = 128, window = 384):
    data_IR_filterd = x
    t= range(len(data_IR_filterd))
    coeffs = pywt.wavedec(data_IR_filterd, 'db4', level=4)
    thresholds = [np.std(c) * 3 for c in coeffs]
    coeffs_thresholded = [pywt.threshold(c, t, mode='soft') for c, t in zip(coeffs, thresholds)]

    reconstructed_signal = pywt.waverec(coeffs_thresholded, 'db4')
    threshold_reconstructed = np.mean(reconstructed_signal) + 2 * np.std(reconstructed_signal)
    anomalous_points = np.where(np.abs(reconstructed_signal) > threshold_reconstructed)[0]

    anomalous_regions = []
    start = None

    for i in range(1, len(anomalous_points)):
        if start is None:
            start = anomalous_points[i-1]
        if anomalous_points[i] - anomalous_points[i-1] > 1:
            anomalous_regions.append((start, anomalous_points[i-1]))
            start = None
    
    if start is not None:  # For the case where the last point is also anomalous
        anomalous_regions.append((start, anomalous_points[-1]))

    L = frame  #

    segments = [range(i, i + L) for i in range(0, len(data_IR_filterd), L)]

    discard_segments = []
    
    for start, end in anomalous_regions:
        for segment in segments:
            if start <= segment[-1] and end >= segment[0]:
                discard_segments.append(segment)

    # remaining_signal = np.delete(data_IR_filterd, np.concatenate(discard_segments))

    length_threshold = window
    
    boundaries = [segment[0] for segment in discard_segments] + [discard_segments[-1][-1]]
    
    boundaries = [0] + boundaries + [len(data_IR_filterd) - 1]

    intervals = [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]

    long_intervals = [(start, end) for start, end in intervals if end - start > length_threshold]

    interval_signals = [data_IR_filterd[start:end+1] for start, end in long_intervals]

    com = np.concatenate(interval_signals)
    me = com.mean()
    dom  = np.abs(com-me).max()

    long_interval_signals = [(data_IR_filterd[start:end+1]-me)/dom for start, end in long_intervals]
    
    return long_interval_signals

def transform_data(data):
    Transform = data.copy() 
    # Initialize as None
    IR_matrix = None
    RED_matrix = None
    Glucose_IR = []
    Glucose_RED = []

    for subject in data:

        # Normalization
        data_IR = normalization(Transform[subject]['IR'])
        data_RED = normalization(Transform[subject]['RED'])

        # Filtering
        data_IR = moving_average_filter(data_IR)
        data_RED = moving_average_filter(data_RED)
        glucose_value = Transform[subject]['Glucose']

        # Wavelet Denoising
        data_IR = Anomalous(data_IR)
        data_RED = Anomalous(data_RED)

        # Segment the signals
        # data_IR = segment_signal(data_IR)
        # data_RED = segment_signal(data_RED)


        data_IR = segment_long_intervals(data_IR)
        data_RED = segment_long_intervals(data_RED)
        G_IR = np.full(len(data_IR), glucose_value)
        G_RED = np.full(len(data_RED), glucose_value)


        # Append to the matrices
        if IR_matrix is None:
            IR_matrix = data_IR
            RED_matrix = data_RED
        else:
            IR_matrix = np.vstack((IR_matrix, data_IR))
            RED_matrix = np.vstack((RED_matrix, data_RED))
        
        # Append glucose values
        Glucose_IR = np.append(Glucose_IR, G_IR)
        Glucose_RED = np.append(Glucose_RED, G_RED)

    return IR_matrix, RED_matrix, Glucose_IR, Glucose_RED

def transform_data_Re(data, weight = None):
    Transform = data.copy() 
    # Initialize as None
    IR_matrix = None
    RED_matrix = None
    Glucose_IR = []
    Glucose_RED = []

    for subject in data:

        # Normalization
        data_IR = normalization(Transform[subject]['IR'])
        data_RED = normalization(Transform[subject]['RED'])

        # Filtering
        data_IR = moving_average_filter(data_IR)
        data_RED = moving_average_filter(data_RED)
        glucose_value = Transform[subject]['Glucose']


        # Segment the signals
        data_IR = segment_signal(data_IR, weight,glucose_value)
        data_RED = segment_signal(data_RED,weight,glucose_value)

        G_IR = np.full(len(data_IR), glucose_value)
        G_RED = np.full(len(data_RED), glucose_value)


        # Append to the matrices
        if IR_matrix is None:
            IR_matrix = data_IR
            RED_matrix = data_RED
        else:
            IR_matrix = np.vstack((IR_matrix, data_IR))
            RED_matrix = np.vstack((RED_matrix, data_RED))
        
        # Append glucose values
        Glucose_IR = np.append(Glucose_IR, G_IR)
        Glucose_RED = np.append(Glucose_RED, G_RED)

    return IR_matrix, RED_matrix, Glucose_IR, Glucose_RED



