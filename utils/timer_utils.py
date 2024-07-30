import time

def tic(timing_dict,label="default"):
    timing_dict[label] = time.time()

def toc(timing_dict,label="default"):
    if label in timing_dict:
        elapsed_time = time.time() - timing_dict[label]
        print(f"Elapsed time for {label}: {elapsed_time:.6f} seconds")
    else:
        print(f"No timer found for label: {label}")