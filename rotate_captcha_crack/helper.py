import os

NUM_WORKERS = None


def default_num_workers() -> int:
    global NUM_WORKERS

    if NUM_WORKERS is None:
        if (cpu_count := os.cpu_count()) is None:
            NUM_WORKERS = 0
        else:
            cpu_count = cpu_count >> 1
            if cpu_count > 1:
                # reserve 1 core for other apps
                NUM_WORKERS = cpu_count - 1
            else:
                NUM_WORKERS = 0

    return NUM_WORKERS
