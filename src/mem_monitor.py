stop = False
def monitor(_obj, _func, pid, threshold=100, max_threshold=150, params=()):
    def _run(_obj, _func, params):
        getattr(_obj, _func)(*params)
        global stop
        print(" ", _func, "finishing...")
        stop = True

    def _men_monitor(pid, threshold, max_threshold):
        import psutil
        global stop
        while not stop:
            p1 = psutil.Process(pid)
            mem = p1.memory_full_info()[0] / 1024 ** 3
            if mem > threshold:
                threshold *= 1.1
                if threshold > max_threshold:
                    threshold = max_threshold
                print('')
                print("# pid:", pid)
                print("# update threshold to %.3fG" % threshold)
                print("# need more %.3fG" % (threshold-mem))
                import os
                os.system("kill -19 %s" % pid)
            import time
            time.sleep(3)
    from threading import Thread
    _task = Thread(target=_run, args=[_obj, _func, params],  name="task")
    _mem = Thread(target=_men_monitor, args=[pid, threshold, max_threshold], name="memory_monitor")
    _task.start()
    _mem.start()
    _task.join()
    _mem.join()