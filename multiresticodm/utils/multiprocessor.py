import multiprocessing
import concurrent.futures


class BoundedQueuePoolExecutor:
    def __init__(self, semaphore):
        self.semaphore = semaphore

    def release(self, future):
        self.semaphore.release()

    def submit(self, fn, *args, **kwargs):
        self.semaphore.acquire()
        future = super().submit(fn, *args, **kwargs)
        future.add_done_callback(self.release)
        return future

class BoundedQueueProcessPoolExecutor(BoundedQueuePoolExecutor, concurrent.futures.ProcessPoolExecutor):
    def __init__(self, *args, max_waiting_tasks=None, **kwargs):
        concurrent.futures.ProcessPoolExecutor.__init__(self, *args, **kwargs)
        if max_waiting_tasks is None:
            max_waiting_tasks = self._max_workers
        elif max_waiting_tasks < 0:
            raise ValueError(f'Invalid negative max_waiting_tasks value: {max_waiting_tasks}')
        BoundedQueuePoolExecutor.__init__(self, multiprocessing.BoundedSemaphore(self._max_workers + max_waiting_tasks))