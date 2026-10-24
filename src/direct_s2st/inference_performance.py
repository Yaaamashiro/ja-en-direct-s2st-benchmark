"""Single-process inference measurements, not a steady-state benchmark."""
import threading
import time
import os


class Measurement:
    def __enter__(self):
        import torch
        self.torch = torch
        self.process, self.total_ram, self.rss = None, None, None
        self.memory_failed = False
        self.memory_errors = (OSError,)
        try:
            import psutil
            self.memory_errors = (OSError, getattr(psutil, 'Error', OSError))
            self.process = psutil.Process()
            self.total_ram = psutil.virtual_memory().total
            self.rss = self.process.memory_info().rss
        except (ImportError, *self.memory_errors):
            self.process, self.total_ram, self.rss = None, None, None
            pass  # Memory telemetry is optional; never report unavailable RAM as zero.
        self.cuda = torch.cuda.is_available()
        self.result = {}
        if self.cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.started = time.perf_counter()
        self.inference_started = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        if self.process is None:
            return
        while not self.stop.wait(.05):
            try:
                self.rss = max(self.rss, self.process.memory_info().rss)
            except self.memory_errors:
                self.memory_failed = True
                return

    def loaded(self):
        if self.cuda:
            self.torch.cuda.synchronize()
        self.inference_started = time.perf_counter()

    def __exit__(self, kind, error, traceback):
        try:
            if self.cuda:
                self.torch.cuda.synchronize()
            ended = time.perf_counter()
            if self.process is not None:
                try:
                    self.rss = max(self.rss, self.process.memory_info().rss)
                except self.memory_errors:
                    self.memory_failed = True
            self.result.update(model_load_seconds=(self.inference_started or ended)-self.started,
                               inference_seconds=ended-self.inference_started if self.inference_started else None,
                               process_rss_peak_bytes=self.rss,
                               process_rss_percent_system_ram=100*self.rss/self.total_ram if self.total_ram and self.rss is not None else None,
                               gpu_name=None, gpu_total_bytes=None, torch_allocated_peak_bytes=None,
                               torch_reserved_peak_bytes=None, torch_allocated_percent_total_vram=None,
                               success=kind is None, warmup_runs=0,
                               measurement_session=os.environ.get('S2ST_INTERACTIVE_MEASUREMENT_SESSION'),
                               scope='first-call frontend/generation/vocoder; local serialization included; '
                                     'acquisition/load (which may include downloads), subprocess startup, Drive publish and evaluation ASR excluded')
            if self.memory_failed:
                self.result.update(process_rss_peak_bytes=None, process_rss_percent_system_ram=None)
            if self.cuda:
                total = self.torch.cuda.get_device_properties(0).total_memory
                allocated = self.torch.cuda.max_memory_allocated()
                self.result.update(gpu_name=self.torch.cuda.get_device_name(0), gpu_total_bytes=total,
                                   torch_allocated_peak_bytes=allocated,
                                   torch_reserved_peak_bytes=self.torch.cuda.max_memory_reserved(),
                                   torch_allocated_percent_total_vram=100*allocated/total)
        finally:
            self.stop.set()
            self.thread.join()


def branch_performance(parts, input_seconds, output_seconds):
    """Sequential phases: SUM times, MAX resident memory (never sum peaks)."""
    if not parts or any(not p or not p.get('success') or p.get('inference_seconds') is None for p in parts):
        return None
    seconds = sum(p['inference_seconds'] for p in parts)
    result = dict(inference_seconds=seconds, model_load_seconds=sum(p['model_load_seconds'] for p in parts),
                  input_rtf=seconds/input_seconds if input_seconds else None,
                  output_rtf=seconds/output_seconds if output_seconds else None,
                  warmup_runs=0, sequential_gpu=True)
    for key in ('process_rss_peak_bytes', 'process_rss_percent_system_ram', 'torch_allocated_peak_bytes',
                'torch_reserved_peak_bytes', 'torch_allocated_percent_total_vram'):
        values = [p.get(key) for p in parts if p.get(key) is not None]
        result[key] = max(values) if values else None
    result['gpu_name'] = next((p.get('gpu_name') for p in parts if p.get('gpu_name')), None)
    result['hardware'] = sorted({(p['gpu_name'], p.get('gpu_total_bytes')) for p in parts if p.get('gpu_name')})
    result['measurement_sessions'] = sorted({p['measurement_session'] for p in parts if p.get('measurement_session')})
    result['mixed_measurement_sessions'] = len(result['measurement_sessions']) > 1
    return result
