#
# Instrumentation for local_wake.py: score logging plus audio clips around
# detections and near-misses.
#
# The point is to stop guessing. A false activation at 3am is invisible unless
# something wrote down what the microphone actually heard, and a wake word that
# failed to trigger is invisible unless you can see the score it reached. This
# records both, cheaply enough to leave running for a week on a Pi 3.
#
# Everything here is off unless --capture is passed, and every call from the
# detection loop is a no-op when instrumentation is disabled.
#

import collections
import logging
import os
import queue
import threading
import time
import wave

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MIN_FREE_BYTES = 500 * 1024 * 1024
DISK_CHECK_INTERVAL_S = 3600.0


class AudioRing:
    """Circular int16 buffer addressed by absolute sample index.

    Tracking an absolute, monotonically increasing sample counter means an
    event can say "samples N..M" without any wraparound bookkeeping at the
    call site; extract() works out whether that span is still resident.
    """

    def __init__(self, seconds, sample_rate=SAMPLE_RATE):
        self._n = int(seconds * sample_rate)
        self._buf = np.zeros(self._n, dtype=np.int16)
        self._written = 0  # absolute count of samples ever pushed

    @property
    def position(self):
        return self._written

    def push(self, chunk):
        n = len(chunk)
        if n >= self._n:
            # Keep only what fits, but advance past the dropped prefix first so
            # the "sample s lives at offset s % n" invariant that extract()
            # relies on still holds. Writing the tail at offset 0 instead would
            # silently shift every subsequent extract by `written % n`.
            chunk = chunk[-self._n:]
            self._written += n - self._n
            n = self._n
        start = self._written % self._n
        end = start + n
        if end <= self._n:
            self._buf[start:end] = chunk
        else:
            split = self._n - start
            self._buf[start:] = chunk[:split]
            self._buf[:end - self._n] = chunk[split:]
        self._written += n

    def extract(self, start, end):
        """Samples [start, end) by absolute index, clamped to what is resident."""
        oldest = max(0, self._written - self._n)
        start = max(start, oldest)
        end = min(end, self._written)
        if end <= start:
            return np.zeros(0, dtype=np.int16)
        i = start % self._n
        j = end % self._n
        if i < j:
            return self._buf[i:j].copy()
        return np.concatenate((self._buf[i:], self._buf[:j]))


class ClipWriter:
    """Background WAV writer. Never blocks the audio loop."""

    def __init__(self, maxsize=8):
        self._q = queue.Queue(maxsize=maxsize)
        self.dropped = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, path, samples):
        try:
            self._q.put_nowait((path, samples))
            return True
        except queue.Full:
            # Dropping a clip is always better than stalling capture and
            # causing the overruns this whole exercise was about.
            self.dropped += 1
            return False

    def _run(self):
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            path, samples = item
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                w = wave.open(path, 'wb')
                try:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(SAMPLE_RATE)
                    w.writeframes(samples.tobytes())
                finally:
                    w.close()
            except Exception as e:  # a bad clip must not kill the detector
                logger.error('failed writing clip %s: %s', path, e)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=3)


class Marker:
    """VoiceHat button as a ground-truth signal.

    One gesture, disambiguated by timing:
      - pressed within `fa_window` of a detection  -> "that was wrong"
      - pressed otherwise -> "I just said the wake word and nothing happened"

    The button driver calls back with NO arguments (aiy/_drivers/_button.py
    calls `self.callback()` despite its docstring promising a channel number),
    and it debounces by blocking its own GPIO thread in a polling loop, so the
    callback must do nothing but hand the timestamp over.
    """

    def __init__(self):
        self._q = queue.Queue()
        self.enabled = False
        try:
            import aiy.voicehat
            aiy.voicehat.get_button().on_press(self._on_press)
            self.enabled = True
        except Exception as e:
            logger.warning('button marking unavailable: %s', e)

    def _on_press(self):
        self._q.put(time.time())

    def drain(self):
        out = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                return out


class Instrumentation:
    """Facade used by the detection loop. All methods are cheap."""

    def __init__(self, capture_dir, threshold, near_threshold,
                 clip_pre_s=4.0, clip_post_s=1.5, mark_pre_s=20.0,
                 log_scores='events', keep_days=14, max_clips_per_hour=30,
                 fa_window=20.0, mark_button=False, profile=False,
                 chunk_samples=1280):
        self.dir = capture_dir
        self.threshold = threshold
        self.near_threshold = near_threshold
        self.clip_pre_s = clip_pre_s
        self.clip_post_s = clip_post_s
        self.mark_pre_s = mark_pre_s
        self.log_scores = log_scores
        self.fa_window = fa_window
        self.max_clips_per_hour = max_clips_per_hour
        self.profile = profile
        self._chunk_s = chunk_samples / float(SAMPLE_RATE)

        ring_s = max(clip_pre_s + clip_post_s, mark_pre_s) + 4.0
        self._ring = AudioRing(ring_s)
        self._writer = ClipWriter()
        self._marker = Marker() if mark_button else None

        self._pending = []        # armed events awaiting their post-roll
        self._seq = 0
        self._recent_fires = collections.deque(maxlen=8)
        self._clip_times = collections.deque()
        self._score_tail = collections.deque(
            maxlen=max(1, int(clip_pre_s / self._chunk_s)))
        self._flush_until = 0.0
        self._flush_event = ''
        self._score_header = None

        self._predict_ms = collections.deque(maxlen=256)
        self._summary_due = time.monotonic() + 10.0
        self._summary = collections.defaultdict(list)
        self._disk_due = 0.0
        self._disk_ok = True

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(os.path.join(self.dir, 'scores'), exist_ok=True)
        self._events_path = os.path.join(self.dir, 'events.csv')
        self._ensure_header(
            self._events_path,
            'ts,event_id,type,model,score,threshold,gate,clip,extra\n')
        self._prune(keep_days)
        self.event('startup', '', 0.0, gate='ok', clip='', extra='')

    # --- plumbing ----------------------------------------------------------

    def _ensure_header(self, path, header):
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, 'a') as f:
                f.write(header)

    def _scores_path(self):
        return os.path.join(self.dir, 'scores',
                            'scores-%s.csv' % time.strftime('%Y%m%d'))

    def _prune(self, keep_days):
        cutoff = time.time() - keep_days * 86400
        for sub in ('clips', 'scores'):
            root = os.path.join(self.dir, sub)
            for dirpath, _, names in os.walk(root):
                for name in names:
                    p = os.path.join(dirpath, name)
                    try:
                        if os.path.getmtime(p) < cutoff:
                            os.remove(p)
                    except OSError:
                        pass

    def _disk_ready(self):
        now = time.monotonic()
        if now >= self._disk_due:
            self._disk_due = now + DISK_CHECK_INTERVAL_S
            try:
                st = os.statvfs(self.dir)
                self._disk_ok = st.f_bavail * st.f_frsize > MIN_FREE_BYTES
                if not self._disk_ok:
                    logger.warning('capture disabled: less than %d MB free',
                                   MIN_FREE_BYTES // (1024 * 1024))
            except OSError:
                self._disk_ok = False
        return self._disk_ok

    def _rate_ok(self):
        now = time.monotonic()
        while self._clip_times and now - self._clip_times[0] > 3600:
            self._clip_times.popleft()
        if len(self._clip_times) >= self.max_clips_per_hour:
            return False
        self._clip_times.append(now)
        return True

    def _next_id(self):
        self._seq += 1
        return '%s-%04d' % (time.strftime('%Y%m%d'), self._seq)

    # --- called from the detection loop ------------------------------------

    def _emit_clip(self, p):
        samples = self._ring.extract(p['start'], p['end'])
        want = p['end'] - p['start']
        if len(samples) < want * 0.95:
            logger.warning(
                'clip %s truncated to %.1fs of %.1fs requested: the ring '
                'buffer is too small',
                os.path.basename(p['path']),
                len(samples) / float(SAMPLE_RATE),
                want / float(SAMPLE_RATE))
        self._writer.submit(p['path'], samples)

    def push_audio(self, chunk):
        self._ring.push(chunk)
        done = [p for p in self._pending if self._ring.position >= p['end']]
        if done:
            self._pending = [p for p in self._pending if p not in done]
            for p in done:
                self._emit_clip(p)

        if self._marker:
            for ts in self._marker.drain():
                self._handle_mark(ts)

    def _handle_mark(self, ts):
        recent = [f for f in self._recent_fires if ts - f[0] <= self.fa_window]
        if recent:
            fire_ts, event_id, model, score = recent[-1]
            self.event('fa_tag', model, score, gate='ok', clip='',
                       extra='tags=%s' % event_id)
            logger.info('button: tagged %s (%s) as a false activation',
                        event_id, model)
            return
        event_id = self._next_id()
        clip = self._arm(event_id, 'mark', '', 0.0, pre_s=self.mark_pre_s)
        self.event('mark', '', 0.0, gate='ok', clip=clip, extra='', event_id=event_id)
        logger.info('button: marked a missed wake word (%s), keeping the %.0fs '
                    'before the press', event_id, self.mark_pre_s)

    def _arm(self, event_id, kind, model, score, pre_s=None, post_s=None):
        if not self._disk_ready() or not self._rate_ok():
            return ''
        pre_s = self.clip_pre_s if pre_s is None else pre_s
        post_s = self.clip_post_s if post_s is None else post_s
        now = self._ring.position
        path = os.path.join(
            self.dir, 'clips', time.strftime('%Y%m%d'),
            '%s_%s_%s_s%03d.wav' % (event_id, kind,
                                    model.replace('/', '_') or 'none',
                                    int(round(score * 100))))
        pending = {
            'start': now - int(pre_s * SAMPLE_RATE),
            'end': now + int(post_s * SAMPLE_RATE),
            'path': path,
        }
        if self._ring.position >= pending['end']:
            # Nothing left to wait for (post_s == 0), so write it now rather
            # than leaving it queued until the next chunk arrives.
            self._emit_clip(pending)
        else:
            self._pending.append(pending)
        # Keep the surrounding score rows too, not just the audio.
        self._flush_until = time.time() + self.clip_post_s
        self._flush_event = event_id
        return os.path.relpath(path, self.dir)

    def log_frame(self, scores, raw_scores, vad, rms, peak, predict_ms):
        if predict_ms is not None:
            self._predict_ms.append(predict_ms)
        for name, score in scores.items():
            # Accumulate the UNGATED score. Summarising the gated one made the
            # line read "max=0.00" whenever a gate was active, which hid
            # exactly the thing the summary exists to show.
            self._summary[name].append(float(raw_scores.get(name, score)))

        names = sorted(scores)
        if self._score_header is None:
            self._score_header = (
                'ts,event_id,' + ','.join(names) + ',' +
                ','.join('raw_' + n for n in names) + ',rms,peak,predict_ms\n')

        now = time.time()
        row = '%.3f,%s,%s,%s,%.1f,%d,%.2f\n' % (
            now,
            self._flush_event if now < self._flush_until else '',
            ','.join('%.4f' % scores.get(k, 0.0) for k in names),
            ','.join('%.4f' % raw_scores.get(k, float('nan')) for k in names),
            rms, peak,
            predict_ms if predict_ms is not None else 0.0)

        if self.log_scores == 'all' or now < self._flush_until:
            self._append_score(row)
        else:
            self._score_tail.append(row)

        if time.monotonic() >= self._summary_due:
            self._emit_summary()

    def _append_score(self, row):
        """Single write path, so the header cannot be skipped.

        The pre-roll flush used to bypass this, which left the first line of
        every score file being a data row.
        """
        path = self._scores_path()
        if self._score_header:
            self._ensure_header(path, self._score_header)
        try:
            with open(path, 'a') as f:
                f.write(row)
        except OSError as e:
            logger.error('score log write failed: %s', e)

    def _emit_summary(self):
        self._summary_due = time.monotonic() + 10.0
        if not self._summary:
            return
        parts = []
        for name, vals in sorted(self._summary.items()):
            a = np.array(vals)
            parts.append('%s max=%.2f mean=%.2f p95=%.2f'
                         % (name, a.max(), a.mean(), np.percentile(a, 95)))
        if self._predict_ms:
            p = np.array(self._predict_ms)
            parts.append('predict p50=%.0fms p95=%.0fms rtf=%.2f'
                         % (np.percentile(p, 50), np.percentile(p, 95),
                            np.percentile(p, 50) / (self._chunk_s * 1000.0)))
        if self.profile:
            logger.info('summary: %s', '  '.join(parts))
        self._summary.clear()

    def flush_score_tail(self, event_id):
        """Write the buffered pre-roll score rows, stamped with the event."""
        rows = list(self._score_tail)
        self._score_tail.clear()
        for row in rows:
            head, _, rest = row.partition(',')
            _, _, tail = rest.partition(',')
            self._append_score('%s,%s,%s' % (head, event_id, tail))

    def event(self, kind, model, score, gate='ok', clip='', extra='',
              event_id=None):
        event_id = event_id or self._next_id()
        line = '%.3f,%s,%s,%s,%.4f,%.2f,%s,%s,%s\n' % (
            time.time(), event_id, kind, model, score, self.threshold,
            gate, clip, extra)
        try:
            with open(self._events_path, 'a') as f:
                f.write(line)
        except OSError as e:
            logger.error('event log write failed: %s', e)
        return event_id

    def on_fire(self, model, score):
        event_id = self._next_id()
        # No post-roll: main() closes the AudioSource before calling this, so
        # nothing pushes audio again until after the whole assistant
        # round-trip. A post-roll here would splice several seconds of
        # unrelated later audio onto the end of the clip. The wake word is
        # already complete at detection anyway, so the pre-roll has it all.
        clip = self._arm(event_id, 'fire', model, score, post_s=0.0)
        self.event('fire', model, score, gate='ok', clip=clip,
                   event_id=event_id)
        self.flush_score_tail(event_id)
        self._recent_fires.append((time.time(), event_id, model, score))
        return event_id

    def on_near_miss(self, model, score, gate='below_threshold'):
        event_id = self._next_id()
        clip = self._arm(event_id, 'near', model, score)
        self.event('near', model, score, gate=gate, clip=clip,
                   event_id=event_id)
        self.flush_score_tail(event_id)
        return event_id

    def on_deaf(self, start):
        self.event('deaf_start' if start else 'deaf_end', '', 0.0)

    def close(self):
        self._emit_summary()
        self._writer.close()
        if self._writer.dropped:
            logger.warning('%d clips dropped (writer queue full)',
                           self._writer.dropped)
