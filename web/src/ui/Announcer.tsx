// Screen-reader announcements (WCAG 4.1.3): run start / every 10th iteration / done / stopped / error, and the
// notices over the viewport. Two persistent live regions, so the text is announced even though the banners come and go.
import { useEffect, useState } from 'react';
import { useStore } from '../state/store';

/** Iterations between progress announcements: often enough to follow, rare enough not to drown everything else. */
export const ANNOUNCE_EVERY = 10;

// the same text twice in a row would not be re-announced: alternate a trailing no-break space
const fresh = (prev: string, text: string) => (prev === text ? `${text} ` : text);

export function Announcer() {
  const [polite, setPolite] = useState('');
  const [assertive, setAssertive] = useState('');

  useEffect(() => {
    const say = (text: string) => setPolite((p) => fresh(p, text));
    const alert = (text: string) => setAssertive((p) => fresh(p, text));
    let queued: ReturnType<typeof setTimeout> | undefined;
    const unsub = useStore.subscribe((s, p) => {
      const r = s.run;
      const q = p.run;
      if (r.status !== q.status) {
        const n = r.history.length;
        const last = r.history[n - 1];
        clearTimeout(queued);
        // every run is "queued" for a moment before it starts: only say so when it really waits
        if (r.status === 'queued')
          queued = setTimeout(() => {
            if (useStore.getState().run.status === 'queued') say('Run queued: waiting for the previous run to finish.');
          }, 1500);
        else if (r.status === 'running') say('Run started.');
        else if (r.status === 'done') say(`Run finished after ${n} iterations${last ? `, compliance ${last.compliance.toPrecision(4)}` : ''}.`);
        else if (r.status === 'cancelled') say(`Run stopped after ${n} iterations.`);
        else if (r.status === 'error') alert(`Run failed: ${r.error ?? 'unknown error'}`);
      } else if (r.status === 'running' && r.history.length > q.history.length) {
        const last = r.history[r.history.length - 1];
        const prevIt = q.history[q.history.length - 1]?.it ?? 0;
        // one message per crossed multiple of ANNOUNCE_EVERY, however the records arrive
        if (last && Math.floor(last.it / ANNOUNCE_EVERY) > Math.floor(prevIt / ANNOUNCE_EVERY)) {
          const max = r.maxIter || s.project.params.max_iter;
          say(`Iteration ${last.it} of ${max}, compliance ${last.compliance.toPrecision(4)}.`);
        }
      }
      if (s.notice !== p.notice && s.notice) (s.notice.kind === 'error' ? alert : say)(s.notice.text);
    });
    return () => {
      clearTimeout(queued);
      unsub();
    };
  }, []);

  return (
    <>
      <div className="visually-hidden" role="status" aria-live="polite" data-testid="live-status">
        {polite}
      </div>
      <div className="visually-hidden" role="alert" aria-live="assertive" data-testid="live-alert">
        {assertive}
      </div>
    </>
  );
}
