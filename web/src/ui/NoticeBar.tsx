import { useStore } from '../state/store';
import { Banner } from './controls';

export function NoticeBar() {
  const notice = useStore((s) => s.notice);
  const busy = useStore((s) => s.busy);
  return (
    <>
      {notice && (
        <Banner kind={notice.kind} testId="notice">
          <span>{notice.text}</span>
          <button type="button" className="banner-close" aria-label="Dismiss" onClick={() => useStore.getState().setNotice(null)}>
            &times;
          </button>
        </Banner>
      )}
      {busy && (
        <p className="dim" data-testid="busy">
          {busy}...
        </p>
      )}
    </>
  );
}
