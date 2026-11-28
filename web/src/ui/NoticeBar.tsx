import { useStore } from '../state/store';
import { Banner } from './controls';
import { Icon } from './icons';
import { MissingMeshes } from './ImportPanel';

/** Messages over the top of the viewport: the last notice and the re-upload prompt for missing geometry. */
export function NoticeBar() {
  const notice = useStore((s) => s.notice);
  return (
    <div className="notices">
      {notice && (
        <Banner kind={notice.kind} testId="notice" live={false}>
          <Icon name={notice.kind === 'info' ? 'info' : 'warn'} size={14} className="banner-icon" />
          <span className="grow">{notice.text}</span>
          <button type="button" className="banner-close" aria-label="Dismiss the notice" onClick={() => useStore.getState().setNotice(null)}>
            <Icon name="close" size={12} />
          </button>
        </Banner>
      )}
      <MissingMeshes />
    </div>
  );
}
