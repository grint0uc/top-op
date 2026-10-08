import { DomainPanel } from './ui/DomainPanel';
import { ImportPanel } from './ui/ImportPanel';
import { LoadsPanel } from './ui/LoadsPanel';
import { NoticeBar } from './ui/NoticeBar';
import { RefModelsPanel } from './ui/RefModelsPanel';
import { ResultsPanel } from './ui/ResultsPanel';
import { RunPanel } from './ui/RunPanel';
import { SupportsPanel } from './ui/SupportsPanel';
import { ViewportHost } from './ui/ViewportHost';
import { useHotkeys } from './ui/hotkeys';

export function App() {
  useHotkeys();
  return (
    <div className="layout">
      <aside className="sidebar">
        <h1 className="brand">top-op</h1>
        <NoticeBar />
        <ImportPanel />
        <DomainPanel />
        <LoadsPanel />
        <SupportsPanel />
        <RefModelsPanel />
        <RunPanel />
        <ResultsPanel />
      </aside>
      <main className="main">
        <ViewportHost />
      </main>
    </div>
  );
}
