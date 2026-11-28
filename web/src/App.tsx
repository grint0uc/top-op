import { useState } from 'react';
import { useAutoVoxelize } from './ui/DomainPanel';
import { ContextMenu, FeatureTree } from './ui/FeatureTree';
import { FileInputs, useWindowDrop } from './ui/ImportPanel';
import { PropertyManager } from './ui/PropertyManager';
import { Ribbon, TitleBar } from './ui/Ribbon';
import { JobMonitor } from './ui/RunPanel';
import { StatusBar } from './ui/StatusBar';
import { ViewportHost } from './ui/ViewportHost';
import { useHotkeys } from './ui/hotkeys';

export function App() {
  const [dropping, setDropping] = useState(false);
  useHotkeys();
  useAutoVoxelize();
  useWindowDrop(setDropping);
  return (
    <div className="app">
      <TitleBar />
      <Ribbon />
      <div className="workspace">
        <aside className="left-pane" aria-label="Specification tree">
          <FeatureTree />
          <JobMonitor />
        </aside>
        <main className="center">
          <ViewportHost dropping={dropping} />
          <FileInputs />
        </main>
        <PropertyManager />
      </div>
      <StatusBar />
      <ContextMenu />
    </div>
  );
}
