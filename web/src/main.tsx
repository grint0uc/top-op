import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App';
import { installPersistence } from './state/persist';
import { useStore } from './state/store';
import { installTheme } from './ui/uiStore';
import './styles.css';

installPersistence(useStore);
installTheme();

const root = document.getElementById('root');
if (!root) throw new Error('#root not found');

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
