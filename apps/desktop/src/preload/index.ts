import { contextBridge, ipcRenderer, webUtils } from 'electron'

const api = {
  connectEngine: (): Promise<{ url: string; token: string }> =>
    ipcRenderer.invoke('engine:connect'),
  openVideo: (): Promise<string | null> => ipcRenderer.invoke('dialog:openVideo'),
  openBackground: (): Promise<string | null> => ipcRenderer.invoke('dialog:openBackground'),
  saveVideo: (inputPath: string): Promise<string | null> =>
    ipcRenderer.invoke('dialog:saveVideo', inputPath),
  reveal: (path: string): Promise<void> => ipcRenderer.invoke('shell:reveal', path),
  /** Absolute path of a file dropped onto the window. */
  pathForFile: (file: File): string => webUtils.getPathForFile(file)
}

export type VibeApi = typeof api

contextBridge.exposeInMainWorld('vibe', api)
