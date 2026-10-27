import { app, shell, BrowserWindow, ipcMain, dialog } from 'electron'
import { basename, dirname, extname, join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import icon from '../../resources/icon.png?asset'
import { Engine } from './engine'

const VIDEO_EXTS = ['mp4', 'mov', 'm4v', 'webm', 'mkv']
const IMAGE_EXTS = ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tif', 'tiff']

const engine = new Engine()

function createWindow(): void {
  const mainWindow = new BrowserWindow({
    width: 1280,
    height: 820,
    minWidth: 960,
    minHeight: 640,
    show: false,
    backgroundColor: '#0e0e10',
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    autoHideMenuBar: true,
    ...(process.platform === 'linux' ? { icon } : {}),
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false
    }
  })

  mainWindow.on('ready-to-show', () => {
    mainWindow.show()
  })

  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url)
    return { action: 'deny' }
  })

  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

function registerIpc(): void {
  ipcMain.handle('engine:connect', () => engine.start())

  ipcMain.handle('dialog:openVideo', async () => {
    const r = await dialog.showOpenDialog({
      properties: ['openFile'],
      filters: [{ name: 'Videos', extensions: VIDEO_EXTS }]
    })
    return r.canceled ? null : r.filePaths[0]
  })

  ipcMain.handle('dialog:openBackground', async () => {
    const r = await dialog.showOpenDialog({
      properties: ['openFile'],
      filters: [{ name: 'Images and videos', extensions: [...IMAGE_EXTS, ...VIDEO_EXTS] }]
    })
    return r.canceled ? null : r.filePaths[0]
  })

  ipcMain.handle('dialog:saveVideo', async (_e, inputPath: string) => {
    const stem = basename(inputPath, extname(inputPath))
    const r = await dialog.showSaveDialog({
      defaultPath: join(dirname(inputPath), `${stem} - vibe.mp4`),
      filters: [{ name: 'MP4 video', extensions: ['mp4'] }]
    })
    return r.canceled ? null : r.filePath
  })

  ipcMain.handle('shell:reveal', (_e, path: string) => shell.showItemInFolder(path))
}

app.whenReady().then(() => {
  electronApp.setAppUserModelId('app.vibebackground')

  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  registerIpc()
  createWindow()

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('will-quit', () => engine.stop())
