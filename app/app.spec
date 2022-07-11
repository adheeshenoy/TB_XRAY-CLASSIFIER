# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules
from os import getcwd, path


hidden_imports = collect_submodules('h5py')

block_cipher = None


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[(path.join(getcwd(), 'unweighted_model.h5'), '.'),
           (path.join(getcwd(), 'weighted_model.h5'), '.'),
           (path.join(getcwd(), 'media/empty.png'), 'media'),
           (path.join(getcwd(), 'media/logo.png'), 'media'),
     ],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='app',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='app',
)
app = BUNDLE(
    coll,
    name='app.app',
    icon=path.join(getcwd(), 'media/large_logo.png'),
    bundle_identifier=None,
)
