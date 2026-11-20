; Inno Setup script for CamTrack (built by .github/workflows/windows-release.yml)
#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif

[Setup]
AppId={{6C1E3F2A-9B7D-4F5E-A1C3-2D8E7B9F0A41}
AppName=CamTrack
AppVersion={#AppVersion}
AppPublisher=CamTrack
DefaultDirName={autopf}\CamTrack
DefaultGroupName=CamTrack
DisableProgramGroupPage=yes
OutputDir=..\release
OutputBaseFilename=CamTrack-Setup-{#AppVersion}-win64
SetupIconFile=camtrack.ico
UninstallDisplayIcon={app}\CamTrack.exe
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequiredOverridesAllowed=dialog
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\CamTrack\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\CamTrack"; Filename: "{app}\CamTrack.exe"
Name: "{group}\CamTrack (open in browser)"; Filename: "{app}\CamTrack.exe"; Parameters: "--browser"
Name: "{autodesktop}\CamTrack"; Filename: "{app}\CamTrack.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\CamTrack.exe"; Description: "Launch CamTrack"; Flags: nowait postinstall skipifsilent
