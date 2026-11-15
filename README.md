# ⛳ Swing Lab

Golf-Schwunganalyse als Web-App (PWA) fürs iPhone, ohne App Store.

**Funktionen**
- Live-Kamera mit Skelett-Overlay (MediaPipe Pose, läuft komplett auf dem Gerät)
- Ganzkörper-Check: Die Session wird erst scharf geschaltet, wenn Füße, Kopf und Hände im Bild sind
- Automatische Schwungerkennung: Ansprechen → Schwung → Auswertung, mit Piepton
- Schlüsselpositionen P1, P3, P4, P6, P7, P10 als Standbilder
- Kennzahlen mit Richtwert-Korridoren (Ampel): Tempo, Oberkörperneigung, Neigungsverlust (Early Extension), Hüft- und Kopfbewegung, Ellbogen am Top, Schulter-/Hüftdrehung (3D-Schätzung)
- Zeitlupen-Replay mit Skelett
- Video-Import (z. B. Zeitlupe aus der Kamera-App) und Referenz-Schwung (Pro-Clip) zum Nebeneinanderlegen
- KI-Coach: Kennzahlen und Positionsbilder gehen an Claude (eigener API-Key, wird nur lokal gespeichert)

**Aufnahme:** Down-the-Line, Handy hochkant, Linse auf Hüfthöhe, 2–3 m hinter dir auf der Linie Hände → Ziel.

**Hinweise**
- 2D-Messung aus einer Kamera: Die Werte eignen sich für Trends und Vergleiche, sie sind keine Labormessung. Die Werte mit „Schätzung“ sind besonders grob.
- Die Richtwerte sind grobe Orientierungswerte aus Lehrliteratur und Studien zu Tour-Spielern (u. a. GolfTEC SwingTRU).
- Schwünge werden lokal auf dem Gerät gespeichert (IndexedDB) und bleiben nach dem Schließen erhalten.

Dateien: `index.html` (UI), `app.js` (Kamera, Pose, Session, KI), `analysis.js` (Phasenerkennung und Kennzahlen).
