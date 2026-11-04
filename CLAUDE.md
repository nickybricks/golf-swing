# Swing Lab – Projektkontext

Golf-Schwunganalyse als PWA fürs iPhone (Nick, iPhone 17 Pro Max, Rechtshänder-Erkennung automatisch, Körpergröße 193 cm).
Live: https://nickybricks.github.io/golf-swing/ (GitHub Pages, Branch `main`, Root). Push auf `main` = Deploy nach ca. 1 Min.

## Architektur
- `index.html`: UI (4 Screens: home, session, results, detail), CSS inline
- `app.js`: Kamera (getUserMedia), MediaPipe Pose Landmarker (CDN jsdelivr `@mediapipe/tasks-vision@1.2.0`, Modelle von storage.googleapis), Live-Loop, Ring-Buffer für Bilder, Video-Import (Seek pro Frame), Ergebnis-/Detail-Ansicht, KI-Coach (Claude Messages API direkt aus dem Browser, Header `anthropic-dangerous-direct-browser-access`)
- `analysis.js`: reine Logik ohne DOM – `detectEvents` (P1/P3/P4/P6/P7/P10 aus Handhöhe/-tempo), `computeMetrics`, `CORRIDORS` (Richtwerte), `LiveSegmenter` (wait → address → swing → cooldown)

## Entscheidungen
- Perspektive: Down-the-Line (Kamera hinter dem Spieler, Blick zum Ziel), Handy hochkant, ca. 2,5–3 m Abstand
- Keine Persistenz: Schwünge nur im Speicher; Einstellungen + API-Key in localStorage
- API-Key bleibt vorerst lokal auf dem Handy; Supabase-Proxy (Edge Function + Passwort) ist für später geplant
- Kein Key und keine fremden Videos im Repo (Repo ist öffentlich). Referenz-Clips lädt Nick selbst lokal in der App.
- iOS-Webkamera max. 60 fps; für Zeitlupen-Analyse eigenes Slo-Mo-Video importieren

## Getestet
- Synthetischer Schwung (Node) und echter Clip (Playwright/Chromium mit Fake-Kamera): Phasenerkennung, Live-Segmentierung, Import, Referenzvergleich, KI-Request (gemockt)
- Noch nicht auf dem iPhone/auf der Range getestet. Messwerte schwanken (derselbe Clip: Neigungsverlust −2° vs. −8°) → Trends über mehrere Schwünge betrachten

## Offene Ideen
- Zuschneiden beim Import (Start/Ende wählen) + Zeitlupen-Schalter für korrekte ms-Werte
- Supabase-Proxy für den API-Key
- Speichern von Sessions (IndexedDB oder Supabase)
