# VISION — Golden Demo Launcher (CLI / OpenCV window fallback)
#
# Runs the exact known-good demo configuration:
#   - Video: data/videos/shreyas1.mp4 (proven 100% face recognition)
#   - Zones: configs/zones_demo.yaml (calibrated to this video's actual
#            subject position — fires INTRUSION + LOITERING)
#   - ANPR disabled (no verified-legible plate in any shipped video; avoids
#            any risk of showing fabricated OCR text)
#   - No --db-uri: persistence stays in-memory, no PostgreSQL required
#
# This is the safe fallback demo path if the FastAPI/React dashboard is not
# available or not stable at demo time — it is still live inference, just
# shown in the plain OpenCV window instead of the dashboard.

& ".\.venv\Scripts\python.exe" -m src.main `
    --video data/videos/shreyas1.mp4 `
    --zones configs/zones_demo.yaml `
    --loitering-duration 3 `
    --disable-anpr
