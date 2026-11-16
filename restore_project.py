
import os
import shutil
import py_compile

BACKUP_PATH = "/content/drive/MyDrive/GlobalStock_AI/backup/source_code"

FILES = ['app.py', 'fundamentals.py', 'financials.py', 'research_assistant.py', 'comparison.py', 'portfolio_analytics.py', 'market_overview.py', 'sector_analysis.py']

print("========== GLOBALSTOCK AI RECOVERY ==========")

for filename in FILES:
    source = os.path.join(BACKUP_PATH, filename)
    destination = os.path.join("/content", filename)

    if os.path.exists(source):
        shutil.copy2(source, destination)
        py_compile.compile(destination, doraise=True)
        print("RESTORED:", filename)
    else:
        print("MISSING:", filename)

print("\\nRecovery process completed!")
