"""
Update GLoRCQ results to Google Sheets.
Uses saved OAuth token from google-docs-mcp.
"""
import json
import os
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

SPREADSHEET_ID = "1Tknzu-nJ9d-yqA1zYAkYBRI-4AZ7NpXX7unSLxpTmdY"
TOKEN_PATH = os.path.expanduser("~/.config/google-docs-mcp/token.json")
CLIENT_ID = os.environ["GOOGLE_CLIENT_ID"]
CLIENT_SECRET = os.environ["GOOGLE_CLIENT_SECRET"]
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

def get_creds():
    with open(TOKEN_PATH) as f:
        token_data = json.load(f)
    creds = Credentials(
        token=token_data.get("access_token"),
        refresh_token=token_data.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        scopes=SCOPES,
    )
    if not creds.valid:
        creds.refresh(Request())
    return creds

def read_zeroshot(json_path):
    with open(json_path) as f:
        d = json.load(f)
    tasks = ["arc_challenge", "arc_easy", "winogrande", "hellaswag", "piqa"]
    results = {}
    for t in tasks:
        r = d["task_results"].get(t, {})
        results[t] = round(r.get("value", 0) * 100, 2)
    results["average"] = round(d["task_results"]["average"]["value"] * 100, 2)
    return results

def append_rows(service, rows):
    body = {"values": rows}
    service.spreadsheets().values().append(
        spreadsheetId=SPREADSHEET_ID,
        range="A1",
        valueInputOption="USER_ENTERED",
        insertDataOption="INSERT_ROWS",
        body=body,
    ).execute()
    print(f"Appended {len(rows)} rows.")

def main():
    creds = get_creds()
    service = build("sheets", "v4", credentials=creds)

    # Read E5 zero-shot results
    e5_zs = read_zeroshot("/home/qyyang/repo/GLoRCQ/logs/zeroshot_e5_ubits8_iter3.json")

    # All experiment data to append
    # Format: [Model, PPL, Bits, ARC-C, ARC-E, WinoG, HellaS, PIQA, Avg, Notes]
    rows = [
        # Section header
        ["=== Explore experiments (base: SOTA PPL=7.62, 3.1931 bits) ==="],
        ["Model", "PPL", "Bits", "ARC-C", "ARC-E", "WinoG", "HellaS", "PIQA", "Avg (5-task)", "Notes"],
        # SOTA baseline
        ["SOTA repro_v2_lora2", 7.62, 3.1931, 41.21, 64.56, 66.54, 71.90, 77.53, 64.35,
         "rank=32,rdown=512,rattn=512,u4/u_attn8/sv8,iter2"],
        # E1
        ["E1 rank=128 gate/up", 7.53, 3.4513, "", "", "", "", "", "",
         "rank=128,rdown=512,rattn=512,u4sv8,iter2"],
        # E2
        ["E2 u_bits=8 MoE", 7.57, 3.2270, "", "", "", "", "", "",
         "rank=32,rdown=512,rattn=512,u8sv8,iter2"],
        # E3
        ["E3 n_lora_iter=3", 7.59, 3.1931, "", "", "", "", "", "",
         "rank=32,rdown=512,rattn=512,u4sv8,iter3"],
        # E4
        ["E4 rank=128+iter3", 7.50, 3.4513, "", "", "", "", "", "",
         "rank=128,rdown=512,rattn=512,u4sv8,iter3"],
        # E5 (final method) — with zero-shot
        ["E5 u8+iter3 (FINAL)", 7.55, 3.2270,
         e5_zs["arc_challenge"], e5_zs["arc_easy"], e5_zs["winogrande"],
         e5_zs["hellaswag"], e5_zs["piqa"], e5_zs["average"],
         "rank=32,rdown=512,rattn=512,u8sv8,iter3 — FINAL METHOD"],
        # Speed
        [],
        ["=== Inference Speed (Qwen1.5-MoE-A2.7B, A100 80G, batch=1) ==="],
        ["Method", "tok/s", "Speedup vs FP16", "GPU Memory", "Notes"],
        ["FP16 (HuggingFace)", 4.6, "1.00×", "~28 GB", "baseline"],
        ["GLoRCQ Standard", 11.3, "2.46×", "~10 GB", "single 24GB GPU deployable"],
        ["GLoRCQ CUDA Graph", 28.3, "6.17×", "~10 GB", "fastest mode"],
    ]

    append_rows(service, rows)
    print("Done! Spreadsheet updated.")
    print(f"E5 zero-shot results: {e5_zs}")

if __name__ == "__main__":
    main()
