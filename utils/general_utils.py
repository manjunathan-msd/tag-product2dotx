import pandas as pd

def downloadable_url(google_sheet_url):
    base_url = "https://docs.google.com/spreadsheets/d/"
    export_url_suffix = "/export?gid="
    format_suffix = "&format=csv"

    # Extract the document ID and gid from the URL
    try:
        doc_id_start = google_sheet_url.index(base_url) + len(base_url)
        doc_id_end = google_sheet_url.index('/edit')
        doc_id = google_sheet_url[doc_id_start:doc_id_end]
        
        gid_start = google_sheet_url.index('gid=') + len('gid=')
        gid_end = google_sheet_url.index('#', gid_start) if '#' in google_sheet_url[gid_start:] else len(google_sheet_url)
        gid = google_sheet_url[gid_start:gid_end]
    except ValueError:
        raise ValueError("The provided URL is not in the expected format.")

    # Construct the export URL
    export_url = f"{base_url}{doc_id}{export_url_suffix}{gid}{format_suffix}"
    return export_url