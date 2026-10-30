"""Deliberately simple SYNTHETIC fixture; never a real-world benchmark."""

import pandas as pd


def synthetic_data() -> pd.DataFrame:
    phrases = {
        "Billing": ["invoice charge", "payment receipt", "billing statement"],
        "Account Access": ["login password", "account locked", "sign in reset"],
        "Technical Issue": ["application crash", "server error", "software broken"],
    }
    rows = []
    for label, variants in phrases.items():
        for i in range(30):
            rows.append(
                {
                    "ticket_id": f"synthetic-{label}-{i}",
                    "ticket_text": f"Synthetic example {i}: help with {variants[i % 3]} please",
                    "category": label,
                }
            )
    return pd.DataFrame(rows)
